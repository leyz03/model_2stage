#!/usr/bin/env python3
"""
build_two_stage_toy.py — scenario pools for the two-stage LIME-OPR toy (model_2stage).

First stage: task prices, chosen before any demand is seen.
Second stage: the whole day (stages 1..T) of OD demand is revealed at once.

So a scenario is a full-day trajectory xi = (d^1, ..., d^T); d^1 is random here
(model_v2 fixed it as the deterministic root).  Deterministic data (network, fleet,
workers, R, C_p, phi) and the Poisson rates are read from reference/model_v2_toy/toy_instance.json
(a frozen copy of the model_v2 toy), never re-typed, so the two-stage toy differs from the v2 toy only in its randomness.

Sampling contract (02_instances.md §2):
  * inverse-CDF, one uniform per draw (also for lambda = 0)  -- same sampler as model_v2
  * one RNG substream per trajectory, seed = pool_seed*100000 + k  -> first K of a pool
    is a valid K-scenario sample, and K-prefixes are nested
  * within a trajectory: t ascending, then (i, j) ascending
  * stdlib random.Random only

Two independent pools per variant: `train` (SAA problems) and `eval` (out-of-sample
evaluation of a fixed price vector).  The JSON is the authoritative instance.

Run:  python3 model_2stage/toy/build_two_stage_toy.py
"""

import importlib.util
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
V2_JSON = os.path.join(HERE, "..", "reference", "model_v2_toy", "toy_instance.json")
V2_BUILDER = os.path.join(HERE, "..", "reference", "model_v2_toy", "build_toy_tree.py")

_spec = importlib.util.spec_from_file_location("build_toy_tree", V2_BUILDER)
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
poisson_icdf = _v2.poisson_icdf                 # the model_v2 sampler, reused as is

POOLS = {"train": (20261003, 1000), "eval": (20261004, 5000)}
VARIANTS = ("toy-A", "toy-B")

# First-stage price menu (integer, so the stability Big-M Q2 stays integral).
# The v2 toy used p = 4 for every task; it is on the menu as the baseline.
PRICE_MENU = list(range(0, 9))
PRICE_BASELINE = 4


def trajectory(seed, lam, n, T):
    rng = random.Random(seed)
    return [[[poisson_icdf(rng.random(), lam[t][i][j]) for j in range(n)]
             for i in range(n)] for t in range(T)]


def main():
    v2 = json.load(open(V2_JSON))
    det = v2["deterministic"]
    n, T, Q3 = det["N"], det["T"], det["B_total"]
    Q1 = det["W_total"]
    Q2 = max(PRICE_MENU) + max(map(max, det["d_ij"])) + max(map(max, det["c_ij"]))

    out = {
        "instance_family": "LIME-OPR two-stage toy (prices first, full-day demand second)",
        "spec": "model_2stage/01_model.md, model_2stage/02_instances.md",
        "deterministic_source": "reference/model_v2_toy/toy_instance.json (deterministic block, read at load time)",
        "sampler": {"kind": "poisson-inverse-cdf", "rng": "python random.Random",
                    "uniforms_per_draw": 1,
                    "trajectory_substream_seed": "pool_seed*100000 + k (k = 1..K)",
                    "draw_order": "per trajectory: t, then i, j ascending",
                    "pools": {name: {"seed": s, "size": m} for name, (s, m) in POOLS.items()}},
        "pricing": {"menu": PRICE_MENU, "baseline": PRICE_BASELINE,
                    "structures": {"uniform": "one price for every task",
                                   "type": "one price for moves (j!=k), one for swaps (j=k)",
                                   "od": "one price per (j,k)"}},
        "constants": {"Q1": Q1, "Q2": Q2, "Q3": Q3},
        "variants": {},
    }

    for var in VARIANTS:
        lam = v2["variants"][var]["lambda_ijt"]
        entry = {"lambda_ijt": lam, "pools": {}}
        for pool, (seed, size) in POOLS.items():
            scen = [trajectory(seed * 100000 + k, lam, n, T) for k in range(1, size + 1)]
            # (SC-5): every node's total demand fits the fleet, in every scenario and stage
            worst = max(sum(x[t][i]) for x in scen for t in range(T) for i in range(n))
            if worst > Q3:
                raise SystemExit("(SC-5) violated in %s/%s: node demand %d > Q3 = %d"
                                 % (var, pool, worst, Q3))
            dbar = [[[max(x[t][i][j] for x in scen) for j in range(n)] for i in range(n)]
                    for t in range(T)]
            mean_total = sum(sum(map(sum, x[t])) for x in scen for t in range(T)) / size
            entry["pools"][pool] = {"scenarios": scen, "d_bar": dbar, "max_node_demand": worst,
                                    "mean_daily_trips": round(mean_total, 4)}
            print("%-6s %-5s  K=%4d  max node demand %2d <= Q3=%d  mean trips/day %.2f"
                  % (var, pool, size, worst, Q3, mean_total))
        out["variants"][var] = entry

    path = os.path.join(HERE, "two_stage_toy.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()

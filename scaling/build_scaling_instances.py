#!/usr/bin/env python3
"""
build_scaling_instances.py — synthetic two-stage instances for the n / T / K_s scaling study.

    python3 model_2stage/scaling/build_scaling_instances.py            # default grid
    python3 model_2stage/scaling/build_scaling_instances.py --n 3 5 --T 4 8

One JSON per (n, T): model_2stage/scaling/instances/syn_n{n}_T{T}.json, readable by
load_ts_instance (it carries its own `deterministic` block).

Family (densities per node are held fixed, so only the size changes):
  * n nodes uniform in the unit square (network seed depends on n only, so the T sweep
    shares one network);  dist = Euclidean / sqrt(2) in [0, 1]
  * rides  c_ij = ceil(3 dist) in {1,2,3} (i != j);  c_ii = 3 (swap / loop time, as the toy)
  * worker travel  d_ij = ceil(2 dist) in {1,2} (i != j);  d_ii = 0
  * fleet B = 4n (4 bikes per node, all on level 1), workers W = 2n (2 per node)
  * R = 10, C_p = 5, phi = 0.15 on every OD (the toy economics)
  * demand: 1.4 trips per node per period on average, two peaks (morning / evening) with
    reversed direction: lambda_ij^t ∝ o_i(t) e_j(t) exp(-dist_ij / 0.4), lambda_ii = 0,
    o = a, e = b in the first half of the day and swapped in the second half
  * price menu 0 .. max d + max c + 4  (the toy's 0..8 rule: max threshold + 4)

Scenarios: full-day trajectories, sampled exactly as model_2stage/toy (inverse-CDF Poisson,
one uniform per draw, one substream per trajectory, seed = pool_seed*100000 + k, so the
first K of a pool is a K-scenario sample).  Pools: train (200) and eval (500).
"""

import argparse
import importlib.util
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "build_toy_tree", os.path.join(HERE, "..", "reference", "model_v2_toy", "build_toy_tree.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
poisson_icdf = _v2.poisson_icdf

BASE_SEED = 20261004
TRAIN, EVAL = 200, 500
RATE = 1.4


def network(n):
    rng = random.Random(BASE_SEED * 100 + n)
    xy = [(rng.random(), rng.random()) for _ in range(n)]
    a = [0.5 + rng.random() for _ in range(n)]          # origin weight (morning)
    b = [0.5 + rng.random() for _ in range(n)]          # destination weight (morning)
    dist = [[math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) / math.sqrt(2)
             for j in range(n)] for i in range(n)]
    c = [[3 if i == j else max(1, math.ceil(3 * dist[i][j])) for j in range(n)] for i in range(n)]
    d = [[0 if i == j else max(1, math.ceil(2 * dist[i][j])) for j in range(n)] for i in range(n)]
    return xy, a, b, dist, c, d


def rates(n, T, a, b, dist):
    prof = []
    for t in range(T):
        tau = (t + 0.5) / T
        prof.append(0.6 + 0.8 * math.exp(-((tau - 0.25) / 0.12) ** 2)
                    + 0.8 * math.exp(-((tau - 0.75) / 0.12) ** 2))
    mean = sum(prof) / T
    lam = []
    for t in range(T):
        o, e = (a, b) if (t + 0.5) / T < 0.5 else (b, a)
        w = [[0.0 if i == j else o[i] * e[j] * math.exp(-dist[i][j] / 0.4) for j in range(n)]
             for i in range(n)]
        tot = sum(map(sum, w))
        scale = RATE * n * prof[t] / mean / tot
        lam.append([[round(w[i][j] * scale, 6) for j in range(n)] for i in range(n)])
    return lam


def pool(seed, size, lam, n, T):
    out = []
    for k in range(1, size + 1):
        rng = random.Random(seed * 100000 + k)
        out.append([[[poisson_icdf(rng.random(), lam[t][i][j]) for j in range(n)]
                     for i in range(n)] for t in range(T)])
    return out


def build(n, T):
    xy, a, b, dist, c, d = network(n)
    lam = rates(n, T, a, b, dist)
    B, W = 4 * n, 2 * n
    menu = list(range(0, max(map(max, d)) + max(map(max, c)) + 5))
    det = {"N": n, "T": T, "B_total": B, "W_total": W, "A0": [4] * n, "U0": [0] * n, "W0": [2] * n,
           "R_ij": [[10.0] * n for _ in range(n)], "C_p": 5.0, "d_ij": d, "c_ij": c,
           "phi_ij": [[0.15] * n for _ in range(n)]}
    pools = {}
    for name, (s, m) in {"train": (BASE_SEED + 1, TRAIN), "eval": (BASE_SEED + 2, EVAL)}.items():
        scen = pool(s * 1000 + 10 * n + T, m, lam, n, T)
        worst = max(sum(x[t][i]) for x in scen for t in range(T) for i in range(n))
        if worst > B:
            raise SystemExit("(SC-5) violated for n=%d T=%d: node demand %d > %d" % (n, T, worst, B))
        pools[name] = {"scenarios": scen, "max_node_demand": worst,
                       "mean_daily_trips": round(sum(sum(map(sum, x[t])) for x in scen
                                                     for t in range(T)) / m, 3)}
    out = {"instance_family": "LIME-OPR two-stage synthetic scaling family",
           "spec": "model_2stage/scaling/build_scaling_instances.py (docstring)",
           "deterministic": det, "coordinates": xy,
           "pricing": {"menu": menu, "baseline": 4},
           "constants": {"Q1": W, "Q3": B},
           "variants": {"syn": {"lambda_ijt": lam, "pools": pools}}}
    os.makedirs(os.path.join(HERE, "instances"), exist_ok=True)
    path = os.path.join(HERE, "instances", "syn_n%d_T%d.json" % (n, T))
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("n=%2d T=%2d  B=%3d W=%2d  menu 0..%d  trips/day %.1f  max node demand %d  %.1f MB"
          % (n, T, B, W, menu[-1], pools["train"]["mean_daily_trips"], pools["train"]["max_node_demand"],
             os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[3, 5, 8, 12, 16, 20])
    ap.add_argument("--T", type=int, nargs="+", default=[4, 8, 12])
    args = ap.parse_args()
    for n in args.n:
        for T in args.T:
            build(n, T)

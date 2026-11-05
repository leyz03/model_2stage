#!/usr/bin/env python3
"""
build_6min_instances.py — synthetic instances at the real time resolution (LAGRANGIAN_TD.md §3).

    python3 model_2stage/fixed_price/build_6min_instances.py --n 5 8 10 --T 24 36 48 60

Writes model_2stage/fixed_price/instances/syn_n{n}_T{T}_m6.json (read with --tag _m6).
One period = 6 minutes (adjacent-zone rides average 6.7 min).  The horizon is the first T periods
of the day from 06:00, so the model's start state (fresh fleet on the top battery level, no bike or
worker in transit, no open task) is the start of operations.

  * network: the scaling family's n nodes (scaling/build_scaling_instances.py, seed depends on n only)
  * rides: a ride to the nearest zone takes 6.7 min on average, so minutes_ij = 6.7 dist_ij / (mean
    nearest-zone distance) and c_ij = round(minutes_ij / 6) clipped to 1..4 periods (96.5% of real
    trips end within 24 min); swaps c_ii = 2 (12 min)
  * worker travel d_ij = round(2/3 minutes_ij / 6) clipped to 1..3, d_ii = 0 (the family's d ≈ 2c/3)
  * fleet 4 bikes and 2 workers per node, R = 10, C_p = 5, phi = 0.15 per ride (the family's economics)
  * demand: TRIPS trips per bike per day (default 4) spread over the 240 periods of the day with
    the profile 0.15 at night (22:00-06:00), 0.6 by day, +0.8 Gaussian peaks (sd 1.5 h) at 08:00
    and 18:00; directions o = a, e = b before 13:00 and reversed after (the family's two-peak
    reversal); lambda_ij ∝ o_i e_j exp(-dist_ij / 0.4), lambda_ii = 0
  * menu 0 .. max d + max c + 4
Scenarios are sampled as in the family (inverse-CDF Poisson, one substream per trajectory);
pools train (20) and eval (20).
"""

import argparse
import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "build_scaling_instances", os.path.join(HERE, "..", "scaling", "build_scaling_instances.py"))
fam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fam)

DAY, START = 240, 60          # periods per day, first period of the horizon (06:00)


def profile(t):
    h = (t + 0.5) * 24 / DAY
    base = 0.6 if 6 <= h < 22 else 0.15
    return base + 0.8 * math.exp(-((h - 8) / 1.5) ** 2) + 0.8 * math.exp(-((h - 18) / 1.5) ** 2)


def build(n, T, trips):
    xy, a, b, dist, _, _ = fam.network(n)
    near = sum(min(dist[i][j] for j in range(n) if j != i) for i in range(n)) / n
    mins = [[6.7 * dist[i][j] / near for j in range(n)] for i in range(n)]
    c = [[2 if i == j else min(4, max(1, round(mins[i][j] / 6))) for j in range(n)] for i in range(n)]
    d = [[0 if i == j else min(3, max(1, round(2 * mins[i][j] / 18))) for j in range(n)] for i in range(n)]
    total = trips * 4 * n                                   # trips per day
    prof = [profile(t) for t in range(DAY)]
    scale_day = total / sum(prof)
    lam = []
    for t in range(START, START + T):
        h = (t + 0.5) * 24 / DAY
        o, e = (a, b) if h < 13 else (b, a)
        w = [[0.0 if i == j else o[i] * e[j] * math.exp(-dist[i][j] / 0.4) for j in range(n)] for i in range(n)]
        tot = sum(map(sum, w))
        lam.append([[round(w[i][j] / tot * prof[t] * scale_day, 6) for j in range(n)] for i in range(n)])
    Bt, W = 4 * n, 2 * n
    det = {"N": n, "T": T, "B_total": Bt, "W_total": W, "A0": [4] * n, "U0": [0] * n, "W0": [2] * n,
           "R_ij": [[10.0] * n for _ in range(n)], "C_p": 5.0, "d_ij": d, "c_ij": c,
           "phi_ij": [[0.15] * n for _ in range(n)]}
    pools = {}
    for name, (s, m) in {"train": (fam.BASE_SEED + 11, 20), "eval": (fam.BASE_SEED + 12, 20)}.items():
        scen = fam.pool(s * 1000 + 10 * n + T, m, lam, n, T)
        worst = max(sum(x[t][i]) for x in scen for t in range(T) for i in range(n))
        if worst > Bt:
            raise SystemExit("(SC-5) violated: node demand %d > %d" % (worst, Bt))
        pools[name] = {"scenarios": scen, "max_node_demand": worst,
                       "mean_trips": round(sum(sum(map(sum, x[t])) for x in scen for t in range(T)) / m, 3)}
    menu = list(range(0, max(map(max, d)) + max(map(max, c)) + 5))
    out = {"instance_family": "LIME-OPR two-stage synthetic, 6-minute periods from 06:00",
           "spec": "model_2stage/fixed_price/build_6min_instances.py (docstring)",
           "deterministic": det, "coordinates": xy,
           "pricing": {"menu": menu, "baseline": 4},
           "constants": {"Q1": W, "Q3": Bt, "minutes_per_period": 6, "start": "06:00", "trips_per_bike_day": trips},
           "variants": {"syn": {"lambda_ijt": lam, "pools": pools}}}
    os.makedirs(os.path.join(HERE, "instances"), exist_ok=True)
    path = os.path.join(HERE, "instances", "syn_n%d_T%d_m6.json" % (n, T))
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    mean_c = sum(c[i][j] for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    print("n=%2d T=%2d (06:00-%02d:%02d)  trips in horizon %.1f (%.2f per bike)  mean ride %.2f periods  "
          "c 1..%d  d 1..%d  %s" % (n, T, (START + T) * 6 // 60, (START + T) * 6 % 60,
                                    pools["train"]["mean_trips"], pools["train"]["mean_trips"] / Bt, mean_c,
                                    max(map(max, c)), max(map(max, d)), os.path.basename(path)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[5, 8, 10])
    ap.add_argument("--T", type=int, nargs="+", default=[24, 36, 48, 60])
    ap.add_argument("--trips", type=float, default=4.0, help="trips per bike per day")
    args = ap.parse_args()
    for n in args.n:
        for T in args.T:
            build(n, T, args.trips)

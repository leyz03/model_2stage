#!/usr/bin/env python3
"""Summarize Lagrangian time-decomposition runs (results/<job>/lagrangian_td.jsonl) against the paired
:pre / tier runs and the historical feasible values on the same scenarios.

    python3 model_2stage/fixed_price/summarize_lag.py <lag job...> [--tier=<job>...] [--history=<job>...]
                                                      [--ref=<job>...]

Defaults: --tier = 39696957 39696956 39697587 39697971, --history = 39662998 39664108.
--ref adds run_ref_bounds.jl jobs (whole-day LP, time-limited tier MILP, relax-and-fix) for any instance
family (key includes the instance tag, e.g. _m6).
Best LB = best feasible value of any method (tier, :pre, relax-and-fix, exact, Lagrangian forward pass).
OPT is shown when :pre or tier proved optimality.  Gaps are (UB - best LB) / best LB.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
args = sys.argv[1:]
tier_jobs = [a.split("=", 1)[1] for a in args if a.startswith("--tier=")] or \
    ["39696957", "39696956", "39697587", "39697971"]
hist_jobs = [a.split("=", 1)[1] for a in args if a.startswith("--history=")] or ["39662998", "39664108"]
ref_jobs = [a.split("=", 1)[1] for a in args if a.startswith("--ref=")]
lag_jobs = [a for a in args if not a.startswith("--")]

lbs = defaultdict(list)            # (n, T, price, s) -> feasible values
opt = {}
tier = {}                          # (n, T, price, s) -> tier row
for job in tier_jobs:
    p = RESULTS / job / "tier_compare.jsonl"
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        r = json.loads(line)
        key = (r["n"], r["T"], "", r["price"], r["scenario"])
        if r["feasible"] and r["value"] is not None:
            lbs[key].append(r["value"])
        if r["optimal"]:
            opt[key] = r["value"]
        if r["method"] == "tier":
            tier[key] = r
for job in hist_jobs:
    p = RESULTS / job / "fixed_price.jsonl"
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("mode") != "td" or r.get("phi", "instance") != "instance" or r.get("value") is None:
            continue
        key = (r["n"], r["T"], "", r["price"], r["scenario"])
        if r["method"].startswith("rf_") and r.get("ok", True) and not r.get("failed"):
            lbs[key].append(r["value"])
        if r["method"] == "ex":
            lbs[key].append(r["value"])
            if r.get("optimal"):
                opt[key] = r["value"]

tier_tl = {}
for job in ref_jobs:
    for line in (RESULTS / job / "ref_bounds.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["n"], r["T"], r.get("tag", ""), r["price"], r["scenario"])
        if r["method"] == "tier":
            tier[key] = r
            tier_tl[key] = r.get("tl")
            if r.get("feasible") and r.get("value") is not None:
                lbs[key].append(r["value"])
            if r.get("optimal"):
                opt[key] = r["value"]
        elif r["method"].startswith("rf_") and r.get("ok") and r.get("value") is not None:
            lbs[key].append(r["value"])

lag = defaultdict(dict)            # (n, T, tag, price, s) -> variant -> row
for job in lag_jobs:
    for line in (RESULTS / job / "lagrangian_td.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (r["n"], r["T"], r.get("tag", ""), r["price"], r["scenario"])
        lag[key][r["variant"]] = r
        for v in (r.get("fwd"), r.get("fwd_myopic")):
            if v is not None and r.get("fwd_ok", True):
                lbs[key].append(v)

fmt = lambda v, d=2: "–" if v is None else f"{v:.{d}f}"
gap = lambda ub, lb: None if ub is None or lb is None else max(0.0, 100 * (ub - lb) / max(1.0, abs(lb)))
mean = lambda xs: statistics.mean(xs) if xs else None

variants = sorted({v for d in lag.values() for v in d})
print("Per scenario (gaps in % of the best LB; tier = utility-tier MILP with its time limit, 1 thread):")
print("| n | T | inst | s | variant | B | nλ | best LB | OPT | LP gap | tier gap (s) | Lag@300 gap | Lag final gap | Lag s | it | stop | fwd gap | myopic gap |")
print("|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
agg = defaultdict(lambda: defaultdict(list))
for key in sorted(lag, key=lambda k: (k[2], k[1], k[0], k[3], k[4])):
    n, T, tag, price, s = key
    lb = max(lbs[key]) if lbs[key] else None
    t = tier.get(key)
    tub = t["bound"] if t else None
    for v in variants:
        r = lag[key].get(v)
        if r is None:
            continue
        lpg, tg = gap(r["lp_whole"], lb), gap(tub, lb)
        g3, gf = gap(r["ub_300"], lb), gap(r["ub"], lb)
        fg = None if r["fwd"] is None else 100 * (lb - r["fwd"]) / max(1, abs(lb))
        mg = None if r.get("fwd_myopic") is None else 100 * (lb - r["fwd_myopic"]) / max(1, abs(lb))
        ts = None if t is None else t.get("solve_s")
        print(f"| {n} | {T} | {tag or 'syn'} | {s} | {v} | {r['B']} | {r['nlambda']} | {fmt(lb, 0)} | "
              f"{fmt(opt.get(key), 0)} | {fmt(lpg)} | {fmt(tg)} ({fmt(ts, 0)}) | {fmt(g3)} | {fmt(gf)} | "
              f"{fmt(r['lag_wall'], 0)} | {r['it']} | {r.get('stop', 'converged' if r['converged'] else '?')} | "
              f"{fmt(fg)} | {fmt(mg)} |")
        a = agg[(n, T, tag, price, v)]
        closed = None if lpg is None or gf is None or lpg <= 1e-9 else max(0.0, 100 * (lpg - gf) / lpg)
        for name, val in (("lp", lpg), ("tier", tg), ("l300", g3), ("lf", gf), ("fwd", fg), ("myo", mg),
                          ("wall", r["lag_wall"]), ("it", r["it"]), ("bs", r["block_max_s"]), ("closed", closed),
                          ("ts", ts), ("load", max(r["boundary_load"]) if r.get("boundary_load") else None)):
            if val is not None:
                a[name].append(val)
        a["conv"].append(1 if r["converged"] else 0)
        a["tier_beaten"].append(1 if tub is not None and r["ub"] < tub - 1e-6 else 0)

print("\nMeans per configuration and variant:")
print("| n | T | inst | variant | scen | max boundary load | LP gap | tier gap | tier s | Lag@300 gap | Lag final gap | LP gap closed % | Lag < tier UB | converged | mean s | mean it | max block s | fwd gap | myopic gap |")
print("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for (n, T, tag, price, v), a in sorted(agg.items(), key=lambda k: (k[0][2], k[0][1], k[0][0], k[0][4])):
    print(f"| {n} | {T} | {tag or 'syn'} | {v} | {len(a['conv'])} | {fmt(max(a['load']) if a['load'] else None, 1)} | "
          f"{fmt(mean(a['lp']))} | {fmt(mean(a['tier']))} | {fmt(mean(a['ts']), 0)} | "
          f"{fmt(mean(a['l300']))} | {fmt(mean(a['lf']))} | {fmt(mean(a['closed']), 0)} | {sum(a['tier_beaten'])} | "
          f"{sum(a['conv'])} | {fmt(mean(a['wall']), 0)} | {fmt(mean(a['it']), 0)} | "
          f"{fmt(max(a['bs']) if a['bs'] else None, 1)} | {fmt(mean(a['fwd']))} | {fmt(mean(a['myo']))} |")

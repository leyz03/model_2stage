#!/usr/bin/env python3
"""Summarize paired runs; optional --history=<job> adds prior valid lower bounds from fixed_price.jsonl."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
paired = defaultdict(dict)
history = defaultdict(list)
for job in (arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--history=")):
    for line in (RESULTS / job / "fixed_price.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("mode") == "td" and (row["method"].startswith("rf_") and row.get("ok") or
                                         row["method"] == "ex" and row.get("value") is not None):
            if row.get("value") is not None:
                history[(row["n"], row["T"], row["price"], row["scenario"])].append(row["value"])
for job in (arg for arg in sys.argv[1:] if not arg.startswith("--history=")):
    for line in (RESULTS / job / "tier_compare.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["n"], row["T"], row["price"], row["scenario"], job)
        paired[key][row["method"]] = row

groups = defaultdict(list)
for (n, T, price, scenario, job), d in paired.items():
    if set(d) == {"pre", "tier"}:
        groups[(n, T, price)].append((scenario, job, d["pre"], d["tier"]))

fmt = lambda v: "–" if v is None else f"{v:.2f}"
avg = lambda vals: statistics.mean(vals) if vals else None

print("| n | T | price | paired | equal opt | pre opt | tier opt | pre solver s | tier solver s | pre wall s | tier wall s | pre/tier solver | bins pre/tier |")
print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for key, rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
    eq = sum(a["optimal"] and b["optimal"] and abs(a["value"] - b["value"]) <= 1e-6 * max(1, abs(a["value"]))
             for _, _, a, b in rows)
    ap, at = avg([a["solve_s"] for _, _, a, _ in rows]), avg([b["solve_s"] for _, _, _, b in rows])
    wp, wt = avg([a["wall_s"] for _, _, a, _ in rows]), avg([b["wall_s"] for _, _, _, b in rows])
    bins = f"{rows[0][2]['nbin']}/{rows[0][3]['nbin']}"
    print(f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} | {eq} | "
          f"{sum(a['optimal'] for _, _, a, _ in rows)} | {sum(b['optimal'] for _, _, _, b in rows)} | "
          f"{fmt(ap)} | {fmt(at)} | {fmt(wp)} | {fmt(wt)} | {fmt(ap / at if at else None)} | {bins} |")

print("\nUnresolved scenarios: common LB = best incumbent plus optional historical feasible value; common-LB gaps compare upper bounds directly.")
print("| n | T | price | s | job | common LB | pre LB | pre UB | pre own gap % | tier LB | tier UB | tier own gap % | pre/tier common-LB gap % |")
print("|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
for (n, T, price), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])):
    for scenario, job, a, b in rows:
        if a["optimal"] and b["optimal"]:
            continue
        lbs = [r["value"] for r in (a, b) if r["value"] is not None]
        lbs += history[(n, T, price, scenario)]
        common = max(lbs) if lbs else None
        cg = [max(0, 100 * (r["bound"] - common) / max(1, abs(common)))
              if common is not None and r["bound"] is not None else None for r in (a, b)]
        print(f"| {n} | {T} | {price} | {scenario} | {job} | {fmt(common)} | "
              f"{fmt(a['value'])} | {fmt(a['bound'])} | {fmt(100*a['gap'] if a['gap'] is not None else None)} | "
              f"{fmt(b['value'])} | {fmt(b['bound'])} | {fmt(100*b['gap'] if b['gap'] is not None else None)} | "
              f"{fmt(cg[0])}/{fmt(cg[1])} |")

#!/usr/bin/env python3
"""
summarize.py — tables from the scaling runs (model_2stage/results/<job>/scaling.jsonl).

    python3 model_2stage/scaling/summarize.py 39618904 39618905 ...     # job ids (default: all)
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def load(jobs):
    files = [os.path.join(RES, j, "scaling.jsonl") for j in jobs] if jobs else \
        sorted(glob.glob(os.path.join(RES, "*", "scaling.jsonl")))
    rows = []
    for f in files:
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def oracle_table(rows):
    rs = [r for r in rows if r["mode"] == "oracle"]
    if not rs:
        return
    print("\n## per-scenario MILP at p = 4 (solver seconds per MILP, mean of the scenarios; binaries before presolve)\n")
    print("| n | T | full s | pre s | lazy s | relax s | bin full | bin pre | lazy cuts | relax - exact | not optimal |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by = {}
    for r in rs:
        by.setdefault((r["n"], r["T"]), {})[r["form"]] = r
    for (n, T) in sorted(by):
        d = by[(n, T)]
        g = lambda f, k, fmt="%.3f": (fmt % d[f][k]) if f in d and d[f].get(k) is not None else "-"
        nopt = sum(d[f].get("not_optimal", 0) for f in d)
        ex = next((d[f]["mean_value"] for f in ("full", "pre") if f in d and d[f].get("mean_value") is not None), None)
        gap = ("%.2f" % (d["relax"]["mean_value"] - ex)) if "relax" in d and ex is not None else "-"
        print("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
            n, T, g("full", "solver_s_mean"), g("pre", "solver_s_mean"), g("lazy", "solver_s_mean"),
            g("relax", "solver_s_mean"), g("full", "binaries", "%d"), g("pre", "binaries", "%d"),
            g("lazy", "lazy_cuts_mean", "%.1f"), gap, nopt))


def solve_table(rows):
    rs = [r for r in rows if r["mode"] in ("bnb", "ef")]
    if not rs:
        return
    print("\n## two-stage solves (`type` prices; time limit per run in column tl)\n")
    print("| n | T | K_s | method | status | value | ub | gap % | seconds | nodes | exact pts | exact MILPs | relax MILPs | exact solver s | relax solver s | not opt |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rs, key=lambda r: (r["n"], r["T"], r["nscen"], r["mode"], r.get("inner", ""))):
        meth = "bnb-%s" % r["inner"] if r["mode"] == "bnb" else "EF"
        v = r["value"] if r["value"] is not None else float("nan")
        ub = r["ub"] if r["ub"] is not None else float("nan")
        gap = 100 * (ub - v) / abs(v) if v == v and v != 0 else float("nan")
        sec = r["seconds"] + r.get("build_s", 0.0)
        if r["mode"] == "bnb":
            extra = "%d | %d | %d | %d | %.0f | %.0f | %d" % (
                r["nodes"], r["exact_points"], r["exact_milps"], r["relax_milps"], r["exact_solver_s"],
                r["relax_solver_s"], r.get("exact_not_optimal", 0) + r.get("relax_not_optimal", 0))
        else:
            extra = "- | - | - | - | - | - | -"
        print("| %d | %d | %d | %s | %s | %.3f | %.3f | %.2f | %.0f | %s |" % (
            r["n"], r["T"], r["nscen"], meth, r["status"], v, ub, gap, sec, extra))


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    oracle_table(rows)
    solve_table(rows)

"""summarize_decomp.py — markdown tables of TODO #3 runs (int_flow/run_decomp.jl).  Times are Gurobi solve times
(sums over the solves for cg / benders); cg / benders also show the wall time of the method in parentheses; ✗ = not
proved optimal.

    python3 model_2stage/int_flow/summarize_decomp.py <job> [<job> ...]     job = a directory under int_flow/results
Later jobs override earlier ones for the same (day, T, method).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}
for job in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(HERE, "results", job, "*.json"))):
        try:
            r = json.load(open(f))
        except ValueError:                  # still being written
            print("skipped (unreadable):", f, file=sys.stderr)
            continue
        m = r["method"] + ("" if r.get("fixstep", True) else "nofix")
        runs[(r["day"], r["T"], m)] = r

DAYS = sorted({k[0] for k in runs}, key=lambda d: (d != 11, d != 18, d))
TS = sorted({k[1] for k in runs})


def fmt(x, p=0):
    return "–" if x is None else f"{x:,.{p}f}"


def grb(r):
    """Gurobi time of a run: the solve time (full, lp), the sum over the solves (cg, benders)."""
    if r["method"] in ("full", "lp"):
        return r["method_s"]
    if r["method"] == "cg":
        return sum(l["solve_s"] for l in r["log"])
    return r["master_s"] + r["sub_s"]


def tm(r):
    """Gurobi time; cg / benders also the wall time of the method (Gurobi + JuMP: copy, bounds, reduced costs)."""
    if r is None:
        return "–"
    s = fmt(grb(r), 0) + " s"
    if r["method"] in ("cg", "benders"):
        s += f"（{fmt(r['method_s'], 0)}）"
    return s if r.get("proved") else s + " ✗"


print("| 日期 | T | 最优值 | full | lp | cg（墙钟） | cg 不带 fix 步（墙钟） | benders（墙钟） | cg：LP 轮数 / 最后的列 / 全部列 | cg：LP 段 / 之后各步 | benders：轮数（可行割）/ gap |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for d in DAYS:
    for T in TS:
        g = {m: runs.get((d, T, m)) for m in ("full", "lp", "cg", "cgnofix", "benders")}
        if not any(g.values()):
            continue
        vals = [r["value"] for m, r in g.items() if r and m != "lp" and r.get("proved")]
        best = vals[0] if vals else None
        same = all(abs(v - best) < 1e-6 for v in vals) if vals else False
        info = next(r for r in g.values() if r)["day_info"]
        cg, bd = g["cg"], g["benders"]
        cgs = cgt = bds = "–"
        if cg:
            lp_it = sum(1 for l in cg["log"] if l["phase"] == "lp")
            mips = [l for l in cg["log"] if l["phase"] != "lp"]
            cgs = f"{lp_it} / {cg['active']:,} / {cg['columns']:,}"
            lps = sum(l["solve_s"] for l in cg["log"] if l["phase"] == "lp")
            cgt = f"{lps:.0f} s / " + " + ".join(f"{l['phase']} {l['solve_s']:.0f}" for l in mips) + " s"
        if bd:
            gap = None if bd["bound"] is None or bd["value"] is None else bd["bound"] - bd["value"]
            bds = f"{bd['iterations']}（{bd['feas_cuts']}）/ {fmt(gap, 2)}"
        print(f"| {info['date'][5:]}（{info['U0']}） | {T} | {fmt(best)}{'' if same else ' ≠'} | {tm(g['full'])} | "
              f"{tm(g['lp'])} | {tm(cg)} | {tm(g['cgnofix'])} | {tm(bd)} | {cgs} | {cgt} | {bds} |")

bad = [(k, r.get("frac"), r.get("residual"), r.get("blocking_pairs")) for k, r in runs.items()
       if r["method"] != "lp" and r.get("proved") and
       (any(r.get("frac", {}).values()) or (r.get("residual") or 0) > 1e-6 or (r.get("blocking_pairs") or 0) > 0)]
print()
print("solutions with fractional variables, residual or blocking pairs:", bad if bad else "none")

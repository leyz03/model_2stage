"""summarize_lazy_stab.py — TODO #9: the stable-matching rows on demand (run_decomp.jl --stab lazy; cg also --stab
defer: left out of the LP phase, all added back before the MIP, in a table of its own) next to the same
method with every row (--stab tier, the #3 / #7 runs), one table per fleet scale ρ.

    python3 model_2stage/int_flow/summarize_lazy_stab.py <job> [<job> ...]      job = a directory under int_flow/results
Later jobs override earlier ones for the same (ρ, day, T, method, stab).  Lazy rows count every submission
(a group sent again is counted again), so they are an upper bound on the rows needed.  Cells: tier → lazy.  Times are Gurobi solve
times (sums over the solves for cg; lp lazy: sum over the row-generation rounds), cg also the wall time in
parentheses; ✗ = not proved, followed by the gap (bound − value) / value; cg without a proof has the LP bound.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}


def rho_of(r):
    if "fleet_scale" in r:
        return r["fleet_scale"]
    m = re.search(r"fleet(\d{3})", r.get("instance", ""))
    return int(m.group(1)) / 100 if m else 1.0


def root_lp(logfile):
    """(root LP seconds, root LP objective) from the Gurobi log, or (None, None)."""
    try:
        txt = open(logfile).read()
    except OSError:
        return None, None
    m = re.search(r"Root relaxation: objective ([-+.e\d]+), \d+ iterations, ([.\d]+) seconds", txt)
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)


for job in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(HERE, "results", job, "*.json"))):
        try:
            r = json.load(open(f))
        except ValueError:
            print("skipped (unreadable):", f, file=sys.stderr)
            continue
        if r.get("drop"):
            continue
        m = r["method"] + ("" if r.get("fixstep", True) else "nofix")
        if m not in ("full", "lp", "cg", "cgnofix"):
            continue
        r["_root"] = root_lp(f[:-5] + ".log") if r["method"] == "full" else (None, None)
        runs[(rho_of(r), r["day"], r["T"], m, r.get("stab", "tier"))] = r


def fmt(x, p=0):
    return "–" if x is None else f"{x:,.{p}f}"


def grb(r):
    if r["method"] == "full":
        return r["method_s"]
    if r["method"] == "lp":
        return r.get("lp_solve_s", r["method_s"])
    return sum(l["solve_s"] for l in r["log"])


def cell(r):
    if r is None:
        return "–"
    s = fmt(grb(r)) + " s"
    if r["method"] == "cg":
        s += f"（{fmt(r['method_s'])}）"
    if r["method"] != "lp" and not r.get("proved"):
        v, b = r.get("value"), r.get("bound")
        gap = "" if v is None or b is None or v <= 0 else f" {100 * (b - v) / v:.2f}%"
        s += " ✗" + gap
    return s


def pair(a, b):
    return f"{cell(a)} → {cell(b)}"


for rho in sorted({k[0] for k in runs}, reverse=True):
    keys = [k for k in runs if k[0] == rho]
    days = sorted({k[1] for k in keys}, key=lambda d: (d != 11, d != 18, d))
    Ts = sorted({k[2] for k in keys})
    print(f"\n**ρ = {rho:g}**\n")
    print("| 日期 | T | LP | lp | lp lazy：轮数 / 加的组 | full | full 根节点 LP | full lazy：候选解 / 拒绝 / lazy 行 | cg | cg 不带 fix 步 | cg lazy：LP 段加的组 / 拒绝 / lazy 行（占全部行） |")
    print("|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|")
    for d in days:
        for T in Ts:
            g = {(m, s): runs.get((rho, d, T, m, s)) for m in ("full", "lp", "cg", "cgnofix") for s in ("tier", "lazy")}
            if not any(v for (m, s), v in g.items() if s == "lazy"):
                continue
            info = next(v for v in g.values() if v)["day_info"]
            lp = g[("lp", "tier")] or g[("lp", "lazy")]
            lpv = fmt(lp["value"], 1) if lp else "–"
            ll = g[("lp", "lazy")]
            lls = f"{ll['rounds']} / {ll['stab_stats']['groups_active']:,}" if ll else "–"
            if ll and g[("lp", "tier")] and abs(ll["value"] - g[("lp", "tier")]["value"]) > 1e-6 * max(1, abs(ll["value"])):
                lls += " ≠ LP"
            ft, fl = g[("full", "tier")], g[("full", "lazy")]
            root = f"{fmt(ft['_root'][0]) if ft else '–'} → {fmt(fl['_root'][0]) if fl else '–'} s"
            fls = "–"
            if fl:
                st = fl["stab_stats"]
                fls = f"{st['lazy_calls']:,} / {st['lazy_rejects']:,} / {st['lazy_rows']:,}"
            cl = g[("cg", "lazy")] or g[("cgnofix", "lazy")]
            cls = "–"
            if cl:
                st = cl["stab_stats"]
                lpg = sum(l.get("groups_added", 0) for l in cl["log"] if l["phase"] == "lp")
                cls = f"{lpg:,} / {st['lazy_rejects']:,} / {st['lazy_rows']:,}（{100 * st['lazy_rows'] / st['rows_all']:.0f}%）"
            print(f"| {info['date'][5:]}（{info['U0']}） | {T} | {lpv} | {pair(g[('lp', 'tier')], ll)} | {lls} | "
                  f"{pair(ft, fl)} | {root} | {fls} | {pair(g[('cg', 'tier')], g[('cg', 'lazy')])} | "
                  f"{pair(g[('cgnofix', 'tier')], g[('cgnofix', 'lazy')])} | {cls} |")

# TODO #9 (11-04): cg with the rows left out of the LP phase and added back at once before the MIP (--stab defer)
if any(k[4] == "defer" for k in runs):
    print("\n### cg：原写法 / lazy / defer（LP 段不带稳定匹配行，进 MIP 前全部加回）")
    for rho in sorted({k[0] for k in runs if k[4] == "defer"}, reverse=True):
        keys = [k for k in runs if k[0] == rho and k[4] == "defer"]
        days = sorted({k[1] for k in keys}, key=lambda d: (d != 11, d != 18, d))
        Ts = sorted({k[2] for k in keys})
        print(f"\n**ρ = {rho:g}**\n")
        print("| 日期 | T | 方法 | 用时：原写法 | lazy | defer | LP 段：原写法 / lazy / defer | defer：加回的行 / 用时 | defer：MIP 各步 |")
        print("|---|---:|---|---:|---:|---:|---|---|---|")
        for d in days:
            for T in Ts:
                for m in ("cg", "cgnofix"):
                    rt, rl, rd = (runs.get((rho, d, T, m, s)) for s in ("tier", "lazy", "defer"))
                    if rd is None:
                        continue
                    info = rd["day_info"]
                    lps = " / ".join("–" if r is None else f"{r['lp_s']:,.0f}" for r in (rt, rl, rd)) + " s"
                    ab = [l for l in rd["log"] if l["phase"] == "addback"]
                    abs_ = f"{ab[0]['rows']:,} / {ab[0]['add_s']:.0f} s" if ab else "–"
                    mips = " + ".join(f"{l['phase']} {l['solve_s']:.0f}" for l in rd["log"] if l["phase"] not in ("lp", "addback"))
                    print(f"| {info['date'][5:]}（{info['U0']}） | {T} | {'cg' if m == 'cg' else 'cg 不带 fix 步'} | "
                          f"{cell(rt)} | {cell(rl)} | {cell(rd)} | {lps} | {abs_} | {mips or '–'} |")

bad = []
for k, r in runs.items():
    if k[4] == "tier" or r["method"] == "lp" or r.get("value") is None:
        continue
    if any((r.get("frac") or {}).values()) or (r.get("residual") or 0) > 1e-6 or (r.get("blocking_pairs") or 0) > 0:
        bad.append((k, r.get("frac"), r.get("residual"), r.get("blocking_pairs")))
    t = runs.get(k[:4] + ("tier",))
    if t and r.get("proved") and t.get("proved") and abs(r["value"] - t["value"]) > 1e-6 * max(1, abs(t["value"])):
        bad.append((k, "value", r["value"], "tier", t["value"]))
print()
print("lazy / defer runs with fractional variables, residual, blocking pairs or a proved value ≠ tier:", bad if bad else "none")

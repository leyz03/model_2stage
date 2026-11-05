#!/usr/bin/env python3
"""
summarize.py — tables of the fixed-price experiments from results/<job>/fixed_price.jsonl

    python3 model_2stage/fixed_price/summarize.py <job id> [<job id> ...]

stab rows: price of stability (R1 - exact), how often R1's optimum is already stable (plain / with
           tie-break), solver seconds per method, blocking-pair types of plain R1.
td rows:   per (n, T): bounds (LP, R1 with tie-break, exact), relax-and-fix values and gaps to the best
           upper bound, the warm-started exact MILP.  Gap = (UB - value) / UB, UB = best bound of the
           scenario (exact / warm optimum when proven, else min of the bounds found).
"""
import json
import os
import sys
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))


def load(jobs):
    rows = []
    for j in jobs:
        with open(os.path.join(HERE, "..", "results", j, "fixed_price.jsonl")) as f:
            rows += [json.loads(l) for l in f]
    return rows


def fmt(x, nd=2):
    return "–" if x is None else ("%.*f" % (nd, x))


def stab(rows):
    g = defaultdict(dict)
    for r in rows:
        if r["mode"] == "stab":
            g[(r["n"], r["T"], r["price"])].setdefault(r["method"], {})[r["scenario"]] = r
    print("| n | T | price | K_s | exact | R1 − exact | scen. R1>exact | R1 stable | R1+tb stable "
          "| exact s | R1 s | R1+tb s | exact+R1 cut s | verify s (rounds) | verify+tb s (rounds) | bin exact / R1 |")
    print("|" + "---:|" * 16)
    for key in sorted(g, key=lambda k: (k[0] * k[1], k[0], k[2])):
        d = g[key]
        r0, r1, r1tb = d.get("r0", {}), d.get("r1", {}), d.get("r1tb", {})
        sc = sorted(set(r0) & set(r1))
        both = [s for s in sc if r0[s]["optimal"] and r1[s]["optimal"]]
        pos = [r1[s]["value"] - r0[s]["value"] for s in both]
        ex = mean(r0[s]["value"] for s in both) if both else None
        ks = len(r1 or r0)
        t = lambda m: (mean(x["solve_s"] for x in d[m].values()) if m in d else None)
        tr = lambda m: (mean(x["rounds"] for x in d[m].values()) if m in d else None)
        nopt = lambda m: sum(not x["optimal"] for x in d.get(m, {}).values())
        st1 = sum(x["blocking"] == 0 for x in r1.values())
        st2 = sum(x["blocking"] == 0 for x in r1tb.values())
        flag = lambda m: ("" if nopt(m) == 0 else " (%d TL)" % nopt(m))
        print("| %d | %d | %s | %d | %s | %s (%s%%) | %d/%d | %d/%d | %d/%d | %s%s | %s | %s | %s%s | %s (%s)%s | %s (%s)%s | %s / %s |" % (
            key[0], key[1], key[2], ks, fmt(ex, 1), fmt(mean(pos) if pos else None),
            fmt(100 * mean(pos) / ex if pos else None, 2), sum(p > 1e-6 for p in pos), len(pos),
            st1, len(r1), st2, len(r1tb), fmt(t("r0")), flag("r0"), fmt(t("r1"), 3), fmt(t("r1tb"), 3),
            fmt(t("r0cut")), flag("r0cut"), fmt(t("verify")), fmt(tr("verify"), 1), flag("verify"),
            fmt(t("verifytb")), fmt(tr("verifytb"), 1), flag("verifytb"),
            max((x["nbin"] for x in r0.values()), default="–"), max((x["nbin"] for x in r1.values()), default="–")))
    # blocking types of plain R1 and what R1 does differently when it is worth more
    print("\nplain R1: blocking-pair types (summed over scenarios) and ops of R1 vs exact where R1 > exact")
    for key in sorted(g, key=lambda k: (k[0] * k[1], k[0], k[2])):
        d = g[key]
        tot = defaultdict(int)
        for x in d.get("r1", {}).values():
            for k, v in x["blocking_types"].items():
                tot[k] += v
        diffs = []
        for s, x in d.get("r1", {}).items():
            y = d.get("r0", {}).get(s)
            if y and y["optimal"] and x["value"] - y["value"] > 1e-6 and x.get("ops") and y.get("ops"):
                diffs.append({k: x["ops"][k] - y["ops"][k] for k in x["ops"]})
        dd = {k: round(mean(z[k] for z in diffs), 2) for k in diffs[0]} if diffs else {}
        print("  n=%d T=%d %s  %s  R1-exact ops: %s" % (key[0], key[1], key[2], dict(tot), dd))


def td(rows):
    g = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["mode"] == "td":
            g[(r["n"], r["T"], r["price"])][r["scenario"]][r["method"]] = r
    rfnames = sorted({m for r in rows if r["mode"] == "td" for m in [r["method"]] if m.startswith("rf_")})
    print("\nUB = best proven bound; gaps in % of UB; times = wall seconds (mean over scenarios)")
    head = "| n | T | K_s | LP above UB | R1+tb: gap, s (stable) | exact: gap, s | " + " | ".join(
        n[3:] + ": gap, s" for n in rfnames) + " | warm exact: gap, s |"
    print(head)
    print("|" + "---:|" * (6 + len(rfnames) + 1))
    for key in sorted(g, key=lambda k: (k[0] * k[1], k[0])):
        sc = g[key]
        cells = defaultdict(list)
        for s, d in sc.items():
            bounds = [d[m]["bound"] for m in ("ex", "ws", "r1tb") if m in d and d[m].get("bound") is not None]
            bounds.append(d["lp"]["value"])
            ub = min(bounds)
            cells["lp"].append((d["lp"]["value"] - ub) / ub)          # how far the LP bound is above UB
            for m in ["r1tb", "ex", "ws"] + rfnames:
                if m in d and d[m].get("value") is not None:
                    cells[m].append(((ub - d[m]["value"]) / ub, d[m].get("wall_s"), d[m]))
                elif m in d:
                    cells[m].append((None, d[m].get("wall_s"), d[m]))

        def c(m, show_time=True):
            xs = cells.get(m, [])
            if not xs:
                return "pruned"
            gaps = [x[0] for x in xs if x[0] is not None]
            out = fmt(100 * mean(gaps), 2) if len(gaps) == len(xs) else "%s (%d/%d)" % (
                fmt(100 * mean(gaps), 2) if gaps else "–", len(gaps), len(xs))
            if show_time:
                out += ", %s" % fmt(mean(x[1] for x in xs if x[1] is not None), 1)
            if m in ("ex", "ws"):
                no = sum(not x[2]["optimal"] for x in xs)
                out += "" if no == 0 else " [%d TL]" % no
            return out
        r1 = cells.get("r1tb", [])
        r1s = "%s (%d/%d)" % (c("r1tb"), sum(x[2].get("blocking") == 0 for x in r1), len(r1)) if r1 else "pruned"
        print("| %d | %d | %d | %s | %s | %s | %s | %s |" % (
            key[0], key[1], len(sc), fmt(100 * mean(cells["lp"]), 1), r1s, c("ex"),
            " | ".join(c(m) for m in rfnames), c("ws")))


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    if any(r["mode"] == "stab" for r in rows):
        stab(rows)
    if any(r["mode"] == "td" for r in rows):
        td(rows)

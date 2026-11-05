"""summarize_alt_price.py — tables of a TODO #4 job (run_alt_price.jl / slurm_alt_price.sh): per T, the uniform prices,
every alternating run, the best of each, the upper bound and the gap.

    .venv/bin/python -I model_2stage/int_flow/summarize_alt_price.py <job id> [<job id> ...]

Merges every results/<job>/alt_T*.json of the given jobs (one per array task).  Profits are the mean over the days;
moves, swaps, wages and lost riders are summed over the days.
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(jobs):
    by_t = defaultdict(list)
    for job in jobs:
        for f in sorted(glob.glob(os.path.join(HERE, "results", job, "alt_T*.json"))):
            with open(f) as io:
                d = json.load(io)
            by_t[d["T"]].append(d)
    return by_t


def fmt(x, nd=2):
    return "—" if x is None else f"{x:,.{nd}f}"


def num(x):
    return str(round(x) + 0)


def main(jobs):
    for T, ds in sorted(load(jobs).items()):
        uni, runs, ub = {}, [], None
        for d in ds:
            for u in d["uniform"]:
                uni[u["price"]] = u
            runs += d["runs"]
            ub = d["ub"] if d.get("ub") else ub
        days = ds[0]["days"]
        print(f"\n## T = {T}, days {days}\n")
        print("| uniform price | profit | moves | swaps | wages | lost | solve s | optimal |")
        print("|---:|---:|---:|---:|---:|---:|---:|---|")
        for p in sorted(uni):
            u = uni[p]
            print(f"| {p:g} | {fmt(u['value'])} | {num(u['moves'])} | {num(u['swaps'])} | {num(u['wages'])} | "
                  f"{num(u['lost'])} | {num(u['solve_s'])} | {u['optimal']} |")
        best_u = max(uni.values(), key=lambda u: u["value"]) if uni else None
        if runs:
            print("\n| start | unused | O steps | converged | profit | vs best uniform | moves | wages | "
                  "ODs used | their prices (level×count) | monotone |")
            print("|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|")
            for r in sorted(runs, key=lambda r: (r["start"], r["unused"])):
                last = r["trace"][-1]
                final = [pr for _, _, pr in r["used_prices"]]
                levels = ", ".join(f"{pr:g}×{final.count(pr)}" for pr in sorted(set(final)))
                mono = all(t.get("monotone", True) and t.get("start_ok", True) for t in r["trace"])
                gain = r["value"] - best_u["value"] if best_u else None
                print(f"| {r['start']:g} | {r['unused']} | {len(r['trace']) - 1} | {r['converged']} | "
                      f"{fmt(r['value'])} | {fmt(gain)} | {num(last['moves'])} | {num(last['wages'])} | "
                      f"{len(final)} | {levels or '—'} | {mono} |")
        best_r = max(runs, key=lambda r: r["value"]) if runs else None
        print()
        if best_u:
            print(f"- best uniform: {fmt(best_u['value'])} at price {best_u['price']:g} "
                  f"(wages {best_u['wages']:.0f} over the days)")
        if best_r:
            print(f"- best alternating: {fmt(best_r['value'])} from start {best_r['start']:g} ({best_r['unused']})"
                  + (f", {fmt(best_r['value'] - best_u['value'])} over the best uniform" if best_u else ""))
        if ub:
            print(f"- upper bound: {fmt(ub['value'])} (per day {', '.join(fmt(v) for v in ub['per_day'])}; "
                  f"optimal {all(i['optimal'] for i in ub['info'])})")
            if best_r:
                g = ub["value"] - best_r["value"]
                print(f"- gap: {fmt(g)} ({100 * g / abs(ub['value']):.4f}% of the bound)"
                      + (f"; room above the best uniform {fmt(ub['value'] - best_u['value'])}" if best_u else ""))


if __name__ == "__main__":
    main(sys.argv[1:])

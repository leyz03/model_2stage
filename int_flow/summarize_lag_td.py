"""summarize_lag_td.py — TODO #10: one evaluation of the Lagrangian time decomposition (int_flow/run_lag_td.jl).

    python3 -I model_2stage/int_flow/summarize_lag_td.py <lp job> <block job> [--full JOB/TAG] [--sol JOB/TAG ...]

L(λ) = Σ_k (proven bound of block k) at λ from the joint LP, per block form.  "L at t" sums the bounds every block
had after t seconds of its own Gurobi time (read from the B&B table of its log; '–' while some block has none yet),
i.e. what that block time limit would have given with the blocks run in parallel.
--full: a whole-day run (results/JOB/TAG.json + .log) whose bound over time is the reference;
--sol: runs whose best value is the incumbent the gaps are taken against.  gap = (bound − incumbent) / incumbent.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TIMES = (60, 300, 600, 900, 1200, 1800)


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def bounds_over_time(path):
    """(time, bound) of the B&B table lines of a Gurobi log (bound None when shown as '-')."""
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path, errors='ignore'):
        m = re.search(r'\s(\d+)s$', ln.rstrip())
        tok = ln.split()
        gi = next((k for k, t in enumerate(tok) if t.endswith('%')), None)
        if m and gi is not None and gi >= 2:
            b = num(tok[gi - 1])
            b is not None and out.append((int(m.group(1)), b))
    return out


def bound_at(tab, t):
    b = None
    for s, v in tab:
        if s <= t:
            b = v
    return b


def opt_arg(flag):
    out, args = [], sys.argv[1:]
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            out.append(args[i + 1])
    return out


def main():
    lpjob, bjob = sys.argv[1], sys.argv[2]
    lam = sorted(glob.glob(os.path.join(RES, lpjob, 'lambda_*.json')))
    if not lam:
        sys.exit(f'no lambda json in results/{lpjob}')
    L = json.load(open(lam[0]))
    print(f"{L['instance']}, day {L['day']}, T {L['T']}, {L['blocks']} blocks {L['cuts']}, fleet {L['fleet']}")
    print(f"joint LP {L['lp']:,.1f} in {L['lp_s']:.0f} s; multipliers per boundary {L['nlam']}")

    inc = None
    for ref in opt_arg('--sol'):
        r = json.load(open(os.path.join(RES, ref + '.json')))
        if r.get('value') is not None:
            inc = r['value'] if inc is None else max(inc, r['value'])
    full = opt_arg('--full')
    if inc is not None:
        print(f"incumbent (best of --sol) {inc:,.0f}; LP gap {(L['lp'] - inc) / inc * 100:.3f}%")
    if full:
        tab = bounds_over_time(os.path.join(RES, full[0] + '.log'))
        row = []
        for t in (600, 1800, 3600, 7200):
            b = bound_at(tab, t)
            row.append(f"{t} s {b:,.1f}" + (f" ({(b - inc) / inc * 100:.3f}%)" if inc and b else '') if b else f"{t} s –")
        print('whole-day reference ' + full[0] + ': bound ' + '; '.join(row))

    blocks = {}
    for f in glob.glob(os.path.join(RES, bjob, '*_k*_*.json')):
        r = json.load(open(f))
        blocks.setdefault(r['form'], {})[r['k']] = (r, f[:-5] + '.log')
    B = L['blocks']
    print()
    hdr = ['form', 'L(λ)', 'gap', 'L − LP', 'proved', 'max block s', 'wall (LP + max)'] + [f'L at {t} s' for t in TIMES]
    print(' | '.join(hdr))
    for form in ('full', 'xbin', 'bin', 'lp'):
        if form not in blocks:
            continue
        bl = blocks[form]
        miss = [k for k in range(1, B + 1) if k not in bl]
        if miss:
            print(f'{form}: blocks {miss} missing')
            continue
        tot = sum(bl[k][0]['bound'] for k in range(1, B + 1))
        mx = max(bl[k][0]['solve_s'] for k in range(1, B + 1))
        npr = sum(bl[k][0]['proved'] for k in range(1, B + 1))
        at = []
        for t in TIMES:
            if form == 'lp':
                s = tot if mx <= t else None
            else:
                parts = []
                for k in range(1, B + 1):
                    r, log = bl[k]
                    b = r['bound'] if r['proved'] and r['solve_s'] <= t else bound_at(bounds_over_time(log), t)
                    parts.append(b)
                s = sum(parts) if all(p is not None for p in parts) else None
            at.append(f'{s:,.1f}' + (f' ({(s - inc) / inc * 100:.3f}%)' if inc else '') if s is not None else '–')
        gap = f'{(tot - inc) / inc * 100:.3f}%' if inc else '–'
        print(' | '.join([form, f'{tot:,.1f}', gap, f'{tot - L["lp"]:,.1f}', f'{npr}/{B}', f'{mx:.0f}',
                          f'{L["lp_s"] + mx:.0f}'] + at))
        for k in range(1, B + 1):
            r = bl[k][0]
            print(f"    k{k} ({r['ta']}–{r['tb']}): {r['status']}, bound {r['bound']:,.1f}, value "
                  f"{r['value'] if r['value'] is None else format(r['value'], ',.1f')}, {r['solve_s']:.0f} s, "
                  f"nodes {r['nodes']}, in/out states {r['nin']}/{r['nout']}")


if __name__ == '__main__':
    main()

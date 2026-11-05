"""Table of TODO #2 runs (run_partial_int.jl): one row per (day, T), value and time of each form, gaps to the exact optimum.

    python3 model_2stage/int_flow/summarize_partial_int.py <results dir> [<results dir> ...]
"""
import json
import sys
from pathlib import Path

FORMS = ['full', 'xbin', 'x', 'lp']


def cell(r):
    if r is None:
        return 'not run'
    v = r.get('value')
    if v is None:
        return f"{r['status']} (no solution)"
    s = f"{v:,.1f} ({r['solve_s']:,.0f} s)"
    if not r['optimal']:
        g = r.get('gap')
        s += f" {r['status']}, gap {100 * g:.3f}%" if g is not None else f" {r['status']}"
    return s


def pct(a, b):
    return f'{100 * (a - b) / abs(b):+.3f}%' if a is not None and b else '—'


def main():
    runs = {}
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob('d*_T*_*.json')):
            r = json.loads(f.read_text())
            runs[(r['day'], r['T'], r['form'])] = r
    keys = sorted({(d, T) for d, T, _ in runs}, key=lambda k: (runs.get(k + ('full',), runs.get(k + ('lp',)))
                                                               ['day_info']['U0'], k[1]))
    print('| day (U0) | T | full | xbin | x | lp | xbin − full | x − full | lp − full | xbin y fractional before → after cleanup | full with x of x-only |')
    print('|---|---:|---|---|---|---|---:|---:|---:|---|---|')
    for d, T in keys:
        rr = {f: runs.get((d, T, f)) for f in FORMS}
        info = next(r for r in rr.values() if r)['day_info']
        val = {f: (rr[f] or {}).get('value') if (rr[f] or {}).get('optimal') else None for f in FORMS}
        exact = val['full'] if val['full'] is not None else val['xbin']
        cl = (rr['xbin'] or {}).get('cleanup')
        cls = f"{cl['frac_before']} → {cl['frac_after']}, {cl['value']:,.1f}" if cl else '—'
        fx = (rr['x'] or {}).get('fixedx')
        fxs = '—' if not fx else (f"{fx['value']:,.1f} ({fx['status']})" if fx.get('value') is not None
                                  else fx['status'])
        print(f"| {info['date'][5:]} ({info['U0']}) | {T} | " + ' | '.join(cell(rr[f]) for f in FORMS) +
              f" | {pct(val['xbin'], val['full'])} | {pct(val['x'], exact)} | {pct(val['lp'], exact)} | {cls} | {fxs} |")


if __name__ == '__main__':
    main()

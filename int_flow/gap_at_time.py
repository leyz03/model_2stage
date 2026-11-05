"""gap_at_time.py — incumbent, bound and gap of the decomp runs (int_flow/run_decomp.jl) at a given time, read from the
Gurobi logs: what a shorter time limit would have given.

    python3 model_2stage/int_flow/gap_at_time.py <seconds> <job: full lp cg> [<job: cg --fixstep 0>]

full: the B&B table lines with Gurobi time <= seconds.  cg / cg without the fix step: the log has one segment per
solve, in the order of the json log; the time of a solve starts after the earlier ones (solve + reduced-cost time,
the JuMP overhead between solves left out).  Until it proves optimality the bound of cg is the full LP value, known
once the LP phase is done; the incumbent is the best of the start (nothing_start!) and the solves so far.
opt = proved optimal within the time (wall time of the method for cg); gap = (bound - incumbent) / incumbent.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = {11: '06-11', 18: '06-18', 1: '06-01'}


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def table(lines):
    """(time, incumbent, bound) of the B&B table lines; incumbent / bound None when shown as '-'."""
    out = []
    for ln in lines:
        m = re.search(r'\s(\d+)s$', ln.rstrip())
        tok = ln.split()
        gi = next((k for k, t in enumerate(tok) if t.endswith('%')), None)
        if m and gi is not None and gi >= 2:
            out.append((int(m.group(1)), num(tok[gi - 2]), num(tok[gi - 1])))
    return out


def segments(path):
    segs, cur = [], None
    for ln in open(path):
        if ln.startswith('Gurobi Optimizer version'):
            cur = []
            segs.append(cur)
        elif cur is not None:
            cur.append(ln)
    return segs


def at_full(path, t):
    inc = bnd = None
    for s, i, b in table(open(path)):
        if s <= t:
            inc = i if i is not None else inc
            bnd = b if b is not None else bnd
    return inc, bnd


def at_cg(r, path, t):
    segs, log = segments(path), r['log']
    assert len(segs) == len(log), f'{path}: {len(segs)} solves in the log, {len(log)} in the json'
    off, inc, ub = 0.0, r.get('start'), None
    for seg, l in zip(segs, log):
        if off > t:
            break
        if l['phase'] == 'lp':
            ub = l['value'] if off + l['solve_s'] + l['rc_s'] <= t else ub
        elif off + l['solve_s'] <= t:
            inc = max(inc, l['value']) if l.get('value') is not None else inc
        else:
            inc = max([inc] + [i for s, i, _ in table(seg) if off + s <= t and i is not None])
        off += l['solve_s'] + l.get('rc_s', 0.0)
    return inc, ub


def cell(job, tag, t):
    js, lg = (os.path.join(HERE, 'results', job, tag + e) for e in ('.json', '.log'))
    if not os.path.exists(js):
        return 'no result'                      # cg: the LP phase ran out of time (error, no json)
    r = json.load(open(js))
    if r['proved'] and r['method_s'] <= t:
        return f"opt {r['method_s']:,.0f} s"
    inc, bnd = at_full(lg, t) if r['method'] == 'full' else at_cg(r, lg, t)
    if inc is None or bnd is None:
        return f'incumbent {inc}, no bound'
    return f'{inc:,.0f} / {bnd:,.1f}  {100 * (bnd - inc) / abs(inc):.3f}%'


def main():
    t, jobs = float(sys.argv[1]), sys.argv[2:]
    runs = [('full', jobs[0], ''), ('cg', jobs[0], '')] + ([('cg nofix', jobs[1], 'nofix')] if len(jobs) > 1 else [])
    have = {json.load(open(os.path.join(HERE, 'results', jobs[0], f)))['day']
            for f in os.listdir(os.path.join(HERE, 'results', jobs[0])) if f.endswith('_full.json')}
    print(f'at {t:,.0f} s: incumbent / bound  gap')
    print('| 日期 | T | ' + ' | '.join(m for m, _, _ in runs) + ' |')
    print('|---|---:|' + '---:|' * len(runs))
    for d in sorted(have, key=lambda d: (d != 11, d != 18, d)):
        for T in (10, 20, 50, 100):
            if not os.path.exists(os.path.join(HERE, 'results', jobs[0], f'd{d:02d}_T{T:03d}_full.json')):
                continue
            cells = [cell(job, f"d{d:02d}_T{T:03d}_{'full' if m == 'full' else 'cg'}{suf}", t) for m, job, suf in runs]
            print(f"| {DATE.get(d, d)} | {T} | " + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()

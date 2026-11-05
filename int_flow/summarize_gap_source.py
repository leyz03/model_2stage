"""summarize_gap_source.py — markdown tables of TODO #8 (where the LP bound is loose): the variants of
int_flow/run_decomp.jl (--drop S | M | SM, --method nox | bin) next to the full model of the same instance.

    python3 model_2stage/int_flow/summarize_gap_source.py <job> [<job> ...]     job = a directory under int_flow/results
Runs are grouped by fleet scale (from the instance), day, T, method and drop; later jobs override earlier ones.
LP − MIP: one number when the MIP is proved optimal, else LP − bound … LP − incumbent (the gap lies in between).
Only (day, T) with a variant run are shown; root columns are read from the Gurobi log next to each json.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = {11: '06-11', 18: '06-18', 1: '06-01'}
VARIANTS = [('', 'full', '原模型'), ('S', 'full', '去掉服务规则'), ('M', 'full', '去掉稳定匹配'),
            ('SM', 'full', '两者都去掉（网络核）'), ('', 'nox', 'x 连续（任务可拆给多个工人）'),
            ('', 'bin', '只有 β、z 整数（捆绑全部放开）')]


def rho_of(r):
    if 'fleet_scale' in r:
        return r['fleet_scale']
    m = re.search(r'_fleet(\d+)\.json$', r['instance'])
    return int(m.group(1)) / 100 if m else 1.0


runs = {}
for job in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(HERE, 'results', job, '*.json'))):
        try:
            r = json.load(open(f))
        except ValueError:
            print('skipped (unreadable):', f, file=sys.stderr)
            continue
        if r['method'] not in ('full', 'lp', 'nox', 'bin'):
            continue
        r['_log'] = f[:-5] + '.log'
        runs[(rho_of(r), r['day'], r['T'], r['method'], r.get('drop', ''))] = r


def root(r):
    """(root relaxation after presolve, bound at the end of the root cut loop) from the Gurobi log."""
    rel = bd = None
    if r is None or not os.path.exists(r['_log']):
        return rel, bd
    for line in open(r['_log']):
        m = re.match(r'Root relaxation: objective (\S+),', line)
        if m:
            rel = float(m.group(1))
        tok = line.lstrip('H*').split()
        if tok[:2] == ['0', '0']:
            g = next((k for k, v in enumerate(tok) if v.endswith('%')), None)
            if g:
                bd = float(tok[g - 1])
    return rel, bd


def fmt(x, p=0):
    return '–' if x is None else f'{x:,.{p}f}'


def gap(lp, r):
    if r is None or lp is None or r.get('value') is None:
        return '–'
    if r.get('proved'):
        return fmt(lp - r['value'], 1)
    return f"{fmt(lp - r['bound'], 1)} … {fmt(lp - r['value'], 1)}"


for rho in sorted({k[0] for k in runs}, reverse=True):
    print(f'\n**ρ = {rho}**\n')
    print('| 日期 | T | 变体 | LP | 根松弛 | 根节点割后 | MIP 解 | MIP 界 | LP − MIP | 用时（MIP / LP） | 丢掉的乘客 | '
          '服务规则违反 | blocking pair |')
    print('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    keys = sorted({(k[1], k[2]) for k in runs if k[0] == rho and (k[4] or k[3] in ('nox', 'bin'))},
                  key=lambda dt: (dt[0] != 11, dt[0] != 18, dt))
    for d, T in keys:
        first = True
        for drop, meth, name in VARIANTS:
            mip = runs.get((rho, d, T, meth, drop))
            lpr = runs.get((rho, d, T, 'lp', drop))
            if mip is None and (lpr is None or meth != 'full'):
                continue
            lp = lpr['value'] if lpr else None
            t = (fmt(mip['method_s']) + ' s' + ('' if mip.get('proved') else ' ✗') if mip else '–') + \
                (' / ' + fmt(lpr['method_s']) + ' s' if lpr else '')
            ops = (mip or {}).get('ops') or {}
            rel, bd = root(mip)
            print(f"| {DATE.get(d, d) if first else ''} | {T if first else ''} | {name} | {fmt(lp, 1)} | "
                  f"{fmt(rel, 1)} | {fmt(bd, 1)} | "
                  f"{fmt(mip and mip.get('value'))} | {fmt(mip and mip.get('bound'), 1)} | {gap(lp, mip)} | {t} | "
                  f"{fmt(ops.get('lost'))} | {fmt(mip and mip.get('service_violations'))} | "
                  f"{fmt(mip and mip.get('blocking_pairs'))} |")
            first = False

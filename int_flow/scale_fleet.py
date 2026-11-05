"""Smaller fleet for TODO #7 (a tighter instance): from a build_int_instance.py instance, write the same instance with
every morning state scaled by --rho.  Demand, the rider split, workers, prices, R and C_p are unchanged.

    .venv/bin/python -I model_2stage/int_flow/scale_fleet.py --instance <int json> --rho 0.5 --out <new json>

Rounding (largest remainder, per vector): a vector a (one scenario's A0 or U0 over the nodes) becomes b with
sum(b) = round(rho * sum(a)): b = floor(rho * a), then one more bike to the nodes with the largest fractional parts
(ties to the lower node index).  The fleet of scenario s (Q3 of its view) is sum(A0[s]) + sum(U0[s]) of the result.
The 30-day means in `deterministic` (unused under the integer rider split) are scaled the same way.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scale(a, rho):
    a = np.asarray(a, dtype=float)
    x = rho * a
    b = np.floor(x).astype(np.int64)
    left = int(round(rho * a.sum())) - int(b.sum())
    order = np.argsort(-(x - b), kind='stable')
    b[order[:left]] += 1
    assert b.sum() == round(rho * a.sum()) and (b >= 0).all()
    return [int(v) for v in b]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--instance', required=True, help='build_int_instance.py instance json (variant paris, pool train)')
    ap.add_argument('--rho', type=float, required=True, help='fleet scale, 0 < rho <= 1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not 0 < a.rho <= 1:
        ap.error('--rho must lie in (0, 1]')
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')

    raw = json.loads(Path(a.instance).read_text())
    det, pool, split = raw['deterministic'], raw['variants']['paris']['pools']['train'], raw['int_split']
    old = [sum(x) + sum(y) for x, y in zip(pool['A0'], pool['U0'])]
    pool['A0'] = [scale(x, a.rho) for x in pool['A0']]
    pool['U0'] = [scale(x, a.rho) for x in pool['U0']]
    fleet = [sum(x) + sum(y) for x, y in zip(pool['A0'], pool['U0'])]
    split['fleet_per_scenario'] = fleet

    det['A0'] = scale(det['A0'], a.rho)
    det['U0'] = scale(det['U0'], a.rho)
    det['A0_by_level'] = [scale(x, a.rho) for x in det['A0_by_level']]
    det['B_total'] = sum(det['A0']) + sum(det['U0'])
    raw['constants']['Q3'] = det['B_total']

    # (SC-5) of load_ts_instance: no node demand above the fleet
    d = np.array(pool['scenarios'], dtype=np.int64)
    assert (d.sum(axis=3).max(axis=(1, 2)) <= np.array(fleet)).all(), '(SC-5): a node demand exceeds the fleet'

    raw['fleet_scale'] = {
        'rho': a.rho, 'base_instance': str(Path(a.instance)), 'base_sha256': sha(a.instance),
        'rule': 'each scenario A0 / U0 vector scaled by rho, largest-remainder rounding (sum = round(rho * sum))',
        'fleet_per_scenario_before': old,
    }
    raw['instance_family'] += f' | fleet scaled by {a.rho} (int_flow/scale_fleet.py)'
    out.write_text(json.dumps(raw, separators=(',', ':')))
    print(f'{out}: rho {a.rho}, fleet per scenario {min(fleet)}-{max(fleet)} (was {min(old)}-{max(old)})')


if __name__ == '__main__':
    main()

"""Scale of phi_ij from the swap counts (model_2stage TODO #1): one factor alpha for every OD such that the expected
number of rides ending with a dead bike per day, sum_ij alpha phi_ij rides_ij, equals the done battery swaps inside the
instance's domain during its horizon, averaged over its days.  build_int_instance.py --calibration <out> applies it.

    .venv/bin/python model_2stage/int_flow/calibrate_phi_swaps.py --instance <paris_n30_H10.json> --out <summary json>
                                                                  [--basis all|workers]

Swaps: task_type BatterySwapTask, vehicle_type electric, resolution_status done, collection cell (H3 r9) in the domain
(labels.csv of the instance's partition run), collection hour in [start, start + horizon).  --basis all counts every
swapper (juicer, ops and riders who swap themselves); workers leaves out the rider swaps.  The change of the low-battery
stock over the horizon is not counted.  Rides are the instance demand (completed in-domain rides).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / 'UC Berkeley Data Sharing' / 'paris_tasks_2025-06.csv'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--instance', required=True, help='build_instances.py instance json (variant paris, pool train)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--basis', choices=['all', 'workers'], default='all')
    ap.add_argument('--start-hour', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')

    raw = json.loads(Path(a.instance).read_text())
    det, pool = raw['deterministic'], raw['variants']['paris']['pools']['train']
    n, T, dt = det['N'], det['T'], raw['constants']['minutes_per_period']
    hours = T * dt / 60
    assert hours == int(hours)
    lo, hi = a.start_hour, a.start_hour + int(hours)
    dates = pool['dates']
    d = np.array(pool['scenarios'], dtype=np.int64)                       # (S, T, n, n)
    phi = np.array(det['phi_ij'], dtype=float)
    rides = d.sum(axis=(1, 2, 3))
    expected = np.einsum('ij,sij->s', phi, d.sum(axis=1))                  # sum_ij phi_ij rides_ij per day

    run = ROOT / raw['source']['partition_run']
    core = set(pd.read_csv(run / 'labels.csv', usecols=['h3_r9']).h3_r9)
    f = pd.read_csv(TASKS, usecols=['task_type', 'vehicle_type', 'resolution_status', 'collector_type',
                                    'collection_timestamp_local', 'collection_h3_r9'])
    f = f[f.task_type.eq('BatterySwapTask') & f.vehicle_type.eq('electric') & f.resolution_status.eq('done')]
    t = pd.to_datetime(f.collection_timestamp_local)
    keep = f.collection_h3_r9.isin(core) & t.dt.hour.between(lo, hi - 1) & t.dt.strftime('%Y-%m-%d').isin(dates)
    f, t = f[keep], t[keep]
    day = t.dt.strftime('%Y-%m-%d')
    rider = f.collector_type.eq('rider')
    per_day = pd.DataFrame({'all': day.value_counts(), 'workers': day[~rider].value_counts()}).reindex(dates).fillna(0)
    swaps = per_day[a.basis].to_numpy(float)
    scale = float(swaps.mean() / expected.mean())
    alt = 'workers' if a.basis == 'all' else 'all'
    assert scale * phi.max() <= 1, 'scaled phi exceeds 1'
    summary = {
        'instance': str(Path(a.instance)), 'instance_sha256': hashlib.sha256(Path(a.instance).read_bytes()).hexdigest(),
        'tasks': str(TASKS.relative_to(ROOT)), 'window_hours': [lo, hi], 'basis': a.basis, 'phi_scale': scale,
        'phi_scale_other_basis': {alt: float(per_day[alt].mean() / expected.mean())},
        'swaps_per_day_mean': {k: float(per_day[k].mean()) for k in ('all', 'workers')},
        'swaps_by_collector_type': f.collector_type.value_counts().to_dict(),
        'expected_dead_per_day_mean_before': float(expected.mean()), 'rides_per_day_mean': float(rides.mean()),
        'ride_weighted_phi_before': float(expected.sum() / rides.sum()),
        'ride_weighted_phi_after': float(scale * expected.sum() / rides.sum()),
        'per_day': {dates[s]: {'rides': int(rides[s]), 'expected_dead_before': float(expected[s]),
                               'swaps_all': int(per_day['all'].iloc[s]), 'swaps_workers': int(per_day['workers'].iloc[s])}
                    for s in range(len(dates))},
        'not_counted': 'change of the low-battery stock over the window; swaps outside the domain or the window'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f'{out}: window {lo}:00-{hi}:00, {len(dates)} days, rides/day {rides.mean():.0f}')
    print(f'  swaps/day: all {per_day["all"].mean():.1f}, workers {per_day["workers"].mean():.1f}; '
          f'expected dead/day before {expected.mean():.1f}')
    print(f'  basis {a.basis}: phi scale {scale:.4f} (ride-weighted phi {summary["ride_weighted_phi_before"]:.4f} -> '
          f'{summary["ride_weighted_phi_after"]:.4f}); basis {alt}: {summary["phi_scale_other_basis"][alt]:.4f}')


if __name__ == '__main__':
    main()

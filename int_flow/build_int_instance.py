"""Integer rider split for model_2stage TODO #1 (no battery levels, every trip flow integer): from a build_instances.py
instance, write an instance whose riders are split into d = d^A + d^U (d^U: the bike ends the ride dead) and whose every
scenario starts from its own day's 08:00 state.  load_ts_instance(path, "paris"; levels = 2) reads it.

    .venv/bin/python model_2stage/int_flow/build_int_instance.py --instance <paris_n30_H10.json> \
        --initial-state <n30/initial_state.npz> --out <new json> [--seed 20261031] [--calibration <calibrate_phi_swaps json>]

Split rule (common random numbers): u[t, i, j, k] ~ U(0, 1) is drawn once with --seed for every period t, OD (i, j) and
rider k = 1..max demand; in every scenario rider k of (i, j, t) ends dead iff u[t, i, j, k] < phi_ij, so d^U is the
number of k <= d_ij^t with u < phi_ij.  Equal demand counts give equal splits in every scenario: only the demand and the
morning state are random.  phi_ij = the instance's phi_ij x --phi-scale (or the scale of --calibration).  The split is
frozen in the output, so it does not depend on the numpy version.

Morning state of scenario s (initial_state.npz of the same build, one entry per day): A0 = rentable bikes of all levels,
U0 = low-battery candidates; bikes in a trip at 08:00 are left out, as in the base instance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--instance', required=True, help='build_instances.py instance json (variant paris, pool train)')
    ap.add_argument('--initial-state', required=True, help='initial_state.npz of the same build (n<N>/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--phi-scale', type=float, default=1.0)
    ap.add_argument('--calibration', default=None, help='calibrate_phi_swaps.py summary json: use its phi_scale')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    scale, cal = a.phi_scale, None
    if a.calibration:
        if a.phi_scale != 1.0:
            ap.error('--calibration sets the scale; do not pass --phi-scale too')
        cal = json.loads(Path(a.calibration).read_text())
        if Path(cal['instance']).resolve() != Path(a.instance).resolve():
            ap.error(f"--calibration was computed on {cal['instance']}, not on --instance")
        scale = cal['phi_scale']

    raw = json.loads(Path(a.instance).read_text())
    det, pool = raw['deterministic'], raw['variants']['paris']['pools']['train']
    n, T = det['N'], det['T']
    d = np.array(pool['scenarios'], dtype=np.int64)                       # (S, T, n, n)
    S = d.shape[0]
    assert d.shape == (S, T, n, n) and (d >= 0).all()
    phi = np.array(det['phi_ij'], dtype=float) * scale
    assert phi.shape == (n, n) and (phi >= 0).all() and (phi <= 1).all(), 'phi_ij x scale must lie in [0, 1]'

    # common random numbers: one uniform per (period, OD, rider index), shared by every scenario
    Dmax = int(d.max())
    u = np.random.default_rng(a.seed).random((T, n, n, Dmax))
    cum = np.concatenate([np.zeros((T, n, n, 1), dtype=np.int64),
                          np.cumsum(u < phi[None, :, :, None], axis=-1)], axis=-1)   # dead among the first k riders
    dU = np.stack([np.take_along_axis(cum, d[s][..., None], axis=-1)[..., 0] for s in range(S)])
    assert (dU >= 0).all() and (dU <= d).all()

    st = np.load(a.initial_state, allow_pickle=False)
    dates = [str(x) for x in st['dates']]
    assert dates == pool['dates'], 'initial_state.npz days differ from the scenario days'
    A0 = st['rentable_by_level'].sum(axis=1).astype(np.int64)             # (S, n): every rentable level
    U0 = st['low_battery'].astype(np.int64)                               # (S, n)
    assert A0.shape == (S, n) and U0.shape == (S, n)
    fleet = A0.sum(1) + U0.sum(1)
    node_D = d.sum(axis=3)                                                # (S, T, n)
    assert (node_D <= fleet[:, None, None]).all(), '(SC-5) a node demand exceeds its day fleet'

    rides = d.sum(axis=(1, 2, 3))
    pool['scenarios_U'] = dU.tolist()
    pool['A0'] = A0.tolist()
    pool['U0'] = U0.tolist()
    det['phi_ij'] = phi.tolist()
    raw['instance_family'] += ' | integer rider split, daily morning state (model_2stage/int_flow)'
    raw['assumptions']['A0'] = 'per scenario: rentable bikes of all levels at 08:00 of that day (pool.A0); ' \
                               'deterministic.A0 is the 30-day mean of the base instance, unused'
    raw['assumptions']['U0'] = 'per scenario: low-battery candidates at 08:00 of that day (pool.U0)'
    raw['assumptions']['dead_riders'] = 'pool.scenarios_U: riders whose bike ends dead, common random numbers ' \
                                        '(int_split.rule)'
    raw['int_split'] = {
        'base_instance': str(Path(a.instance)), 'base_sha256': sha(a.instance),
        'initial_state': str(Path(a.initial_state)), 'initial_state_sha256': sha(a.initial_state),
        'seed': a.seed, 'rng': f'numpy {np.__version__} default_rng(seed).random((T, n, n, Dmax)), Dmax = {Dmax}',
        'rule': 'rider k of OD (i, j) in period t (k = 1..d_ij^t) ends with a dead bike iff u[t, i, j, k] < phi_ij; '
                'every scenario uses the same u',
        'phi_scale': scale, 'calibration': a.calibration, 'in_trip_at_08_left_out': st['in_trip'].sum(1).tolist(),
        'fleet_per_scenario': fleet.tolist(), 'rides_per_scenario': rides.tolist(),
        'dead_riders_per_scenario': dU.sum(axis=(1, 2, 3)).tolist(),
        'expected_dead_riders_per_scenario': [float((phi * d[s].sum(0)).sum()) for s in range(S)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw, separators=(',', ':')))
    ex = np.mean(raw['int_split']['expected_dead_riders_per_scenario'])
    print(f'{out}: {S} scenarios, n = {n}, T = {T}, Dmax = {Dmax}, phi scale {scale:.4f}')
    print(f'  rides/day {rides.mean():.1f}; dead riders/day {dU.sum(axis=(1, 2, 3)).mean():.1f} (expected {ex:.1f})')
    print(f'  A0/day {A0.sum(1).min()}-{A0.sum(1).max()}, U0/day {U0.sum(1).min()}-{U0.sum(1).max()}, '
          f'fleet {fleet.min()}-{fleet.max()}')


if __name__ == '__main__':
    main()

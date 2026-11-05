#!/usr/bin/env python3
"""
build_toy_tree.py — LIME-OPR v2 toy instance + fixed Poisson scenario tree.

Provenance script.  It samples the stage-wise scenario tree ONCE from the
Poisson rates, freezes it, derives every instance-dependent constant the v2
specification needs, verifies the exactness conditions, and writes
`toy_instance.json`.  That JSON is the authoritative instance; Julia/Python
solvers read it and never re-sample.

Tree construction follows Zou, Ahmed & Sun (2019), Sec. 3.2 / 6:
  * stage 1 is the deterministic root: one realization with probability 1
    (d^1 = round-half-up of lambda^1, an instance input, not sampled)
  * stages 2..T: N_b i.i.d. joint realizations each, probability 1/N_b,
    recombining stage-wise independent tree with N_b^(T-1) scenarios

Sampling contract (see 02_scenario_tree.md):
  * one uniform per draw, inverse-CDF   -> stream consumption independent of lambda
  * one RNG substream per stage t       -> the N_b=10 tree is the prefix of N_b=20
  * within a stage: atom l, then (i, j) ascending
  * stdlib `random.Random` only         -> no numpy/scipy version sensitivity

Run:  python3 model_v2/toy/build_toy_tree.py
"""

import json
import math
import os
import random
from fractions import Fraction

# ---------------------------------------------------------------------------
# 1. Deterministic part of the toy instance (inherited from `new model/`)
# ---------------------------------------------------------------------------

N = 3
T = 4

B_TOTAL = 12
A0 = [4, 4, 4]
U0 = [0, 0, 0]

W_TOTAL = 6
W0 = [1, 2, 3]

R = [[10.0] * N for _ in range(N)]          # R_ij
P_JK = [[4.0] * N for _ in range(N)]        # p_jk (fixed parameter in v2)
C_P = 5.0

D_MAT = [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]                          # d_ij : worker deadhead time

C_MAT = [[3, 1, 1],
         [1, 3, 1],
         [1, 1, 3]]                          # c_ij : ride / task execution time

PHI = [[0.15] * N for _ in range(N)]         # phi_ij : battery failure rate

# Poisson rates lambda[t][i][j], 1-indexed stages flattened to 0-indexed lists.
# Off-diagonal means reproduce the demand scale of the `new model/` toy;
# the diagonal is the only structural knob between the two variants below.
LAMBDA_OFFDIAG = {
    "early": [[None, 0.33, 2.00],            # stages 1..3
              [0.33, None, 1.00],
              [0.33, 0.33, None]],
    "late":  [[None, 1.50, 0.50],            # stage 4
              [0.33, None, 0.33],
              [0.50, 0.33, None]],
}

VARIANTS = {
    # toy-A: no closed-loop demand.  Minimal state -- the instance to run SDDiP on.
    "toy-A": {"lambda_diag": 0.0,
              "note": "no self-loop demand; customer pipeline tail r>=2 is provably empty"},
    # toy-B: small closed-loop demand.  Exercises c_ii=3 rides and the r>=2
    # pipeline buckets.  Stage-model verification instance, not an SDDiP run.
    "toy-B": {"lambda_diag": 0.15,
              "note": "self-loop demand on; exercises r>=2 customer pipeline"},
}

BRANCHES_MAIN = 20        # N_b of the instance SDDiP runs on
BRANCHES_EF = 10          # N_b of the extensive-form validation tree (prefix)
SEED = 20260912


def stage_seed(t):
    """Independent substream per stage (t is 1-indexed)."""
    return SEED * 100 + t


# ---------------------------------------------------------------------------
# 2. Poisson inverse-CDF sampler
# ---------------------------------------------------------------------------

def poisson_icdf(u, lam):
    """Smallest m >= 0 with P(X <= m) >= u, for X ~ Poisson(lam). One uniform."""
    if lam <= 0.0:
        return 0
    p = math.exp(-lam)          # pmf at m = 0
    cdf = p
    m = 0
    while cdf < u:
        m += 1
        p *= lam / m
        cdf += p
        if m > 10000:           # numerical guard; unreachable for lam used here
            break
    return m


def root_demand(lam1):
    """Deterministic stage-1 data: round-half-up of the Poisson mean."""
    return [[int(math.floor(lam1[i][j] + 0.5)) for j in range(N)] for i in range(N)]


def build_tree(lam, n_branch):
    """Omega[0] = [d^1] (root); Omega[t] = n_branch joint matrices for t >= 1."""
    omega = [[root_demand(lam[0])]]
    for t in range(1, T):
        rng = random.Random(stage_seed(t + 1))
        stage = []
        for _ in range(n_branch):
            mat = [[0] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    mat[i][j] = poisson_icdf(rng.random(), lam[t][i][j])
            stage.append(mat)
        omega.append(stage)
    return omega


# ---------------------------------------------------------------------------
# 3. Derived constants
# ---------------------------------------------------------------------------

def derive_gamma(phi):
    """Gamma = lcm of the denominators of phi_ij in lowest terms  (SC-1)."""
    g = 1
    for row in phi:
        for v in row:
            den = Fraction(str(v)).limit_denominator(10 ** 6).denominator
            g = g * den // math.gcd(g, den)
    return g


def reduce_mul(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def bits_for(ub):
    return 0 if ub <= 0 else int(math.ceil(math.log2(ub + 1)))


def od_max(omega, t, i, j):
    """max over scenarios of d_ij at stage t (0-indexed t)."""
    return max(s[i][j] for s in omega[t])


def node_max(omega, t, i):
    """max over scenarios of the NODE TOTAL sum_j d_ij at stage t."""
    return max(sum(s[i][j] for j in range(N)) for s in omega[t])


def analyse(name, variant, omega, gamma):
    q_phi = [[int(round(gamma * PHI[i][j])) for j in range(N)] for i in range(N)]
    c_max = max(max(r) for r in C_MAT)
    h_max = max(D_MAT[i][j] + C_MAT[j][k]
                for i in range(N) for j in range(N) for k in range(N))

    Q1, Q3 = W_TOTAL, B_TOTAL
    Q2 = (max(max(r) for r in P_JK)
          + max(max(r) for r in D_MAT)
          + max(max(r) for r in C_MAT))
    B_scaled = gamma * Q3

    # ---- exactness condition (SC-4): Q3 >= max_k D_i^{t,k} --------------
    cond = []
    for t in range(T):
        for i in range(N):
            cond.append({"t": t + 1, "i": i + 1,
                         "max_node_demand": node_max(omega, t, i),
                         "Q3": Q3,
                         "ok": node_max(omega, t, i) <= Q3})
    cond_ok = all(c["ok"] for c in cond)

    # ---- per-stage physical state bounds (ST-6') ------------------------
    # d_bar[t][i][j] = per-OD tree max, used as a per-OD cap on Y_ij.
    d_bar = [[[od_max(omega, t, i, j) for j in range(N)] for i in range(N)]
             for t in range(T)]

    def dbar(t, i, j):                       # t is 1-indexed here; 0 outside horizon
        return d_bar[t - 1][i][j] if 1 <= t <= T else 0

    bounds = {}   # component key -> list over t=1..T of the physical upper bound

    # A_tilde, U_tilde live at stages 1..T ; pipelines/leftover tails at 1..T
    bounds["A"] = {}
    bounds["U"] = {}
    for j in range(N):
        bounds["A"][j] = [B_scaled] * T
        # U_j^t only ever grew from unavailable returns created before t
        ub_u = []
        for t in range(1, T + 1):
            acc = sum(q_phi[i][j] * dbar(s, i, j)
                      for s in range(1, t) for i in range(N))
            ub_u.append(min(B_scaled, acc))
        bounds["U"][j] = ub_u

    bounds["PA"] = {}
    bounds["PU"] = {}
    for j in range(N):
        for r in range(2, c_max + 1):        # only the r>=2 tail is state
            ub_a, ub_u = [], []
            for t in range(1, T + 1):
                # bucket (j,r) at t = sum over m of inflow into (j,r+m) at t-m
                sa = sum((gamma - q_phi[i][j]) * dbar(t - m, i, j)
                         for m in range(0, c_max - r + 1)
                         for i in range(N) if C_MAT[i][j] == r + m)
                su = sum(q_phi[i][j] * dbar(t - m, i, j)
                         for m in range(0, c_max - r + 1)
                         for i in range(N) if C_MAT[i][j] == r + m)
                ub_a.append(min(B_scaled, sa))
                ub_u.append(min(B_scaled, su))
            bounds["PA"][(j, r)] = ub_a
            bounds["PU"][(j, r)] = ub_u

    bounds["W"] = {j: [Q1] * T for j in range(N)}

    bounds["G"] = {}
    for k in range(N):
        for r in range(2, h_max + 1):
            ub = []
            for t in range(1, T + 1):
                reach = any(D_MAT[i][j] + C_MAT[j][k] == r + m
                            for m in range(0, h_max - r + 1)
                            for i in range(N) for j in range(N))
                ub.append(Q1 if reach else 0)
            bounds["G"][(k, r)] = ub

    bounds["Mrem"] = {(j, k): [Q3] * T for j in range(N) for k in range(N)}

    # ---- horizon-max bit allocation (ST-7') ----------------------------
    # SDDP.jl needs one state vector shape for every node, so bits come from
    # the horizon max; the per-stage bounds still drive the slack caps.
    groups = []
    for grp, unit in (("A", "bike"), ("U", "bike"), ("PA", "bike"),
                      ("PU", "bike"), ("W", "count"), ("G", "count"),
                      ("Mrem", "count")):
        n_comp = n_drop = n_bits = 0
        pad = 0
        for _, per_t in sorted(bounds[grp].items(), key=lambda kv: str(kv[0])):
            ub = max(per_t)
            n_comp += 1
            b = bits_for(ub)
            if b == 0:
                n_drop += 1
                continue
            n_bits += b
            pad += (2 ** b - 1) - ub
        groups.append({"group": grp, "unit": unit, "components": n_comp,
                       "dropped_zero": n_drop, "bits": n_bits,
                       "padding_codes": pad})

    components = []
    for grp in ("A", "U", "PA", "PU", "W", "G", "Mrem"):
        for key, per_t in sorted(bounds[grp].items(), key=lambda kv: str(kv[0])):
            idx = list(key) if isinstance(key, tuple) else [key]
            # 1-based node indices; the pipeline remaining-time r stays as is
            one_based = [idx[0] + 1] + ([idx[1] + 1] if grp == "Mrem" else idx[1:])
            components.append({"group": grp, "index": one_based,
                               "ub_per_t": per_t, "ub": max(per_t),
                               "bits": bits_for(max(per_t))})

    total_bits = sum(g["bits"] for g in groups)
    live = sum(g["components"] - g["dropped_zero"] for g in groups)

    # padding fraction on the widest bike component
    bike_ub = B_scaled
    bike_b = bits_for(bike_ub)
    pad_frac = ((2 ** bike_b) - (bike_ub + 1)) / float(2 ** bike_b)

    # ---- initial constant cuts (IC-1 baseline, IC-2 tightened) ---------
    R_max = max(max(r) for r in R)
    theta_v1, theta_tight = [0.0] * (T + 1), [0.0] * (T + 1)
    for t in range(T, 0, -1):
        # baseline: serve every OD's max demand for free
        rev_v1 = sum(R[i][j] * dbar(t, i, j) for i in range(N) for j in range(N))
        # tightened: at most B_total rides can start in one stage
        caps = sorted(((R[i][j], dbar(t, i, j))
                       for i in range(N) for j in range(N)), reverse=True)
        room, rev_t = B_TOTAL, 0.0
        for r_ij, cap in caps:
            take = min(cap, room)
            rev_t += r_ij * take
            room -= take
            if room <= 0:
                break
        theta_v1[t - 1] = theta_v1[t] - rev_v1
        theta_tight[t - 1] = theta_tight[t] - rev_t

    # ---- A2 penalty calibration (rho_t) --------------------------------
    p_max = max(max(r) for r in P_JK)
    d_bar_stage = [sum(dbar(t, i, j) for i in range(N) for j in range(N))
                   for t in range(1, T + 1)]
    # Per stage the served volume cannot exceed B_total (every ride takes a bike),
    # so only min(B_total, d_bar_s) units can swing between served and lost.
    def stage_range(s):
        return (R_max + C_P) * min(B_TOTAL, d_bar_stage[s - 1]) + W_TOTAL * p_max

    delta_c, delta_c_loose, rho, rho_loose = [], [], [], []
    for t in range(1, T + 1):
        dc = sum(stage_range(s) for s in range(t, T + 1))
        dc_loose = ((R_max + C_P) * sum(d_bar_stage[s - 1] for s in range(t, T + 1))
                    + W_TOTAL * p_max * (T - t + 1))
        delta_c.append(dc)
        delta_c_loose.append(dc_loose)
        rho.append(int(math.floor(dc)) + 1)      # rho_t * eps_t > Delta C_t, eps_t = 1
        rho_loose.append(int(math.floor(dc_loose)) + 1)

    # ---- A2 elastic row census ------------------------------------------
    n, n2, n3 = N, N * N, N * N * N
    eq = {"D1": n, "D3-sum": n, "D4": n,
          "D7": n * c_max, "D8": n * c_max, "G1": n * h_max,
          "M1": n2, "M2": n2}
    ineq = {"D2": 4 * n, "D3-cap": n2, "P1": n, "P2": n, "M3": n2, "W1": n,
            "S1": 4 * n3, "S2": n3, "S3": n3, "S6": n3, "S4": n, "S5": n}
    eq_tr = {"A-transition": n, "U-transition": n, "W-transition": n,
             "Mrem-out": n2}
    n_eq = sum(eq.values()) + sum(eq_tr.values())
    n_ineq = sum(ineq.values())
    n_slack = 2 * n_eq + n_ineq + 2 * live

    # ---- duplicate atoms -------------------------------------------------
    dup = []
    for t in range(T):
        uniq = {}
        for s in omega[t]:
            key = tuple(tuple(row) for row in s)
            uniq[key] = uniq.get(key, 0) + 1
        dup.append({"t": t + 1, "K": len(omega[t]), "distinct": len(uniq),
                    "max_multiplicity": max(uniq.values())})

    return {
        "name": name,
        "note": variant["note"],
        "lambda_diag": variant["lambda_diag"],
        "gamma": gamma,
        "q_phi": q_phi,
        "C_max": c_max,
        "H_max": h_max,
        "Q1": Q1, "Q2": Q2, "Q3": Q3,
        "B_scaled": B_scaled,
        "exactness_condition": {"rows": cond, "all_ok": cond_ok},
        "state": {"groups": groups, "components": components, "total_bits": total_bits,
                  "live_components": live,
                  "bike_component_bits": bike_b,
                  "bike_padding_fraction": pad_frac},
        "theta_baseline": theta_v1[:T + 1],
        "theta_tightened": theta_tight[:T + 1],
        "delta_C": delta_c,
        "delta_C_loose": delta_c_loose,
        "rho": rho,
        "rho_loose": rho_loose,
        "d_bar_stage": d_bar_stage,
        "d_bar_od": d_bar,
        "a2_rows": {"equalities": n_eq, "inequalities": n_ineq,
                    "slack_variables": n_slack},
        "duplicates": dup,
        "tree": {"branches_per_stage": [len(st) for st in omega],
                 "scenarios": reduce_mul([len(st) for st in omega]),
                 "nodes": sum(reduce_mul([len(st) for st in omega[:t + 1]])
                              for t in range(T)),
                 "backward_solves_per_path": sum(len(st) for st in omega[1:])},
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    gamma = derive_gamma(PHI)

    out = {
        "instance_family": "LIME-OPR v2 toy",
        "spec": "model_v2/01_problem_definition.md, model_v2/02_scenario_tree.md",
        "sampler": {"kind": "poisson-inverse-cdf", "rng": "python random.Random",
                    "stage1": "deterministic root, d1 = floor(lambda1 + 0.5)",
                    "stage_substream_seed": "SEED*100 + t",
                    "draw_order": "per stage: atom, then i, j ascending",
                    "uniforms_per_draw": 1, "seed": SEED,
                    "branches_main": BRANCHES_MAIN, "branches_ef": BRANCHES_EF},
        "deterministic": {
            "N": N, "T": T, "B_total": B_TOTAL, "W_total": W_TOTAL,
            "A0": A0, "U0": U0, "W0": W0,
            "R_ij": R, "p_jk": P_JK, "C_p": C_P,
            "d_ij": D_MAT, "c_ij": C_MAT, "phi_ij": PHI,
        },
        "gamma": gamma,
        "variants": {},
    }

    report = []
    jobs = [("toy-A", BRANCHES_MAIN), ("toy-A-ef", BRANCHES_EF), ("toy-B", BRANCHES_MAIN)]
    for name, n_branch in jobs:
        variant = VARIANTS[name.replace("-ef", "")]
        lam = []
        for t in range(1, T + 1):
            base = LAMBDA_OFFDIAG["late" if t == T else "early"]
            lam.append([[variant["lambda_diag"] if i == j else base[i][j]
                         for j in range(N)] for i in range(N)])
        omega = build_tree(lam, n_branch)
        info = analyse(name, variant, omega, gamma)
        out["variants"][name] = {
            "lambda_ijt": lam,
            "Omega": omega,
            "probabilities": [[1.0 / len(st)] * len(st) for st in omega],
            "derived": {k: v for k, v in info.items() if k != "d_bar_od"},
            "d_bar_od": info["d_bar_od"],
        }
        report.append(info)

    with open(os.path.join(here, "toy_instance.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    # ---- console summary -------------------------------------------------
    print("Gamma = %d   (lcm of phi denominators; phi = 0.15 = 3/20)" % gamma)
    for info in report:
        print("\n=== %s : %s" % (info["name"], info["note"]))
        print("  C_max=%d  H_max=%d  Q1=%d  Q2=%g  Q3=%d  Gamma*Q3=%d"
              % (info["C_max"], info["H_max"], info["Q1"], info["Q2"],
                 info["Q3"], info["B_scaled"]))
        print("  exactness  max_k D_i^{t,k} <= Q3 : %s"
              % ("PASS" if info["exactness_condition"]["all_ok"] else "FAIL"))
        worst = max(info["exactness_condition"]["rows"],
                    key=lambda r: r["max_node_demand"])
        print("    worst row: t=%d i=%d  max_k D_i = %d  (Q3=%d)"
              % (worst["t"], worst["i"], worst["max_node_demand"], worst["Q3"]))
        print("  state encoding:")
        for g in info["state"]["groups"]:
            print("    %-5s comp=%2d  dropped(ub=0)=%2d  bits=%3d  padding_codes=%d"
                  % (g["group"], g["components"], g["dropped_zero"],
                     g["bits"], g["padding_codes"]))
        print("    total bits n_t = %d   live components = %d"
              % (info["state"]["total_bits"], info["state"]["live_components"]))
        print("    bike component: %d bits, padding fraction %.1f%%"
              % (info["state"]["bike_component_bits"],
                 100 * info["state"]["bike_padding_fraction"]))
        print("  theta baseline  = %s" % [round(v, 1) for v in info["theta_baseline"]])
        print("  theta tightened = %s" % [round(v, 1) for v in info["theta_tightened"]])
        print("  Delta C = %s   (loose: %s)"
              % ([round(v, 1) for v in info["delta_C"]],
                 [round(v, 1) for v in info["delta_C_loose"]]))
        print("  rho     = %s   (loose: %s)" % (info["rho"], info["rho_loose"]))
        print("  d_bar per stage (sum_ij max_k d_ij) = %s" % info["d_bar_stage"])
        print("  A2 rows: eq=%d ineq=%d -> slacks=%d"
              % (info["a2_rows"]["equalities"], info["a2_rows"]["inequalities"],
                 info["a2_rows"]["slack_variables"]))
        tr = info["tree"]
        print("  tree: branches=%s  scenarios=%d  nodes=%d  backward solves/path=%d"
              % (tr["branches_per_stage"], tr["scenarios"], tr["nodes"],
                 tr["backward_solves_per_path"]))
        for d in info["duplicates"]:
            print("    stage %d: K=%d distinct=%d max_mult=%d"
                  % (d["t"], d["K"], d["distinct"], d["max_multiplicity"]))

    print("\nwrote %s" % os.path.join(here, "toy_instance.json"))


if __name__ == "__main__":
    main()

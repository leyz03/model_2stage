"""
test_sparse_td.jl  —  relax-and-fix and the Lagrangian time decomposition on the sparse formulation (form = :sparse)
(doc/2026-10-15/modify_1_*); Gurobi, MIPGap 0.

    julia --project=model_2stage/env model_2stage/test/test_sparse_td.jl [paris=<instance json>]

D0  exact tier optimum: sparse = dense (the reference OPT of the case)
D1  period_ints of a sparse block lists every free integer / binary variable exactly once
D2  the linked sparse block model (copies z_{b+1} = y_b, 2 and 3 blocks) has OPT
D3  joint block LP: sparse = dense value, same multiplier layout
D4  L(λ) at the dense LP multipliers: sparse blocks = dense blocks, >= OPT
D5  td_forward(form = :sparse) at those λ is feasible: td_verify gives the same value, no blocking pair,
    conservation residual 0, value <= OPT (dense forward value printed alongside)
D6  relax_and_fix(form = :sparse) rf6-3p (rf4-2p when T < 12), with tie-breaking and with the window MIP start
    (start = true): feasible (no blocking pair, residual 0, block profit = returned value), value <= OPT
    (dense :pre value printed alongside); D5 is also run with start = true
D7  continue_start!: from the morning state the start is a feasible integral point whose profit is the returned
    value; from the optimum's state at every period t0 >= 2 (optimum values before t0) it is again feasible
With paris=..., the case is periods 1..12 of the first scenario at price 4 (grid 50).
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))

using Printf
using Gurobi

const GENV = Gurobi.Env()
opt(; tol = 1e-9) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 2, "IntFeasTol" => tol, "FeasibilityTol" => tol)
qp() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "Threads" => 2)

SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
M6(n, T) = joinpath(@__DIR__, "..", "fixed_price", "instances", "syn_n$(n)_T$(T)_m6.json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b; tol = 1e-6) = abs(a - b) <= tol * max(1.0, abs(b))

cases = Any[(load_ts_instance(SYN(3, 6), "syn"; levels = 3, nscen = 1), "n3T6", 1e-9),
            (load_ts_instance(M6(5, 24), "syn"; levels = 3, nscen = 1), "n5T24m6", 1e-9)]
for a in ARGS
    startswith(a, "paris=") || continue
    full = load_ts_instance(a[7:end], "paris"; levels = 3, nscen = 1, grid = 50)
    push!(cases, (window_instance(full, 1, 1, 12), "paris_n$(full.n)_t1-12", 1e-6))
end

for (inst, tag, tol) in cases
    o() = opt(; tol)
    T, n = inst.T, inst.n
    P = fill(4.0, n, n)
    println("== $tag  n=$n T=$T Γ=$(inst.Γ)"); flush(stdout)
    vs, ms, bs, is = solve_tier_sparse(inst, 1, P, o())
    vd, = solve_tier(inst, 1, P, o())
    OPT = vs
    check(is.optimal && close_to(vs, vd), @sprintf("D0 %s  sparse %.4f  dense %.4f", tag, vs, vd))

    # D1: period_ints partitions the free integer variables
    seen = VariableRef[]
    for t in 1:T
        append!(seen, [v for v in period_ints(bs, t) if !is_fixed(v)])
    end
    free_int = Set(v for v in all_variables(ms) if (is_integer(v) || is_binary(v)) && !is_fixed(v))
    check(length(seen) == length(Set(seen)) && Set(seen) == free_int,
          @sprintf("D1 %s  period_ints %d  free integer variables %d", tag, length(seen), length(free_int)))
    ms = nothing; bs = nothing

    for nb in (2, 3)
        cuts = td_cuts(T, cld(T, nb))
        # D2
        m, blks, link = td_joint(inst, 1, P, cuts, o(); form = :sparse)
        optimize!(m)
        check(JuMP.termination_status(m) == MOI.OPTIMAL && close_to(objective_value(m), OPT),
              @sprintf("D2 %s %d blocks  joint %.4f  OPT %.4f", tag, length(cuts), objective_value(m), OPT))
        # D3
        lps, λs, = td_lp_multipliers(inst, 1, P, cuts, o(); form = :sparse)
        lpd, λd, = td_lp_multipliers(inst, 1, P, cuts, o())
        check(close_to(lps, lpd) && length.(λs) == length.(λd),
              @sprintf("D3 %s %d blocks  LP sparse %.4f  dense %.4f  multipliers %s", tag, length(cuts), lps, lpd,
                       string(length.(λs))))
        # D4
        λ = reduce(vcat, λd; init = Float64[])
        Ls, = lagrangian_td(inst, 1, P, o(), qp(); cuts, λ0 = λ, maxit = 0, budget = 1800.0, block_tl = 1800.0,
                            form = :sparse, verbose = false)
        Ld, = lagrangian_td(inst, 1, P, o(), qp(); cuts, λ0 = λ, maxit = 0, budget = 1800.0, block_tl = 1800.0,
                            verbose = false)
        check(close_to(Ls, Ld) && Ls >= OPT - 1e-6 * abs(OPT) && Ls <= lps + 1e-6 * abs(lps),
              @sprintf("D4 %s %d blocks  L(λ_LP) sparse %.4f  dense %.4f  OPT %.4f  LP %.4f", tag, length(cuts), Ls, Ld,
                       OPT, lps))
        # D5
        fv, plan, fi = td_forward(inst, 1, P, o(), cuts, λ; block_tl = 600.0, form = :sparse)
        vv, nbp, res = fi.ok ? td_verify(inst, 1, P, o(), plan; form = :sparse) : (-Inf, -1, Inf)
        fd, = td_forward(inst, 1, P, o(), cuts, λ; block_tl = 600.0)
        check(fi.ok && close_to(vv, fv) && nbp == 0 && res < 1e-6 && fv <= OPT + 1e-6 * abs(OPT),
              @sprintf("D5 %s %d blocks  forward sparse %.4f  verify %.4f  blocking %d  residual %.1e  (dense %.4f)  OPT %.4f",
                       tag, length(cuts), fv, vv, nbp, res, fd, OPT))
        fv2, plan2, fi2 = td_forward(inst, 1, P, o(), cuts, λ; block_tl = 600.0, form = :sparse, start = true)
        vv2, nbp2, res2 = fi2.ok ? td_verify(inst, 1, P, o(), plan2; form = :sparse) : (-Inf, -1, Inf)
        check(fi2.ok && close_to(vv2, fv2) && nbp2 == 0 && res2 < 1e-6 && fv2 <= OPT + 1e-6 * abs(OPT),
              @sprintf("D5 %s %d blocks start  forward sparse %.4f  verify %.4f  blocking %d  residual %.1e  OPT %.4f",
                       tag, length(cuts), fv2, vv2, nbp2, res2, OPT))
    end

    # D6
    w, st = T >= 12 ? (6, 3) : (4, 2)
    for (tb, stt) in ((false, false), (true, false), (false, true))
        v, m, b, info = relax_and_fix(inst, 1, P, o(); w, step = st, look = :lp, post_penalty = 1e-3, tiebreak = tb,
                                      win_tl = 600.0, form = :sparse, start = stt)
        bp = info.ok ? blocking_pairs_sparse(inst, b, P) : [nothing]
        res = info.ok ? conservation_residual(inst, b) : Inf
        vd, = relax_and_fix(inst, 1, P, o(); w, step = st, look = :lp, post_penalty = 1e-3, tiebreak = tb, win_tl = 600.0)
        check(info.ok && isempty(bp) && res < 1e-6 && close_to(value(b.profit), v) && v <= OPT + 1e-6 * abs(OPT),
              @sprintf("D6 %s rf%d-%dp%s%s  sparse %.4f  blocking %d  residual %.1e  windows %d  (dense %.4f)  OPT %.4f",
                       tag, w, st, tb ? " tiebreak" : "", stt ? " start" : "", v, length(bp), res, info.windows, vd, OPT))
    end

    # D7
    point(m) = Dict(v => something(start_value(v), is_fixed(v) ? fix_value(v) : NaN) for v in all_variables(m))
    function feasible(m, pt)
        any(isnan, values(pt)) && return false
        isempty(primal_feasibility_report(m, pt; atol = 1e-6)) || return false
        return all(abs(pt[v] - round(pt[v])) < 1e-9 for v in all_variables(m) if is_integer(v) || is_binary(v))
    end
    m = Model()
    b = add_scenario_sparse!(m, inst, inst.scen[1], P, P; stability = :tier)
    pv = continue_start!(inst, b, 1, initial_state(inst, b))
    pt = point(m)
    check(feasible(m, pt) && close_to(value(v -> pt[v], b.profit), pv) && pv <= OPT + 1e-6 * abs(OPT),
          @sprintf("D7 %s  start from the morning feasible, profit %.4f  OPT %.4f", tag, pv, OPT))
    v, m, b, = solve_tier_sparse(inst, 1, P, o())
    opt_val = Dict(x => value(x) for x in all_variables(m))
    bad, ntask = Int[], 0
    for t0 in 2:T
        st0 = read_state(inst, b, t0; val = x -> opt_val[x])
        ntask += sum(st0.Mrem) + sum(st0.Gin)
        for x in all_variables(m)
            set_start_value(x, opt_val[x])
        end
        continue_start!(inst, b, t0, st0)
        feasible(m, point(m)) || push!(bad, t0)
    end
    check(isempty(bad), @sprintf("D7 %s  start from the optimum's state at t0 = 2..%d feasible (failed at %s; open tasks + workers in transit summed over t0: %d)",
                                 tag, T, string(bad), ntask))
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)

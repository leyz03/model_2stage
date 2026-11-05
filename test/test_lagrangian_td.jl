"""
test_lagrangian_td.jl  —  correctness of src/lagrangian_td.jl (HiGHS by default, no Gurobi session)

    julia --project=model_2stage/env model_2stage/test/test_lagrangian_td.jl [gurobi]

L1  every boundary state of the exact optimum respects the `state_keys` bounds (zero where ub = 0)
L2  the linked block model (copies z_{b+1} = y_b) has the exact tier optimum, for 2 and 3 blocks
L3  λ from the joint LP: LP-relaxed blocks give L_LP(λ) = LP value; MILP blocks give OPT <= L(λ) <= LP
L4  the bundle's bound is >= OPT, <= L(λ_LP), non-increasing along the trace; starts from λ_LP and
    from 0 converge to the same dual value; one block gives OPT
L5  td_forward (with λ and myopic) is feasible: re-evaluated in the whole-day model with the
    same value, no blocking pair, conservation residual 0, value <= OPT
L6  td_cuts / td_cuts_adaptive cover 1..T with lengths in range
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))

if "gurobi" in ARGS
    using Gurobi
    const GENV = Gurobi.Env()
    opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => 1, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
    qp() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "Threads" => 1)
else
    using HiGHS
    opt() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
    qp() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "threads" => 1)
end

SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b; tol = 1e-6) = abs(a - b) <= tol * max(1.0, abs(b))

"""L(λ) with every block solved as given (MILP, or LP when `lp`): Σ bounds."""
function lag_value(inst, s, P, cuts, λs; lp = false)
    total = 0.0
    for (k, (ta, tb)) in enumerate(cuts)
        m = Model(opt())
        blk = add_td_block!(m, inst, s, P, ta, tb)
        obj = AffExpr(0.0) + blk.b.profit
        k < length(cuts) && (obj += sum(λs[k] .* blk.y))
        k > 1 && (obj -= sum(λs[k-1] .* blk.z))
        @objective(m, Max, obj)
        lp && relax_integrality(m)
        optimize!(m)
        total += lp ? objective_value(m) : objective_bound(m)
    end
    return total
end

for (n, T) in ((3, 6), (3, 8))
    inst = load_ts_instance(SYN(n, T), "syn"; levels = 3, nscen = 2)
    println("== syn n=$n T=$T Γ=$(inst.Γ)")
    for (pn, P) in (("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])), s in 1:2
        lab = "n$(n)T$T $pn s=$s"
        opt_val, _, b, info = solve_tier(inst, s, P, opt())
        bad = 0
        for t in 2:T, key in state_keys(inst, s, P, t)
            v = value(statevar(b, key, t))
            (v <= key.ub + 1e-6) || (bad += 1)
        end
        check(info.optimal && bad == 0, @sprintf("L1 %s  OPT %.4f  bound violations %d", lab, opt_val, bad))

        for cuts in (td_cuts(T, cld(T, 2)), td_cuts(T, 2))
            mj, blks, _ = td_joint(inst, s, P, cuts, opt())
            optimize!(mj)
            check(close_to(objective_value(mj), opt_val),
                  @sprintf("L2 %s  cuts %s  joint %.4f vs OPT %.4f", lab, cuts, objective_value(mj), opt_val))
        end

        cuts = td_cuts(T, 3)
        lpv, duals, _, _ = td_lp_multipliers(inst, s, P, cuts, opt())
        llp = lag_value(inst, s, P, cuts, duals; lp = true)
        lmip = lag_value(inst, s, P, cuts, duals)
        check(close_to(llp, lpv; tol = 1e-5) && lmip >= opt_val - 1e-6 && lmip <= lpv + 1e-6,
              @sprintf("L3 %s  LP %.4f  L_LP(λ) %.4f  L(λ) %.4f  OPT %.4f", lab, lpv, llp, lmip, opt_val))

        ub, λb, li = lagrangian_td(inst, s, P, opt(), qp(); cuts, budget = 120.0, verbose = false)
        bests = [r.best for r in li.trace]
        check(ub >= opt_val - 1e-6 && ub <= lmip + 1e-6 && all(diff(bests) .<= 1e-9),
              @sprintf("L4 %s  bundle UB %.4f (L(λ_LP) %.4f, OPT %.4f)  it %d  converged %s",
                       lab, ub, lmip, opt_val, li.it, li.converged))

        ub0, _, li0 = lagrangian_td(inst, s, P, opt(), qp(); cuts, λ0 = :zero, budget = 120.0, verbose = false)
        check(li.converged && li0.converged && abs(ub - ub0) <= 2e-3 * max(1.0, abs(ub)),
              @sprintf("L4 %s  converged dual from λ_LP %.4f (it %d) and from 0 %.4f (it %d) agree",
                       lab, ub, li.it, ub0, li0.it))

        for (tag, lam) in (("λ", λb), ("myopic", nothing))
            fv, plan, fi = td_forward(inst, s, P, opt(), cuts, lam)
            vv, nbp, res = td_verify(inst, s, P, opt(), plan)
            check(fi.ok && close_to(vv, fv) && nbp == 0 && res < 1e-6 && fv <= opt_val + 1e-6,
                  @sprintf("L5 %s %s  forward %.4f  re-evaluated %.4f  blocking %d  residual %.1e  OPT %.4f",
                           lab, tag, fv, vv, nbp, res, opt_val))
        end
    end
    ub1, _, _ = lagrangian_td(inst, 1, fill(4.0, n, n), opt(), qp(); cuts = [(1, T)], verbose = false)
    v1 = solve_tier(inst, 1, fill(4.0, n, n), opt())[1]
    check(close_to(ub1, v1), @sprintf("L4 n%dT%d one block  UB %.4f  OPT %.4f", n, T, ub1, v1))
end

for (T, w) in ((9, 3), (8, 4), (10, 4), (12, 4), (12, 5), (240, 4))
    c = td_cuts(T, w)
    ok = c[1][1] == 1 && c[end][2] == T && all(c[k+1][1] == c[k][2] + 1 for k in 1:length(c)-1) &&
         maximum(b - a + 1 for (a, b) in c) - minimum(b - a + 1 for (a, b) in c) <= 1
    load = [abs(sin(t)) for t in 1:T]
    B = cld(T, w)
    ca = td_cuts_adaptive(load, T, B; wmin = 3, wmax = 6)
    ok &= length(ca) == B && ca[1][1] == 1 && ca[end][2] == T &&
          all(3 <= b - a + 1 <= 6 for (a, b) in ca) && all(ca[k+1][1] == ca[k][2] + 1 for k in 1:B-1)
    check(ok, "L6 T=$T w=$w  uniform $(length(c)) blocks, adaptive $(length(ca)) blocks")
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")

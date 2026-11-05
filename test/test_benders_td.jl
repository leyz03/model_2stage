"""
test_benders_td.jl  —  correctness of src/benders_td.jl (HiGHS by default, no Gurobi session)

    julia --project=model_2stage/env model_2stage/test/test_benders_td.jl [gurobi]

B1  every cut over-estimates the optimal tail: α + β·y* >= profit of periods tb+1..T of the exact optimum,
    where y* is the optimum's state after the block that holds the cut (LP and strengthened cuts)
B2  LB <= OPT <= UB; the best forward plan re-evaluated in the whole-day model has value LB,
    no blocking pair, conservation residual 0
B3  one block (no cuts): LB = UB = OPT
"""

include(joinpath(@__DIR__, "..", "src", "benders_td.jl"))

if "gurobi" in ARGS
    using Gurobi
    const GENV = Gurobi.Env()
    opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => 1, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
else
    using HiGHS
    opt() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
end

SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
M6(n, T) = joinpath(@__DIR__, "..", "fixed_price", "instances", "syn_n$(n)_T$(T)_m6.json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b; tol = 1e-6) = abs(a - b) <= tol * max(1.0, abs(b))

cases = [(SYN(3, 6), 3, 6, 2), (SYN(3, 8), 3, 8, 2), (M6(5, 24), 5, 24, 1)]
for (path, n, T, ns) in cases
    inst = load_ts_instance(path, "syn"; levels = 3, nscen = ns)
    println("== $(basename(path))  Γ=$(inst.Γ)")
    prices = T <= 8 ? (("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])) :
                      (("u4", fill(4.0, n, n)),)
    for (pn, P) in prices, s in 1:ns
        lab = "n$(n)T$T $pn s=$s"
        optv, _, bw, info = solve_tier(inst, s, P, opt())
        info.optimal || (check(false, "$lab exact not optimal"); continue)
        tail(t) = t > T ? 0.0 : sum(value(bw.pprofit[u]) for u in t:T)
        ystar(tb) = [value(statevar(bw, key, tb + 1)) for key in state_keys(inst, s, P, tb + 1) if key.ub > 0]

        for w in (1, 3), cut in (:lp, :sb)
            cuts = td_cuts(T, w)
            lb, ub, plan, bi = benders_td(inst, s, P, opt(); cuts, cut, budget = 120.0, maxit = 8, verbose = false)
            worst = Inf
            for (k, α, β) in bi.cutlog
                tb = cuts[k][2]
                worst = min(worst, α + sum(β .* ystar(tb)) - tail(tb + 1))
            end
            check(worst >= -1e-6 * max(1.0, abs(optv)),
                  @sprintf("B1 %s w=%d %s  %d cuts, min(cut − tail*) %.3g", lab, w, cut, length(bi.cutlog), worst))
            vv, nbp, res = td_verify(inst, s, P, opt(), plan)
            check(lb <= optv + 1e-6 && ub >= optv - 1e-6 && close_to(vv, lb) && nbp == 0 && res <= 1e-6,
                  @sprintf("B2 %s w=%d %s  LB %.4f  OPT %.4f  UB %.4f  verify %.4f  blocking %d  it %d (%s)",
                           lab, w, cut, lb, optv, ub, vv, nbp, bi.it, bi.stop))
        end
        lb, ub, _, _ = benders_td(inst, s, P, opt(); cuts = [(1, T)], budget = 120.0, verbose = false)
        check(close_to(lb, optv) && close_to(ub, optv), @sprintf("B3 %s one block  LB %.4f  UB %.4f  OPT %.4f", lab, lb, ub, optv))
    end
end
println(fails == 0 ? "ALL PASS" : "$fails FAILED")

"""
test_baseline.jl  —  correctness of the baseline (model_2stage/04_algorithm.md); needs Gurobi (lazy callbacks)

    julia --project=model_2stage/env model_2stage/test/test_baseline.jl

B1  :full == :pre == :lazy scenario by scenario, at several fixed prices, on the toy and on
    a synthetic instance; the price-box relaxation is ≥ the exact value.
B2  price_bnb equals brute force over the whole menu: toy `type` (3 scenarios, known optimum
    150.3333 from smoke_test T3) and toy / synthetic `uniform`.
B3  with a short time limit price_bnb stops on time (deadline checked inside a node) and its
    upper bound is still ≥ the optimum.
"""

include(joinpath(@__DIR__, "..", "src", "price_bnb.jl"))

using Gurobi, Printf

const GENV = Gurobi.Env()
grb() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 1, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
const SYN = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n5_T4.json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1))

println("B1  formulations agree")
for (path, var) in ((TOY, "toy-A"), (SYN, "syn"))
    inst = load_ts_instance(path, var; levels = 3, nscen = 3)
    for pr in (1.0, 2.0, 4.0, 7.0), s in 1:3
        P = fill(pr, inst.n, inst.n)
        vf, _, _, inf_f = solve_scenario(inst, s, P, grb(); stability = :full)
        vp, _, _, inf_p = solve_scenario(inst, s, P, grb(); stability = :pre)
        vl, _, _, inf_l = solve_scenario(inst, s, P, grb(); stability = :lazy)
        vr, _ = solve_relax(inst, s, P, P, grb())
        check(abs(vf - vp) < 1e-6 && abs(vf - vl) < 1e-6 && vr >= vf - 1e-6,
              @sprintf("%s p=%g s=%d  full %.3f pre %.3f lazy %.3f (%d cuts) relax %.3f  bin %d/%d/%d",
                       var, pr, s, vf, vp, vl, inf_l.ncuts, vr, inf_f.nbin, inf_p.nbin, inf_l.nbin))
    end
end

println("B2  price_bnb vs brute force")
function brute(inst)
    best, arg = -Inf, nothing
    for pt in Iterators.product(ntuple(_ -> eachindex(inst.menu), inst.ngroups)...)
        v, _ = evaluate_prices(inst, price_matrix(inst, collect(pt)), grb(); stability = :pre)
        v > best + 1e-9 && ((best, arg) = (v, inst.menu[collect(pt)]))
    end
    return best, arg
end
for (path, var, str, ns) in ((TOY, "toy-A", :type, 3), (TOY, "toy-A", :uniform, 5), (SYN, "syn", :uniform, 3))
    inst = load_ts_instance(path, var; levels = 3, nscen = ns, structure = str)
    for inner in (:pre, :lazy)
        res = price_bnb(inst, grb(); inner, log = false)
        bv, bp = brute(inst)
        check(res.status == :optimal && abs(res.value - bv) < 1e-6,
              @sprintf("%s %s K_s=%d inner=%s: bnb %.4f at %s (cands %s, %d nodes, %d exact pts)  brute %.4f at %s",
                       var, str, ns, inner, res.value, string(res.prices), string(res.candidates),
                       res.stats.nodes, res.stats.exact_points, bv, string(bp)))
    end
end
println("B3  time limit is honoured inside a node and the reported bound stays valid")
let inst = load_ts_instance(SYN, "syn"; levels = 3, nscen = 20, structure = :type)
    full = price_bnb(inst, grb(); inner = :pre, log = false)
    for tl in (0.5, 5.0, 20.0)
        res = price_bnb(inst, grb(); inner = :pre, tl, log = false)
        ok = res.seconds < tl + 15 && res.ub >= full.value - 1e-6 &&
             (res.status == :optimal ? abs(res.value - full.value) < 1e-6 : res.value <= full.value + 1e-6)
        check(ok, @sprintf("tl %.1f s: %s in %.1f s, value %.4f, ub %.4f (optimum %.4f, %.1f s)",
                           tl, res.status, res.seconds, res.value, res.ub, full.value, full.seconds))
    end
end

println(fails == 0 ? "ALL PASS" : "$fails FAILURE(S)")

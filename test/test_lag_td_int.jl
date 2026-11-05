"""
test_lag_td_int.jl  —  TODO #10 (int_flow/lag_td.jl): the Lagrangian time decomposition on the integer rider split,
blocks solved one by one in the forms :full, :xbin, :bin, :lp.  Split toy / syn n3T6 instances of test_decomp.jl
(2 scenarios × 3 prices each), 2 and 3 blocks.  Gurobi.

    julia --project=model_2stage/env model_2stage/test/test_lag_td_int.jl

J1  linked blocks (td_joint, form = :sparse): MIP value = whole-period tier MIP value, LP value = its LP value
J2  λ from the joint LP: LP value = whole LP; the :lp blocks at λ sum to it (L_LP(λ_LP) = LP)
J3  at λ_LP: OPT ≤ Σ :full ≤ Σ :bin ≤ LP, and Σ :xbin = Σ :full (x and binaries integral suffice, TODO #2)
J4  any λ (λ_LP + noise, λ = 0): Σ :full ≥ OPT, Σ :bin ≥ Σ :full, Σ :xbin = Σ :full
J5  the keys of λ survive a JSON round trip and match the incoming / outgoing states of a block built alone
"""

include(joinpath(@__DIR__, "..", "int_flow", "lag_td.jl"))

using Printf, Gurobi, JSON, Random

const GENV = Gurobi.Env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 2, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))
geq(a, b) = a >= b - 1e-6 * max(1.0, abs(b))

with_split(inst, scenU) =
    TSInstance(inst.name * "/int", inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, inst.Q3, inst.A0, inst.U0, inst.W0, inst.menu, inst.structure,
               inst.group, inst.ngroups, inst.scen, inst.vterm, scenU, [inst.A0 for _ in inst.scen],
               [inst.U0 for _ in inst.scen])
pattern_split(inst) = [[[(ξ[t][i, j] * (3i + 5j + 7t)) % (ξ[t][i, j] + 1) for i in 1:inst.n, j in 1:inst.n]
                        for t in 1:inst.T] for ξ in inst.scen]
prices(n) = [("u2", fill(2.0, n, n)), ("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])]

function whole(v, P; lp = false)
    m = Model(opt())
    b = add_scenario_sparse!(m, v, v.scen[1], P, P; stability = :tier)
    @objective(m, Max, b.profit)
    lp && relax_integrality(m)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("whole: $(JuMP.termination_status(m))")
    return objective_value(m)
end

"""Σ_k of the block optima at λ in one form."""
function lagsum(v, P, cuts, λ, form)
    tot = 0.0
    for k in eachindex(cuts)
        m, _ = lag_block(v, P, cuts, k, λ, opt(); form)
        optimize!(m)
        JuMP.termination_status(m) == MOI.OPTIMAL || error("block $k $form: $(JuMP.termination_status(m))")
        tot += objective_value(m)
    end
    return tot
end

Random.seed!(20261104)
for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T)")
    for s in eachindex(old.scen), (pn, P) in prices(old.n), B in (2, 3)
        v = scenario_view(dead, s)
        v.T >= B || continue
        cuts = lag_cuts(v.T, B)
        opt_v, lp_v = whole(v, P), whole(v, P; lp = true)
        lbl = @sprintf("%s s%d %s B=%d", tag, s, pn, B)
        # J1
        mj, _, _ = td_joint(v, 1, P, cuts, opt(); form = :sparse)
        optimize!(mj)
        jm = objective_value(mj)
        relax_integrality(mj)
        optimize!(mj)
        jl = objective_value(mj)
        check(close_to(jm, opt_v) && close_to(jl, lp_v),
              @sprintf("J1 %s  joint MIP %.4f = OPT %.4f, joint LP %.4f = LP %.4f", lbl, jm, opt_v, jl, lp_v))
        # J2
        lpv, λ, keys, _ = lag_lambda(v, P, cuts, opt())
        llp = lagsum(v, P, cuts, λ, :lp)
        check(close_to(lpv, lp_v) && close_to(llp, lp_v),
              @sprintf("J2 %s  λ LP %.4f = LP %.4f, Σ LP blocks %.4f", lbl, lpv, lp_v, llp))
        # J3
        lf, lx, lb = (lagsum(v, P, cuts, λ, f) for f in (:full, :xbin, :bin))
        check(geq(lf, opt_v) && geq(lb, lf) && geq(lp_v, lb) && close_to(lx, lf),
              @sprintf("J3 %s  at λ_LP: OPT %.4f ≤ full %.4f ≤ bin %.4f ≤ LP %.4f, xbin %.4f", lbl, opt_v, lf, lb,
                       lp_v, lx))
        # J4
        for (nm, λ2) in (("noise", [l .+ (2 .* rand(length(l)) .- 1) for l in λ]), ("zero", [zero(l) for l in λ]))
            f2, x2, b2 = (lagsum(v, P, cuts, λ2, f) for f in (:full, :xbin, :bin))
            check(geq(f2, opt_v) && geq(b2, f2) && close_to(x2, f2),
                  @sprintf("J4 %s  λ %-5s: full %.4f ≥ OPT %.4f, bin %.4f, xbin %.4f", lbl, nm, f2, opt_v, b2, x2))
        end
        # J5
        js = JSON.parse(JSON.json([lag_keys_json(kk) for kk in keys]))
        ok = true
        for k in eachindex(cuts)
            _, blk = lag_block(v, P, cuts, k, λ, opt(); form = :full)
            k > 1 && (ok &= lag_same_keys(blk.zkeys, js[k-1]))
            k < B && (ok &= lag_same_keys(blk.ykeys, js[k]))
        end
        check(ok, @sprintf("J5 %s  %d boundary key lists (%s states) round-trip and match the blocks", lbl, B - 1,
                           join(length.(keys), "/")))
    end
end
println(fails == 0 ? "ALL PASS" : "$fails FAILED")

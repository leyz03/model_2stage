"""
test_gap_source.jl  —  TODO #8 (where the LP bound is loose): the model variants of int_flow/run_decomp.jl
(--drop S | M | SM, --method nox | bin) on the split toy / syn n3T6 instances of test_decomp.jl (Gurobi).

    julia --project=model_2stage/env model_2stage/test/test_gap_source.jl

G1  drop_service_rule! deletes exactly the 2 n T service rule rows
G2  each variant is a relaxation of the full MIP: MIP_full ≤ MIP_S, MIP_M ≤ MIP_SM; MIP_full ≤ MIP_nox ≤ MIP_bin ≤ LP_full;
    LP ≥ MIP of every variant; LP_full ≤ LP_S, LP_M ≤ LP_SM
G3  the full MIP solution has no service rule violation and no blocking pair; a variant solution worth more than
    the full MIP breaks what that variant dropped (S: a service violation, M: a blocking pair, SM: either)
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "..", "int_flow", "decomp.jl"))

using Printf, Gurobi

const GENV = Gurobi.Env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 2, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
le(a, b) = a <= b + 1e-6 * max(1.0, abs(b))

with_split(inst, scenU) =
    TSInstance(inst.name * "/int", inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, inst.Q3, inst.A0, inst.U0, inst.W0, inst.menu, inst.structure,
               inst.group, inst.ngroups, inst.scen, inst.vterm, scenU, [inst.A0 for _ in inst.scen],
               [inst.U0 for _ in inst.scen])
pattern_split(inst) = [[[(ξ[t][i, j] * (3i + 5j + 7t)) % (ξ[t][i, j] + 1) for i in 1:inst.n, j in 1:inst.n]
                        for t in 1:inst.T] for ξ in inst.scen]
prices(n) = [("u2", fill(2.0, n, n)), ("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])]

"""Solve a variant as run_decomp.jl builds it: drop ∈ "", "S", "M", "SM"; form ∈ :full, :lp, :nox, :bin."""
function variant(v, P, drop, form)
    m = Model(opt())
    b = add_scenario_sparse!(m, v, v.scen[1], P, P; stability = 'M' in drop ? :none : :tier)
    nd = 'S' in drop ? drop_service_rule!(m, b) : 0
    form === :lp || nothing_start!(v, b)
    @objective(m, Max, b.profit)
    form === :full || relax_ints!(b, form)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("$drop $form: $(JuMP.termination_status(m))")
    return (value = objective_value(m), b = b, dropped = nd)
end

for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T)")
    for s in eachindex(old.scen), (pn, P) in prices(old.n)
        v = scenario_view(dead, s)
        mip = Dict(d => variant(v, P, d, :full) for d in ("", "S", "M", "SM"))
        lp = Dict(d => variant(v, P, d, :lp).value for d in ("", "S", "M", "SM"))
        nox = variant(v, P, "", :nox).value
        bin = variant(v, P, "", :bin).value
        f = mip[""].value
        say = @sprintf("s%d %s: MIP full %.1f S %.1f M %.1f SM %.1f nox %.1f bin %.1f | LP full %.1f S %.1f M %.1f SM %.1f",
                       s, pn, f, mip["S"].value, mip["M"].value, mip["SM"].value, nox, bin, lp[""], lp["S"], lp["M"],
                       lp["SM"])
        println("  ", say)
        check(mip["S"].dropped == 2 * v.n * v.T && mip["SM"].dropped == 2 * v.n * v.T, "G1 $tag s$s $pn: 2nT rows")
        check(le(f, mip["S"].value) && le(f, mip["M"].value) && le(mip["S"].value, mip["SM"].value) &&
              le(mip["M"].value, mip["SM"].value) && le(f, nox) && le(nox, bin) && le(bin, lp[""]) &&
              all(le(mip[d].value, lp[d]) for d in keys(lp)) &&
              le(lp[""], lp["S"]) && le(lp[""], lp["M"]) && le(lp["S"], lp["SM"]) && le(lp["M"], lp["SM"]),
              "G2 $tag s$s $pn: relaxation order")
        bf = mip[""].b
        check(isempty(service_violations(v, bf)) && isempty(blocking_pairs_sparse(v, bf, P)),
              "G3 $tag s$s $pn: full solution keeps the service rule and stability")
        sv(d) = length(service_violations(v, mip[d].b))
        bp(d) = length(blocking_pairs_sparse(v, mip[d].b, P))
        ok = (le(mip["S"].value, f) || sv("S") > 0) && (le(mip["M"].value, f) || bp("M") > 0) &&
             (le(mip["SM"].value, f) || sv("SM") + bp("SM") > 0)
        check(ok, @sprintf("G3 %s s%d %s: a variant above full breaks what it dropped (S %d viol, M %d bp, SM %d + %d)",
                           tag, s, pn, sv("S"), bp("M"), sv("SM"), bp("SM")))
    end
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)

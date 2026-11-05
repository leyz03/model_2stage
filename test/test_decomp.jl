"""
test_decomp.jl  —  TODO #3 (int_flow/decomp.jl): column generation on x and Benders over (x, β, z) with the LP
subproblem, on the split toy / syn n3T6 instances of test_partial_int.jl (Gurobi: Benders needs Farkas rays).

    julia --project=model_2stage/env model_2stage/test/test_decomp.jl

D1  cg_solve: LP phase ends at the full LP value; final value = full MIP value, proved; the solution is integral,
    residual 0, no blocking pair; from the d0 start with and without the fix step, and from all columns.  Where
    LP > MIP (syn n3T6) the restricted MIP falls short of the LP bound and the reduced-cost-fixing step runs
D2  benders_solve: value = full MIP value, proved; LP phase ends at the full LP value; the subproblem solution at
    the best v is integral, residual 0, no blocking pair
D3  every Benders cut is valid at the full MIP optimum v*: optimality cuts ≥ full value, feasibility cuts ≤ 0
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
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

# the split instances of test_int_flow.jl / test_partial_int.jl
with_split(inst, scenU) =
    TSInstance(inst.name * "/int", inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, inst.Q3, inst.A0, inst.U0, inst.W0, inst.menu, inst.structure,
               inst.group, inst.ngroups, inst.scen, inst.vterm, scenU, [inst.A0 for _ in inst.scen],
               [inst.U0 for _ in inst.scen])
pattern_split(inst) = [[[(ξ[t][i, j] * (3i + 5j + 7t)) % (ξ[t][i, j] + 1) for i in 1:inst.n, j in 1:inst.n]
                        for t in 1:inst.T] for ξ in inst.scen]
prices(n) = [("u2", fill(2.0, n, n)), ("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])]

function solved(v, P, form)
    m = Model(opt())
    b = add_scenario_sparse!(m, v, v.scen[1], P, P; stability = :tier)
    form === :lp || nothing_start!(v, b)
    @objective(m, Max, b.profit)
    relax_ints!(b, form)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("$form: $(JuMP.termination_status(m))")
    return objective_value(m), m, b
end

clean(v, b, P) = (frac = nfrac(value.(int_vars(b))) + nfrac(value.(x_vars(b))) + nfrac(value.(bin_vars(b))),
                  res = conservation_residual(v, b), bp = length(blocking_pairs_sparse(v, b, P)))

for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T)")
    for s in eachindex(old.scen), (pn, P) in prices(old.n)
        v = scenario_view(dead, s)
        vf, mf, bf = solved(v, P, :full)
        vstar = value.(master_vars(bf))
        vl, = solved(v, P, :lp)
        # D1
        for (init, fixstep) in ((:d0, true), (:d0, false), (:all, true))
            r = cg_solve(v, P, opt(); init, fixstep, verbose = false)
            c = clean(v, r.block, P)
            nmip = count(l -> l.phase != "lp", r.log)
            check(close_to(r.lp_bound, vl) && close_to(r.value, vf) && r.proved && c.frac == 0 && c.res < 1e-6 && c.bp == 0,
                  @sprintf("D1 %s s%d %s %-3s fix %-5s cg %.4f = full %.4f (lp %.4f / %.4f), proved %s, %d/%d columns, %d lp + %d fix/mip steps; frac %d res %.1e bp %d",
                           tag, s, pn, init, fixstep, r.value, vf, r.lp_bound, vl, r.proved, r.active, r.columns,
                           length(r.log) - nmip, nmip, c.frac, c.res, c.bp))
        end
        # D2
        r = benders_solve(v, P, opt(); keep_cuts = true, verbose = false, tl = 600.0)
        c = clean(v, r.block, P)
        nfeas = count(l -> l.cut == "feas", r.log)
        check(close_to(r.value, vf) && r.proved && close_to(r.lp_bound, vl) && c.frac == 0 && c.res < 1e-6 && c.bp == 0,
              @sprintf("D2 %s s%d %s  benders %.4f (ub %.4f) = full %.4f, lp %.4f / %.4f, %d iterations (%d feasibility); frac %d res %.1e bp %d",
                       tag, s, pn, r.value, r.bound, vf, r.lp_bound, vl, length(r.log), nfeas, c.frac, c.res, c.bp))
        # D3
        okc = all(r.cuts) do k
            a = k.c0 + sum(k.coef .* vstar)
            k.kind === :opt ? a >= vf - 1e-6 * max(1.0, abs(vf)) : a <= 1e-6
        end
        check(okc, @sprintf("D3 %s s%d %s  %d cuts valid at the full optimum", tag, s, pn, length(r.cuts)))
    end
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")

"""
test_partial_int.jl  —  TODO #2 (int_flow/partial_int.jl): integrality on some variables only, sparse tier block on
the integer rider split.

    julia --project=model_2stage/env model_2stage/test/test_partial_int.jl [gurobi]

P1  relax_ints! leaves the right groups integral (:full / :xbin / :x / :lp) and restore_ints! undoes it
P2  :xbin = :full (the derivation of weekly/2026-10-31_11-06/TODO.md "#2 核对"); cleanup! (x and binaries fixed,
    simplex) keeps the value, every general integer integral, conservation residual 0, no blocking pair
P3  :full <= :x <= :lp (relaxations); the exact model with the x of the :x optimum fixed is <= :full when feasible
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "..", "int_flow", "partial_int.jl"))

using Printf

if "gurobi" in ARGS
    using Gurobi
    const GENV = Gurobi.Env()
    opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => 2, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
    const LP_ATTRS = ("Method" => 1,)
else
    using HiGHS
    opt() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
    const LP_ATTRS = ("solver" => "simplex",)
end

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

# the split instances of test_int_flow.jl
with_split(inst, scenU) =
    TSInstance(inst.name * "/int", inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, inst.Q3, inst.A0, inst.U0, inst.W0, inst.menu, inst.structure,
               inst.group, inst.ngroups, inst.scen, inst.vterm, scenU, [inst.A0 for _ in inst.scen],
               [inst.U0 for _ in inst.scen])
pattern_split(inst) = [[[(ξ[t][i, j] * (3i + 5j + 7t)) % (ξ[t][i, j] + 1) for i in 1:inst.n, j in 1:inst.n]
                        for t in 1:inst.T] for ξ in inst.scen]
prices(n) = [("u2", fill(2.0, n, n)), ("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])]

function build(v, P, form)
    m = Model(opt())
    b = add_scenario_sparse!(m, v, v.scen[1], P, P; stability = :tier)
    form === :lp || nothing_start!(v, b)
    @objective(m, Max, b.profit)
    return m, b, relax_ints!(b, form)
end

function solved(v, P, form)
    m, b, r = build(v, P, form)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("$form: $(JuMP.termination_status(m))")
    return objective_value(m), m, b, r
end

# P1: the groups
let old = load_ts_instance(TOY, "toy-A"; levels = 2, nscen = 2, phi = 0)
    v = scenario_view(with_split(old, pattern_split(old)), 1)
    P = fill(4.0, v.n, v.n)
    for form in (:full, :xbin, :x, :lp)
        m, b, r = build(v, P, form)
        ix = all(is_integer, x_vars(b)) == (form !== :lp)
        ib = all(is_binary, bin_vars(b)) == (form in (:full, :xbin))
        ii = all(is_integer, int_vars(b)) == (form === :full) && (form === :full || !any(is_integer, int_vars(b)))
        bb = form in (:full, :xbin) || all(v -> lower_bound(v) == 0 && upper_bound(v) == 1, bin_vars(b))
        restore_ints!(r)
        back = all(is_integer, x_vars(b)) && all(is_binary, bin_vars(b)) && all(is_integer, int_vars(b)) &&
               !any(has_upper_bound, bin_vars(b))
        check(ix && ib && ii && bb && back,
              @sprintf("P1 %-4s  x int %s, binaries %s, general ints %s, [0,1] bounds %s, restored %s (%d x, %d bin, %d int)",
                       form, ix, ib, ii, bb, back, length(x_vars(b)), length(bin_vars(b)), length(int_vars(b))))
    end
end

for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T)")
    for s in eachindex(old.scen), (pn, P) in prices(old.n)
        v = scenario_view(dead, s)
        vf, mf, bf = solved(v, P, :full)
        # P2
        vb, mb, bb = solved(v, P, :xbin)
        cl = cleanup!(mb, bb; lp_attrs = LP_ATTRS)
        res = conservation_residual(v, bb)
        bp = length(blocking_pairs_sparse(v, bb, P))
        check(close_to(vb, vf) && cl.status == "OPTIMAL" && close_to(cl.value, vf) && cl.frac_after == 0 &&
              res < 1e-6 && bp == 0,
              @sprintf("P2 %s s%d %s  xbin %.4f = full %.4f; cleanup %.4f, fractional %d -> %d, residual %.1e, blocking %d",
                       tag, s, pn, vb, vf, cl.value, cl.frac_before, cl.frac_after, res, bp))
        # P3
        vx, mx, bx, rx = solved(v, P, :x)
        vl, = solved(v, P, :lp)
        xv = round.(value.(x_vars(bx)))
        restore_ints!(rx)
        foreach((u, a) -> fix(u, a; force = true), x_vars(bx), xv)
        optimize!(mx)
        st = JuMP.termination_status(mx)
        fx = st == MOI.OPTIMAL ? objective_value(mx) : NaN
        okfx = st == MOI.INFEASIBLE || (st == MOI.OPTIMAL && fx <= vf + 1e-6 * max(1, abs(vf)))
        check(vf <= vx + 1e-6 * max(1, abs(vx)) && vx <= vl + 1e-6 * max(1, abs(vl)) && okfx,
              @sprintf("P3 %s s%d %s  full %.4f <= x %.4f <= lp %.4f; full with that x: %s %.4f",
                       tag, s, pn, vf, vx, vl, st, fx))
    end
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")

"""
test_lazy_stab.jl  —  TODO #9 (int_flow/lazy_stab.jl): the stable-matching rows on demand, on the split toy /
syn n3T6 instances of test_decomp.jl (2 scenarios × 3 prices each, Gurobi: lazy-constraint callback).

    julia --project=model_2stage/env model_2stage/test/test_lazy_stab.jl

L1  all groups added to the block built without stability = the tier block: same number of rows and of z, same LP
    and MIP value
L2  the checks match the model: at the MIP optimum without stability rows, separate_int finds a group iff
    blocking_pairs_sparse finds a pair; at the tier MIP optimum (all groups as rows) separate_int and at its LP
    optimum separate_lp find nothing
L3  full MIP with the lazy callback: value = tier MIP value, optimal, integral, residual 0, no blocking pair
L4  row generation on the full LP (cg_solve init = :all, stab = :lazy, lp_only): value = tier LP value
L5  cg_solve with stab = :lazy (d0 start, with and without the fix step): value = tier MIP value, proved, LP phase
    = tier LP value, integral, residual 0, no blocking pair
L6  as L5 with sep_tol = −2: every group added in the LP phase (the add-rows / dual-simplex re-solve path, which the
    small instances never reach otherwise: their LP sees no stable-matching row)
L7  cg_solve with stab = :defer (LP phase as L5, every remaining group added as rows before the MIP, no callback):
    value = tier MIP value, proved, LP phase = tier LP value, every group active at the end, integral, residual 0,
    no blocking pair (with and without the fix step)
At the end: the callback rejected a candidate in at least one case (else L3 / L5 test nothing).
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "..", "int_flow", "decomp.jl"))        # also partial_int.jl, lazy_stab.jl

using Printf, Gurobi

const GENV = Gurobi.Env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 2, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
rejects = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

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

"""Block without stability rows plus every z; `groups = :all` adds every group, `:none` none; `lazy` the callback."""
function solved_none(v, P; groups::Symbol = :none, lp::Bool = false, lazy::Bool = false)
    m = Model(opt())
    b = add_scenario_sparse!(m, v, v.scen[1], P, P; stability = :none)
    S = stab_init!(m, v, b)
    groups === :all && add_all_groups!(m, S)
    lp || nothing_start!(v, b)
    @objective(m, Max, b.profit)
    if lp
        relax_ints!(b, :lp)
        relax_z!(S)
    end
    lazy && lazy_stab!(m, S)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("none/$groups: $(JuMP.termination_status(m))")
    return objective_value(m), m, b, S
end

nrows(m) = sum(num_constraints(m, F, S) for (F, S) in list_of_constraint_types(m) if F <: GenericAffExpr)
clean(v, b, P) = (frac = nfrac(value.(int_vars(b))) + nfrac(value.(x_vars(b))) + nfrac(value.(bin_vars(b))),
                  res = conservation_residual(v, b), bp = length(blocking_pairs_sparse(v, b, P)))

for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T)")
    for s in eachindex(old.scen), (pn, P) in prices(old.n)
        v = scenario_view(dead, s)
        vf, mf, bf = solved(v, P, :full)
        vl, ml, = solved(v, P, :lp)
        # L1
        va, ma, ba, Sa = solved_none(v, P; groups = :all)
        vla, mla, = solved_none(v, P; groups = :all, lp = true)
        check(nrows(ma) == nrows(mf) && length(Sa.z) == bf.ntier && close_to(va, vf) && close_to(vla, vl),
              @sprintf("L1 %s s%d %s  all groups: %d rows = tier %d, %d z = %d, MIP %.4f = %.4f, LP %.4f = %.4f",
                       tag, s, pn, nrows(ma), nrows(mf), length(Sa.z), bf.ntier, va, vf, vla, vl))
        # L2
        vn, mn, bn, Sn = solved_none(v, P)
        nbp = length(blocking_pairs_sparse(v, bn, P))
        gn = separate_int(Sn, value)
        ga = separate_int(Sa, value; skip_active = false)
        _, mlp, blp, Slp = solved_none(v, P; groups = :all, lp = true)
        gl = separate_lp(Slp, value; skip_active = false)
        check((nbp > 0) == !isempty(gn) && isempty(ga) && isempty(gl),
              @sprintf("L2 %s s%d %s  no stability: MIP %.4f, %d blocking pairs, %d violated groups; tier: %d / LP %d",
                       tag, s, pn, vn, nbp, length(gn), length(ga), length(gl)))
        # L3
        vz, mz, bz, Sz = solved_none(v, P; lazy = true)
        c = clean(v, bz, P)
        global rejects += Sz.nreject
        check(close_to(vz, vf) && c.frac == 0 && c.res < 1e-6 && c.bp == 0,
              @sprintf("L3 %s s%d %s  full lazy %.4f = full %.4f; %d candidates, %d rejected, %d lazy rows; frac %d res %.1e bp %d",
                       tag, s, pn, vz, vf, Sz.ncalls, Sz.nreject, Sz.nlazy, c.frac, c.res, c.bp))
        # L4
        r = cg_solve(v, P, opt(); init = :all, stab = :lazy, lp_only = true, verbose = false)
        check(close_to(r.lp_bound, vl),
              @sprintf("L4 %s s%d %s  LP row generation %.4f = LP %.4f, %d rounds, %d of %d groups",
                       tag, s, pn, r.lp_bound, vl, length(r.log), count(r.stab.active), ngroups(r.stab)))
        # L5
        for fixstep in (true, false)
            r = cg_solve(v, P, opt(); init = :d0, fixstep, stab = :lazy, verbose = false)
            c = clean(v, r.block, P)
            global rejects += r.stab.nreject
            check(close_to(r.lp_bound, vl) && close_to(r.value, vf) && r.proved && c.frac == 0 && c.res < 1e-6 && c.bp == 0,
                  @sprintf("L5 %s s%d %s fix %-5s cg lazy %.4f = full %.4f (lp %.4f / %.4f), proved %s, %d/%d columns, groups %d (LP phase %d), %d rejected; frac %d res %.1e bp %d",
                           tag, s, pn, fixstep, r.value, vf, r.lp_bound, vl, r.proved, r.active, r.columns,
                           count(r.stab.active), sum(l.groups_added for l in r.log if l.phase == "lp"),
                           r.stab.nreject, c.frac, c.res, c.bp))
        end
        # L6
        r = cg_solve(v, P, opt(); init = :d0, stab = :lazy, sep_tol = -2.0, verbose = false)
        c = clean(v, r.block, P)
        lpg = sum(l.groups_added for l in r.log if l.phase == "lp")
        check(lpg == ngroups(r.stab) && close_to(r.lp_bound, vl) && close_to(r.value, vf) && r.proved && c.frac == 0 &&
              c.res < 1e-6 && c.bp == 0,
              @sprintf("L6 %s s%d %s  cg lazy, every group added in the LP phase (%d of %d, %d LP rounds): %.4f = full %.4f, lp %.4f / %.4f, proved %s; frac %d res %.1e bp %d",
                       tag, s, pn, lpg, ngroups(r.stab), count(l -> l.phase == "lp", r.log), r.value, vf, r.lp_bound,
                       vl, r.proved, c.frac, c.res, c.bp))
        # L7
        for fixstep in (true, false)
            r = cg_solve(v, P, opt(); init = :d0, fixstep, stab = :defer, verbose = false)
            c = clean(v, r.block, P)
            ab = [l for l in r.log if l.phase == "addback"]
            check(close_to(r.lp_bound, vl) && close_to(r.value, vf) && r.proved && c.frac == 0 && c.res < 1e-6 &&
                  c.bp == 0 && count(r.stab.active) == ngroups(r.stab) && length(ab) == 1,
                  @sprintf("L7 %s s%d %s fix %-5s cg defer %.4f = full %.4f (lp %.4f / %.4f), proved %s, %d of %d groups at the end (%d added back, %d rows); frac %d res %.1e bp %d",
                           tag, s, pn, fixstep, r.value, vf, r.lp_bound, vl, r.proved, count(r.stab.active),
                           ngroups(r.stab), isempty(ab) ? -1 : ab[1].groups_added, isempty(ab) ? -1 : ab[1].rows,
                           c.frac, c.res, c.bp))
        end
    end
end

check(rejects > 0, "the lazy callback rejected $rejects candidates in total (> 0: L3 / L5 exercised it)")
println(fails == 0 ? "ALL PASS" : "$fails FAILED")

"""
test_int_flow.jl  —  the integer rider split (no battery levels, every trip flow integer; model_2stage/int_flow,
doc/2026-10-31/modify_2_*): `scenario_view`, YU in add_scenario! / add_scenario_sparse!, the MIP starts, time blocks.

    julia --project=model_2stage/env model_2stage/test/test_int_flow.jl [gurobi] [paris=<int instance json>] [tw=<periods>]

I1  (paris=) the instance file: 0 <= scenU <= scen; common random numbers (equal demand of (t, i, j) in two scenarios
    -> equal split, more demand -> no fewer dead riders); every view has its own morning state and fleet
I2  a split with no dead rider = the old model with φ = 0 (Γ = 1), scenario by scenario: sparse tier, sparse R1, dense :pre
I3  a split with dead riders: sparse tier = dense :pre, sparse R1 = dense price-box relaxation at (p, p); conservation
    residual 0; YU integral, YU <= scenU, Yod - YU <= ξ - scenU; dead riders are served
I4  nothing_start! (R1 and tier rows) and continue_start! from the morning state are feasible integral points, <= optimum
I5  the linked sparse block model (2 blocks, window_instance) has the whole-day optimum
I6  (paris=) periods 1..tw (default 6) of scenario 1 at the baseline price: the start is feasible, R1 solves with
    residual 0 and value >= the start; building a block on the 30-scenario instance itself is refused
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl

using Printf

if "gurobi" in ARGS
    using Gurobi
    const GENV = Gurobi.Env()
    opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => 2, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
else
    using HiGHS
    opt() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
end

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

"""`inst` with the rider split `scenU` and its own morning state repeated for every scenario."""
with_split(inst, scenU) =
    TSInstance(inst.name * "/int", inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, inst.Q3, inst.A0, inst.U0, inst.W0, inst.menu, inst.structure,
               inst.group, inst.ngroups, inst.scen, inst.vterm, scenU, [inst.A0 for _ in inst.scen],
               [inst.U0 for _ in inst.scen])

"""A deterministic split with dead riders on most busy ODs."""
pattern_split(inst) = [[[(ξ[t][i, j] * (3i + 5j + 7t)) % (ξ[t][i, j] + 1) for i in 1:inst.n, j in 1:inst.n]
                        for t in 1:inst.T] for ξ in inst.scen]

function start_ok(inst, P, stability; cont = false)
    m = Model()
    b = add_scenario_sparse!(m, inst, inst.scen[1], P, P; stability)
    cont ? continue_start!(inst, b, 1, initial_state(inst, b)) : nothing_start!(inst, b)
    pt = Dict(v => something(start_value(v), is_fixed(v) ? fix_value(v) : NaN) for v in all_variables(m))
    any(isnan, values(pt)) && return false, -Inf
    rep = primal_feasibility_report(m, pt; atol = 1e-6)
    ints = all(abs(pt[v] - round(pt[v])) < 1e-9 for v in all_variables(m) if is_integer(v) || is_binary(v))
    return isempty(rep) && ints, value(vv -> pt[vv], b.profit)
end

function split_ok(inst, b)
    YU, Y = value.(b.YU), value.(b.Yod)
    ξ, ξU = inst.scen[1], inst.scenU[1]
    ok = all(abs(v - round(v)) < 1e-6 for v in YU)
    for t in 1:inst.T, i in 1:inst.n, j in 1:inst.n
        ok &= YU[i, j, t] <= ξU[t][i, j] + 1e-6 && Y[1, i, j, t] - YU[i, j, t] <= ξ[t][i, j] - ξU[t][i, j] + 1e-6
    end
    return ok, sum(YU)
end

prices(n) = [("u2", fill(2.0, n, n)), ("u4", fill(4.0, n, n)), ("t25", [j == k ? 5.0 : 2.0 for j in 1:n, k in 1:n])]

for (path, var, tag) in ((TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"))
    old = load_ts_instance(path, var; levels = 2, nscen = 2, phi = 0)
    zero = with_split(old, [[zeros(Int, old.n, old.n) for _ in 1:old.T] for _ in old.scen])
    dead = with_split(old, pattern_split(old))
    println("== $tag  n=$(old.n) T=$(old.T) Γ=$(old.Γ)  dead riders $(sum(sum(sum, s) for s in dead.scenU)) of $(sum(sum(sum, s) for s in old.scen))")
    for s in eachindex(old.scen), (pn, P) in prices(old.n)
        # I2: no dead rider = old model, φ = 0
        vz, = solve_tier_sparse(scenario_view(zero, s), 1, P, opt())
        vo, = solve_tier_sparse(old, s, P, opt())
        rz, = solve_r1_sparse(scenario_view(zero, s), 1, P, opt())
        ro, = solve_r1_sparse(old, s, P, opt())
        dz, = solve_scenario(scenario_view(zero, s), 1, P, opt(); stability = :pre)
        check(close_to(vz, vo) && close_to(rz, ro) && close_to(dz, vo),
              @sprintf("I2 %s s%d %s  zero split: tier %.4f = old %.4f, R1 %.4f = %.4f, dense %.4f", tag, s, pn, vz, vo, rz, ro, dz))
        # I3: dead riders, sparse = dense
        v = scenario_view(dead, s)
        vs, _, bs, is = solve_tier_sparse(v, 1, P, opt())
        vd, _, bd, = solve_scenario(v, 1, P, opt(); stability = :pre)
        rs, = solve_r1_sparse(v, 1, P, opt())
        rd, = solve_relax(v, 1, P, P, opt())
        oks, ys = split_ok(v, bs)
        okd, yd = split_ok(v, bd)
        res = max(conservation_residual(v, bs), conservation_residual(v, bd))
        check(is.optimal && close_to(vs, vd) && close_to(rs, rd) && res < 1e-6 && oks && okd && ys > 0,
              @sprintf("I3 %s s%d %s  tier sparse %.4f dense %.4f, R1 %.4f / %.4f, residual %.1e, dead served %.0f / %.0f",
                       tag, s, pn, vs, vd, rs, rd, res, ys, yd))
        # I4: MIP starts
        for (stab, best) in ((:none, rs), (:tier, vs)), cont in (false, true)
            ok, val = start_ok(v, P, stab; cont)
            check(ok && val <= best + 1e-6, @sprintf("I4 %s s%d %s %s %s  start feasible, %.1f <= %.1f", tag, s, pn,
                                                     stab, cont ? "continue_start!" : "nothing_start!", val, best))
        end
        # I5: linked blocks = whole day
        T = v.T
        m, = td_joint(v, 1, P, [(1, T ÷ 2), (T ÷ 2 + 1, T)], opt(); form = :sparse)
        optimize!(m)
        vj = objective_value(m)
        check(JuMP.termination_status(m) == MOI.OPTIMAL && close_to(vj, vs),
              @sprintf("I5 %s s%d %s  two linked blocks %.4f = whole day %.4f", tag, s, pn, vj, vs))
    end
end

pa = findfirst(a -> startswith(a, "paris="), ARGS)
if pa !== nothing
    path = ARGS[pa][7:end]
    iw = findfirst(a -> startswith(a, "tw="), ARGS)
    tw = iw === nothing ? 6 : parse(Int, ARGS[iw][4:end])
    raw = JSON.parsefile(path)
    full = load_ts_instance(path, "paris"; levels = 2)
    S, n, T = length(full.scen), full.n, full.T
    println("== $(basename(path))  $S scenarios, n=$n T=$T Γ=$(full.Γ)")
    # I1: the data
    okb = all(all(0 .<= full.scenU[s][t] .<= full.scen[s][t]) for s in 1:S, t in 1:T)
    pairs = [(full.scen[s1][t], full.scen[s2][t], full.scenU[s1][t], full.scenU[s2][t]) for s1 in 1:S, s2 in 1:S, t in 1:T if s1 < s2]
    crn = all(all(u1[d1 .== d2] .== u2[d1 .== d2]) for (d1, d2, u1, u2) in pairs)
    mono = all(all(u1[d1 .< d2] .<= u2[d1 .< d2]) && all(u2[d2 .< d1] .<= u1[d2 .< d1]) for (d1, d2, u1, u2) in pairs)
    fleet = raw["int_split"]["fleet_per_scenario"]
    views = [scenario_view(full, s) for s in 1:S]
    okv = all(views[s].Q3 == fleet[s] && sum(views[s].A0) + sum(views[s].U0) == fleet[s] && views[s].A0 == full.A0s[s]
              for s in 1:S)
    dead = [sum(sum, full.scenU[s]) for s in 1:S]
    check(okb && crn && mono && okv && dead == raw["int_split"]["dead_riders_per_scenario"],
          @sprintf("I1 bounds %s, common random numbers %s, monotone %s, views %s; dead riders/day %.1f, fleet %d-%d",
                   okb, crn, mono, okv, sum(dead) / S, minimum(fleet), maximum(fleet)))
    # I6: a short window of scenario 1
    P = fill(Float64(raw["pricing"]["baseline"]), n, n)
    refused = try
        add_scenario_sparse!(Model(), full, full.scen[1], P, P); false
    catch e
        occursin("scenario_view", sprint(showerror, e))
    end
    w = window_instance(full, 1, 1, tw)
    ok, v0 = start_ok(w, P, :none)
    t1 = @elapsed (r, _, b, info) = solve_r1_sparse(w, 1, P, opt())
    okw, ys = split_ok(w, b)
    res = conservation_residual(w, b)
    check(refused && ok && info.optimal && r >= v0 - 1e-6 && okw && res < 1e-6,
          @sprintf("I6 periods 1-%d: refused on the full instance %s, start %.1f feasible %s, R1 %.1f (%.1f s, %d vars), dead served %.0f, residual %.1e",
                   tw, refused, v0, ok, r, t1, info.nvar, ys, res))
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")

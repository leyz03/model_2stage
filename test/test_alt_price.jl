"""
test_alt_price.jl  —  TODO #4 (int_flow/alt_price.jl): the P step of the alternating method, its MIP start and the
upper bound, on the integer rider split.

    julia --project=model_2stage/env model_2stage/test/test_alt_price.jl [gurobi]

A1  least_prices on random difference rows (4 ODs, menu 0..5) = brute force: :keep is the least feasible point with
    the unused ODs at their price; :follow has the used ODs at the least point with every OD free and each unused OD at
    the largest feasible level <= its price
A2  day 11, periods 1..20, menu 0..6, from uniform 6: the operations have no blocking pair and are individually
    rational at the P step's price q <= 6; lowering any used OD by one level breaks one of the two (so price_rows says the same as
    blocking_pairs_sparse); the MIP start at q is feasible with value = the old value + the wage saved; the O step at q
    is >= that value
A3  discrim_ub (menu 0..6) >= the exact value at uniform 4, at uniform 6 and at q

Price 12 allows 25,903 worker-task triples (price 6: 11,263; price 4: 4,310); one T = 20 solve at 12 took > 10 min
and 22 GB, so the test stays at 6.
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "..", "int_flow", "alt_price.jl"))

using Printf, Random

if "gurobi" in ARGS
    using Gurobi
    const GENV = Gurobi.Env()
    opt(s = 0) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => 8, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
else
    using HiGHS
    opt(s = 0) = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
end

const INST = joinpath(@__DIR__, "..", "int_flow", "instances", "paris_n30_H10_int_workers.json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

println("A1  least_prices = brute force on random rows")
let rng = MersenneTwister(20261101), menu = collect(0.0:5.0), n = 2, bad = 0, cases = 0
    menus = uniform_menus(n, menu)
    grid = vec(collect(Iterators.product(ntuple(_ -> menu, n * n)...)))
    for case in 1:300
        p0 = rand(rng, menu, n, n)
        used = rand(rng, Bool, n, n)
        any(used) || (used[1] = true)
        lb = [used[jk] ? rand(rng, menu[menu .<= p0[jk]]) - rand(rng, 0:1) : -Inf for jk in CartesianIndices(p0)]
        ub = [rand(rng) < 0.3 ? p0[jk] + rand(rng, 0:2) : Inf for jk in CartesianIndices(p0)]
        diff = Dict{NTuple{4,Int},Float64}()
        for _ in 1:rand(rng, 0:5)
            a = rand(rng, findall(vec(used)))
            b = rand(rng, 1:n*n)
            a == b && continue
            ja, ka = Tuple(CartesianIndices(p0)[a])
            jb, kb = Tuple(CartesianIndices(p0)[b])
            diff[(ja, ka, jb, kb)] = p0[a] - p0[b] - rand(rng, 0:2)
        end
        rows = (lb = lb, ub = ub, vol = Float64.(used), diff = diff)
        rows_ok(p0, menus, rows) || error("generator")
        pts = [reshape(collect(g), n, n) for g in grid]
        feas = filter(p -> rows_ok(p, menus, rows), pts)
        # :keep
        fk = filter(p -> all(used[jk] || p[jk] == p0[jk] for jk in eachindex(p)), feas)
        lk = [minimum(p[jk] for p in fk) for jk in CartesianIndices(p0)]
        qk = least_prices(p0, menus, rows; unused = :keep)
        # :follow
        lf = [minimum(p[jk] for p in feas) for jk in CartesianIndices(p0)]
        qf = least_prices(p0, menus, rows; unused = :follow)
        rest = filter(p -> all(!used[jk] || p[jk] == lf[jk] for jk in eachindex(p)) && all(p .<= p0), feas)
        hi = [used[jk] ? lf[jk] : maximum(p[jk] for p in rest) for jk in CartesianIndices(p0)]
        cases += 1
        (qk == lk && lk in fk && qf == hi && rows_ok(qf, menus, rows)) || (bad += 1)
    end
    check(bad == 0, "$cases random cases, $bad mismatches")
end

println("A2  P step on day 11, T = 20, from uniform 6")
full = load_ts_instance(INST, "paris"; levels = 2)
w = window_instance(full, 11, 1, 20)
n, d, c = w.n, w.d, w.c
menus = uniform_menus(n, 0:6)
function solved(p)
    m = Model(opt())
    b = add_scenario_sparse!(m, w, w.scen[1], p, p; stability = :tier)
    nothing_start!(w, b)
    @objective(m, Max, b.profit)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("$(JuMP.termination_status(m))")
    return objective_value(m), m, b
end
ir_ok(b, p) = all(p[j, k] >= d[i, j] + c[j, k] - 1e-9 for (e, (i, j, k)) in enumerate(b.trips), t in 1:w.T
                  if value(b.x[e, t]) > 0.5)
stable_at(b, p) = ir_ok(b, p) && isempty(blocking_pairs_sparse(w, b, p))
phi = fill(6.0, n, n)
vhi, mhi, bhi = solved(phi)
sn = snapshot(bhi)
rows = price_rows([w], [sn])
ntask = sum(values(sn.x); init = 0)
@printf("  uniform 6: value %.2f, %d tasks, %d ODs used, %d difference rows, %d idle bounds\n", vhi, ntask,
        count(rows.vol .> 0), length(rows.diff), count(isfinite, rows.ub))
check(ntask > 0, "the window has tasks at price 6")
check(stable_at(bhi, phi), "the operations are stable at 6")
for rule in (:keep, :follow)
    q = least_prices(phi, menus, rows; unused = rule)
    check(all(q .<= phi) && stable_at(bhi, q), "$rule: q <= 6 and the operations stay stable and IR at q")
    used = findall(rows.vol .> 0)
    tight = 0
    for jk in used
        h = findfirst(==(q[jk]), menus[jk])
        h == 1 && (tight += 1; continue)
        q2 = copy(q); q2[jk] = menus[jk][h-1]
        stable_at(bhi, q2) || (tight += 1)
    end
    check(tight == length(used), "$rule: lowering any of the $(length(used)) used ODs by one level breaks stability or IR")
    saved = sum((phi .- q) .* task_volume(sn))
    m2 = Model(opt())
    b2 = add_scenario_sparse!(m2, w, w.scen[1], q, q; stability = :tier)
    start_from!(b2, sn)
    @objective(m2, Max, b2.profit)
    pt = Dict(v => something(start_value(v), NaN) for v in all_variables(m2))
    check(all(isfinite, values(pt)), "$rule: every variable has a start value")
    rep = primal_feasibility_report(m2, pt; atol = 1e-6)
    check(isempty(rep), "$rule: the start is feasible at q ($(length(rep)) violated rows)")
    v0 = value(v -> pt[v], b2.profit)
    check(close_to(v0, vhi + saved), @sprintf("%s: start value %.2f = %.2f + wage saved %.2f", rule, v0, vhi, saved))
    vals, _, infos = ostep([w], q, opt; starts = [sn])
    check(vals[1] >= v0 - 1e-6 && infos[1].blocking == 0, @sprintf("%s: O step at q %.2f >= %.2f, no blocking pair",
                                                                     rule, vals[1], v0))
    global qkeep = rule === :keep ? q : qkeep
    global vkeep = rule === :keep ? vals[1] : vkeep
end

println("A3  upper bound")
v4, _, _ = solved(fill(4.0, n, n))
ub, info = discrim_ub(w, menus, opt())
@printf("  ub %.2f (value %.2f, paid %.2f): uniform 4 %.2f, uniform 6 %.2f, keep q %.2f\n", ub, info.value, info.paid,
        v4, vhi, vkeep)
check(ub >= max(v4, vhi, vkeep) - 1e-6, "discrim_ub >= uniform 4, uniform 6 and the P step's price")

println(fails == 0 ? "ALL PASS" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)

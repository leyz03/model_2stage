"""
test_sparse.jl  —  correctness of src/sparse_scenario.jl and price_bnb(...; sparse = true)

    julia --project=model_2stage/env model_2stage/test/test_sparse.jl [gurobi] [paris=<instance json>]

S1  sparse tier value (direct and compact rows) = dense utility-tier value = dense :pre value; no blocking pair; conservation residual 0
S2  sparse R1 = dense R1, sparse price-box relaxation = dense relaxation (two boxes)
S3  price_bnb(sparse = true) = price_bnb(:pre): same optimal value (toy type, syn_n5_T4 uniform)
S4  (with paris=...) real-data instance, first train scenario at the baseline price: sparse tier = dense tier
S5  the "post no task" MIP start (nothing_start!) satisfies every row, bound and integrality; its profit <= the optimum
"""

include(joinpath(@__DIR__, "..", "src", "price_bnb.jl"))

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

cases = [(TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"), (SYN(5, 4), "syn", "n5T4")]
prices(inst) = [("u2", fill(2.0, inst.n, inst.n)), ("u4", fill(4.0, inst.n, inst.n)),
                ("u8", fill(8.0, inst.n, inst.n)),
                ("t25", [j == k ? 5.0 : 2.0 for j in 1:inst.n, k in 1:inst.n]),
                ("od", [Float64(1 + (3j + 2k) % 6) for j in 1:inst.n, k in 1:inst.n])]

for (path, var, tag) in cases
    inst = load_ts_instance(path, var; levels = 3, nscen = 2)
    println("== $tag  n=$(inst.n) T=$(inst.T) Γ=$(inst.Γ)")
    for (pn, P) in prices(inst), s in 1:2
        lab = "$tag $pn s=$s"
        vpre, = solve_scenario(inst, s, P, opt(); stability = :pre)
        vtier, = solve_tier(inst, s, P, opt())
        for form in (:direct, :compact)
            vs, _, bs, is = solve_tier_sparse(inst, s, P, opt(); tier_form = form)
            bp = blocking_pairs_sparse(inst, bs, P)
            check(is.optimal && close_to(vs, vpre) && close_to(vs, vtier) && isempty(bp) &&
                  conservation_residual(inst, bs) < 1e-6 && close_to(sum(value.(bs.pprofit)), vs),
                  @sprintf("S1 %s %s  sparse %.4f  tier %.4f  pre %.4f  blocking %d  triples %d / %d",
                           lab, form, vs, vtier, vpre, length(bp), is.ntrip, inst.n^3))
        end
        vr1, = solve_r1(inst, s, P, opt())
        vr1s, = solve_r1_sparse(inst, s, P, opt())
        check(close_to(vr1s, vr1) && vr1s >= vpre - 1e-6, @sprintf("S2 %s  R1 sparse %.4f  dense %.4f", lab, vr1s, vr1))
    end
    for (a, b) in ((2.0, 4.0), (0.0, 8.0))
        A, B = fill(a, inst.n, inst.n), fill(b, inst.n, inst.n)
        vd, = solve_relax(inst, 1, A, B, opt())
        vsp, = solve_relax_sparse(inst, 1, A, B, opt())
        check(close_to(vsp, vd), @sprintf("S2 %s box [%g, %g]  sparse %.4f  dense %.4f", tag, a, b, vsp, vd))
    end
end

for (path, var, structure, ns) in ((TOY, "toy-A", :type, 3), (SYN(5, 4), "syn", :uniform, 3))
    inst = load_ts_instance(path, var; levels = 3, nscen = ns, structure)
    rd = price_bnb(inst, opt(); inner = :pre, log = false)
    rs = price_bnb(inst, opt(); sparse = true, log = false)
    check(rd.status == :optimal && rs.status == :optimal && close_to(rs.value, rd.value) && close_to(rs.ub, rs.value),
          @sprintf("S3 %s %s  sparse %.4f %s  pre %.4f %s  (%d vs %d exact points)", basename(path), structure,
                   rs.value, string(rs.prices), rd.value, string(rd.prices), rs.stats.exact_points, rd.stats.exact_points))
end

# S5: the "post no task" MIP start satisfies every constraint and integrality (R1 and direct tier rows)
function start_ok(inst, s, P, stability)
    m = Model()
    b = add_scenario_sparse!(m, inst, inst.scen[s], P, P; stability)
    nothing_start!(inst, b)
    @objective(m, Max, b.profit)
    pt = Dict(v => something(start_value(v), is_fixed(v) ? fix_value(v) : NaN) for v in all_variables(m))
    any(isnan, values(pt)) && return false, -Inf
    rep = primal_feasibility_report(m, pt; atol = 1e-6)
    ints = all(abs(pt[v] - round(pt[v])) < 1e-9 for v in all_variables(m) if is_integer(v) || is_binary(v))
    return isempty(rep) && ints, value(vv -> pt[vv], b.profit)
end
for (path, var, tag) in cases
    inst = load_ts_instance(path, var; levels = 3, nscen = 2)
    for (pn, P) in prices(inst), stab in (:none, :tier)
        ok, v = start_ok(inst, 1, P, stab)
        vbest = stab === :tier ? solve_tier_sparse(inst, 1, P, opt())[1] : solve_r1_sparse(inst, 1, P, opt())[1]
        check(ok && v <= vbest + 1e-6, @sprintf("S5 %s %s %s  start feasible, value %.1f <= optimum %.1f", tag, pn, stab, v, vbest))
    end
end

pa = findfirst(a -> startswith(a, "paris="), ARGS)
if pa !== nothing
    path = ARGS[pa][7:end]
    inst = load_ts_instance(path, "paris"; levels = 3, nscen = 1)
    P = fill(Float64(JSON.parsefile(path)["pricing"]["baseline"]), inst.n, inst.n)
    t1 = @elapsed (vt, mt, _, it) = solve_tier(inst, 1, P, opt())
    ok, v = start_ok(inst, 1, P, :tier)
    check(ok, @sprintf("S5 %s  start feasible, value %.1f", basename(path), v))
    for form in (:direct, :compact)
        t2 = @elapsed (vs, ms, bs, is) = solve_tier_sparse(inst, 1, P, opt(); tier_form = form)
        check(it.optimal && is.optimal && close_to(vs, vt) && conservation_residual(inst, bs) < 1e-6,
              @sprintf("S4 %s %s  sparse %.1f (%.1f s, %d vars)  dense tier %.1f (%.1f s, %d vars)", basename(path),
                       form, vs, t2, num_variables(ms), vt, t1, num_variables(mt)))
    end
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")

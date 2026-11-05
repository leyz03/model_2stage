"""
test_fixed_price.jl  —  correctness of src/fixed_price.jl (HiGHS by default, no Gurobi session)

    julia --project=model_2stage/env model_2stage/test/test_fixed_price.jl [gurobi]

F1  the exact (:pre) optimum has no blocking pair; per-period profits add up to the profit
F2  row generation (solve_verify) = exact :pre value
F3  R1 >= exact, and R1 = exact whenever R1's optimum has no blocking pair
F4  relax-and-fix (:pre and :verify, several windows) is feasible and stable, value <= exact,
    conservation residual 0; with w = T it equals the exact value
F5  the exact MILP warm-started from the relax-and-fix solution reaches the exact value
F6  the exact MILP with the cut profit <= R1 value reaches the exact value
F7  utility-tier stable matching = exact :pre, with no blocking pairs and valid conservation
F8  utility-tier stability also matches :pre for heterogeneous OD prices
F2/F3 are also run with the tie-breaking objective (same profit; R1's optimum more often stable)
"""

include(joinpath(@__DIR__, "..", "src", "fixed_price.jl"))

using Printf

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

const TOY = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
SYN(n, T) = joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(n)_T$(T).json")
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1); flush(stdout))
close_to(a, b) = abs(a - b) <= 1e-6 * max(1.0, abs(b))

cases = [(TOY, "toy-A", "toy"), (SYN(3, 6), "syn", "n3T6"), (SYN(5, 4), "syn", "n5T4")]
prices(inst) = [("u2", fill(2.0, inst.n, inst.n)), ("u4", fill(4.0, inst.n, inst.n)),
                ("u8", fill(8.0, inst.n, inst.n)),
                ("t25", [j == k ? 5.0 : 2.0 for j in 1:inst.n, k in 1:inst.n])]

for (path, var, tag) in cases
    inst = load_ts_instance(path, var; levels = 3, nscen = 2)
    println("== $tag  n=$(inst.n) T=$(inst.T) Γ=$(inst.Γ)")
    for (pn, P) in prices(inst), s in 1:2
        lab = "$tag $pn s=$s"
        v0, m0, b0, _ = solve_scenario(inst, s, P, opt(); stability = :pre)
        bp0 = blocking_pairs(inst, b0, P)
        check(isempty(bp0) && close_to(sum(value.(b0.pprofit)), v0),
              @sprintf("F1 %s  exact %.4f  blocking %d", lab, v0, length(bp0)))
        vtier, _, btier, itier = solve_tier(inst, s, P, opt())
        bptier = blocking_pairs(inst, btier, P)
        check(itier.optimal && close_to(vtier, v0) && isempty(bptier) &&
              conservation_residual(inst, btier) < 1e-6,
              @sprintf("F7 %s  tier %.4f vs pre %.4f  blocking %d  bin %d vs %d",
                       lab, vtier, v0, length(bptier), itier.nbin, count_free_binaries(m0)))
        vv, _, bv, iv = solve_verify(inst, s, P, opt())
        check(close_to(vv, v0) && isempty(blocking_pairs(inst, bv, P)),
              @sprintf("F2 %s  verify %.4f (%d rounds, %d rows)", lab, vv, iv.rounds, iv.rows))
        v1, _, b1, _ = solve_r1(inst, s, P, opt())
        bp1 = blocking_pairs(inst, b1, P)
        check(v1 >= v0 - 1e-6 && (!isempty(bp1) || close_to(v1, v0)),
              @sprintf("F3 %s  R1 %.4f  blocking %d", lab, v1, length(bp1)))
        vt, _, bt, it = solve_r1(inst, s, P, opt(); tiebreak = true)
        bpt = blocking_pairs(inst, bt, P)
        check(close_to(vt, v1) && it.bound >= v1 - 1e-6 && (!isempty(bpt) || close_to(vt, v0)),
              @sprintf("F3 %s  R1 tie-break %.4f (bound %.4f)  blocking %d", lab, vt, it.bound, length(bpt)))
        vvt, _, bvt, ivt = solve_verify(inst, s, P, opt(); tiebreak = true)
        check(close_to(vvt, v0) && isempty(blocking_pairs(inst, bvt, P)),
              @sprintf("F2 %s  verify tie-break %.4f (%d rounds, %d rows)", lab, vvt, ivt.rounds, ivt.rows))
        vu, _, _, iu = solve_exact_ub(inst, s, P, opt(), v1)
        check(close_to(vu, v0) && iu.optimal, @sprintf("F6 %s  exact with R1 cut %.4f", lab, vu))
    end
    P = fill(4.0, inst.n, inst.n)
    for s in 1:2
        v0, _, _, _ = solve_scenario(inst, s, P, opt(); stability = :pre)
        for (w, st, look, stab, tb) in ((1, 1, :lp, :pre, false), (2, 1, :lp, :pre, false), (3, 2, :lp, :verify, false),
                                        (3, 2, :lp, :verify, true), (2, 1, :none, :pre, false), (2, 2, 1, :verify, false),
                                        (inst.T, 1, :lp, :pre, false), (inst.T, 1, :lp, :verify, true))
            vr, mr, br, ir = relax_and_fix(inst, s, P, opt(); w, step = st, look, stability = stab, tiebreak = tb)
            res = conservation_residual(inst, br)
            nb = length(blocking_pairs(inst, br, P))
            ok = ir.ok && vr <= v0 + 1e-6 && res < 1e-6 && nb == 0 && (w < inst.T || close_to(vr, v0))
            check(ok, @sprintf("F4 %s s=%d RF(w=%d,step=%d,look=%s,%s,tb=%s) %.4f vs exact %.4f  windows %d rows %d  res %.1e  blocking %d",
                               tag, s, w, st, string(look), stab, tb, vr, v0, ir.windows, ir.rows, res, nb))
            if w == 2 && st == 1 && look === :lp
                vw, _, _, iw = solve_warm(inst, s, P, opt(), value.(all_variables(mr)))
                check(close_to(vw, v0) && iw.optimal, @sprintf("F5 %s s=%d warm start from RF %.4f -> %.4f", tag, s, vr, vw))
            end
        end
    end
end

inst_od = load_ts_instance(TOY, "toy-A"; levels = 3, nscen = 2)
P_od = [Float64(mod(2 * j + 3 * k, 9)) for j in 1:inst_od.n, k in 1:inst_od.n]
for s in 1:2
    v0, _, _, i0 = solve_scenario(inst_od, s, P_od, opt(); stability = :pre)
    vt, _, bt, it = solve_tier(inst_od, s, P_od, opt())
    check(i0.optimal && it.optimal && close_to(vt, v0) && isempty(blocking_pairs(inst_od, bt, P_od)),
          @sprintf("F8 toy OD s=%d  tier %.4f vs pre %.4f", s, vt, v0))
end

println(fails == 0 ? "ALL PASS" : "$fails FAILED")
exit(fails == 0 ? 0 : 1)

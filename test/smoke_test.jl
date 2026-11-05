"""
smoke_test.jl  —  correctness checks for model_2stage/src/two_stage_bl.jl (HiGHS, no Gurobi)

    julia --project=model_2stage/env model_2stage/test/smoke_test.jl [nscen_equiv] [nscen_ef]

T1  per-scenario equivalence: with p ≡ 4 on one full-day trajectory, the second-stage MILP
    equals the deterministic equivalent of the EXP-BATTLVL SDDP.jl model built on that
    single path (binary states, its own Q2) -- for K = 2 (= model_v2) and K = 3.
T2  conservation residual 0 on every period of every solved block.
T3  extensive form with prices (:type structure) == brute-force grid over the menu,
    each grid point evaluated scenario by scenario.
"""

include(joinpath(@__DIR__, "..", "src", "two_stage_bl.jl"))

using HiGHS, Printf

const PATH = joinpath(@__DIR__, "..", "toy", "two_stage_toy.json")
const N_EQ = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const N_EF = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
quiet() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false,
                                    "mip_rel_gap" => 0.0, "mip_abs_gap" => 0.0)
fails = 0
check(ok, msg) = (ok ? println("  PASS  ", msg) : (println("  FAIL  ", msg); global fails += 1))

println("T1/T2  second-stage MILP vs EXP-BATTLVL deterministic equivalent (p ≡ 4)")
for K in (2, 3)
    inst = load_ts_instance(PATH, "toy-A"; levels = K, nscen = N_EQ)
    raw = JSON.parsefile(joinpath(@__DIR__, "..", "reference", "model_v2_toy", "toy_instance.json"))["deterministic"]
    p4 = fill(4.0, inst.n, inst.n)
    for s in 1:N_EQ
        ξ = inst.scen[s]
        val, _, b = solve_scenario(inst, s, p4, quiet())
        bl = make_bl_instance(; name = "eq", K, move = :top,
            Δ = [rationalize(BigInt, x; tol = 1e-12) * (4 // 5) for x in _mat(raw["phi_ij"], Float64)],
            θL = 1 // 5, grid = 20, R = inst.R, p = p4, Cp = inst.Cp, d = inst.d, c = inst.c,
            A0 = Int.(raw["A0"]), U0 = Int.(raw["U0"]), W0 = Int.(raw["W0"]),
            Ω = [[ξ[t]] for t in 1:inst.T], prob = [[1.0] for _ in 1:inst.T])
        bl.Γ == inst.Γ || error("Γ mismatch")
        pg = build_bl_model(bl; optimizer = quiet())
        de = SDDP.deterministic_equivalent(pg, quiet())
        optimize!(de)
        ref = -objective_value(de)
        check(abs(val - ref) < 1e-6, @sprintf("K=%d scen %d: two-stage block %.4f  BATTLVL DE %.4f", K, s, val, ref))
        res = conservation_residual(inst, b)
        check(res < 1e-6, @sprintf("K=%d scen %d: conservation residual %.2e", K, s, res))
    end
end

println("T3  extensive form vs grid search (:type prices, K = 3, $(N_EF) scenarios)")
inst = load_ts_instance(PATH, "toy-A"; levels = 3, nscen = N_EF, structure = :type)
t_ef = @elapsed begin
    m, price, blocks = build_ef(inst, quiet())
    optimize!(m)
end
ef_val = objective_value(m)
ef_choice = [findfirst(l -> value(price.z[g, l]) > 0.5, eachindex(inst.menu)) for g in 1:inst.ngroups]
@printf("  EF: %.4f  prices (move, swap) = (%g, %g)  %.1f s\n", ef_val,
        inst.menu[ef_choice[1]], inst.menu[ef_choice[2]], t_ef)
for (s, b) in enumerate(blocks)
    check(conservation_residual(inst, b) < 1e-6, "EF scen $s conservation")
end
best, arg = -Inf, (0, 0)
t_grid = @elapsed for a in eachindex(inst.menu), w in eachindex(inst.menu)
    v, _ = evaluate_prices(inst, price_matrix(inst, [a, w]), quiet())
    v > best + 1e-9 && (global best, arg = v, (a, w))
end
@printf("  grid: %.4f  at (move, swap) = (%g, %g)  %.1f s\n", best, inst.menu[arg[1]], inst.menu[arg[2]], t_grid)
check(abs(best - ef_val) < 1e-6, "EF optimum equals grid optimum")
v_ef_choice, _ = evaluate_prices(inst, price_matrix(inst, ef_choice), quiet())
check(abs(v_ef_choice - ef_val) < 1e-6, "EF price vector re-evaluated scenario by scenario gives the EF value")

println(fails == 0 ? "ALL PASS" : "$fails FAILURE(S)")

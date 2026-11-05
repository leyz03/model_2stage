"""
diag_stability.jl  —  one scenario where stability costs profit: what R1 does that the exact model may not

    julia --project=model_2stage/env model_2stage/fixed_price/diag_stability.jl --n 3 --T 8 --price u4 --s 2

Solves R1 with tie-break and the exact (:pre) MILP with HiGHS (no Gurobi session) and prints, per
period, lost riders, worker assignments (i -> task jk, utility, travel d_ij) and idle workers, plus the
blocking pairs of the R1 solution.
"""

include(joinpath(@__DIR__, "..", "src", "fixed_price.jl"))

using HiGHS, Printf

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const NN, TT, S = parse(Int, _sarg("--n", "3")), parse(Int, _sarg("--T", "8")), parse(Int, _sarg("--s", "2"))
const PRICE = _sarg("--price", "u4")
hs() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
                                 "mip_abs_gap" => 0.0, "threads" => 1, "mip_feasibility_tolerance" => 1e-9)
price_of(spec, n) = spec[1] == 'u' ? fill(parse(Float64, spec[2:end]), n, n) :
    (mv_sw = parse.(Float64, split(spec[2:end], '_')); [j == k ? mv_sw[2] : mv_sw[1] for j in 1:n, k in 1:n])

inst = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(NN)_T$(TT).json"), "syn";
                        levels = 3, nscen = S)
P = price_of(PRICE, inst.n)
println("d = ", inst.d, "\nc = ", inst.c)

function show(tag, inst, b)
    n = inst.n
    x, W, L, Mt, mt = round.(Int, value.(b.x)), round.(Int, value.(b.W)), round.(Int, value.(b.L)),
                      round.(Int, value.(b.Mt)), round.(Int, value.(b.mtil))
    Itot = [sum(value(b.A[ℓ, i, t]) for ℓ in 1:inst.K-1) / inst.Γ for i in 1:n, t in 1:inst.T]
    @printf("%s  profit %.1f\n", tag, value(b.profit))
    for t in 1:inst.T
        D = [sum(inst.scen[S][t][i, :]) for i in 1:n]
        tasks = String[]
        for i in 1:n, j in 1:n, k in 1:n
            x[i, j, k, t] > 0 && push!(tasks, @sprintf("%d×(%d→%d%s by w@%d, d=%d u=%g)", x[i, j, k, t], j, k,
                                                       j == k ? " swap" : "", i, inst.d[i, j], P[j, k] - inst.d[i, j] - inst.c[j, k]))
        end
        idle = [W[i, t] - sum(x[i, :, :, t]) for i in 1:n]
        left = sum(Mt[:, :, t]) - sum(mt[:, :, t])
        @printf("  t=%2d  demand %s  bikes %s  lost %s  workers %s idle %s  open-left %d  %s\n", t, string(D),
                string(round.(Itot[:, t]; digits = 1)), string([L[i, t] for i in 1:n]), string([W[i, t] for i in 1:n]), string(idle), left, join(tasks, "; "))
    end
end

v1, _, b1, _ = solve_r1(inst, S, P, hs(); tiebreak = true)
show("R1 (no stability, tie-break)", inst, b1)
for q in blocking_pairs(inst, b1, P)
    println("  blocking: worker@$(q.i) ($(q.worker)) wants $(q.j)→$(q.k) at t=$(q.t), u=$(q.u), job side $(q.job)")
end
v0, _, b0, _ = solve_scenario(inst, S, P, hs(); stability = :pre)
show("exact", inst, b0)
@printf("R1 %.1f  exact %.1f  price of stability %.1f\n", v1, v0, v1 - v0)

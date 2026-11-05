"""
run_grid.jl  —  exact SAA optimum of the two-stage toy by enumerating every price vector

    julia --project=model_2stage/env model_2stage/run_grid.jl --variant toy-A --levels 3 \\
        --structure type --nscen 100 --neval 1000

Only for low-dimensional price structures (`uniform`: |menu| points, `type`: |menu|^2).
Each point is evaluated exactly, one MILP per scenario, so the result is the SAA optimum
v_K with no branch-and-bound gap.  The best point and the v2 baseline p ≡ 4 are then
evaluated on the eval pool.  One Gurobi Env = one WLS session.
Writes results/<SLURM_JOB_ID or local>/grid_<variant>_K<levels>_<structure>_n<nscen>.csv and .json.
"""

include(joinpath(@__DIR__, "src", "two_stage_bl.jl"))

using Gurobi, Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const VARIANT   = _sarg("--variant", "toy-A")
const LEVELS    = parse(Int, _sarg("--levels", "3"))
const STRUCTURE = Symbol(_sarg("--structure", "type"))
const NSCEN     = parse(Int, _sarg("--nscen", "100"))
const NEVAL     = parse(Int, _sarg("--neval", "1000"))
const PATH      = joinpath(@__DIR__, "toy", "two_stage_toy.json")
const OUT_DIR = joinpath(@__DIR__, "results", get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT_DIR)

const ENV = Gurobi.Env()
grb() = optimizer_with_attributes(() -> Gurobi.Optimizer(ENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 1, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

inst = load_ts_instance(PATH, VARIANT; levels = LEVELS, nscen = NSCEN, structure = STRUCTURE)
inst.ngroups <= 2 || error("grid search only for uniform/type structures")
Lm = eachindex(inst.menu)
points = vec(collect(Iterators.product(ntuple(_ -> Lm, inst.ngroups)...)))
@printf("GRID %s  Γ=%d  scenarios=%d  points=%d\n", inst.name, inst.Γ, NSCEN, length(points)); flush(stdout)

rows = String[]
best, best_pt = -Inf, nothing
t_all = @elapsed for pt in points
    t = @elapsed (μ, vals) = evaluate_prices(inst, price_matrix(inst, collect(pt)), grb())
    prices = inst.menu[collect(pt)]
    push!(rows, join(vcat(prices, μ, t), ","))
    @printf("  prices %-12s  mean profit %9.4f  (%.1f s)\n", string(Int.(prices)), μ, t); flush(stdout)
    if μ > best + 1e-9
        global best, best_pt = μ, collect(pt)
    end
end
p_star = price_matrix(inst, best_pt)
@printf("best %.4f at prices %s  (%d points, %.0f s)\n", best, string(inst.menu[best_pt]), length(points), t_all)

hdr = inst.ngroups == 1 ? "price" : "p_move,p_swap"
f = joinpath(OUT_DIR, "grid_$(VARIANT)_K$(LEVELS)_$(STRUCTURE)_n$(NSCEN)")
open(io -> (println(io, hdr, ",mean_profit,seconds"); foreach(r -> println(io, r), rows)), f * ".csv", "w")

evinst = load_ts_instance(PATH, VARIANT; levels = LEVELS, pool = "eval", nscen = NEVAL, structure = STRUCTURE)
function oos(p)
    μ, vals = evaluate_prices(evinst, p, grb())
    return Dict("mean" => μ, "halfwidth" => 1.96 * std(vals) / sqrt(length(vals)))
end
e_star, e_base = oos(p_star), oos(fill(4.0, inst.n, inst.n))
@printf("out-of-sample (%d)  p*: %.4f ± %.4f   p≡4: %.4f ± %.4f\n", NEVAL,
        e_star["mean"], e_star["halfwidth"], e_base["mean"], e_base["halfwidth"])
open(io -> JSON.print(io, Dict("instance" => inst.name, "nscen" => NSCEN, "neval" => NEVAL,
    "v_K" => best, "price_choice" => inst.menu[best_pt], "seconds" => t_all,
    "oos_p_star" => e_star, "oos_baseline_p4" => e_base), 2), f * ".json", "w")
println("wrote ", f, ".{csv,json}")

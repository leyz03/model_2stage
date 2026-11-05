"""
run_ef.jl  —  extensive form of the two-stage toy (prices first, full-day demand second)

    julia --project=model_2stage/env model_2stage/run_ef.jl --variant toy-A --levels 3 \\
        --structure type --nscen 100 --neval 1000 --tl 3600

1. Solves the EF on the first `nscen` trajectories of the train pool (SAA value v_K and
   the SAA-optimal price vector), with Gurobi MIPGap = 0.
2. Evaluates that price vector and the v2 baseline p ≡ 4 exactly on the first `neval`
   trajectories of the independent eval pool (one MILP per scenario).
All Gurobi models share one Env, i.e. one WLS session.
Writes results/<SLURM_JOB_ID or local>/ef_<variant>_K<levels>_<structure>_n<nscen>.json.
"""

include(joinpath(@__DIR__, "src", "two_stage_bl.jl"))

using Gurobi, Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const VARIANT   = _sarg("--variant", "toy-A")
const LEVELS    = parse(Int, _sarg("--levels", "3"))
const STRUCTURE = Symbol(_sarg("--structure", "type"))
const NSCEN     = parse(Int, _sarg("--nscen", "100"))
const NEVAL     = parse(Int, _sarg("--neval", "1000"))
const TL        = parse(Float64, _sarg("--tl", "3600"))
const THREADS   = parse(Int, _sarg("--threads", "2"))
const PATH      = joinpath(@__DIR__, "toy", "two_stage_toy.json")
const OUT_DIR = joinpath(@__DIR__, "results", get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT_DIR)

const ENV = Gurobi.Env()
grb(; tl = 600.0, threads = 1) = optimizer_with_attributes(() -> Gurobi.Optimizer(ENV),
    "OutputFlag" => 0, "MIPGap" => 0.0, "Threads" => threads, "TimeLimit" => tl,
    "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

inst = load_ts_instance(PATH, VARIANT; levels = LEVELS, nscen = NSCEN, structure = STRUCTURE)
@printf("EF %s  Γ=%d  scenarios=%d  menu=%s  groups=%d\n", inst.name, inst.Γ, NSCEN,
        string(Int.(inst.menu)), inst.ngroups); flush(stdout)

t_build = @elapsed (m, price, blocks) = build_ef(inst, grb(; tl = TL, threads = THREADS))
set_optimizer_attribute(m, "OutputFlag", 1)
t_solve = @elapsed optimize!(m)
st = JuMP.termination_status(m)
v_K, bound = objective_value(m), objective_bound(m)
choice = [findfirst(l -> value(price.z[g, l]) > 0.5, eachindex(inst.menu)) for g in 1:inst.ngroups]
p_star = price_matrix(inst, choice)
cons = maximum(conservation_residual(inst, b) for b in blocks)
@printf("EF status %s  v_K %.4f  bound %.4f  build %.1f s  solve %.1f s  vars %d  conservation %.1e\n",
        st, v_K, bound, t_build, t_solve, num_variables(m), cons)
println("prices p*[j,k] = ", p_star); flush(stdout)

evinst = load_ts_instance(PATH, VARIANT; levels = LEVELS, pool = "eval", nscen = NEVAL, structure = STRUCTURE)
function oos(p)
    t = @elapsed (μ, vals) = evaluate_prices(evinst, p, grb())
    hw = 1.96 * std(vals) / sqrt(length(vals))
    return (mean = μ, halfwidth = hw, seconds = t)
end
e_star = oos(p_star)
e_base = oos(fill(4.0, inst.n, inst.n))
in_base, _ = evaluate_prices(inst, fill(4.0, inst.n, inst.n), grb())
@printf("in-sample  p*: %.4f   p≡4: %.4f\n", v_K, in_base)
@printf("out-of-sample (%d)  p*: %.4f ± %.4f   p≡4: %.4f ± %.4f\n", NEVAL,
        e_star.mean, e_star.halfwidth, e_base.mean, e_base.halfwidth)

out = Dict("instance" => inst.name, "gamma" => inst.Γ, "nscen" => NSCEN, "neval" => NEVAL,
    "status" => string(st), "v_K" => v_K, "bound" => bound, "build_s" => t_build,
    "solve_s" => t_solve, "num_variables" => num_variables(m), "conservation" => cons,
    "price_choice" => inst.menu[choice], "p_star" => [p_star[j, :] for j in 1:inst.n],
    "in_sample_baseline_p4" => in_base,
    "oos_p_star" => Dict(pairs(e_star)), "oos_baseline_p4" => Dict(pairs(e_base)))
f = joinpath(OUT_DIR, "ef_$(VARIANT)_K$(LEVELS)_$(STRUCTURE)_n$(NSCEN).json")
open(io -> JSON.print(io, out, 2), f, "w")
println("wrote ", f)

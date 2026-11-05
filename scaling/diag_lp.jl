"""
diag_lp.jl  —  why does the per-scenario MILP get hard with T?  LP bounds with HiGHS (no Gurobi session)

    julia --project=model_2stage/env model_2stage/scaling/diag_lp.jl --n 3 --T 4+6+8+10+12 --nscen 3

At p = 4 for every task, per scenario:
  lp_exact   LP relaxation of the exact (:pre) scenario MILP
  lp_relax   LP relaxation of the no-stability relaxation
The MILP optimum comes from the oracle runs (results/*/scaling.jsonl, Gurobi); the gap
LP − MILP shows how loose the LP gets as T grows, and lp_exact − lp_relax how much of the
LP bound the stable-matching rows contribute.
Appends to results/<SLURM_JOB_ID or local>/diag_lp.jsonl.
"""

include(joinpath(@__DIR__, "..", "src", "two_stage_bl.jl"))

using HiGHS, Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const NN    = parse(Int, _sarg("--n", "3"))
const TS    = parse.(Int, split(_sarg("--T", "4+8+12"), r"[,+]"))
const NSCEN = parse(Int, _sarg("--nscen", "3"))
const PHI   = (x = _sarg("--phi", ""); isempty(x) ? nothing : parse(Float64, x))
const MILP  = "--milp" in ARGS                 # also solve the exact MILP with HiGHS
const MILP_TL = parse(Float64, _sarg("--mip_tl", "600"))
const OUT   = joinpath(@__DIR__, "..", "results", get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT)

function lp_value(inst, s, P; relax)
    m = Model(optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false))
    b = relax ? add_scenario!(m, inst, inst.scen[s], (wage = P, ir = P)) :
                add_scenario!(m, inst, inst.scen[s], P; stability = :pre)
    @objective(m, Max, b.profit)
    relax_integrality(m)
    optimize!(m)
    return objective_value(m), num_variables(m), num_constraints(m; count_variable_in_set_constraints = false)
end

for T in TS
    path = joinpath(@__DIR__, "instances", "syn_n$(NN)_T$(T).json")
    isfile(path) || (println("missing ", path); continue)
    inst = load_ts_instance(path, "syn"; levels = 3, nscen = NSCEN, phi = PHI)
    P = fill(4.0, inst.n, inst.n)
    le = [lp_value(inst, s, P; relax = false) for s in 1:NSCEN]
    lr = [lp_value(inst, s, P; relax = true) for s in 1:NSCEN]
    rec = Dict("n" => NN, "T" => T, "nscen" => NSCEN, "lp_exact" => mean(first.(le)),
               "lp_relax" => mean(first.(lr)), "vars" => le[1][2], "rows" => le[1][3],
               "phi" => PHI === nothing ? "instance" : PHI, "gamma" => inst.Γ)
    if MILP
        hs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
                                       "mip_abs_gap" => 0.0, "time_limit" => MILP_TL, "threads" => 1)
        res = [begin
                   m = Model(hs); b = add_scenario!(m, inst, inst.scen[s], P; stability = :pre)
                   @objective(m, Max, b.profit); optimize!(m)
                   (objective_value(m), solve_time(m), JuMP.termination_status(m) == MOI.OPTIMAL)
               end for s in 1:NSCEN]
        rec["milp"] = mean(first.(res)); rec["milp_s"] = mean(getindex.(res, 2))
        rec["milp_all_optimal"] = all(last.(res))
        @printf("        MILP(HiGHS) %9.3f  %.2f s/MILP  all optimal %s  LP gap %.1f%%\n", rec["milp"], rec["milp_s"],
                rec["milp_all_optimal"], 100 * (rec["lp_exact"] - rec["milp"]) / rec["milp"])
    end
    @printf("n=%d T=%2d Γ=%d  LP exact %9.3f  LP relax %9.3f  vars %d rows %d\n", NN, T, inst.Γ, rec["lp_exact"],
            rec["lp_relax"], rec["vars"], rec["rows"])
    open(io -> (JSON.print(io, rec); println(io)), joinpath(OUT, "diag_lp.jsonl"), "a")
end

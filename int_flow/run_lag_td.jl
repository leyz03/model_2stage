"""
run_lag_td.jl  —  TODO #10 (2026-11-04): one evaluation of the Lagrangian time decomposition (int_flow/lag_td.jl) on
periods 1..T of one day of the integer-split instance at the baseline price.  Phase lp solves the joint LP of the
linked blocks and writes λ; phase block solves one block alone at that λ (one job per block and form).

    julia --project=model_2stage/env model_2stage/int_flow/run_lag_td.jl --phase lp    --day 11 --T 100 --blocks 5 [opts]
    julia --project=model_2stage/env model_2stage/int_flow/run_lag_td.jl --phase block --day 11 --T 100 --blocks 5 \\
          --k 3 --form bin --lam <lambda json> --tl 1800 [opts]

--instance (default int_flow/instances/paris_n30_H10_int_workers.json), --threads 8, --tl (s; lp 7200, block 1800),
--form full | xbin | bin | lp (block integrality: all / x and β, z / only β, z / none).
Gurobi, MIPGap 0, IntFeasTol = FeasibilityTol = 1e-9 (as #2).  Output in results/<array job id or job id>/:
  lp     lambda_dDD_TTTT_BB.json (LP value, seconds, cuts, λ per boundary with its state keys) and its Gurobi log
  block  dDD_TTTT_BB_kK_<form>.json (bound, value, status, seconds, nodes) and its Gurobi log (bound over time)
L(λ) = Σ_k bound of block k ≥ the optimum (int_flow/summarize_lag_td.py).
"""

include(joinpath(@__DIR__, "lag_td.jl"))

using Gurobi, Printf, JSON

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const INSTANCE = _sarg("--instance", joinpath(@__DIR__, "instances", "paris_n30_H10_int_workers.json"))
const PHASE    = Symbol(_sarg("--phase", "lp"))
const DAY      = parse(Int, _sarg("--day", "11"))
const TT       = parse(Int, _sarg("--T", "100"))
const BLOCKS   = parse(Int, _sarg("--blocks", "5"))
const KBLK     = parse(Int, _sarg("--k", "1"))
const FORM     = Symbol(_sarg("--form", "full"))
const LAM      = _sarg("--lam", "")
const THREADS  = parse(Int, _sarg("--threads", "8"))
const TL       = parse(Float64, _sarg("--tl", PHASE === :lp ? "7200" : "1800"))
const JOB      = get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "local"))
const OUT_DIR  = joinpath(@__DIR__, "results", JOB)
const BASE     = @sprintf("d%02d_T%03d_B%d", DAY, TT, BLOCKS)
const TAG      = PHASE === :lp ? "lambda_" * BASE : @sprintf("%s_k%d_%s", BASE, KBLK, FORM)
PHASE in (:lp, :block) || error("--phase must be lp or block")
PHASE === :block && (isempty(LAM) && error("--phase block needs --lam"); FORM in LAG_FORMS || error("bad --form"))
mkpath(OUT_DIR)

"""A token of the site license (Savio token server); waits while all tokens are taken."""
function grb_env(; tries = 60)
    for k in 1:tries
        try
            return Gurobi.Env()
        catch e
            k == tries && rethrow()
            println("Gurobi.Env failed (try $k): ", sprint(showerror, e)); flush(stdout)
            sleep(60)
        end
    end
end
const GENV = grb_env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 1, "MIPGap" => 0.0,
    "Threads" => THREADS, "TimeLimit" => TL, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9,
    "LogFile" => joinpath(OUT_DIR, TAG * ".log"))
_clean(x::AbstractFloat) = isfinite(x) ? x : nothing
_clean(x) = x

raw = JSON.parsefile(INSTANCE)
full = load_ts_instance(INSTANCE, "paris"; levels = 2)
P = fill(Float64(raw["pricing"]["baseline"]), full.n, full.n)
w = window_instance(full, DAY, 1, TT)
cuts = lag_cuts(TT, BLOCKS)
out = Dict{String,Any}("instance" => basename(INSTANCE), "day" => DAY, "T" => TT, "blocks" => BLOCKS,
                       "cuts" => [collect(c) for c in cuts], "price" => P[1, 1], "threads" => THREADS, "tl" => TL,
                       "job" => JOB, "phase" => string(PHASE), "fleet" => w.Q3)
println("$TAG: $(basename(INSTANCE)), day $DAY, T $TT, cuts $cuts, fleet $(w.Q3)"); flush(stdout)

if PHASE === :lp
    lpv, λ, keys, secs = lag_lambda(w, P, cuts, opt())
    out["lp"] = lpv
    out["lp_s"] = secs
    out["lam"] = λ
    out["keys"] = [lag_keys_json(kk) for kk in keys]
    out["nlam"] = length.(λ)
    @printf("joint LP %.4f in %.1f s, %s multipliers\n", lpv, secs, join(length.(λ), "/"))
else
    js = JSON.parsefile(LAM)
    js["blocks"] == BLOCKS && js["T"] == TT && js["day"] == DAY && js["instance"] == basename(INSTANCE) ||
        error("$LAM is for another run")
    λ = [Float64.(l) for l in js["lam"]]
    t0 = time()
    m, blk = lag_block(w, P, cuts, KBLK, λ, opt(); form = FORM)
    KBLK > 1 && (lag_same_keys(blk.zkeys, js["keys"][KBLK-1]) || error("incoming keys differ from $LAM"))
    KBLK < BLOCKS && (lag_same_keys(blk.ykeys, js["keys"][KBLK]) || error("outgoing keys differ from $LAM"))
    build_s = time() - t0
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("block $KBLK $FORM: $st")
    hasv = result_count(m) > 0 && primal_status(m) == MOI.FEASIBLE_POINT
    attr(a) = try MOI.get(m, a) catch; NaN end
    merge!(out, Dict("k" => KBLK, "ta" => cuts[KBLK][1], "tb" => cuts[KBLK][2], "form" => string(FORM),
                     "lam_file" => LAM, "lp" => js["lp"], "status" => string(st), "proved" => st == MOI.OPTIMAL,
                     "bound" => _clean(objective_bound(m)), "value" => hasv ? objective_value(m) : nothing,
                     "profit" => hasv ? value(blk.b.profit) : nothing, "solve_s" => solve_time(m),
                     "build_s" => build_s, "nodes" => _clean(Float64(attr(MOI.NodeCount()))),
                     "nvars" => num_variables(m), "ncons" => num_constraints(m; count_variable_in_set_constraints = false),
                     "nin" => length(blk.z), "nout" => length(blk.y)))
    @printf("block %d (%d–%d) %s: %s, bound %.4f, value %s, %.1f s\n", KBLK, cuts[KBLK]..., FORM, st,
            objective_bound(m), hasv ? @sprintf("%.4f", objective_value(m)) : "–", solve_time(m))
end
open(io -> JSON.print(io, out, 1), joinpath(OUT_DIR, TAG * ".json"), "w")
println("RESULT ", JSON.json(Dict(k => v for (k, v) in out if k != "lam" && k != "keys")))

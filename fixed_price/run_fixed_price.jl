"""
run_fixed_price.jl  —  fixed-price per-scenario experiments (model_2stage/fixed_price/README.md)

    julia --project=model_2stage/env model_2stage/fixed_price/run_fixed_price.jl --mode stab --n 5 --T 8 --price u4

MODE
  stab  how much stable matching costs, per scenario: R1 (no stability) and its blocking pairs, with and
        without the tie-breaking objective (r1, r1tb); the exact :pre MILP (r0); the exact MILP with the
        cut profit <= R1 (r0cut); row generation without / with tie-break (verify, verifytb)
  td    time decomposition, per scenario: LP bound, R1 with tie-break (upper bound, exact if its optimum
        is stable), exact :pre MILP (--tl), relax-and-fix variants (--rf), and the exact MILP
        warm-started from the best relax-and-fix solution (--tl)

--price  u<v> = every task v (u4);  t<move>_<swap> = moves / swaps (t2_5)
--rf     relax-and-fix variants w-step-look-stability[-tb] joined by "/", look = lp | none | <h>,
         stability = pre | verify, -tb = tie-breaking objective
         (default "2-1-lp-pre/3-1-lp-verify/4-2-lp-pre/3-1-none-pre")
--tl 300 (exact / warm MILP), --win_tl 60 (one window), --rf_tl 900 (one relax-and-fix run),
--nscen 3, --phi x (diagnostic: every φ = x), --solver gurobi|highs, --threads 1,
--only_rf (td: skip R1 / exact / warm start, e.g. extra relax-and-fix variants on scenarios an earlier job bounded)
Instance: model_2stage/scaling/instances/syn_n{n}_T{T}.json.  Every solve appends one JSON line
to results/<SLURM_JOB_ID or local>/fixed_price.jsonl.

Pruning (memory: prune dominated experiments): a method is skipped at (n, T) when it already
failed at some (n', T') with n' <= n, T' <= T, same mode / price / phi, in this job or in the
jobs listed in PRUNE_JOBS.  Failure: an exact solve that is not optimal, or a relax-and-fix
run without a solution.  PRUNE=0 disables.
"""

include(joinpath(@__DIR__, "..", "src", "fixed_price.jl"))

using Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const MODE    = _sarg("--mode", "stab")
const NN      = parse(Int, _sarg("--n", "5"))
const TT      = parse(Int, _sarg("--T", "8"))
const NSCEN   = parse(Int, _sarg("--nscen", "3"))
const PRICE   = _sarg("--price", "u4")
const RFSPEC  = _sarg("--rf", "2-1-lp-pre/3-1-lp-verify/4-2-lp-pre/3-1-none-pre")
const TL      = parse(Float64, _sarg("--tl", "300"))
const WIN_TL  = parse(Float64, _sarg("--win_tl", "60"))
const RF_TL   = parse(Float64, _sarg("--rf_tl", "900"))
const PHI     = (x = _sarg("--phi", ""); isempty(x) ? nothing : parse(Float64, x))
const ONLY_RF = "--only_rf" in ARGS          # td: LP and relax-and-fix only (bounds come from an earlier job)
const SOLVER  = _sarg("--solver", "gurobi")
const THREADS = parse(Int, _sarg("--threads", "1"))
const JOB     = get(ENV, "SLURM_JOB_ID", "local")
const RESULTS = joinpath(@__DIR__, "..", "results")
const OUT     = joinpath(RESULTS, JOB, "fixed_price.jsonl")
mkpath(dirname(OUT))

if SOLVER == "gurobi"
    using Gurobi
    const GENV = Gurobi.Env()
    opt(; tl = TL) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
        "Threads" => THREADS, "TimeLimit" => tl, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
else
    using HiGHS
    opt(; tl = TL) = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false, "mip_rel_gap" => 0.0,
        "mip_abs_gap" => 0.0, "time_limit" => tl, "threads" => THREADS, "mip_feasibility_tolerance" => 1e-9)
end

function price_of(spec, n)
    spec[1] == 'u' && return fill(parse(Float64, spec[2:end]), n, n)
    spec[1] == 't' || error("price spec $spec")
    mv, sw = parse.(Float64, split(spec[2:end], '_'))
    return [j == k ? sw : mv for j in 1:n, k in 1:n]
end

_clean(x::AbstractFloat) = isfinite(x) ? x : nothing
_clean(x::NamedTuple) = Dict(string(k) => _clean(v) for (k, v) in pairs(x))
_clean(x::AbstractVector) = map(_clean, x)
_clean(x::Symbol) = string(x)
_clean(x) = x
function record(d)
    d = Dict(k => _clean(v) for (k, v) in d)
    open(io -> (JSON.print(io, d); println(io)), OUT, "a")
    println("RECORD ", JSON.json(d)); flush(stdout)
end

const PHITAG = PHI === nothing ? "instance" : string(PHI)
base(s) = Dict("mode" => MODE, "n" => NN, "T" => TT, "price" => PRICE, "phi" => PHITAG, "scenario" => s,
               "solver" => SOLVER, "threads" => THREADS, "job" => JOB, "tl" => TL)

"""true when `method` failed at a dominated (n', T') <= (n, T) in this job or PRUNE_JOBS (not at (n, T) itself)"""
function pruned(method)
    get(ENV, "PRUNE", "1") == "1" || return false
    for job in unique(vcat([JOB], split(get(ENV, "PRUNE_JOBS", ""))))
        f = joinpath(RESULTS, job, "fixed_price.jsonl")
        isfile(f) || continue
        for line in eachline(f)
            r = JSON.parse(line)
            get(r, "mode", "") == MODE && get(r, "method", "") == method && get(r, "price", "") == PRICE &&
                get(r, "phi", "") == PHITAG && get(r, "failed", false) == true || continue
            (r["n"], r["T"]) != (NN, TT) && r["n"] <= NN && r["T"] <= TT || continue
            println("  prune $method: failed at n=$(r["n"]) T=$(r["T"]) (job $job)")
            return true
        end
    end
    return false
end

inst = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(NN)_T$(TT).json"), "syn";
                        levels = 3, nscen = NSCEN, phi = PHI)
P = price_of(PRICE, inst.n)
@printf("%s  n=%d T=%d K_s=%d  Γ=%d  price %s  tl %g\n", MODE, inst.n, inst.T, NSCEN, inst.Γ, PRICE, TL); flush(stdout)

counts(bp) = Dict("idle_open" => count(q -> q.worker === :idle && q.job === :open, bp),
                  "idle_far" => count(q -> q.worker === :idle && q.job === :far, bp),
                  "worse_open" => count(q -> q.worker === :worse && q.job === :open, bp),
                  "worse_far" => count(q -> q.worker === :worse && q.job === :far, bp),
                  "periods" => length(unique(q.t for q in bp)))

if MODE == "stab"
    methods = ("r1", "r1tb", "r0", "r0cut", "verify", "verifytb")
    skip = Dict(mt => pruned(mt) for mt in methods)
    for s in 1:NSCEN
        v1 = NaN
        for (mt, tb) in (("r1", false), ("r1tb", true))
            skip[mt] && continue
            tw = time()
            v, _, b1, i1 = solve_r1(inst, s, P, opt(); tiebreak = tb)
            mt == "r1" && i1.optimal && (v1 = v)
            bp = i1.feasible ? blocking_pairs(inst, b1, P) : NamedTuple[]
            record(merge(base(s), Dict("method" => mt, "value" => v, "bound" => i1.bound, "solve_s" => i1.solve_s,
                "wall_s" => time() - tw, "optimal" => i1.optimal, "failed" => !i1.optimal, "nbin" => i1.nbin,
                "blocking" => length(bp), "blocking_types" => counts(bp),
                "ops" => i1.feasible ? op_stats(inst, b1) : nothing)))
        end
        if !skip["r0"]
            tw = time()
            v0, m0, b0, i0 = solve_scenario(inst, s, P, opt(); stability = :pre)
            record(merge(base(s), Dict("method" => "r0", "value" => v0, "bound" => objective_bound(m0),
                "solve_s" => i0.solve_s, "wall_s" => time() - tw, "optimal" => i0.optimal, "failed" => !i0.optimal,
                "nbin" => i0.nbin, "ops" => i0.feasible ? op_stats(inst, b0) : nothing)))
        end
        if !skip["r0cut"] && isfinite(v1)
            tw = time()
            vc, _, _, ic = solve_exact_ub(inst, s, P, opt(), v1)
            record(merge(base(s), Dict("method" => "r0cut", "value" => vc, "bound" => ic.bound, "solve_s" => ic.solve_s,
                "wall_s" => time() - tw, "optimal" => ic.optimal, "failed" => !ic.optimal, "r1" => v1)))
        end
        for (mt, tb) in (("verify", false), ("verifytb", true))
            skip[mt] && continue
            tw = time()
            vv, _, _, iv = solve_verify(inst, s, P, opt(); deadline = time() + TL, tiebreak = tb)
            record(merge(base(s), Dict("method" => mt, "value" => vv, "solve_s" => iv.solve_s,
                "wall_s" => time() - tw, "optimal" => iv.optimal, "failed" => !iv.optimal,
                "rounds" => iv.rounds, "rows" => iv.rows)))
        end
    end
elseif MODE == "td"
    specs = map(split(RFSPEC, '/')) do sp
        f = split(sp, '-')
        w, st, lk, stb = f[1:4]
        (name = "rf_" * sp, w = parse(Int, w), step = parse(Int, st),
         look = lk == "lp" ? :lp : lk == "none" ? :none : parse(Int, lk), stab = Symbol(stb),
         tb = length(f) >= 5 && f[5] == "tb")
    end
    skip = Dict(mt => ONLY_RF || pruned(mt) for mt in ["r1tb", "ex", "ws"])
    merge!(skip, Dict(sp.name => pruned(sp.name) for sp in specs))
    for s in 1:NSCEN
        lp, lps = solve_lp(inst, s, P, opt())
        record(merge(base(s), Dict("method" => "lp", "value" => lp, "solve_s" => lps, "failed" => false)))
        if !skip["r1tb"]                       # upper bound; exact when its optimum is stable
            tw = time()
            v1, _, b1, i1 = solve_r1(inst, s, P, opt(); tiebreak = true)
            record(merge(base(s), Dict("method" => "r1tb", "value" => v1, "bound" => i1.bound, "solve_s" => i1.solve_s,
                "wall_s" => time() - tw, "optimal" => i1.optimal, "failed" => !i1.optimal,
                "blocking" => i1.feasible ? length(blocking_pairs(inst, b1, P)) : nothing)))
        end
        if !skip["ex"]
            tw = time()
            v0, m0, _, i0 = solve_scenario(inst, s, P, opt(); stability = :pre)
            record(merge(base(s), Dict("method" => "ex", "value" => v0, "bound" => objective_bound(m0),
                "solve_s" => i0.solve_s, "wall_s" => time() - tw, "optimal" => i0.optimal, "failed" => !i0.optimal,
                "nbin" => i0.nbin)))
        end
        best, bestval = nothing, -Inf
        for sp in specs
            skip[sp.name] && continue
            tw = time()
            vr, mr, br, ir = relax_and_fix(inst, s, P, opt(; tl = WIN_TL); w = sp.w, step = sp.step, look = sp.look,
                                           stability = sp.stab, win_tl = WIN_TL, deadline = time() + RF_TL,
                                           tiebreak = sp.tb)
            wall = time() - tw
            if ir.ok && vr > bestval
                bestval, best = vr, value.(all_variables(mr))
            end
            record(merge(base(s), Dict("method" => sp.name, "value" => vr, "solve_s" => ir.solve_s, "wall_s" => wall,
                "windows" => ir.windows, "rounds" => ir.rounds, "rows" => ir.rows, "worst_win_s" => ir.worst_win_s,
                "ok" => ir.ok, "failed" => !ir.ok, "failed_at" => ir.failed_at, "w" => sp.w, "step" => sp.step,
                "look" => string(sp.look), "stab" => string(sp.stab), "tiebreak" => sp.tb,
                "blocking" => ir.ok ? length(blocking_pairs(inst, br, P)) : nothing)))
        end
        if best !== nothing && !skip["ws"]
            tw = time()
            vw, mw, _, iw = solve_warm(inst, s, P, opt(), best)
            record(merge(base(s), Dict("method" => "ws", "value" => vw, "bound" => iw.bound, "start" => bestval,
                "solve_s" => iw.solve_s, "wall_s" => time() - tw, "optimal" => iw.optimal, "failed" => !iw.optimal)))
        end
    end
else
    error("unknown mode $MODE")
end
println("done"); flush(stdout)

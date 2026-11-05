"""
run_tier_compare.jl — paired fixed-price :pre vs utility-tier MILPs on the same scenarios.
Run via slurm_tier_compare.sh; writes results/<job>/tier_compare.jsonl.
"""

include(joinpath(@__DIR__, "..", "src", "fixed_price.jl"))
using Printf

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : ARGS[i+1]
const NN = parse(Int, _arg("--n", "3"))
const TT = parse(Int, _arg("--T", "4"))
const NSCEN = parse(Int, _arg("--nscen", "3"))
const PRICE = _arg("--price", "u4")
const TL = parse(Float64, _arg("--tl", "300"))
const SOLVER = _arg("--solver", "gurobi")
const THREADS = parse(Int, _arg("--threads", "1"))
const JOB = get(ENV, "SLURM_JOB_ID", "local")
const OUT = joinpath(@__DIR__, "..", "results", JOB, "tier_compare.jsonl")
mkpath(dirname(OUT))

if SOLVER == "gurobi"
    using Gurobi
    const GENV = Gurobi.Env()
    opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0,
        "MIPGap" => 0.0, "Threads" => THREADS, "TimeLimit" => TL,
        "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
elseif SOLVER == "highs"
    using HiGHS
    opt() = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false,
        "mip_rel_gap" => 0.0, "mip_abs_gap" => 0.0, "threads" => THREADS,
        "time_limit" => TL, "mip_feasibility_tolerance" => 1e-9)
else
    error("unknown solver $SOLVER")
end

function price_of(spec, n)
    spec[1] == 'u' && return fill(parse(Float64, spec[2:end]), n, n)
    spec[1] == 't' || error("price spec $spec")
    mv, sw = parse.(Float64, split(spec[2:end], '_'))
    return [j == k ? sw : mv for j in 1:n, k in 1:n]
end

_clean(v::AbstractFloat) = isfinite(v) ? v : nothing
_clean(v) = v
function record(row)
    safe = Dict(k => _clean(v) for (k, v) in row)
    open(io -> (JSON.print(io, safe); println(io)), OUT, "a")
    println("RECORD ", JSON.json(safe)); flush(stdout)
end

inst = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n$(NN)_T$(TT).json"),
                        "syn"; levels = 3, nscen = NSCEN)
P = price_of(PRICE, NN)
@printf("tier comparison  n=%d T=%d scenarios=%d price=%s tl=%g solver=%s\n",
        NN, TT, NSCEN, PRICE, TL, SOLVER); flush(stdout)

# Exclude one-time Julia compilation from paired build/solve timings.
warm = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n3_T4.json"),
                        "syn"; levels = 3, nscen = 1)
warm_price = fill(4.0, warm.n, warm.n)
solve_scenario(warm, 1, warm_price, opt(); stability = :pre)
solve_tier(warm, 1, warm_price, opt())
println("warm-up complete"); flush(stdout)

for s in 1:NSCEN
    answers = Dict{String,NamedTuple}()
    for method in (isodd(s) ? ("pre", "tier") : ("tier", "pre"))
        tw = time()
        if method == "pre"
            value, model, block, info = solve_scenario(inst, s, P, opt(); stability = :pre)
            bound = objective_bound(model)
            feasible, optimal, nbin, ntier = info.feasible, info.optimal, info.nbin, 0
        else
            value, model, block, info = solve_tier(inst, s, P, opt())
            bound = info.bound
            feasible, optimal, nbin, ntier = info.feasible, info.optimal, info.nbin, info.ntier
        end
        wall = time() - tw
        blocking = feasible ? length(blocking_pairs(inst, block, P)) : nothing
        conservation = feasible ? conservation_residual(inst, block) : nothing
        feasible && (blocking != 0 || conservation > 1e-6) &&
            error("$method n=$NN T=$TT s=$s: unstable or conservation residual $conservation")
        gap = feasible && isfinite(bound) ? max(0.0, (bound - value) / max(1.0, abs(value))) : nothing
        answers[method] = (value = value, bound = bound, feasible = feasible, optimal = optimal)
        record(Dict("job" => JOB, "n" => NN, "T" => TT, "scenario" => s, "price" => PRICE,
                    "solver" => SOLVER, "threads" => THREADS, "tl" => TL, "method" => method,
                    "value" => value, "bound" => bound, "gap" => gap, "optimal" => optimal,
                    "feasible" => feasible, "blocking" => blocking, "conservation" => conservation,
                    "nbin" => nbin, "ntier" => ntier, "solve_s" => info.solve_s, "wall_s" => wall))
    end
    pre, tier = answers["pre"], answers["tier"]
    if pre.optimal && tier.optimal
        abs(pre.value - tier.value) <= 1e-6 * max(1.0, abs(pre.value)) ||
            error("n=$NN T=$TT s=$s: pre $(pre.value) != tier $(tier.value)")
        @printf("MATCH n=%d T=%d s=%d value=%.4f\n", NN, TT, s, pre.value); flush(stdout)
    elseif pre.feasible && tier.feasible
        tier.value <= pre.bound + 1e-6 && pre.value <= tier.bound + 1e-6 ||
            error("n=$NN T=$TT s=$s: proven bounds contradict the other method's feasible value")
    end
end
println("done"); flush(stdout)

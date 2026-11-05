"""
run_ref_bounds.jl — reference bounds for the Lagrangian runs (LAGRANGIAN_TD.md):
whole-day LP, utility-tier MILP with a time limit (UB and LB), and relax-and-fix (LB).
Writes results/<job>/ref_bounds.jsonl.

    julia --project=model_2stage/env model_2stage/fixed_price/run_ref_bounds.jl --n 5 --T 24 --scen 1-2 \\
          --price u4 --tl 600 [--tag _m6 --instdir model_2stage/fixed_price/instances] [--rf 4-2-8]
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : ARGS[i+1]
const NN = parse(Int, _arg("--n", "5"))
const TT = parse(Int, _arg("--T", "24"))
const SCENS = let spec = _arg("--scen", "1-2")
    parse(Int, split(spec, '-')[1]):parse(Int, split(spec, '-')[end])
end
const PRICE = _arg("--price", "u4")
const TL = parse(Float64, _arg("--tl", "600"))
const RF = (v = _arg("--rf", "4-2-8"); v == "none" ? nothing : parse.(Int, split(v, '-')))  # window, step, LP look-ahead
const INSTDIR = _arg("--instdir", joinpath(@__DIR__, "..", "scaling", "instances"))
const TAG = _arg("--tag", "")
const PHI = (v = _arg("--phi", "instance"); v == "instance" ? nothing : parse(Float64, v))   # 0: Γ = 1 diagnostic
const SKIP_MILP = _arg("--milp", "1") == "0"
const JOB = get(ENV, "SLURM_JOB_ID", "local")
const OUT = joinpath(@__DIR__, "..", "results", JOB, "ref_bounds.jsonl")
mkpath(dirname(OUT))

using Gurobi
const GENV = Gurobi.Env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 1, "TimeLimit" => TL, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

_clean(v::AbstractFloat) = isfinite(v) ? v : nothing
_clean(v) = v
record(row) = (safe = Dict(k => _clean(v) for (k, v) in row);
               open(io -> (JSON.print(io, safe); println(io)), OUT, "a");
               println("RECORD ", JSON.json(safe)); flush(stdout))

P = PRICE[1] == 'u' ? fill(parse(Float64, PRICE[2:end]), NN, NN) : error("price $PRICE")
inst = load_ts_instance(joinpath(INSTDIR, "syn_n$(NN)_T$(TT)$(TAG).json"), "syn"; levels = 3,
                        nscen = maximum(SCENS), phi = PHI)
let w = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n3_T6.json"), "syn"; levels = 3, nscen = 1)
    solve_tier(w, 1, fill(4.0, 3, 3), opt()); lp_load(w, 1, fill(4.0, 3, 3), opt())
    relax_and_fix(w, 1, fill(4.0, 3, 3), opt(); w = 4, step = 2, look = 2)
end
base = Dict("job" => JOB, "n" => NN, "T" => TT, "tag" => TAG * (PHI === nothing ? "" : "_phi$(PHI)"),
            "Gamma" => inst.Γ, "price" => PRICE, "tl" => TL)
for s in SCENS
    t0 = time()
    _, lpv = lp_load(inst, s, P, opt())
    record(merge(base, Dict("scenario" => s, "method" => "lp", "bound" => lpv, "wall_s" => time() - t0)))
    if !SKIP_MILP
        t0 = time()
        v, m, b, info = solve_tier(inst, s, P, opt())
        record(merge(base, Dict("scenario" => s, "method" => "tier", "value" => v, "bound" => info.bound,
                                "optimal" => info.optimal, "feasible" => info.feasible,
                                "solve_s" => info.solve_s, "wall_s" => time() - t0, "nbin" => info.nbin)))
    end
    RF === nothing && continue
    t0 = time()
    v, _, b, info = relax_and_fix(inst, s, P, opt(); w = RF[1], step = RF[2], look = RF[3], win_tl = 120.0)
    nbp = info.ok ? length(blocking_pairs(inst, b, P)) : -1
    record(merge(base, Dict("scenario" => s, "method" => "rf_$(join(RF, '-'))", "value" => v, "ok" => info.ok,
                            "blocking" => nbp, "solve_s" => info.solve_s, "wall_s" => time() - t0)))
end
println("done"); flush(stdout)

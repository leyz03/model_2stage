"""
run_lagrangian_td.jl — Lagrangian time decomposition (src/lagrangian_td.jl) on fixed-price scenarios.
Run via slurm_lagrangian_td.sh; writes results/<job>/lagrangian_td.jsonl (one row per scenario and variant).

Variants (joined by '/'): <cut><w>[n][z]
  cut  u = balanced blocks of about w periods,  a = same number of blocks, boundaries at low
       in-transit + open-task periods of the whole-day LP (lengths ceil(0.75w)..ceil(1.5w), 3..6 for w = 4)
  n    unscaled prox (all weights 1) instead of per bike / worker / task units
  z    λ0 = 0 instead of the joint-LP shadow prices
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : ARGS[i+1]
const NN = parse(Int, _arg("--n", "3"))
const TT = parse(Int, _arg("--T", "8"))
const SCENS = let spec = _arg("--scen", "1-3")
    occursin('-', spec) ? (parse(Int, split(spec, '-')[1]):parse(Int, split(spec, '-')[2])) :
                          parse.(Int, split(spec, '.'))
end
const PRICE = _arg("--price", "u4")
const VARIANTS = split(_arg("--variants", "u4"), '/')
const BUDGET = parse(Float64, _arg("--budget", "600"))
const BLOCK_TL = parse(Float64, _arg("--block-tl", "60"))
const BLOCK_GAP = parse(Float64, _arg("--block-gap", "1e-6"))
const MYOPIC = _arg("--myopic", "1") == "1"
const INSTDIR = _arg("--instdir", joinpath(@__DIR__, "..", "scaling", "instances"))
const TAG = _arg("--tag", "")                # e.g. _m6 for fixed_price/instances/syn_n5_T24_m6.json
const PHI = (v = _arg("--phi", "instance"); v == "instance" ? nothing : parse(Float64, v))   # 0: Γ = 1 diagnostic
const JOB = get(ENV, "SLURM_JOB_ID", "local")
const OUT = joinpath(@__DIR__, "..", "results", JOB, "lagrangian_td.jsonl")
mkpath(dirname(OUT))

using Gurobi
const GENV = Gurobi.Env()
blk_opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0,
    "MIPGap" => BLOCK_GAP, "Threads" => 1, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
qp_opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "Threads" => 1)

function price_of(spec, n)
    spec[1] == 'u' && return fill(parse(Float64, spec[2:end]), n, n)
    mv, sw = parse.(Float64, split(spec[2:end], '_'))
    return [j == k ? sw : mv for j in 1:n, k in 1:n]
end

_clean(v::AbstractFloat) = isfinite(v) ? v : nothing
_clean(v::AbstractVector) = _clean.(v)
_clean(v) = v
function record(row)
    safe = Dict(k => _clean(v) for (k, v) in row)
    open(io -> (JSON.print(io, safe); println(io)), OUT, "a")
    println("RECORD ", JSON.json(Dict(k => v for (k, v) in safe if k != "trace"))); flush(stdout)
end

function parse_variant(v)
    m = match(r"^([ua])(\d+)([nz]*)$", v)
    m === nothing && error("variant $v")
    return (cut = m[1] == "u" ? :uniform : :adaptive, w = parse(Int, m[2]),
            scaled = !occursin('n', m[3]), λ0 = occursin('z', m[3]) ? :zero : :lp)
end

inst = load_ts_instance(joinpath(INSTDIR, "syn_n$(NN)_T$(TT)$(TAG).json"), "syn"; levels = 3,
                        nscen = maximum(SCENS), phi = PHI)
P = price_of(PRICE, NN)
@printf("Lagrangian TD  n=%d T=%d scenarios=%s price=%s variants=%s budget=%g block_tl=%g gap=%g\n",
        NN, TT, collect(SCENS), PRICE, join(VARIANTS, ","), BUDGET, BLOCK_TL, BLOCK_GAP); flush(stdout)

# compile everything once on a small instance
let w = load_ts_instance(joinpath(@__DIR__, "..", "scaling", "instances", "syn_n3_T6.json"), "syn";
                         levels = 3, nscen = 1)
    pw = fill(4.0, 3, 3)
    _, λw, _ = lagrangian_td(w, 1, pw, blk_opt(), qp_opt(); cuts = td_cuts(6, 3), budget = 20.0, verbose = false)
    _, plan, _ = td_forward(w, 1, pw, blk_opt(), td_cuts(6, 3), λw)
    td_verify(w, 1, pw, blk_opt(), plan)
    lp_load(w, 1, pw, blk_opt())
    println("warm-up complete"); flush(stdout)
end

for s in SCENS, vs in VARIANTS
    v = parse_variant(vs)
    tw = time()
    B = cld(TT, v.w)
    load, lpw = lp_load(inst, s, P, blk_opt())
    cuts = v.cut === :uniform ? td_cuts(TT, v.w) :
        td_cuts_adaptive(load, TT, B; wmin = cld(3v.w, 4), wmax = cld(3v.w, 2))
    @printf("== n=%d T=%d s=%d %s  cuts %s  boundary load %s\n", NN, TT, s, vs, cuts,
            [round(load[b[1]]; digits = 2) for b in cuts[2:end]]); flush(stdout)
    ub, λb, info = lagrangian_td(inst, s, P, blk_opt(), qp_opt(); cuts, λ0 = v.λ0, scaled = v.scaled,
                                 block_tl = BLOCK_TL, budget = BUDGET)
    tf = time()
    fv, plan, fi = td_forward(inst, s, P, blk_opt(), cuts, λb; block_tl = BLOCK_TL)
    vv, nbp, res = fi.ok ? td_verify(inst, s, P, blk_opt(), plan) : (-Inf, -1, Inf)
    fwd_s = time() - tf
    fi.ok && (abs(vv - fv) > 1e-6 * max(1, abs(fv)) || nbp != 0 || res > 1e-6) &&
        error("forward plan n=$NN T=$TT s=$s: value $fv re-evaluated $vv, blocking $nbp, residual $res")
    myo = MYOPIC ? td_forward(inst, s, P, blk_opt(), cuts, nothing; block_tl = BLOCK_TL)[1] : nothing
    tr = info.trace
    ub_at(sec) = (k = findlast(r -> r.wall <= sec, tr)) === nothing ? Inf : tr[k].best
    record(Dict("job" => JOB, "n" => NN, "T" => TT, "tag" => TAG, "phi" => PHI === nothing ? "instance" : PHI, "Gamma" => inst.Γ, "scenario" => s, "price" => PRICE, "variant" => vs,
                "cut" => String(v.cut), "w" => v.w, "scaled" => v.scaled, "lambda0" => String(v.λ0),
                "cuts" => [[a, b] for (a, b) in cuts], "boundary_load" => [load[b[1]] for b in cuts[2:end]],
                "B" => info.B, "nlambda" => info.nλ, "lp" => info.lp, "lp_whole" => lpw,
                "ub" => ub, "ub_300" => ub_at(300.0), "ub_600" => ub_at(600.0),
                "it" => info.it, "serious" => info.serious, "converged" => info.converged, "stop" => String(info.stop),
                "ncut" => info.ncut, "lag_wall" => info.wall, "build_s" => info.build_s, "lp_s" => info.lp_s,
                "block_solver_s" => sum(r.ssum for r in tr), "block_max_s" => maximum(r.smax for r in tr),
                "evals_with_tl" => count(r -> r.ntl > 0, tr), "budget" => BUDGET, "block_tl" => BLOCK_TL,
                "block_gap" => BLOCK_GAP, "fwd" => fv, "fwd_ok" => fi.ok, "fwd_tl" => fi.ntl, "fwd_s" => fwd_s,
                "fwd_myopic" => myo, "wall" => time() - tw,
                "trace" => [[r.it, r.wall, r.ub, r.best, r.ssum, r.smax, r.ntl] for r in tr]))
end
println("done"); flush(stdout)

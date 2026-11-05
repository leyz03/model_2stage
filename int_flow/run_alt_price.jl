"""
run_alt_price.jl  —  one run of TODO #4: per-OD prices by alternating (int_flow/alt_price.jl) on periods 1..T of a few
days of the integer-split instance, against the best uniform price.

    julia --project=model_2stage/env model_2stage/int_flow/run_alt_price.jl --T 20 [opts]

1. uniform: all days at p = θ (exact tier model) for every level θ of the menu (--uniform all) or only for the
   explicit starts (--uniform starts);
2. alternating from the uniform prices in --starts, for each P-step rule in --unused, until the P step changes nothing
   or --maxit O steps; every O step starts from the previous operations;
3. upper bound for any per-OD prices on the menu (`discrim_ub`): --ub 1, 0 or only (nothing else).

--days "11,18,1" (#2's days), --menu "0:12" or a list "0,2,4,6" (every OD the same levels),
--starts all | best | top<k> (the k best uniform levels) | none | levels "6" or "4,6,8", --unused "keep,follow",
--maxit 10, --instance (default int_flow/instances/paris_n30_H10_int_workers.json), --tl 7200 (s per MILP),
--threads 8.  Gurobi, MIPGap 0, IntFeasTol = FeasibilityTol = 1e-9.  Writes results/<job id>/<tag>.json (tag
alt_T<T>[_s<starts>][_ub]) and the Gurobi log (all solves) next to it; <job id> is the array job id for array tasks.
One array task per start (slurm_alt_price.sh) and summarize_alt_price.py merges them.
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "alt_price.jl"))

using Gurobi, Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const INSTANCE = _sarg("--instance", joinpath(@__DIR__, "instances", "paris_n30_H10_int_workers.json"))
const TT       = parse(Int, _sarg("--T", "10"))
const DAYS     = parse.(Int, split(_sarg("--days", "11,18,1"), ","))
const MENU_ARG = _sarg("--menu", "0:12")
const STARTS   = _sarg("--starts", "all")
const UNUSED   = Symbol.(split(_sarg("--unused", "keep,follow"), ","))
const MAXIT    = parse(Int, _sarg("--maxit", "10"))
const TL       = parse(Float64, _sarg("--tl", "7200"))
const THREADS  = parse(Int, _sarg("--threads", "8"))
const UNIFORM  = _sarg("--uniform", "all")
const UB_ARG   = _sarg("--ub", "1")
const EXPLICIT = all(ch -> isdigit(ch) || ch in ",.", STARTS)
const JOB      = get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "local"))
const OUT_DIR  = joinpath(@__DIR__, "results", JOB)
const TAG      = @sprintf("alt_T%03d", TT) * (EXPLICIT ? "_s" * replace(STARTS, "," => "-") : "") *
                 (UB_ARG == "only" ? "_ub" : "")
mkpath(OUT_DIR)

parse_menu(s) = occursin(":", s) ? Float64.((:)(parse.(Int, split(s, ":"))...)) : parse.(Float64, split(s, ","))

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
opt(s = 0) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "LogToConsole" => 0, "MIPGap" => 0.0,
    "Threads" => THREADS, "TimeLimit" => TL, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9,
    "LogFile" => joinpath(OUT_DIR, TAG * ".log"))

_clean(x::AbstractFloat) = isfinite(x) ? x : nothing
_clean(x::NamedTuple) = Dict(string(k) => _clean(v) for (k, v) in pairs(x))
_clean(x::AbstractVector) = [_clean(v) for v in x]
_clean(x::AbstractDict) = Dict(string(k) => _clean(v) for (k, v) in x)
_clean(x) = x

raw = JSON.parsefile(INSTANCE)
full = load_ts_instance(INSTANCE, "paris"; levels = 2)
const MENU = parse_menu(MENU_ARG)
const N = full.n
menus = uniform_menus(N, MENU)
ws = [window_instance(full, d, 1, TT) for d in DAYS]
dates = [raw["variants"]["paris"]["pools"]["train"]["dates"][d] for d in DAYS]
println("== $TAG  days $DAYS ($(join(dates, ", ")))  n=$N  menu $MENU  starts $STARTS  uniform $UNIFORM  ",
        "unused $UNUSED  ub $UB_ARG  threads $THREADS  tl $TL s"); flush(stdout)

# O steps by price (several starts can reach the same price)
cache = Dict{Vector{Float64},Any}()
nsolve = Ref(0)
function solve_at(p; starts = nothing)
    key = vec(p)
    haskey(cache, key) && return cache[key]
    vals, snaps, infos = ostep(ws, p, opt; starts)
    nsolve[] += length(ws)
    cache[key] = (vals = vals, snaps = snaps, infos = infos)
    return cache[key]
end
summ(r) = (value = mean(r.vals), vals = r.vals, optimal = all(i.optimal for i in r.infos),
           solve_s = sum(i.solve_s for i in r.infos), moves = sum(i.ops.moves for i in r.infos),
           swaps = sum(i.ops.swaps for i in r.infos), wages = sum(i.ops.wages for i in r.infos),
           lost = sum(i.ops.lost for i in r.infos), blocking = sum(i.blocking for i in r.infos),
           start = [i.start for i in r.infos])

# 1. uniform
UB_ARG == "only" || UNIFORM == "all" || EXPLICIT || error("--uniform starts needs explicit --starts")
uni = []
for θ in (UB_ARG == "only" ? Float64[] : UNIFORM == "all" ? MENU : parse.(Float64, split(STARTS, ",")))
    r = solve_at(fill(θ, N, N))
    s = summ(r)
    push!(uni, merge((price = θ,), s))
    @printf("uniform %4.1f: %.2f  (moves %d swaps %d wages %.0f lost %d, %.1f s%s)\n", θ, s.value, s.moves, s.swaps,
            s.wages, s.lost, s.solve_s, s.optimal ? "" : ", NOT OPTIMAL"); flush(stdout)
end
order = sortperm([u.value for u in uni]; rev = true)
best_uni = isempty(uni) ? nothing : uni[order[1]]
starts = UB_ARG == "only" || STARTS == "none" ? Float64[] : EXPLICIT ? parse.(Float64, split(STARTS, ",")) :
         STARTS == "all" ? MENU : STARTS == "best" ? [best_uni.price] :
         startswith(STARTS, "top") ? [uni[h].price for h in order[1:parse(Int, STARTS[4:end])]] :
         error("--starts $STARTS")

# 2. alternating
pricestats(p, vol) = (used = count(vol .> 0), levels = Dict(string(l) => count(jk -> vol[jk] > 0 && p[jk] == l,
                      eachindex(p)) for l in MENU if any(jk -> vol[jk] > 0 && p[jk] == l, eachindex(p))))
runs = []
for θ in starts, rule in UNUSED
    p = fill(θ, N, N)
    r = solve_at(p)
    trace = Any[merge((it = 0, p_changed = 0), summ(r))]
    @printf("start %4.1f %-6s it 0: %.2f\n", θ, rule, mean(r.vals)); flush(stdout)
    conv = false
    for it in 1:MAXIT
        rows = price_rows(ws, r.snaps)
        tp = @elapsed q = least_prices(p, menus, rows; unused = rule)
        nchg = count(q .!= p)
        if nchg == 0
            conv = true
            break
        end
        # the old operations at q: still stable (checked on the rows), wage lower
        trimmed = [r.vals[s] + sum((p .- q) .* task_volume(r.snaps[s])) for s in eachindex(ws)]
        r2 = solve_at(q; starts = r.snaps)
        s2 = summ(r2)
        mono = all(r2.vals .>= trimmed .- 1e-6)
        push!(trace, merge((it = it, p_changed = nchg, pstep_s = tp, trimmed = mean(trimmed), monotone = mono,
                            start_ok = all(abs.(s2.start .- trimmed) .<= 1e-6 .* max.(1.0, abs.(trimmed))),
                            prices = pricestats(q, rows.vol)), s2))
        @printf("start %4.1f %-6s it %d: %d prices changed, trimmed %.2f -> %.2f  (moves %d wages %.0f%s%s)\n", θ, rule,
                it, nchg, mean(trimmed), s2.value, s2.moves, s2.wages, mono ? "" : ", NOT MONOTONE",
                trace[end].start_ok ? "" : ", start value differs"); flush(stdout)
        p, r = q, r2
    end
    push!(runs, (start = θ, unused = string(rule), converged = conv, value = mean(r.vals), trace = trace,
                 prices = [(j, k, p[j, k]) for j in 1:N, k in 1:N if p[j, k] != θ],
                 used_prices = [(j, k, p[j, k]) for j in 1:N, k in 1:N
                                if any(task_volume(sn)[j, k] > 0 for sn in r.snaps)]))
end
best_run = isempty(runs) ? nothing : runs[argmax([x.value for x in runs])]

# 3. upper bound
ub = nothing
if UB_ARG in ("1", "only")
    ubs = [discrim_ub(w, menus, opt()) for w in ws]
    ub = (value = mean(first.(ubs)), per_day = first.(ubs), info = last.(ubs))
    @printf("upper bound (no stable matching, every worker paid the menu level at its threshold): %.2f\n", ub.value)
end

out = Dict{String,Any}("tag" => TAG, "job" => JOB, "instance" => basename(INSTANCE), "days" => DAYS, "dates" => dates,
    "T" => TT, "menu" => MENU, "starts" => starts, "uniform_set" => UNIFORM, "unused" => string.(UNUSED),
    "maxit" => MAXIT, "tl" => TL, "threads" => THREADS, "mip_solves" => nsolve[], "uniform" => uni, "best_uniform" => best_uni, "runs" => runs,
    "best" => best_run === nothing ? nothing : (start = best_run.start, unused = best_run.unused,
                                                 value = best_run.value), "ub" => ub)
open(io -> JSON.print(io, _clean(out), 1), joinpath(OUT_DIR, TAG * ".json"), "w")
best_uni === nothing || @printf("best uniform %.2f (price %.1f)\n", best_uni.value, best_uni.price)
best_run === nothing || @printf("best alternating %.2f (start %.1f, %s)\n", best_run.value, best_run.start,
                                best_run.unused)

"""
run_scaling.jl  —  n / T / K_s scaling runs of the two-stage baseline (model_2stage/04_algorithm.md)

    julia --project=model_2stage/env model_2stage/scaling/run_scaling.jl --mode MODE --n 5 --T 8 [opts]

MODE
  oracle  per-scenario MILP cost at one fixed price (`--price`, default 4 for every task):
          for each formulation in --inner (default full,pre,lazy) and the price-box relaxation
          at the same price, solve the first --nscen scenarios; checks that every formulation
          gives the same value.
  bnb     two-stage baseline: price_bnb with --inner (one formulation), --nscen scenarios.
  ef      extensive form (build_ef, full stability) on --nscen scenarios.

Common options: --structure type|uniform|od (default type), --levels 3, --tl 1800 (seconds,
for bnb / ef), --mip_tl 600 (per small MILP), --solver gurobi|highs, --threads 1.
Instance: model_2stage/scaling/instances/syn_n{n}_T{T}.json (or --instance PATH, --variant).
Every run appends one JSON line to results/<SLURM_JOB_ID or local>/scaling.jsonl.
"""

include(joinpath(@__DIR__, "..", "src", "price_bnb.jl"))

using Printf, Statistics

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const MODE      = _sarg("--mode", "oracle")
const NN        = parse(Int, _sarg("--n", "5"))
const TT        = parse(Int, _sarg("--T", "8"))
const INSTANCE  = _sarg("--instance", joinpath(@__DIR__, "instances", "syn_n$(NN)_T$(TT).json"))
const VARIANT   = _sarg("--variant", "syn")
const STRUCTURE = Symbol(_sarg("--structure", "type"))
const LEVELS    = parse(Int, _sarg("--levels", "3"))
const NSCEN     = parse(Int, _sarg("--nscen", "10"))
const INNER     = Symbol.(split(_sarg("--inner", MODE == "oracle" ? "full,pre,lazy" : "pre"), r"[,/]"))   # "/" survives sbatch --export
const PRICE     = parse(Float64, _sarg("--price", "4"))
const TL        = parse(Float64, _sarg("--tl", "1800"))
const MIP_TL    = parse(Float64, _sarg("--mip_tl", "600"))
const SOLVER    = _sarg("--solver", "gurobi")
const THREADS   = parse(Int, _sarg("--threads", "1"))
const OUT_DIR   = joinpath(@__DIR__, "..", "results", get(ENV, "SLURM_JOB_ID", "local"))
mkpath(OUT_DIR)

if SOLVER == "gurobi"
    using Gurobi
    const GENV = Gurobi.Env()
    opt(; tl = MIP_TL, threads = THREADS) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV),
        "OutputFlag" => 0, "MIPGap" => 0.0, "Threads" => threads, "TimeLimit" => tl,
        "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)
else
    using HiGHS
    opt(; tl = MIP_TL, threads = THREADS) = optimizer_with_attributes(HiGHS.Optimizer,
        "output_flag" => false, "mip_rel_gap" => 0.0, "mip_abs_gap" => 0.0, "time_limit" => tl,
        "threads" => threads)
end

# JSON has no NaN / Inf: a formulation without an incumbent within its time limit is written as null
_clean(x::AbstractFloat) = isfinite(x) ? x : nothing
_clean(x::AbstractVector) = map(_clean, x)
_clean(x) = x
record(d) = (d = Dict(k => _clean(v) for (k, v) in d);
             open(io -> (JSON.print(io, d); println(io)), joinpath(OUT_DIR, "scaling.jsonl"), "a");
             println("RECORD ", JSON.json(d)); flush(stdout))

inst = load_ts_instance(INSTANCE, VARIANT; levels = LEVELS, nscen = NSCEN, structure = STRUCTURE)
base = Dict("mode" => MODE, "n" => inst.n, "T" => inst.T, "nscen" => NSCEN, "structure" => string(STRUCTURE),
            "levels" => LEVELS, "gamma" => inst.Γ, "solver" => SOLVER, "threads" => THREADS,
            "instance" => basename(INSTANCE), "job" => get(ENV, "SLURM_JOB_ID", "local"))
@printf("%s  %s  n=%d T=%d K_s=%d  Γ=%d  menu=%d\n", MODE, basename(INSTANCE), inst.n, inst.T, NSCEN,
        inst.Γ, length(inst.menu)); flush(stdout)

if MODE == "oracle"
    P = fill(PRICE, inst.n, inst.n)
    ref = nothing
    for form in vcat(INNER, [:relax])
        global ref
        vals, walls, solves, cuts, nbins, nopt = Float64[], Float64[], Float64[], Int[], Int[], 0
        for s in 1:NSCEN
            tw = time()
            if form === :relax
                v, info = solve_relax(inst, s, P, P, opt())
                c = 0
            else
                v, _, _, info = solve_scenario(inst, s, P, opt(); stability = form)
                c = info.ncuts
            end
            push!(vals, v); push!(walls, time() - tw); push!(solves, info.solve_s)
            push!(cuts, c); push!(nbins, info.nbin); info.optimal || (nopt += 1)
        end
        # comparisons only over scenarios every formulation solved to optimality
        if form !== :relax
            ref === nothing && (ref = vals)
            both = isfinite.(vals) .& isfinite.(ref)
            maxdiff = any(both) ? maximum(abs.(vals[both] .- ref[both])) : NaN
        else
            both = isfinite.(ref)
            maxdiff = any(both) ? minimum(vals[both] .- ref[both]) : NaN   # relaxation must be ≥ the exact value
        end
        vals = [isfinite(v) ? v : NaN for v in vals]      # no incumbent within the time limit
        @printf("  %-6s mean value %10.4f  solver %.3f s/MILP (max %.3f)  wall %.3f s/MILP  binaries %d  cuts %.1f  check %.2e\n",
                form, mean(vals), mean(solves), maximum(solves), mean(walls), maximum(nbins), mean(cuts), maxdiff)
        record(merge(base, Dict("form" => string(form), "price" => PRICE, "mean_value" => mean(vals),
            "solver_s_mean" => mean(solves), "solver_s_max" => maximum(solves), "wall_s_mean" => mean(walls),
            "binaries" => maximum(nbins), "lazy_cuts_mean" => mean(cuts), "not_optimal" => nopt, "mip_tl" => MIP_TL,
            (form === :relax ? "min_relax_minus_exact" : "max_abs_diff_vs_first") => maxdiff)))
    end
elseif MODE == "bnb"
    res = price_bnb(inst, opt(); inner = INNER[1], tl = TL, log = false)
    st = res.stats
    @printf("  bnb[%s] %s  value %.4f  ub %.4f  prices %s  %.1f s  nodes %d  exact %d pts / %d MILPs  relax %d pts / %d MILPs  cands %s\n",
            INNER[1], res.status, res.value, res.ub, string(res.prices), res.seconds, st.nodes, st.exact_points,
            st.exact_milps, st.relax_points, st.relax_milps, string(res.candidates))
    record(merge(base, Dict("inner" => string(INNER[1]), "status" => string(res.status), "value" => res.value,
        "ub" => res.ub, "prices" => res.prices, "seconds" => res.seconds, "candidates" => res.candidates,
        "menu" => res.menu, "nodes" => st.nodes, "exact_points" => st.exact_points, "exact_milps" => st.exact_milps,
        "exact_s" => st.exact_s, "exact_solver_s" => st.exact_solver_s, "relax_points" => st.relax_points,
        "relax_milps" => st.relax_milps, "relax_s" => st.relax_s, "relax_solver_s" => st.relax_solver_s,
        "lazy_cuts" => st.lazy_cuts, "max_nbin_exact" => st.max_nbin_exact, "max_nbin_relax" => st.max_nbin_relax,
        "exact_not_optimal" => st.exact_not_optimal, "relax_not_optimal" => st.relax_not_optimal, "mip_tl" => MIP_TL,
        "tl" => TL)))
elseif MODE == "ef"
    tb = @elapsed (m, price, blocks) = build_ef(inst, opt(; tl = TL))
    optimize!(m)
    stt = JuMP.termination_status(m)
    has = primal_status(m) == MOI.FEASIBLE_POINT
    val = has ? objective_value(m) : NaN
    choice = has ? [inst.menu[findfirst(l -> value(price.z[g, l]) > 0.5, eachindex(inst.menu))] for g in 1:inst.ngroups] : Float64[]
    @printf("  ef %s  value %.4f  bound %.4f  prices %s  build %.1f s  solve %.1f s  vars %d  binaries %d\n",
            stt, val, objective_bound(m), string(choice), tb, solve_time(m), num_variables(m), count_free_binaries(m))
    record(merge(base, Dict("status" => string(stt), "value" => val, "ub" => objective_bound(m), "prices" => choice,
        "build_s" => tb, "seconds" => solve_time(m), "num_variables" => num_variables(m),
        "binaries" => count_free_binaries(m), "tl" => TL)))
end

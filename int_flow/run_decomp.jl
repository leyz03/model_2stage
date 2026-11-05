"""
run_decomp.jl  —  one run of TODO #3: periods 1..T of one day of the integer-split instance at the baseline price,
exact tier model (stable matching), solved by one method (int_flow/decomp.jl).

    julia --project=model_2stage/env model_2stage/int_flow/run_decomp.jl --day 18 --T 100 --method cg [opts]

--method full | lp | cg | benders | nox | bin   (the MIP as in #2 / its LP / column generation on x / Benders over x
                    and binaries / TODO #8: the MIP with x continuous / with only β, z integral)
--instance (default int_flow/instances/paris_n30_H10_int_workers.json), --tl 7200 (s, whole method), --threads 8,
--init d0 | all (cg start columns), --fixstep 1 | 0 (cg: fix the integral part of the LP solution first),
--maxit 100000 (benders iterations),
--drop S | M | SM (full / lp / nox / bin, TODO #8: without the service rule rows / without stable matching / both).
--stab tier | lazy (full / lp / cg, TODO #9, int_flow/lazy_stab.jl: the stable-matching rows on demand; tag _lazyM)
    full  no stable-matching row at the start, every z built; a lazy-constraint callback adds the rows of the groups
          each integral candidate violates (one branch and bound, exact)
    lp    row generation: LP without those rows, add the groups no z ∈ [0, 1] satisfies, re-solve (dual simplex)
    cg    the LP phase adds rows when no column is left, the MIP phase adds the rest by the callback
--stab defer (cg only, TODO #9 11-04; tag _deferM): the same LP phase, then every remaining stable-matching row is
          added at once before the MIP phase (no callback)
Gurobi, MIPGap 0, IntFeasTol = FeasibilityTol = 1e-9 (as #2).  Writes results/<array job id or job id>/<tag>.json and
the Gurobi log next to it.  method_s = wall time of the method after the model is built (full / lp: Gurobi time).
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "decomp.jl"))

using Gurobi, Printf

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const INSTANCE = _sarg("--instance", joinpath(@__DIR__, "instances", "paris_n30_H10_int_workers.json"))
const DAY      = parse(Int, _sarg("--day", "18"))
const TT       = parse(Int, _sarg("--T", "10"))
const METHOD   = Symbol(_sarg("--method", "cg"))
const TL       = parse(Float64, _sarg("--tl", "7200"))
const THREADS  = parse(Int, _sarg("--threads", "8"))
const INIT     = Symbol(_sarg("--init", "d0"))
const MAXIT    = parse(Int, _sarg("--maxit", "100000"))
const FIXSTEP  = _sarg("--fixstep", "1") == "1"
const DROP     = _sarg("--drop", "")
const STAB     = Symbol(_sarg("--stab", "tier"))
const JOB      = get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "local"))
const OUT_DIR  = joinpath(@__DIR__, "results", JOB)
const TAG      = @sprintf("d%02d_T%03d_%s", DAY, TT, METHOD) * (METHOD === :cg && !FIXSTEP ? "nofix" : "") *
                 (isempty(DROP) ? "" : "_drop" * DROP) * (STAB === :tier ? "" : "_" * string(STAB) * "M")
METHOD in (:full, :lp, :cg, :benders, :nox, :bin) || error("--method must be full, lp, cg, benders, nox or bin")
DROP in ("", "S", "M", "SM") || error("--drop must be S, M or SM")
isempty(DROP) || METHOD in (:full, :lp, :nox, :bin) || error("--drop only with full, lp, nox or bin")
STAB in (:tier, :lazy, :defer) || error("--stab must be tier, lazy or defer")
STAB === :tier || (METHOD in (:full, :lp, :cg) && isempty(DROP)) || error("--stab lazy only with full, lp or cg, no --drop")
STAB === :defer && METHOD !== :cg && error("--stab defer only with cg")
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
opt(; log = TAG) = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 1, "MIPGap" => 0.0,
    "Threads" => THREADS, "TimeLimit" => TL, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9,
    "LogFile" => joinpath(OUT_DIR, log * ".log"))

_clean(x::AbstractFloat) = isfinite(x) ? x : nothing
_clean(x::NamedTuple) = Dict(string(k) => _clean(v) for (k, v) in pairs(x))
_clean(x::AbstractVector) = [_clean(v) for v in x]
_clean(x) = x
attr(m, a, d = NaN) = try MOI.get(m, a) catch; d end

raw = JSON.parsefile(INSTANCE)
full = load_ts_instance(INSTANCE, "paris"; levels = 2)
P = fill(Float64(raw["pricing"]["baseline"]), full.n, full.n)
w = window_instance(full, DAY, 1, TT)
day = (date = raw["variants"]["paris"]["pools"]["train"]["dates"][DAY], A0 = sum(w.A0), U0 = sum(w.U0), fleet = w.Q3,
       riders = sum(sum, w.scen[1]), dead_riders = sum(sum, w.scenU[1]))
println("== $TAG  $(day.date)  n=$(w.n) T=$(w.T)  fleet $(day.fleet) (U0 $(day.U0))  riders $(day.riders) ",
        "(dead $(day.dead_riders))  threads $THREADS  tl $TL s"); flush(stdout)

out = Dict{String,Any}("tag" => TAG, "job" => JOB, "instance" => basename(INSTANCE), "day" => DAY, "T" => TT,
    "method" => string(METHOD), "threads" => THREADS, "tl" => TL, "price" => P[1, 1], "day_info" => _clean(day),
    "drop" => DROP, "fleet_scale" => get(get(raw, "fleet_scale", Dict()), "rho", 1.0), "stab" => string(STAB))

if METHOD === :full && STAB === :lazy                 # TODO #9: one branch and bound, stable-matching rows lazily
    m = Model(opt())
    tb = @elapsed begin
        b = add_scenario_sparse!(m, w, w.scen[1], P, P; stability = :none)
        S = stab_init!(m, w, b)
        nothing_start!(w, b)
        @objective(m, Max, b.profit)
        lazy_stab!(m, S)
    end
    out["sizes"] = _clean(_sizes(m))
    optimize!(m)
    st = JuMP.termination_status(m)
    out["status"] = string(st)
    out["value"] = primal_status(m) == MOI.FEASIBLE_POINT ? objective_value(m) : NaN
    out["bound"] = attr(m, MOI.ObjectiveBound())
    out["proved"] = st == MOI.OPTIMAL
    out["build_s"] = tb
    out["method_s"] = solve_time(m)
    out["nodes"] = attr(m, MOI.NodeCount(), -1)
    out["stab_stats"] = _clean(stab_stats(S))
elseif METHOD === :lp && STAB === :lazy               # TODO #9: row generation on the full LP
    r = cg_solve(w, P, opt(); init = :all, tl = TL, stab = :lazy, lp_only = true)
    m, b = r.model, r.block
    out["sizes"] = _clean(_sizes(m))
    out["value"] = out["bound"] = out["lp_bound"] = r.lp_bound
    out["proved"] = true
    out["status"] = "OPTIMAL"
    out["build_s"] = r.build_s
    out["method_s"] = r.method_s
    out["lp_solve_s"] = sum(l.solve_s for l in r.log)
    out["rounds"] = length(r.log)
    out["log"] = _clean(r.log)
    out["stab_stats"] = _clean(stab_stats(r.stab))
elseif METHOD in (:full, :lp, :nox, :bin)
    m = Model(opt())
    tb = @elapsed begin
        b = add_scenario_sparse!(m, w, w.scen[1], P, P; stability = 'M' in DROP ? :none : :tier)
        'S' in DROP && (out["dropped_service_rows"] = drop_service_rule!(m, b))
        METHOD === :lp || nothing_start!(w, b)
        @objective(m, Max, b.profit)
        METHOD === :full || relax_ints!(b, METHOD)
    end
    out["sizes"] = _clean(_sizes(m))
    optimize!(m)
    st = JuMP.termination_status(m)
    out["status"] = string(st)
    out["value"] = primal_status(m) == MOI.FEASIBLE_POINT ? objective_value(m) : NaN
    out["bound"] = METHOD === :lp ? out["value"] : attr(m, MOI.ObjectiveBound())
    out["proved"] = st == MOI.OPTIMAL
    out["build_s"] = tb
    out["method_s"] = solve_time(m)
    out["nodes"] = attr(m, MOI.NodeCount(), -1)
elseif METHOD === :cg
    r = cg_solve(w, P, opt(); init = INIT, tl = TL, fixstep = FIXSTEP, stab = STAB)
    m, b = r.model, r.block
    out["sizes"] = _clean(_sizes(m))
    r.stab === nothing || (out["stab_stats"] = _clean(stab_stats(r.stab)))
    for k in (:value, :bound, :lp_bound, :proved, :build_s, :lp_s, :method_s, :start, :active, :columns)
        out[string(k)] = getfield(r, k)
    end
    out["log"] = _clean(r.log)
    out["status"] = r.proved ? "OPTIMAL" : string(JuMP.termination_status(m))
    out["init"] = string(INIT)
    out["fixstep"] = FIXSTEP
else
    r = benders_solve(w, P, opt(log = TAG * "_sub"); mopt = opt(log = TAG * "_master"), tl = TL, maxit = MAXIT)
    m, b = r.model, r.block
    out["sizes"] = _clean(_sizes(m))
    out["master_sizes"] = _clean(_sizes(r.master))
    for k in (:value, :bound, :lp_bound, :proved, :build_s, :method_s, :master_s, :sub_s, :nmaster)
        out[string(k)] = getfield(r, k)
    end
    out["iterations"] = length(r.log)
    out["feas_cuts"] = count(l -> l.cut == "feas", r.log)
    out["log"] = _clean(r.log)
    out["status"] = r.proved ? "OPTIMAL" : "NOT_PROVED"
end

feas = METHOD !== :lp && primal_status(m) == MOI.FEASIBLE_POINT && (METHOD !== :benders || out["value"] > -Inf)
if feas
    out["frac"] = Dict("x" => nfrac(value.(x_vars(b))), "bin" => nfrac(value.(bin_vars(b))),
                       "int" => nfrac(value.(int_vars(b))))
    out["ops"] = _clean(op_stats_sparse(w, b, P))
    out["residual"] = conservation_residual(w, b)
    out["blocking_pairs"] = length(blocking_pairs_sparse(w, b, P))
    out["service_violations"] = length(service_violations(w, b))
end
out = Dict(k => _clean(v) for (k, v) in out)
open(io -> JSON.print(io, out, 1), joinpath(OUT_DIR, TAG * ".json"), "w")
println("RESULT ", JSON.json(out))

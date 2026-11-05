"""
run_partial_int.jl  —  one run of TODO #2: periods 1..T of one day of the integer-split instance at the baseline price,
exact tier model (stable matching) with the integrality kept on some variables only (int_flow/partial_int.jl).

    julia --project=model_2stage/env model_2stage/int_flow/run_partial_int.jl --day 18 --T 100 --form xbin [opts]

--form full | xbin | x | lp     (all integer / x and binaries / x only / LP)
--instance (default int_flow/instances/paris_n30_H10_int_workers.json), --tl 7200 (s), --threads 8,
--fixedx_tl 1800 (s; form x: the exact model with the x of the relaxed optimum fixed).
Gurobi, MIPGap 0, IntFeasTol = FeasibilityTol = 1e-9.  Writes results/<array job id or job id>/<day>_T<T>_<form>.json
and the Gurobi log next to it.
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "partial_int.jl"))

using Gurobi, Printf

_sarg(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]
const INSTANCE  = _sarg("--instance", joinpath(@__DIR__, "instances", "paris_n30_H10_int_workers.json"))
const DAY       = parse(Int, _sarg("--day", "18"))
const TT        = parse(Int, _sarg("--T", "10"))
const FORM      = Symbol(_sarg("--form", "full"))
const TL        = parse(Float64, _sarg("--tl", "7200"))
const THREADS   = parse(Int, _sarg("--threads", "8"))
const FIXEDX_TL = parse(Float64, _sarg("--fixedx_tl", "1800"))
const JOB       = get(ENV, "SLURM_ARRAY_JOB_ID", get(ENV, "SLURM_JOB_ID", "local"))
const OUT_DIR   = joinpath(@__DIR__, "results", JOB)
const TAG       = @sprintf("d%02d_T%03d_%s", DAY, TT, FORM)
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
_clean(x::NamedTuple) = Dict(string(k) => _clean(v) for (k, v) in pairs(x))
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

m = Model(opt())
tb = @elapsed begin
    b = add_scenario_sparse!(m, w, w.scen[1], P, P; stability = :tier)
    if FORM !== :lp
        w.vterm > 0 ? continue_start!(w, b, 1, initial_state(w, b)) : nothing_start!(w, b)
    end
    @objective(m, Max, b.profit)
    relaxed = relax_ints!(b, FORM)
end
v0 = FORM === :lp ? NaN : value(v -> something(start_value(v), 0.0), b.profit)
sz = _sizes(m)
@printf("built in %.1f s: %d vars (%d int, %d bin), %d rows; start %.2f\n", tb, sz.nvar, sz.nint, sz.nbin, sz.ncon, v0)
flush(stdout)

optimize!(m)
st = JuMP.termination_status(m)
feas = primal_status(m) == MOI.FEASIBLE_POINT
out = Dict{String,Any}("tag" => TAG, "job" => JOB, "instance" => basename(INSTANCE), "day" => DAY, "T" => TT,
    "form" => string(FORM), "threads" => THREADS, "tl" => TL, "price" => P[1, 1], "build_s" => tb, "start" => v0,
    "status" => string(st), "optimal" => st == MOI.OPTIMAL, "value" => feas ? objective_value(m) : NaN,
    "bound" => attr(m, MOI.ObjectiveBound()), "gap" => attr(m, MOI.RelativeGap()), "solve_s" => solve_time(m),
    "nodes" => attr(m, MOI.NodeCount(), -1), "day_info" => _clean(day), "sizes" => _clean(sz))
if feas
    out["frac"] = Dict("x" => nfrac(value.(x_vars(b))), "bin" => nfrac(value.(bin_vars(b))),
                       "int" => nfrac(value.(int_vars(b))))
    out["ops"] = _clean(op_stats_sparse(w, b, P))
end
if feas && FORM === :xbin                   # integral trip flows at the same value
    cl = cleanup!(m, b)
    out["cleanup"] = _clean(cl)
end
if feas && FORM in (:full, :xbin) && (FORM === :full || out["cleanup"]["frac_after"] == 0)
    out["residual"] = conservation_residual(w, b)
    out["blocking_pairs"] = length(blocking_pairs_sparse(w, b, P))
end
if feas && FORM === :x                      # is the x of the relaxed optimum feasible for the exact model, at what value
    xs, xv = x_vars(b), round.(value.(x_vars(b)))
    restore_ints!(relaxed)
    foreach((v, a) -> fix(v, a; force = true), xs, xv)
    set_attribute(m, "TimeLimit", FIXEDX_TL)
    optimize!(m)
    st2 = JuMP.termination_status(m)
    out["fixedx"] = Dict("status" => string(st2), "solve_s" => solve_time(m),
        "value" => primal_status(m) == MOI.FEASIBLE_POINT ? objective_value(m) : nothing,
        "bound" => _clean(attr(m, MOI.ObjectiveBound())))
end
out = Dict(k => _clean(v) for (k, v) in out)
open(io -> JSON.print(io, out, 1), joinpath(OUT_DIR, TAG * ".json"), "w")
println("RESULT ", JSON.json(out))

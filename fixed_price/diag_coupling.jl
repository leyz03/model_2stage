"""
diag_coupling.jl — how much state crosses each period boundary in the exact (tier) optimum (LAGRANGIAN_TD.md §3).
Per scenario: rides and tasks per period, and at every boundary t = 2..T the bikes still on the road (ΣPin/Γ),
workers on tasks (ΣGin) and posted-but-unmatched tasks (ΣMrem), as shares of the fleet / workforce.
Writes results/<job>/coupling.jsonl.

    julia --project=model_2stage/env model_2stage/fixed_price/diag_coupling.jl --n 5 --T 36 --scen 1-3 \\
          [--tag _m6 --instdir model_2stage/fixed_price/instances] [--tl 300]
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : ARGS[i+1]
const NN = parse(Int, _arg("--n", "5"))
const TT = parse(Int, _arg("--T", "36"))
const SCENS = let spec = _arg("--scen", "1-3")
    parse(Int, split(spec, '-')[1]):parse(Int, split(spec, '-')[end])
end
const PRICE = _arg("--price", "u4")
const TL = parse(Float64, _arg("--tl", "300"))
const INSTDIR = _arg("--instdir", joinpath(@__DIR__, "..", "scaling", "instances"))
const TAG = _arg("--tag", "")
const JOB = get(ENV, "SLURM_JOB_ID", "local")
const OUT = joinpath(@__DIR__, "..", "results", JOB, "coupling.jsonl")
mkpath(dirname(OUT))

using Gurobi
const GENV = Gurobi.Env()
opt() = optimizer_with_attributes(() -> Gurobi.Optimizer(GENV), "OutputFlag" => 0, "MIPGap" => 0.0,
    "Threads" => 1, "TimeLimit" => TL, "IntFeasTol" => 1e-9, "FeasibilityTol" => 1e-9)

P = fill(parse(Float64, PRICE[2:end]), NN, NN)
inst = load_ts_instance(joinpath(INSTDIR, "syn_n$(NN)_T$(TT)$(TAG).json"), "syn"; levels = 3, nscen = maximum(SCENS))
for s in SCENS
    v, m, b, info = solve_tier(inst, s, P, opt())
    info.feasible || continue
    Γ, T = inst.Γ, inst.T
    road = [sum(value.(b.Pin[:, :, :, t])) / Γ for t in 2:T]
    busy = [sum(value.(b.Gin[:, :, t])) for t in 2:T]
    open_ = [sum(value.(b.Mrem[:, :, t])) for t in 2:T]
    rides = [sum(value.(b.Yod[:, :, :, t])) for t in 1:T]
    tasks = [sum(value.(b.mtil[:, :, t])) for t in 1:T]
    row = Dict("job" => JOB, "n" => NN, "T" => TT, "tag" => TAG, "scenario" => s, "price" => PRICE,
               "value" => v, "optimal" => info.optimal, "fleet" => inst.Q3, "workers" => inst.Q1,
               "rides" => sum(rides), "tasks" => sum(tasks), "rides_per_bike" => sum(rides) / inst.Q3,
               "road" => road, "busy" => busy, "open" => open_,
               "mean_road_share" => sum(road) / length(road) / inst.Q3,
               "mean_busy_share" => sum(busy) / length(busy) / inst.Q1,
               "mean_open" => sum(open_) / length(open_))
    open(io -> (JSON.print(io, row); println(io)), OUT, "a")
    @printf("n=%d T=%d s=%d  value %.0f  rides %.0f (%.2f per bike)  tasks %.0f  on road %.1f%% of fleet  busy %.1f%% of workers  open %.2f\n",
            NN, TT, s, v, sum(rides), sum(rides) / inst.Q3, sum(tasks), 100 * row["mean_road_share"],
            100 * row["mean_busy_share"], row["mean_open"]); flush(stdout)
end
println("done")

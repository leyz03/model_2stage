"""
fixed_price.jl  —  per-scenario solvers at a fixed price (model_2stage/fixed_price/README.md)

With p fixed the two-stage SAA separates into K_s independent T-period MILPs, so everything
here works on one scenario block built by `add_scenario!`.

  solve_r1          the no-stability relaxation R1 (individual rationality kept), returns the block
  solve_tier        exact stable matching with one binary per worker node, positive utility tier, period
  blocking_pairs    stability check of a solved block: every (i, jk, t) that blocks, and why
  solve_verify      exact value by row generation: drop (S1)/(S6) of the u > 0 pairs, solve, add the
                    rows of the blocking pairs of the optimum, re-solve until the optimum is stable
  relax_and_fix     forward time decomposition: periods in the window integer, later periods LP
                    (or out of the objective), fix the first `step` periods of the window, move on
  solve_warm        exact :pre MILP with a MIP start (the relax-and-fix solution)
  op_stats          what a solved block does (tasks, idle workers, open jobs left, lost riders)
"""

include(joinpath(@__DIR__, "two_stage_bl.jl"))

"""Integer decision variables of period `t` of a block (the last index of every container is t).
A dense block (`add_scenario!`) has the stability binaries δ, η, ζ; a sparse one (`add_scenario_sparse!`) its tier
binaries per period in `zt`."""
function period_ints(b, t)
    vs = VariableRef[]
    for f in (:Ycum, :Yod, :YU, :L, :I, :rA, :β, :mhat, :Mt, :mtil, :x, :δ, :η, :ζ)
        haskey(b, f) && getfield(b, f) !== nothing || continue
        c = getfield(b, f)
        nd = ndims(c)
        for v in c[ntuple(k -> k == nd ? t : Colon(), nd)...]
            push!(vs, v)
        end
    end
    haskey(b, :zt) && append!(vs, b.zt[t])
    return vs
end

"""Incoming-state variables of period `t` of a block (A, U, W, Mrem, Pin, Gin)."""
state_vars(b, t) = VariableRef[vec(b.A[:, :, t]); vec(b.U[:, t]); vec(b.W[:, t]); vec(b.Mrem[:, :, t]);
                               vec(b.Pin[:, :, :, t]); vec(b.Gin[:, :, t])]

"""
Tie-breaking term for a fixed integer price: ε (Σ u_ijk x_ijk - Σ (M_jk - m̃_jk)), i.e. among
equally profitable solutions prefer the ones where workers get high-utility tasks (which also
puts closer workers first, u contains -d_ij) and no posted job is left open.  |term| <= 0.4, so
with integer R, C_p and p (integer profits) it never changes the optimal profit.  `periods`
restricts it (relax-and-fix).  Returns `nothing` when the data are not integer.
"""
function tiebreak_term(inst::TSInstance, b, p::AbstractMatrix; periods = 1:inst.T)
    all(isinteger, inst.R) && isinteger(inst.Cp) && all(isinteger, p) || return nothing
    n, d, c = inst.n, inst.d, inst.c
    u = [p[j, k] - d[i, j] - c[j, k] for i in 1:n, j in 1:n, k in 1:n]
    ε = 0.4 / (inst.T * (inst.Q1 * max(1.0, maximum(u)) + inst.Q3))
    pos = haskey(b, :trips) ?                          # sparse block: x[e, t] over the allowed triples
        sum((u[i, j, k] * b.x[e, t] for (e, (i, j, k)) in enumerate(b.trips), t in periods if u[i, j, k] > 0);
            init = AffExpr(0.0)) :
        sum((u[i, j, k] * b.x[i, j, k, t] for i in 1:n, j in 1:n, k in 1:n, t in periods if u[i, j, k] > 0);
            init = AffExpr(0.0))
    open_left = sum(b.Mt[j, k, t] - b.mtil[j, k, t] for j in 1:n, k in 1:n, t in periods)
    return ε * (pos - open_left)
end

_profit_objective(inst, b, p, tiebreak; periods = 1:inst.T, horizon = inst.T) =
    (obj = sum(b.pprofit[t] for t in 1:horizon);
     # the end-of-day battery penalty (sparse, inst.vterm > 0) belongs to the whole-day horizon only
     horizon == inst.T && hasproperty(b, :term) && b.term !== nothing && (obj = obj - inst.vterm * b.term.δ);
     tb = tiebreak ? tiebreak_term(inst, b, p; periods) : nothing;
     tb === nothing ? obj : obj + tb)

"""R1 of scenario `s` at the fixed price `p`: no stable matching, workers still refuse u < 0.
`tiebreak = true` adds `tiebreak_term`; the returned value is always the profit."""
function solve_r1(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; deadline::Float64 = Inf,
                  tiebreak::Bool = false)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    b = add_scenario!(m, inst, inst.scen[s], (wage = p, ir = p))
    @objective(m, Max, _profit_objective(inst, b, p, tiebreak))
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("R1, scenario $s: $st")
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    bound = tiebreak ? floor(objective_bound(m) + 0.4 + 1e-9) : objective_bound(m)
    return (feas ? value(b.profit) : -Inf), m, b,
           (solve_s = solve_time(m), bound = bound, optimal = st == MOI.OPTIMAL, feasible = feas,
            nbin = count_free_binaries(m))
end

"""
Add an exact fixed-price stable-matching formulation to an R1 block.  At utility tier u > 0,
B = W_i - sum(x_ijk : u_ijk >= u) counts idle workers and workers on worse jobs.  Stability
requires the jk pool to be full of workers at distance <= d_ij whenever B > 0.  A single
binary per (i, distinct positive utility, t) expresses this implication.  Negative-utility
assignments are already forbidden by the R1 block; zero utility cannot form a blocking pair.
"""
function add_tier_stability!(m::Model, inst::TSInstance, b, p::AbstractMatrix)
    n, T, d, c = inst.n, inst.T, inst.d, inst.c
    u = [p[j, k] - d[i, j] - c[j, k] for i in 1:n, j in 1:n, k in 1:n]
    levels = [sort(unique(u[i, j, k] for j in 1:n, k in 1:n if u[i, j, k] > 0)) for i in 1:n]
    high_jobs = [[[(j, k) for j in 1:n, k in 1:n if u[i, j, k] >= v]
                  for v in levels[i]] for i in 1:n]
    level_index = [Dict(v => h for (h, v) in enumerate(levels[i])) for i in 1:n]
    near = Dict((j, dist) => [i2 for i2 in 1:n if d[i2, j] <= dist]
                for j in 1:n for dist in unique(d[:, j]))
    tier = Dict{NTuple{3,Int},VariableRef}()
    for t in 1:T, i in 1:n, h in eachindex(levels[i])
        z = @variable(m, binary = true, base_name = "tier_$(i)_$(h)_$(t)")
        tier[(i, h, t)] = z
        @constraint(m, b.W[i, t] - sum(b.x[i, j, k, t] for (j, k) in high_jobs[i][h]) <= inst.Q1 * z)
    end
    for t in 1:T, i in 1:n, j in 1:n, k in 1:n
        u[i, j, k] > 0 || continue
        z = tier[(i, level_index[i][u[i, j, k]], t)]
        H = sum(b.x[i2, j, k, t] for i2 in near[(j, d[i, j])])
        @constraint(m, b.Mt[j, k, t] - H <= inst.Q3 * (1 - z))
    end
    return tier
end

"""Exact single-scenario solver using utility-tier stable matching; returns the unperturbed profit."""
function solve_tier(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; deadline::Float64 = Inf)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    b = add_scenario!(m, inst, inst.scen[s], (wage = p, ir = p))
    tier = add_tier_stability!(m, inst, b, p)
    @objective(m, Max, b.profit)
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("tier, scenario $s: $st")
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    return (feas ? objective_value(m) : -Inf), m, b,
           (solve_s = solve_time(m), bound = objective_bound(m), optimal = st == MOI.OPTIMAL,
            feasible = feas, nbin = count_free_binaries(m), ntier = length(tier))
end

"""Exact :pre MILP of scenario `s` with the valid cut profit <= ub (ub = the R1 value): B&B stops as
soon as it finds a stable solution worth ub."""
function solve_exact_ub(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer, ub::Float64; deadline::Float64 = Inf)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    b = add_scenario!(m, inst, inst.scen[s], p; stability = :pre)
    @objective(m, Max, b.profit)
    @constraint(m, b.profit <= ub)
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("exact with R1 cut, scenario $s: $st")
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    return (feas ? objective_value(m) : -Inf), m, b,
           (solve_s = solve_time(m), bound = objective_bound(m), optimal = st == MOI.OPTIMAL)
end

"""
    blocking_pairs(inst, b, p; periods = 1:T) -> Vector of NamedTuples

Blocking pairs of the solved block `b` at price `p`, by the model's own definition (S1)-(S6):
(i, jk, t) blocks iff node i has workers, u_ijk = p_jk - d_ij - c_jk exceeds the utility its
workers settle for (0 if one of them is idle, else the worst task they took), and the pool M_jk
is not filled by workers at distance <= d_ij.  No blocking pair <=> the solution is feasible
for the exact model with the same x, M, W (so R1 = R0 whenever R1's optimum has none).
`worker` = :idle / :worse and `job` = :open (pool not filled at all) / :far (filled, partly by
workers farther than d_ij) say which of the two sides fails.
"""
function blocking_pairs(inst::TSInstance, b, p::AbstractMatrix; periods = 1:inst.T)
    n, d, c = inst.n, inst.d, inst.c
    xv = round.(Int, value.(b.x))
    Wv = round.(Int, value.(b.W))
    Mv = round.(Int, value.(b.Mt))
    out = NamedTuple[]
    for t in periods, i in 1:n
        Wv[i, t] == 0 && continue
        idle = Wv[i, t] > sum(xv[i, :, :, t])
        worst = idle ? 0.0 : minimum(p[j, k] - d[i, j] - c[j, k] for j in 1:n, k in 1:n if xv[i, j, k, t] > 0)
        for j in 1:n, k in 1:n
            u = p[j, k] - d[i, j] - c[j, k]
            u > worst + 1e-9 || continue
            H = sum(xv[i2, j, k, t] for i2 in 1:n if d[i2, j] <= d[i, j])
            H < Mv[j, k, t] || continue
            push!(out, (i = i, j = j, k = k, t = t, u = u, worker = idle ? :idle : :worse,
                        job = sum(xv[:, j, k, t]) < Mv[j, k, t] ? :open : :far))
        end
    end
    return out
end

"""(S1)/(S6) rows of one u > 0 pair, as `add_scenario!` writes them for `:pre`."""
function add_stability_rows!(m::Model, inst::TSInstance, b, i, j, k, t)
    u, Q3 = b.u[i, j, k], inst.Q3
    H = sum(b.x[i2, j, k, t] for i2 in 1:inst.n if inst.d[i2, j] <= inst.d[i, j])
    v, δ, Mt, s = b.v[i, j, k, t], b.δ[i, j, k, t], b.Mt[j, k, t], b.s[i, t]
    @constraint(m, v <= H)
    @constraint(m, v <= Mt)
    @constraint(m, v <= Q3 * δ)
    @constraint(m, v >= Mt - Q3 * (1 - δ))
    @constraint(m, s >= u - u * δ)
end

"""
Re-solve `m` until its optimum has no blocking pair in `periods`, adding the (S1)/(S6) rows of
every blocking pair found (the block must come from `stability = :lazy`).  Returns
(ok, optimal, rounds, rows, solver_s): ok = the last solution is stable (a feasible solution of
the exact model), optimal = it is also optimal.  A time limit with an unstable incumbent gives up.
"""
function verify_loop!(m::Model, inst::TSInstance, b, p; periods = 1:inst.T, deadline::Float64 = Inf,
                      added = Set{NTuple{4,Int}}(), win_tl::Float64 = Inf)
    rounds, rows, solver = 0, 0, 0.0
    while true
        set_time_limit_sec(m, max(1.0, min(win_tl, deadline - time())))
        optimize!(m)
        rounds += 1
        solver += solve_time(m)
        opt = JuMP.termination_status(m) == MOI.OPTIMAL
        primal_status(m) == MOI.FEASIBLE_POINT || return (ok = false, optimal = false, rounds, rows, solver_s = solver)
        bp = blocking_pairs(inst, b, p; periods)
        isempty(bp) && return (ok = true, optimal = opt, rounds, rows, solver_s = solver)
        opt || return (ok = false, optimal = false, rounds, rows, solver_s = solver)
        for q in bp
            key = (q.i, q.j, q.k, q.t)
            key in added && error("pair $key blocks although its rows are in the model")
            add_stability_rows!(m, inst, b, q.i, q.j, q.k, q.t)
            push!(added, key)
            rows += 1
        end
    end
end

"""
Exact value of scenario `s` by row generation (no callback, any MILP solver): start from the
`:lazy` formulation without lazy rows (= R1 plus the worker-side rows), add the rows of the
blocking pairs of each optimum and re-solve.  Terminates because every round adds at least
one of the finitely many pairs; the final optimum is stable, hence optimal for R0.
"""
function solve_verify(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; deadline::Float64 = Inf,
                      tiebreak::Bool = false)
    m = Model(optimizer)
    b = add_scenario!(m, inst, inst.scen[s], p; stability = :lazy)
    @objective(m, Max, _profit_objective(inst, b, p, tiebreak))
    r = verify_loop!(m, inst, b, p; deadline)
    val = r.ok ? value(b.profit) : -Inf
    return val, m, b, (solve_s = r.solver_s, rounds = r.rounds, rows = r.rows, optimal = r.optimal)
end

function _relax!(v::VariableRef, kind::Symbol)
    if kind === :bin
        unset_binary(v)
        set_lower_bound(v, 0.0)
        set_upper_bound(v, 1.0)
    else
        unset_integer(v)
    end
end

_restore!(v::VariableRef, kind::Symbol) = kind === :bin ? set_binary(v) : set_integer(v)

"""
    relax_and_fix(inst, s, p, optimizer; w = 3, step = 1, look = :lp, stability = :pre,
                  win_tl = 60.0, deadline = Inf, form = :dense)

Forward relax-and-fix on one scenario.  `form = :sparse` builds the block with `add_scenario_sparse!`
(src/sparse_scenario.jl must be loaded): worker assignments only where individual rationality allows them,
`stability` :tier (utility tiers, the default there) or :none (R1).  `form = :dense` (default) is `add_scenario!`
with `stability` :pre (default), :full, :lazy rows left out, or :verify.
`start = true` (sparse only) gives every window a MIP start: the fixed periods at their values and the rest of the
day continued from the window's incoming state by `continue_start!` (post no new task), so no window is left
without an incumbent (on the Paris instances Gurobi finds none by itself at n = 50).  Window [t0, t0+w-1] keeps its integer variables;
periods before it are fixed, periods after it are LP.  `look` sets the objective:
  :lp    whole horizon (the LP tail values the outgoing state)
  :none  window periods only (myopic rolling horizon)
  h::Int window plus h LP periods
After each window solve the first `step` periods are fixed (all of them in the last window).
`stability = :verify` builds the :lazy formulation and runs the row-generation loop inside each
window (only the window's periods are checked; earlier periods were checked when they were in it).
`tiebreak = true` adds `tiebreak_term` over the window's periods.
`post_penalty` > 0 subtracts post_penalty·Σ m̂ over the window's periods (the posting penalty of `td_forward`:
a posted task costs nothing inside the window but locks a bike for the periods after it).
The result is a feasible solution of the exact model, so its value is a lower bound on Q(p, ξ).
Returns (value, model, block, stats); value = -Inf when a window found no solution in win_tl.
"""
function relax_and_fix(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; w::Int = 3, step::Int = 1,
                       look = :lp, stability = nothing, win_tl::Float64 = 60.0, deadline::Float64 = Inf,
                       tiebreak::Bool = false, post_penalty::Float64 = 0.0, form::Symbol = :dense,
                       start::Bool = false)
    1 <= step <= w || error("need 1 <= step <= w")
    form in (:dense, :sparse) || error("form must be :dense or :sparse")
    start && form !== :sparse && error("start = true needs form = :sparse")
    stability = something(stability, form === :sparse ? :tier : :pre)
    T = inst.T
    m = Model(optimizer)
    if form === :sparse
        isdefined(@__MODULE__, :add_scenario_sparse!) || error("form = :sparse needs src/sparse_scenario.jl")
        stability in (:tier, :none) || error("form = :sparse supports stability :tier or :none")
        b = add_scenario_sparse!(m, inst, inst.scen[s], p, p; stability)
    else
        b = add_scenario!(m, inst, inst.scen[s], p; stability = stability === :verify ? :lazy : stability)
    end
    ints = [period_ints(b, t) for t in 1:T]
    kind = Dict{VariableRef,Symbol}()
    for t in 1:T, v in ints[t]
        is_fixed(v) && continue
        kind[v] = is_binary(v) ? :bin : :int
        _relax!(v, kind[v])
    end
    added = Set{NTuple{4,Int}}()
    st = start ? initial_state(inst, b) : nothing    # incoming state of the current window (start = true)
    t0, windows, rounds, rows, solver, worst_win = 1, 0, 0, 0, 0.0, 0.0
    while t0 <= T
        t1 = min(T, t0 + w - 1)
        for t in t0:t1, v in ints[t]
            haskey(kind, v) && !is_fixed(v) && !(is_binary(v) || is_integer(v)) && _restore!(v, kind[v])
        end
        horizon = look === :lp ? T : look === :none ? t1 : min(T, t1 + Int(look))
        obj = _profit_objective(inst, b, p, tiebreak; periods = t0:t1, horizon)
        post_penalty > 0 && (obj = obj - post_penalty * sum(b.mhat[:, :, t0:t1]))
        @objective(m, Max, obj)
        start && continue_start!(inst, b, t0, st)
        if stability === :verify
            r = verify_loop!(m, inst, b, p; periods = t0:t1, deadline, added, win_tl)
            ok, rounds, rows, ws = r.ok, rounds + r.rounds, rows + r.rows, r.solver_s
        else
            set_time_limit_sec(m, max(1.0, min(win_tl, deadline - time())))
            optimize!(m)
            rounds += 1
            ws = solve_time(m)
            ok = primal_status(m) == MOI.FEASIBLE_POINT
        end
        solver += ws
        worst_win = max(worst_win, ws)
        windows += 1
        ok || return -Inf, m, b, (solve_s = solver, windows, rounds, rows, worst_win_s = worst_win, ok = false,
                                  failed_at = t0)
        tfix = t1 == T ? T : t0 + step - 1
        vals = [(v, round(value(v))) for t in t0:tfix for v in ints[t] if haskey(kind, v)]   # read all, then modify
        if start           # the fixed periods keep their values as the start of the later windows
            keep = [(v, value(v)) for t in t0:tfix for v in state_vars(b, t)]
            st = tfix < T ? read_state(inst, b, tfix + 1) : nothing
        end
        for (v, val) in vals
            fix(v, val; force = true)
        end
        if start
            for (v, val) in vals
                set_start_value(v, val)
            end
            for (v, val) in keep
                set_start_value(v, val)
            end
        end
        t0 = tfix + 1
    end
    @objective(m, Max, b.profit)              # every integer is fixed: an LP that only evaluates
    set_time_limit_sec(m, max(60.0, win_tl))
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("relax-and-fix, scenario $s: final evaluation $(JuMP.termination_status(m))")
    return objective_value(m), m, b, (solve_s = solver, windows, rounds, rows, worst_win_s = worst_win, ok = true,
                                      failed_at = 0)
end

"""Exact :pre MILP of scenario `s` started from `start` (values of all_variables of a model built the same way)."""
function solve_warm(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer, start::Vector{Float64};
                    deadline::Float64 = Inf)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    b = add_scenario!(m, inst, inst.scen[s], p; stability = :pre)
    @objective(m, Max, b.profit)
    vars = all_variables(m)
    length(vars) == length(start) || error("start vector does not match the model")
    set_start_value.(vars, start)
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("warm start, scenario $s: $st")
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    return (feas ? objective_value(m) : -Inf), m, b,
           (solve_s = solve_time(m), bound = objective_bound(m), optimal = st == MOI.OPTIMAL)
end

"""LP relaxation of the exact (:pre) scenario MILP (an upper bound)."""
function solve_lp(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer)
    m = Model(optimizer)
    b = add_scenario!(m, inst, inst.scen[s], p; stability = :pre)
    @objective(m, Max, b.profit)
    relax_integrality(m)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("LP, scenario $s: $(JuMP.termination_status(m))")
    return objective_value(m), solve_time(m)
end

"""What a solved block does over the day."""
function op_stats(inst::TSInstance, b)
    n, T = inst.n, inst.T
    x, mt, M, W, L = value.(b.x), value.(b.mtil), value.(b.Mt), value.(b.W), value.(b.L)
    sw = sum(mt[j, j, t] for j in 1:n, t in 1:T)
    return (moves = sum(mt) - sw, swaps = sw, lost = sum(L),
            idle = sum(W[i, t] - sum(x[i, :, :, t]) for i in 1:n, t in 1:T),
            open_left = sum(M .- mt), wage_tasks = sum(mt))
end

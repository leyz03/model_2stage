"""
lagrangian_td.jl  —  Lagrangian time decomposition of one fixed-price scenario
(model_2stage/fixed_price/LAGRANGIAN_TD.md)

The day is cut into blocks of a few periods.  Every block keeps the full physics and the
utility-tier stable matching (`add_tier_stability!`).  Block b > 1 starts from its own integer
copy z_b of the incoming state; y_b is the state after block b.  The copy equalities
z_{b+1} = y_b are dualised with multipliers λ_b, so for every λ

    L(λ) = Σ_b max { profit_b + λ_b·y_b − λ_{b−1}·z_b }  ≥  Q(p, ξ),

and the blocks are independent MILPs.  A time-limited block contributes its proven bound,
never its incumbent.

  td_cuts / td_cuts_adaptive   block boundaries: balanced lengths, or low in-transit + open-task periods
  state_keys                   boundary state components with valid bounds (zero = cannot be nonzero)
  add_td_block!                one block, in its own model or inside a joint model
  td_joint, td_lp_multipliers  the linked block model; λ from its LP relaxation
  lagrangian_td                proximal bundle on L(λ), one cutting-plane model per block,
                               prox weights scaled per state class (bikes / workers / pools)
  td_forward                   primal heuristic: blocks in order, incoming state fixed, outgoing priced by λ
  td_verify                    re-evaluates a forward solution in the whole-day model

Every function that builds blocks takes `form = :dense` (`add_scenario!` + `add_tier_stability!`, the original) or
`form = :sparse` (`add_scenario_sparse!` with tier rows: worker assignments only where individual rationality allows
them, the formulation the real-data instances need).  Both give the same block MILPs and LPs; the state keys and
hence the multiplier vector are the same.
"""

include(joinpath(@__DIR__, "sparse_scenario.jl"))

using Printf

const StateKey = NamedTuple{(:f, :idx, :ub, :cls, :scale),Tuple{Symbol,Tuple,Float64,Symbol,Float64}}

"""Balanced blocks: ceil(T / w) blocks whose lengths differ by at most one (longer ones first)."""
function td_cuts(T::Int, w::Int)
    B = cld(T, w)
    lens = [T ÷ B + (k <= T % B ? 1 : 0) for k in 1:B]
    ends = cumsum(lens)
    return [(e - l + 1, e) for (l, e) in zip(lens, ends)]
end

"""
B blocks with lengths in wmin:wmax whose boundaries (the incoming periods 2..T of blocks 2..B)
minimise Σ load[t]; dynamic programme over the boundary positions.
"""
function td_cuts_adaptive(load::AbstractVector, T::Int, B::Int; wmin::Int = 3, wmax::Int = 6)
    B * wmin <= T <= B * wmax || error("no $B blocks of length $wmin:$wmax cover T = $T")
    best = fill(Inf, B + 1, T + 1)          # best[k+1, e+1]: k blocks covering periods 1..e
    arg = zeros(Int, B + 1, T + 1)
    best[1, 1] = 0.0
    for k in 1:B, e in 1:T, l in wmin:wmax
        e - l >= 0 || continue
        prev = best[k, e-l+1]
        isfinite(prev) || continue
        c = prev + (e < T ? load[e+1] : 0.0)  # boundary = first period of the next block
        if c < best[k+1, e+1]
            best[k+1, e+1], arg[k+1, e+1] = c, l
        end
    end
    isfinite(best[B+1, T+1]) || error("adaptive cuts infeasible")
    cuts, e = Tuple{Int,Int}[], T
    for k in B:-1:1
        l = arg[k+1, e+1]
        pushfirst!(cuts, (e - l + 1, e))
        e -= l
    end
    return cuts
end

"""Instance with only periods ta:tb of scenario s (local periods 1..tb-ta+1, one scenario, its morning state and
fleet: `scenario_view`)."""
function window_instance(inst::TSInstance, s::Int, ta::Int, tb::Int)
    v = scenario_view(inst, s)
    TSInstance(inst.name, inst.n, tb - ta + 1, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
               inst.Cmax, inst.Hmax, inst.Q1, v.Q3, v.A0, v.U0, inst.W0, inst.menu,
               inst.structure, inst.group, inst.ngroups, [v.scen[1][ta:tb]], tb == inst.T ? inst.vterm : 0.0,
               isempty(v.scenU) ? v.scenU : [v.scenU[1][ta:tb]], v.A0s, v.U0s)
end

"""
Components of the incoming state of global period t >= 2, as (f, idx, ub, cls, scale):
`f` the block field (A, U, W, Mrem, Pin, Gin), `ub` a bound implied by the dynamics,
`cls` :bike / :worker / :pool, `scale` the units per bike/worker/task (A, U, Pin are Γ-scaled).
Pin is bounded by the riders that can still be on the way (Σ_ℓ Yod ≤ ξ, q ≤ max_ℓ q), Gin is
zero unless some admissible task (u ≥ 0) of the right duration started in an earlier period.
"""
function state_keys(inst::TSInstance, s::Int, p::AbstractMatrix, t::Int)
    n, K, Γ, d, c, q, ξ = inst.n, inst.K, inst.Γ, inst.d, inst.c, inst.q, inst.scen[s]
    keys = StateKey[]
    bikes, workers = Float64(Γ * inst.Q3), Float64(inst.Q1)
    for ℓ in 1:K-1, i in 1:n
        push!(keys, (f = :A, idx = (ℓ, i), ub = bikes, cls = :bike, scale = Float64(Γ)))
    end
    for i in 1:n
        push!(keys, (f = :U, idx = (i,), ub = bikes, cls = :bike, scale = Float64(Γ)))
    end
    for i in 1:n
        push!(keys, (f = :W, idx = (i,), ub = workers, cls = :worker, scale = 1.0))
    end
    for j in 1:n, k in 1:n
        push!(keys, (f = :Mrem, idx = (j, k), ub = Float64(inst.Q3), cls = :pool, scale = 1.0))
    end
    for ℓ2 in 1:K, j in 1:n, r in 2:max(inst.Cmax, 2)
        ub = 0
        for m in 0:inst.Cmax, i in 1:n          # a ride of length r+m that started in t-1-m
            t - 1 - m >= 1 && c[i, j] == r + m || continue
            ub += qmax(inst, ℓ2, i, j) * ξ[t-1-m][i, j]
        end
        push!(keys, (f = :Pin, idx = (ℓ2, j, r), ub = Float64(min(ub, bikes)), cls = :bike, scale = Float64(Γ)))
    end
    for k in 1:n, r in 2:max(inst.Hmax, 2)
        ok = any(t - 1 - m >= 1 && d[i, j] + c[j, k] == r + m && p[j, k] - d[i, j] - c[j, k] >= 0
                 for m in 0:inst.Hmax, i in 1:n, j in 1:n)
        push!(keys, (f = :Gin, idx = (k, r), ub = ok ? workers : 0.0, cls = :worker, scale = 1.0))
    end
    return keys
end

statevar(b, key, tl) = getfield(b, key.f)[key.idx..., tl]

"""Whether block `b` has a variable for `key`: a sparse block's worker pipeline only has the allowed task durations
(state_keys bounds every longer one by 0)."""
hasstate(b, key) = key.f !== :Gin || key.idx[2] in axes(b.Gin, 2)

"""
Block ta:tb of scenario s at price p, added to model `m` (objective not set).  Block 1 starts from
the morning state; a later block starts from integer copies z (bounded by `state_keys`, bikes and
workers conserved).  Every block but the last also carries its outgoing state y.  Returns
(b, sub, z, y, zkeys, ykeys, tier) with z, y aligned with zkeys, ykeys (the bound > 0 components).
"""
function add_td_block!(m::Model, inst::TSInstance, s::Int, p::AbstractMatrix, ta::Int, tb::Int; tag = "",
                       form::Symbol = :dense)
    sub = window_instance(inst, s, ta, tb)
    first, last = ta == 1, tb == inst.T
    if form === :sparse
        b = add_scenario_sparse!(m, sub, sub.scen[1], p, p; stability = :tier, tag, init = first ? :fixed : :free,
                                 carry = !last)
        tier = b.z
    elseif form === :dense
        b = add_scenario!(m, sub, sub.scen[1], (wage = p, ir = p); tag, init = first ? :fixed : :free,
                          carry = !last)
        tier = add_tier_stability!(m, sub, b, p)
    else
        error("form must be :dense or :sparse")
    end
    z, zkeys = VariableRef[], StateKey[]
    if !first
        for key in state_keys(inst, s, p, ta)
            if !hasstate(b, key)
                key.ub == 0 || error("state $key has a bound but no variable in the sparse block")
                continue
            end
            v = statevar(b, key, 1)
            if key.ub == 0
                fix(v, 0; force = true)
            else
                set_integer(v)
                set_upper_bound(v, key.ub)
                push!(z, v); push!(zkeys, key)
            end
        end
        Γ = inst.Γ
        @constraint(m, sum(b.A[:, :, 1]) + sum(b.U[:, 1]) + sum(b.Pin[:, :, :, 1]) +
                       Γ * sum(b.Mrem[:, :, 1]) + Γ * sum(b.Gin[:, :, 1]) == Γ * sub.Q3)   # the scenario's fleet
        @constraint(m, sum(b.W[:, 1]) + sum(b.Gin[:, :, 1]) == inst.Q1)
    end
    y, ykeys = VariableRef[], StateKey[]
    if !last
        for key in state_keys(inst, s, p, tb + 1)
            key.ub == 0 && continue
            push!(y, statevar(b, key, sub.T + 1)); push!(ykeys, key)
        end
    end
    return (b = b, sub = sub, z = z, y = y, zkeys = zkeys, ykeys = ykeys, tier = tier, ta = ta, tb = tb)
end

"""All blocks in one model with the copy equalities z_{b+1} = y_b (= the whole-day model)."""
function td_joint(inst::TSInstance, s::Int, p::AbstractMatrix, cuts, optimizer; form::Symbol = :dense)
    m = Model(optimizer)
    blks = [add_td_block!(m, inst, s, p, ta, tb; tag = "_b$k", form) for (k, (ta, tb)) in enumerate(cuts)]
    link = [@constraint(m, blks[k+1].z .- blks[k].y .== 0) for k in 1:length(cuts)-1]
    @objective(m, Max, sum(blk.b.profit for blk in blks))
    return m, blks, link
end

"""
LP of the linked block model: its value (an upper bound) and λ_b = d(LP value)/d(rhs) of
z_{b+1} - y_b = 0 (the value of one more unit of incoming state), so that the LP-relaxed blocks
give L_LP(λ) = LP value.  JuMP's `shadow_price` drops the sign of equality duals, so λ is taken
from `dual`: for a maximisation MOI duals are -d(value)/d(rhs).
"""
function td_lp_multipliers(inst::TSInstance, s::Int, p::AbstractMatrix, cuts, optimizer; form::Symbol = :dense)
    m, blks, link = td_joint(inst, s, p, cuts, optimizer; form)
    relax_integrality(m)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("joint LP: $(JuMP.termination_status(m))")
    return objective_value(m), [-dual.(c) for c in link], m, blks
end

"""In-transit bikes (in bikes), workers on tasks and open tasks at the start of each period of a solved
whole-day block: the load used by `td_cuts_adaptive`."""
function boundary_load(inst::TSInstance, b)
    T, Γ = inst.T, inst.Γ
    return [t == 1 ? 0.0 : sum(value.(b.Pin[:, :, :, t])) / Γ + sum(value.(b.Gin[:, :, t])) +
                           sum(value.(b.Mrem[:, :, t])) for t in 1:T]
end

"""Whole-day LP (tier formulation) of scenario s and its `boundary_load`."""
function lp_load(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; form::Symbol = :dense)
    m = Model(optimizer)
    if form === :sparse
        b = add_scenario_sparse!(m, inst, inst.scen[s], p, p; stability = :tier)
    else
        b = add_scenario!(m, inst, inst.scen[s], (wage = p, ir = p))
        add_tier_stability!(m, inst, b, p)
    end
    @objective(m, Max, b.profit)
    relax_integrality(m)
    optimize!(m)
    JuMP.termination_status(m) == MOI.OPTIMAL || error("LP load: $(JuMP.termination_status(m))")
    return boundary_load(inst, b), objective_value(m)
end

"""
    lagrangian_td(inst, s, p, optimizer, master_optimizer; cuts, λ0 = :lp, scaled = true,
                  block_tl = 60.0, budget = 600.0, maxit = 1000, tol = 1e-5, gtol = 0.05)

Proximal bundle on L(λ).  One cutting-plane model per block, built from every solution the
block MILP returns (the solver's pool too); prox term Σ_k w_k (λ_k − center_k)² / 2t with
w_k = scale_k² (`scaled`, i.e. per bike / worker / task units) or 1.  Each evaluation solves all
B blocks; its value is Σ proven block bounds (a valid upper bound even at a block time limit).
Serious step when the bound drops by 10% of the predicted decrease; t doubles after a good
serious step and shrinks by 0.7 after a null step.  Optimality certificate of the center c
(master optimality, ĝ = W(c − λ_new)/t ∈ ∂model(λ_new)):
    L(λ) ≥ U(c) − ε̂ + ĝ·(λ − c)  for all λ,   ε̂ = pred − t ĝ'W⁻¹ĝ.
Stops when ε̂ ≤ tol·|U(c)| and Σ_k |ĝ_k| / scale_k ≤ gtol (aggregate copy mismatch in bikes /
workers / tasks), or at the budget.  A small predicted decrease with a large ĝ only means t
became too small: t is multiplied by 5 instead of stopping.  When the predicted decrease is below
the oracle noise at the center (Σ block bound − Σ block incumbent, i.e. the block MIP gaps) and that
noise is small (≤ 1e-3·|U|), no further decrease can be certified and the run stops (`stop = :noise`).
Without a certificate the run also stops when the best bound improved by less than 0.01% over the
last `stall` evaluations (`:stall`; every bound is valid, only optimality of the dual is not proven).
Cuts with a zero master dual in the last 30 master solves are deleted every 10 iterations.
A block whose MILP returned no solution yet has no cut, so the master is unbounded: the run then stops with the
bounds evaluated so far (`:nocut`, e.g. the Paris blocks at n = 50 within the block time limit).
info.stop ∈ (:converged, :noise, :stall, :budget, :maxit, :nocut).
Returns (ub, λ_best, info) with info.trace one entry per evaluation.
`zstart` (sparse only): incoming states of blocks 2..B (e.g. `td_forward`'s info.ys); every block then gets the
`continue_start!` MIP start from that state, so each block has a feasible solution, hence a cut, from the start.
`on_eval(e)` (optional) is called after every evaluation with e = (wall, ub, best, λbest, improved); time it takes
counts against the budget (used to run the forward heuristic at each new best λ).
"""
function lagrangian_td(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer, master_optimizer;
                       cuts, λ0 = :lp, scaled::Bool = true, block_tl::Float64 = 60.0, budget::Float64 = 600.0,
                       maxit::Int = 1000, tol::Float64 = 1e-5, gtol::Float64 = 0.05, stall::Int = 40,
                       verbose::Bool = true, form::Symbol = :dense, on_eval = nothing, zstart = nothing)
    zstart === nothing || form === :sparse || error("zstart needs form = :sparse")
    t_start = time()
    deadline = t_start + budget
    B = length(cuts)
    blocks = map(cuts) do (ta, tb)
        m = Model(optimizer)
        blk = add_td_block!(m, inst, s, p, ta, tb; form)
        @objective(m, Max, blk.b.profit)
        (m = m, blk...)
    end
    if zstart !== nothing          # feasible MIP start in every block: no new task from the given incoming state
        for (k, blk) in enumerate(blocks)
            st = k == 1 ? initial_state(blk.sub, blk.b) : keys_state(blk.sub, blk.b, blk.zkeys, zstart[k-1])
            continue_start!(blk.sub, blk.b, 1, st)
        end
    end
    build_s = time() - t_start
    sizes = [length(blocks[k].y) for k in 1:B-1]
    off = [0; cumsum(sizes)]
    rng = [off[k]+1:off[k+1] for k in 1:B-1]
    nλ = off[end]
    scl = reduce(vcat, [[key.scale for key in blocks[k].ykeys] for k in 1:B-1]; init = Float64[])
    w = scaled ? scl .^ 2 : ones(nλ)

    lp_value, lp_s = NaN, 0.0
    if λ0 === :lp
        tl = time()
        lp_value, duals, _, _ = td_lp_multipliers(inst, s, p, cuts, optimizer; form)
        λ = reduce(vcat, duals; init = Float64[])
        lp_s = time() - tl
    elseif λ0 === :zero
        λ = zeros(nλ)
    else
        λ = Float64.(λ0)
    end
    length(λ) == nλ || error("λ0 has length $(length(λ)), need $nλ")

    master = Model(master_optimizer)
    @variable(master, Λ[1:nλ])
    @variable(master, θ[1:B])
    seen = [Set{UInt}() for _ in 1:B]
    ncut, it_now = Ref(0), Ref(0)
    pool = NamedTuple{(:con, :k, :h),Tuple{ConstraintRef,Int,UInt}}[]
    lastact = Int[]
    function add_cut!(k, c, y, z)
        h = hash((round(c; digits = 6), round.(y; digits = 6), round.(z; digits = 6)))
        h in seen[k] && return
        push!(seen[k], h)
        ex = AffExpr(c)
        k < B && for (i, v) in zip(rng[k], y)
            add_to_expression!(ex, v, Λ[i])
        end
        k > 1 && for (i, v) in zip(rng[k-1], z)
            add_to_expression!(ex, -v, Λ[i])
        end
        push!(pool, (con = @constraint(master, θ[k] >= ex), k = k, h = h))
        push!(lastact, it_now[])
        ncut[] += 1
    end
    # cuts not active (zero dual) in the last `keep` master solves are deleted every 10 iterations
    function clean_cuts!(it, keep)
        drop = [i for i in eachindex(pool) if it - lastact[i] > keep]
        for i in drop
            delete(master, pool[i].con)
            delete!(seen[pool[i].k], pool[i].h)
        end
        keepi = setdiff(eachindex(pool), drop)
        pool, lastact = pool[keepi], lastact[keepi]
        return length(drop)
    end

    # one oracle call: every block at λ; nothing is returned when the budget ran out mid-way
    function evaluate(λ)
        ub, lb, ssum, smax, ntl, sols = 0.0, 0.0, 0.0, 0.0, 0, Vector{Any}(undef, B)
        for k in 1:B
            blk = blocks[k]
            k < B && for (i, v) in zip(rng[k], blk.y)
                set_objective_coefficient(blk.m, v, λ[i])
            end
            k > 1 && for (i, v) in zip(rng[k-1], blk.z)
                set_objective_coefficient(blk.m, v, -λ[i])
            end
            left = deadline - time()
            left > 1.0 || return nothing
            set_time_limit_sec(blk.m, min(block_tl, left))
            optimize!(blk.m)
            st = JuMP.termination_status(blk.m)
            st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("block $k: $st")
            st == MOI.OPTIMAL || (ntl += 1)
            ub += objective_bound(blk.m)            # proven, also at a time limit
            ts = solve_time(blk.m)
            ssum += ts; smax = max(smax, ts)
            nres = result_count(blk.m)
            if nres == 0 || primal_status(blk.m) != MOI.FEASIBLE_POINT
                lb = -Inf
            else
                lb += objective_value(blk.m)
            end
            for r in 1:nres
                primal_status(blk.m; result = r) == MOI.FEASIBLE_POINT || continue
                c = value(blk.b.profit; result = r)
                y = [value(v; result = r) for v in blk.y]
                z = [value(v; result = r) for v in blk.z]
                add_cut!(k, c, y, z)
                r == 1 && (sols[k] = (y = y, z = z))
            end
        end
        g = all(isassigned(sols, k) for k in 1:B) ?
            reduce(vcat, [sols[k].y .- sols[k+1].z for k in 1:B-1]; init = Float64[]) : zeros(nλ)
        return (ub = ub, lb = lb, ssum = ssum, smax = smax, ntl = ntl, g = g)
    end

    trace = NamedTuple[]
    ev = evaluate(λ)
    ev === nothing && return Inf, λ, (trace = trace, B = B, nλ = nλ, cuts = cuts, lp = lp_value,
                                       build_s = build_s, lp_s = lp_s, it = 0, serious = 0, ncut = ncut[],
                                       converged = false, stop = :budget, wall = time() - t_start)
    center, Uc, noise = copy(λ), ev.ub, ev.ub - ev.lb
    best, λbest = ev.ub, copy(λ)
    on_eval === nothing || on_eval((wall = time() - t_start, ub = ev.ub, best = best, λbest = λbest, improved = true))
    gs = maximum(abs.(ev.g ./ (scaled ? scl : ones(nλ))); init = 0.0)
    t = t0 = 1.0 / max(gs, 1e-3)
    push!(trace, (it = 0, wall = time() - t_start, ub = ev.ub, best = best, lb = ev.lb, pred = NaN, t = t,
                  serious = true, ssum = ev.ssum, smax = ev.smax, ntl = ev.ntl, gnorm = gs))
    verbose && @printf("  it %3d  %7.1fs  L %.4f  best %.4f  (LP %.4f)  blocks %.1fs  tl %d\n",
                       0, time() - t_start, ev.ub, best, lp_value, ev.ssum, ev.ntl)
    nserious, converged, it = 0, nλ == 0, 0           # one block: the evaluation is exact
    stop = converged ? :converged : :maxit
    besthist = [best]
    while !converged && it < maxit && time() < deadline - 1.0
        all(!isempty, seen) || (stop = :nocut; break)
        it += 1
        it % 10 == 0 && clean_cuts!(it, 30)           # before the solve: deleting rows drops the solution
        @objective(master, Min, sum(θ) + sum(w[i] * (Λ[i] - center[i])^2 for i in 1:nλ) / (2t))
        it_now[] = it
        set_time_limit_sec(master, max(1.0, deadline - time()))
        optimize!(master)
        mst = JuMP.termination_status(master)
        mst == MOI.TIME_LIMIT && (stop = :budget; break)
        mst in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("master: $mst")
        for i in eachindex(pool)
            abs(dual(pool[i].con)) > 1e-9 && (lastact[i] = it)
        end
        λn = value.(Λ)
        model_val = sum(value.(θ))
        pred = Uc - model_val
        ĝ = w .* (center .- λn) ./ t
        εagg = pred - t * sum(ĝ .^ 2 ./ w)
        gagg = sum(abs.(ĝ ./ scl))
        tolU = tol * max(1.0, abs(Uc))
        small_noise = noise <= 1e-3 * max(1.0, abs(Uc))
        if pred <= tolU || (small_noise && pred <= noise)
            if εagg <= tolU && gagg <= gtol
                converged, stop = true, :converged
                break
            elseif small_noise && pred <= noise
                stop = :noise
                break
            end
            t *= 5                                   # step too short to certify anything
            continue
        end
        ev = evaluate(λn)
        ev === nothing && (stop = :budget; break)
        improved = ev.ub < best
        if improved
            best, λbest = ev.ub, copy(λn)
        end
        on_eval === nothing || on_eval((wall = time() - t_start, ub = ev.ub, best = best, λbest = λbest, improved))
        push!(besthist, best)
        if length(besthist) > stall && besthist[end-stall] - best <= 1e-4 * max(1.0, abs(best))
            stop = :stall                              # no 0.01% progress in `stall` evaluations
            push!(trace, (it = it, wall = time() - t_start, ub = ev.ub, best = best, lb = ev.lb, pred = pred,
                          t = t, serious = false, ssum = ev.ssum, smax = ev.smax, ntl = ev.ntl, gnorm = gagg))
            break
        end
        serious = ev.ub <= Uc - 0.1 * pred
        if serious
            nserious += 1
            ev.ub <= Uc - 0.5 * pred && (t = min(2t, 1e6))
            center, Uc, noise = λn, ev.ub, ev.ub - ev.lb
        else
            t = max(0.7t, 1e-4 * t0)
        end
        gs = maximum(abs.(ev.g ./ (scaled ? scl : ones(nλ))); init = 0.0)
        push!(trace, (it = it, wall = time() - t_start, ub = ev.ub, best = best, lb = ev.lb, pred = pred, t = t,
                      serious = serious, ssum = ev.ssum, smax = ev.smax, ntl = ev.ntl, gnorm = gagg))
        verbose && @printf("  it %3d  %7.1fs  L %.4f  best %.4f  pred %.4f  ε %.4f  |ĝ| %.3f  %s  t %.3g  blocks %.1fs (max %.1f)  tl %d  cuts %d\n",
                           it, time() - t_start, ev.ub, best, pred, εagg, gagg, serious ? "S" : "n", t, ev.ssum,
                           ev.smax, ev.ntl, ncut[])
        flush(stdout)
    end
    stop === :maxit && it < maxit && (stop = :budget)
    return best, λbest, (trace = trace, B = B, nλ = nλ, cuts = cuts, lp = lp_value, build_s = build_s,
                         lp_s = lp_s, it = it, serious = nserious, ncut = ncut[], converged = converged,
                         stop = stop, wall = time() - t_start)
end

"""
Primal heuristic: solve the blocks in order; block b starts from the state block b-1 actually
left and maximises profit_b + λ_b·y_b (λ = nothing: profit only, i.e. myopic blocks).
A posting penalty of 1e-3 per posted task breaks the ties in favour of not posting: inside a block a
posted task costs nothing until it is matched, but it locks a bike for the blocks after it.
Every block is solved to the time limit; the result is a feasible whole-day plan (see td_verify).
`start = true` (sparse only): every block gets the `continue_start!` MIP start from its incoming state.
Returns (value, plan, info); plan[k] holds the integer decisions of block k.
"""
function td_forward(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer, cuts, λ;
                    block_tl::Float64 = 60.0, deadline::Float64 = Inf, form::Symbol = :dense, start::Bool = false)
    start && form !== :sparse && error("start = true needs form = :sparse")
    B = length(cuts)
    off = 0
    total, plan, prev_y, solver, ntl = 0.0, Any[], nothing, 0.0, 0
    ys = Vector{Float64}[]                          # outgoing state of blocks 1..B-1 (incoming of 2..B)
    for (k, (ta, tb)) in enumerate(cuts)
        m = Model(optimizer)
        blk = add_td_block!(m, inst, s, p, ta, tb; form)
        obj = AffExpr(0.0) + blk.b.profit - 1e-3 * sum(blk.b.mhat)
        if k < B && λ !== nothing
            for (i, v) in enumerate(blk.y)
                add_to_expression!(obj, λ[off+i], v)
            end
        end
        k < B && (off += length(blk.y))
        if k > 1
            for (v, val) in zip(blk.z, prev_y)
                fix(v, val; force = true)
            end
        end
        start && continue_start!(blk.sub, blk.b, 1,
                                 k == 1 ? initial_state(blk.sub, blk.b) : keys_state(blk.sub, blk.b, blk.zkeys, prev_y))
        @objective(m, Max, obj)
        set_time_limit_sec(m, max(1.0, min(block_tl, deadline - time())))
        optimize!(m)
        st = JuMP.termination_status(m)
        st == MOI.OPTIMAL || (ntl += 1)
        primal_status(m) == MOI.FEASIBLE_POINT ||
            return -Inf, plan, (ok = false, failed_at = k, solve_s = solver, ntl = ntl)
        solver += solve_time(m)
        total += value(blk.b.profit)
        prev_y = round.(value.(blk.y))
        k < B && push!(ys, prev_y)
        dec = Dict(f => round.(value.(getfield(blk.b, f)))
                   for f in (:Ycum, :Yod, :YU, :L, :I, :rA, :β, :mhat, :Mt, :mtil, :x) if getfield(blk.b, f) !== nothing)
        push!(plan, (ta = ta, tb = tb, dec = dec))
    end
    return total, plan, (ok = true, failed_at = 0, solve_s = solver, ntl = ntl, ys = ys)
end

"""Incoming state of a sparse block (layout of `initial_state`) from the values of its state keys; keys without a
variable or with bound 0 are zero."""
function keys_state(inst::TSInstance, b, keys, vals)
    n, K = inst.n, inst.K
    st = (A = zeros(Int, K - 1, n), U = zeros(Int, n), W = zeros(Int, n), Mrem = zeros(Int, n, n),
          Pin = zeros(Int, K, n, max(inst.Cmax, 2)), Gin = zeros(Int, n, b.Hmax))
    for (key, v) in zip(keys, vals)
        getfield(st, key.f)[key.idx...] = round(Int, v)
    end
    return st
end

"""
Fix the integer decisions of a `td_forward` plan in the whole-day tier model and re-solve
(only the tier binaries stay free).  Returns (value, blocking pairs, conservation residual).
"""
function td_verify(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer, plan; form::Symbol = :dense)
    m = Model(optimizer)
    if form === :sparse
        b = add_scenario_sparse!(m, inst, inst.scen[s], p, p; stability = :tier)
    else
        b = add_scenario!(m, inst, inst.scen[s], (wage = p, ir = p))
        add_tier_stability!(m, inst, b, p)
    end
    @objective(m, Max, b.profit)
    for part in plan, (f, vals) in part.dec
        full = getfield(b, f)
        nd = ndims(full)
        for tl in 1:(part.tb - part.ta + 1)
            t = part.ta + tl - 1
            for I in CartesianIndices(axes(full)[1:nd-1])
                fix(full[Tuple(I)..., t], vals[Tuple(I)..., tl]; force = true)
            end
        end
    end
    optimize!(m)
    is_solved_and_feasible(m) || return -Inf, -1, Inf
    bp = form === :sparse ? blocking_pairs_sparse(inst, b, p) : blocking_pairs(inst, b, p)
    return objective_value(m), length(bp), conservation_residual(inst, b)
end

"""
benders_td.jl  —  nested Benders (deterministic dual dynamic programming) over the time blocks of one
fixed-price scenario (model_2stage/fixed_price/BENDERS_TD.md)

Blocks as in lagrangian_td.jl (`add_td_block!`): block k > 1 starts from an integer copy z_k of the state
block k-1 left, block k < B carries its outgoing state y_k.  V_k(z) is the best profit of blocks k..B from
state z; θ_{k+1} <= α + β·y_k (one row per cut) is an outer approximation of V_{k+1} kept in block k.

  forward pass   blocks in order, z_k fixed to the state actually left, max profit_k + θ_{k+1} minus the
                 1e-3 posting penalty of `td_forward`; the concatenation is a feasible whole-day plan,
                 so its profit is a lower bound
  LP forward     the same with relaxed integrality: fractional trial states (see `benders_td`)
  backward pass  k = B..2 at the states ŷ_{k-1} of each forward pass:
                 :lp  LP relaxation of block k with its θ_{k+1}: value v, β = d v / d ŷ, cut
                      θ_k <= v + β·(y − ŷ) (valid for the LP value function, hence for the integer one)
                 :sb  also the strengthened Benders cut (Zou, Ahmed & Sun 2019): same β, intercept from
                      L(β) = max profit_k + θ_{k+1} − β·z with z free (a MILP; its proven bound is used),
                      cut θ_k <= L(β) + β·y, added when it is lower than the LP cut at ŷ
  upper bound    block 1 MILP with θ_2 (proven bound); valid because every cut over-estimates V

The posting penalty only enters the forward objective, so the cuts approximate the true profit.
"""

include(joinpath(@__DIR__, "lagrangian_td.jl"))

mutable struct BTDBlock
    m::Model
    blk::Any
    zc::Vector{ConstraintRef}           # z == ŷ (k > 1)
    θ::Union{VariableRef,Nothing}       # value of blocks k+1..B (k < B)
    sm::Union{Model,Nothing}            # z-free copy for strengthened cuts (k > 1)
    sblk::Any
    sθ::Union{VariableRef,Nothing}
    cutvals::Vector{Tuple{Float64,Vector{Float64}}}   # (α, β) of the cuts θ <= α + β·y in this block
end

"""Revenue of periods ta..T: Σ R·ξ, an upper bound on the profit of those periods (θ's initial bound)."""
revenue_tail(inst::TSInstance, s::Int, ta::Int) =
    ta > inst.T ? 0.0 : sum(sum(inst.R .* inst.scen[s][t]) for t in ta:inst.T)

function _btd_model(inst, s, p, ta, tb, optimizer, last; form::Symbol = :dense)
    m = Model(optimizer)
    blk = add_td_block!(m, inst, s, p, ta, tb; form)
    θ = last ? nothing : @variable(m, upper_bound = revenue_tail(inst, s, tb + 1), base_name = "θ")
    return m, blk, θ
end

"""Build one model per block (and its z-free copy when `cut = :sb`)."""
function btd_build(inst::TSInstance, s::Int, p::AbstractMatrix, cuts, optimizer; cut::Symbol = :lp,
                   form::Symbol = :dense)
    B = length(cuts)
    blocks = BTDBlock[]
    for (k, (ta, tb)) in enumerate(cuts)
        m, blk, θ = _btd_model(inst, s, p, ta, tb, optimizer, k == B; form)
        zc = k == 1 ? ConstraintRef[] : [@constraint(m, v == 0) for v in blk.z]
        sm, sblk, sθ = nothing, nothing, nothing
        if cut === :sb && k > 1
            sm, sblk, sθ = _btd_model(inst, s, p, ta, tb, optimizer, k == B; form)
        end
        push!(blocks, BTDBlock(m, blk, zc, θ, sm, sblk, sθ, Tuple{Float64,Vector{Float64}}[]))
    end
    return blocks
end

"""θ <= α + β·y in block k (and in its z-free copy)."""
function btd_add_cut!(b::BTDBlock, α::Float64, β::Vector{Float64})
    @constraint(b.m, b.θ <= α + sum(β[i] * b.blk.y[i] for i in eachindex(β) if β[i] != 0.0; init = 0.0))
    b.sm === nothing ||
        @constraint(b.sm, b.sθ <= α + sum(β[i] * b.sblk.y[i] for i in eachindex(β) if β[i] != 0.0; init = 0.0))
    push!(b.cutvals, (α, β))
    return nothing
end

_objective(b::BTDBlock; penalty::Bool) =
    b.blk.b.profit + (b.θ === nothing ? 0.0 : 1.0 * b.θ) - (penalty ? 1e-3 * sum(b.blk.b.mhat) : 0.0)

function _fix_incoming!(b::BTDBlock, ŷ)
    for (c, v) in zip(b.zc, ŷ)
        set_normalized_rhs(c, v)
    end
end

"""MIP start of a sparse forward block: no new task from its incoming state (`continue_start!`), z = ŷ, and θ = the
lowest of its cuts at the start's outgoing state (capped by θ's upper bound)."""
function btd_start!(b::BTDBlock, ŷ)
    blk = b.blk
    state = ŷ === nothing ? initial_state(blk.sub, blk.b) : keys_state(blk.sub, blk.b, blk.zkeys, ŷ)
    continue_start!(blk.sub, blk.b, 1, state)
    if ŷ !== nothing
        for (v, val) in zip(blk.z, ŷ)
            set_start_value(v, val)
        end
    end
    b.θ === nothing && return
    ys = [something(start_value(v), 0.0) for v in blk.y]
    θs = has_upper_bound(b.θ) ? upper_bound(b.θ) : Inf
    for (α, β) in b.cutvals
        θs = min(θs, α + sum(β .* ys))
    end
    isfinite(θs) && set_start_value(b.θ, θs)
    return
end

"""
    benders_td(inst, s, p, optimizer; cuts, cut = :lp, lp_forward = true, budget = 300.0, block_tl = 60.0,
               sb_tl = 30.0, maxit = 200, tol = 1e-4, stall = 10)

Every iteration: an LP forward pass (blocks with relaxed integrality, fractional states; only when
`lp_forward`), an integer forward pass (the lower bound), a backward pass at the states of each forward
pass, then the upper bound.  Cuts only at the integer forward states stall above the whole-day LP: the
backward LPs move to fractional states where no cut was ever made, and the integer forward pass keeps
visiting the same states.  The LP forward pass makes the cuts converge to (at least) the LP bound.
Stops when (UB − LB) <= tol·|LB|, when neither bound moved by more than tol for `stall` iterations,
after `maxit` iterations or `budget` seconds.  Returns (lb, ub, plan, info); plan is the best integer
forward plan in the `td_forward` format.
`form = :sparse` builds the blocks with `add_scenario_sparse!`; `start = true` (sparse only) gives every integer forward
block the `continue_start!` MIP start from its incoming state and θ the value of its cuts there.  `on_iter(e)` is called
with each trace entry e = (it, t, fwd, lpfwd, lb, ub, ncut).
"""
function benders_td(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; cuts, cut::Symbol = :lp,
                    lp_forward::Bool = true, budget::Float64 = 300.0, block_tl::Float64 = 60.0,
                    sb_tl::Float64 = 30.0, maxit::Int = 200, tol::Float64 = 1e-4, stall::Int = 10,
                    verbose::Bool = true, form::Symbol = :dense, start::Bool = false, on_iter = nothing)
    start && form !== :sparse && error("start = true needs form = :sparse")
    t_start = time()
    deadline = t_start + budget
    left() = deadline - time()
    B = length(cuts)
    blocks = btd_build(inst, s, p, cuts, optimizer; cut, form)
    build_s = time() - t_start
    lb, ub, best_plan = -Inf, Inf, nothing
    trace = NamedTuple[]
    cutlog = Tuple{Int,Float64,Vector{Float64}}[]   # (block that holds the cut, α, β)
    ncut, nsb, nfwd_tl, nsb_tl = 0, 0, 0, 0
    it, quiet, stop = 0, 0, :maxit
    fwd_s, bwd_s, sb_s = 0.0, 0.0, 0.0
    lpfwd = NaN
    nlp_fail = 0                                   # LPs that ended neither optimal nor at the time limit

    # LP forward pass: fractional states of the relaxed blocks
    function forward_lp()
        ŷ = Vector{Vector{Float64}}(undef, B)
        total = 0.0
        for (k, b) in enumerate(blocks)
            k > 1 && _fix_incoming!(b, ŷ[k-1])
            @objective(b.m, Max, _objective(b; penalty = false))
            undo = relax_integrality(b.m)
            set_time_limit_sec(b.m, max(1.0, left()))
            optimize!(b.m)
            st = JuMP.termination_status(b.m)
            if st != MOI.OPTIMAL                  # out of time or solver trouble: no LP trial states this iteration
                undo()
                st == MOI.TIME_LIMIT || (nlp_fail += 1)
                return nothing, total
            end
            total += value(b.blk.b.profit)
            k < B && (ŷ[k] = value.(b.blk.y))
            undo()
        end
        return ŷ, total
    end

    # integer forward pass: a feasible whole-day plan
    function forward_int()
        ŷ = Vector{Vector{Float64}}(undef, B)
        total, plan = 0.0, Any[]
        for (k, b) in enumerate(blocks)
            k > 1 && _fix_incoming!(b, ŷ[k-1])
            @objective(b.m, Max, _objective(b; penalty = true))
            start && btd_start!(b, k == 1 ? nothing : ŷ[k-1])
            set_time_limit_sec(b.m, max(1.0, min(block_tl, left())))
            optimize!(b.m)
            JuMP.termination_status(b.m) == MOI.OPTIMAL || (nfwd_tl += 1)
            primal_status(b.m) == MOI.FEASIBLE_POINT || return nothing, total, plan
            total += value(b.blk.b.profit)
            k < B && (ŷ[k] = round.(value.(b.blk.y)))
            dec = Dict(f => round.(value.(getfield(b.blk.b, f)))
                       for f in (:Ycum, :Yod, :YU, :L, :I, :rA, :β, :mhat, :Mt, :mtil, :x) if getfield(b.blk.b, f) !== nothing)
            push!(plan, (ta = b.blk.ta, tb = b.blk.tb, dec = dec))
        end
        return ŷ, total, plan
    end

    # backward pass at the states ŷ; false when the budget ran out
    function backward(ŷ)
        for k in B:-1:2
            b = blocks[k]
            _fix_incoming!(b, ŷ[k-1])
            @objective(b.m, Max, _objective(b; penalty = false))
            undo = relax_integrality(b.m)
            set_time_limit_sec(b.m, max(1.0, left()))
            optimize!(b.m)
            st = JuMP.termination_status(b.m)
            if st != MOI.OPTIMAL
                undo()
                st == MOI.TIME_LIMIT && return false
                nlp_fail += 1                    # solver trouble: skip this cut (cuts are optional)
                continue
            end
            v = objective_value(b.m)
            β = -dual.(b.zc)                     # d value / d rhs (MOI duals of a maximisation are −d/d rhs)
            undo()
            α = v - sum(β .* ŷ[k-1])
            btd_add_cut!(blocks[k-1], α, β)
            push!(cutlog, (k - 1, α, β))
            ncut += 1
            if cut === :sb
                ts = time()
                sm = b.sm
                @objective(sm, Max, b.sblk.b.profit + (b.sθ === nothing ? 0.0 : 1.0 * b.sθ) -
                                    sum(β[i] * b.sblk.z[i] for i in eachindex(β) if β[i] != 0.0; init = 0.0))
                set_time_limit_sec(sm, max(1.0, min(sb_tl, left())))
                optimize!(sm)
                JuMP.termination_status(sm) == MOI.OPTIMAL || (nsb_tl += 1)
                L = objective_bound(sm)
                sb_s += time() - ts
                if isfinite(L) && L < α - 1e-6 * max(1.0, abs(α))
                    btd_add_cut!(blocks[k-1], L, β)
                    push!(cutlog, (k - 1, L, β))
                    nsb += 1
                end
            end
            left() <= 0 && return false
        end
        return true
    end

    while it < maxit
        it += 1
        t0 = time()
        ŷlp = nothing
        if lp_forward
            ŷlp, lpfwd = forward_lp()
        end
        ŷ, total, plan = forward_int()
        fwd_s += time() - t0
        ŷ === nothing && (stop = left() <= 0 ? :budget : :forward_failed; break)
        lbmoved = total > lb + tol * max(1.0, abs(lb))
        if total > lb
            lb, best_plan = total, plan
        end
        t0 = time()
        done = (ŷlp === nothing || backward(ŷlp)) && backward(ŷ)
        bwd_s += time() - t0
        if !done
            stop = :budget
            push!(trace, (it = it, t = time() - t_start, fwd = total, lpfwd = lpfwd, lb = lb, ub = ub, ncut = ncut))
            break
        end
        # upper bound: block 1 with θ_2
        b1 = blocks[1]
        @objective(b1.m, Max, _objective(b1; penalty = false))
        set_time_limit_sec(b1.m, max(1.0, min(block_tl, left())))
        optimize!(b1.m)
        st1 = JuMP.termination_status(b1.m)
        newub = st1 in (MOI.OPTIMAL, MOI.TIME_LIMIT) ? objective_bound(b1.m) : Inf   # solver trouble: no new bound
        isfinite(newub) || st1 in (MOI.OPTIMAL, MOI.TIME_LIMIT) || (nlp_fail += 1)
        ubmoved = !isfinite(ub) || ub - newub > tol * max(1.0, abs(ub))
        ub = min(ub, newub)
        quiet = (ubmoved || lbmoved) ? 0 : quiet + 1
        push!(trace, (it = it, t = time() - t_start, fwd = total, lpfwd = lpfwd, lb = lb, ub = ub, ncut = ncut))
        on_iter === nothing || on_iter(trace[end])
        verbose && (@printf("    it %3d  %7.1f s  lp-fwd %.3f  fwd %.3f  lb %.3f  ub %.3f  gap %.3f%%  cuts %d (sb %d)\n",
                            it, time() - t_start, lpfwd, total, lb, ub, 100 * (ub - lb) / max(1e-9, abs(lb)), ncut, nsb);
                    flush(stdout))
        if (ub - lb) <= tol * max(1.0, abs(lb))
            stop = :converged; break
        end
        quiet >= stall && (stop = :stall; break)
        left() <= 0 && (stop = :budget; break)
    end
    return lb, ub, best_plan, (it = it, B = B, ncut = ncut, nsb = nsb, nfwd_tl = nfwd_tl, nsb_tl = nsb_tl, nlp_fail = nlp_fail,
                               stop = stop, build_s = build_s, fwd_s = fwd_s, bwd_s = bwd_s, sb_s = sb_s,
                               wall = time() - t_start, trace = trace, cutlog = cutlog)
end

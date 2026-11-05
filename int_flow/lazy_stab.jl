"""
lazy_stab.jl  —  TODO #9: the stable-matching rows of the sparse tier block (`add_tier_direct!`, stability = :tier)
added on demand instead of all at once.

    b = add_scenario_sparse!(m, w, w.scen[1], P, P; stability = :none)
    S = stab_init!(m, w, b)            every tier binary z (in no row yet, start 1 as nothing_start! sets b.z), the
                                       rows of every group as data
    add_groups!(m, S, gs)              add the rows of the groups gs to the model (they become active)
    gs = separate_lp(S, val)           inactive groups whose rows no z ∈ [0, 1] satisfies at the point val
    gs = separate_int(S, val)          inactive groups an integral point violates
    lazy_stab!(m, S)                   lazy-constraint callback: separate_int on every integral candidate, the rows of
                                       the violated groups go to Gurobi as lazy constraints and the groups to
                                       S.pending; flush_pending!(m, S) makes them ordinary rows before the next solve
    relax_z!(S) / restore_z!(S)        z continuous in [0, 1] / binary again

`val` maps a variable to its value (`value`, or `v -> callback_value(cb_data, v)`); `skip_active = false` checks
the active groups too (tests).

Group g = (i, h) at period t: tier h (utility v_h, ascending as in add_tier_direct!) of node i.  Its rows are the rows
add_tier_direct! writes for z[i, h, t]:
    worker row   W[i, t] − Σ_{e ∈ by_i[i], u_e ≥ v_h} x[e, t] ≤ Q1 z
    pool rows    M[j, k, t] − H_e ≤ Q3 (1 − z)      one per triple e = (i, j, k) with u_e = v_h,
                 H_e = Σ_{e2 = (i2, j, k), d[i2, j] ≤ d[i, j]} x[e2, t]
All groups with all their rows give the tier model; any subset of groups gives a relaxation (the other z are free).
LP check:   a z ∈ [0, 1] satisfies the group iff max(lo, 0) ≤ min(hi, 1),
            lo = (W − Σ x) / Q1,   hi = 1 − max_e (M − H_e) / Q3.
Integral check: the worker row forces z = 1 (W − Σ x ≥ 1) while a pool row forces z = 0 (M − H_e ≥ 1).  This is
blocking_pairs_sparse's condition: a worker of i idle or on a task below v_h (worst < v_h) and a pool of tier v_h
not emptied by workers at least as close.
"""

mutable struct StabGroups
    T::Int
    Q1::Float64
    Q3::Float64
    node::Vector{Int}                                  # node i of group kind g
    wx::Vector{Vector{Int}}                            # worker row: the e with u_e ≥ v_h
    pools::Vector{Vector{Tuple{Int,Int,Vector{Int}}}}  # pool rows: (j, k, the e2 of H_e)
    z::Matrix{VariableRef}                             # z[g, t]
    W::Matrix{VariableRef}
    X::Matrix{VariableRef}
    M::Array{VariableRef,3}
    active::BitMatrix                                  # rows of (g, t) in the model
    pending::Set{Tuple{Int,Int}}                       # added by the callback, not yet ordinary rows
    ncalls::Int                                        # integral candidates checked by the callback
    nreject::Int                                       # of them violated
    nlazy::Int                                         # lazy rows submitted
end

ngroups(S::StabGroups) = length(S.node) * S.T
nrows_group(S::StabGroups, g::Int) = 1 + length(S.pools[g])

function stab_init!(m::Model, w::TSInstance, b)
    n, T, d = w.n, w.T, w.d
    trips, by_i, by_jk, u = b.trips, b.by_i, b.by_jk, b.u
    node, wx, pools = Int[], Vector{Int}[], Vector{Tuple{Int,Int,Vector{Int}}}[]
    for i in 1:n
        tiers = sort(unique(u[e] for e in by_i[i] if u[e] > 0))     # as add_tier_direct!
        for v in tiers
            push!(node, i)
            push!(wx, [e for e in by_i[i] if u[e] >= v])
            pl = Tuple{Int,Int,Vector{Int}}[]
            for e in by_i[i]
                u[e] == v || continue
                _, j, k = trips[e]
                push!(pl, (j, k, [e2 for e2 in by_jk[j, k] if d[trips[e2][1], j] <= d[i, j]]))
            end
            push!(pools, pl)
        end
    end
    G = length(node)
    z = [@variable(m, binary = true, base_name = string("z_", node[g], "_", g, "_", t)) for g in 1:G, t in 1:T]
    set_start_value.(z, 1.0)
    E = length(trips)
    W = [b.W[i, t] for i in 1:n, t in 1:T]
    X = [b.x[e, t] for e in 1:E, t in 1:T]
    M = [b.Mt[j, k, t] for j in 1:n, k in 1:n, t in 1:T]
    return StabGroups(T, Float64(w.Q1), Float64(w.Q3), node, wx, pools, z, W, X, M, falses(G, T),
                      Set{Tuple{Int,Int}}(), 0, 0, 0)
end

"""The rows of group (g, t) as built constraints (add_constraint or MOI.submit as lazy)."""
function group_rows(S::StabGroups, g::Int, t::Int)
    i, z = S.node[g], S.z[g, t]
    rows = Any[@build_constraint(S.W[i, t] - sum(S.X[e, t] for e in S.wx[g]) <= S.Q1 * z)]
    for (j, k, H) in S.pools[g]
        push!(rows, @build_constraint(S.M[j, k, t] - sum(S.X[e2, t] for e2 in H) <= S.Q3 * (1 - z)))
    end
    return rows
end

"""Add the rows of the groups `gs` (inactive ones only); returns the number of rows added."""
function add_groups!(m::Model, S::StabGroups, gs)
    nr = 0
    for (g, t) in gs
        S.active[g, t] && continue
        for c in group_rows(S, g, t)
            add_constraint(m, c)
            nr += 1
        end
        S.active[g, t] = true
    end
    return nr
end

add_all_groups!(m::Model, S::StabGroups) = add_groups!(m, S, [(g, t) for g in eachindex(S.node), t in 1:S.T])

function _stab_values(S::StabGroups, val)
    return val.(S.W), val.(S.X), val.(S.M)
end

_xsum(Xv, es, t) = sum(Xv[e, t] for e in es; init = 0.0)

function separate_lp(S::StabGroups, val; tol::Float64 = 1e-7, skip_active::Bool = true)
    Wv, Xv, Mv = _stab_values(S, val)
    out = Tuple{Int,Int}[]
    for t in 1:S.T, g in eachindex(S.node)
        skip_active && S.active[g, t] && continue
        i = S.node[g]
        lo = (Wv[i, t] - _xsum(Xv, S.wx[g], t)) / S.Q1
        hi = 1 - maximum(Mv[j, k, t] - _xsum(Xv, H, t) for (j, k, H) in S.pools[g]) / S.Q3
        max(lo, 0.0) > min(hi, 1.0) + tol && push!(out, (g, t))
    end
    return out
end

function separate_int(S::StabGroups, val; skip_active::Bool = true)
    Wv, Xv, Mv = _stab_values(S, val)
    out = Tuple{Int,Int}[]
    for t in 1:S.T, g in eachindex(S.node)
        skip_active && S.active[g, t] && continue
        i = S.node[g]
        Wv[i, t] - _xsum(Xv, S.wx[g], t) > 0.5 || continue                  # z forced to 1
        any(Mv[j, k, t] - _xsum(Xv, H, t) > 0.5 for (j, k, H) in S.pools[g]) && push!(out, (g, t))   # and to 0
    end
    return out
end

function lazy_stab!(m::Model, S::StabGroups)
    function cb(cb_data)
        callback_node_status(cb_data, m) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        S.ncalls += 1
        # pending groups are checked again: Gurobi may report a candidate that violates lazy rows sent before
        gs = separate_int(S, v -> callback_value(cb_data, v))
        isempty(gs) && return
        S.nreject += 1
        for (g, t) in gs
            for c in group_rows(S, g, t)
                MOI.submit(m, MOI.LazyConstraint(cb_data), c)
                S.nlazy += 1
            end
            push!(S.pending, (g, t))
        end
    end
    MOI.set(m, MOI.LazyConstraintCallback(), cb)
    return S
end

"""Make the groups the callback added ordinary rows (call before the next solve); returns the rows added."""
function flush_pending!(m::Model, S::StabGroups)
    nr = add_groups!(m, S, collect(S.pending))
    empty!(S.pending)
    return nr
end

function relax_z!(S::StabGroups)
    for v in S.z
        unset_binary(v)
        set_lower_bound(v, 0.0)
        set_upper_bound(v, 1.0)
    end
end

function restore_z!(S::StabGroups)
    for v in S.z
        delete_lower_bound(v)
        delete_upper_bound(v)
        set_binary(v)
    end
end

stab_stats(S::StabGroups) = (groups = ngroups(S), groups_active = count(S.active),
    rows_active = sum(nrows_group(S, g) for g in eachindex(S.node), t in 1:S.T if S.active[g, t]; init = 0),
    rows_all = sum(nrows_group(S, g) for g in eachindex(S.node)) * S.T,
    lazy_calls = S.ncalls, lazy_rejects = S.nreject, lazy_rows = S.nlazy)

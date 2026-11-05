"""
alt_price.jl  —  TODO #4: per-OD task prices by alternating (weekly/2026-10-31_11-06/TODO.md "#4 交替求解"), on the
integer rider split (int_flow/README.md) with the sparse tier block.

    vals, snaps, infos = ostep(ws, p, opt; starts)      O step: fixed-price solve of every scenario, MIP start `starts`
    rows = price_rows(ws, snaps)                        the stable matchings of fixed operations as rows on p
    q = least_prices(p, menus, rows; unused = :keep)    P step: cheapest prices on the menus that keep them stable
    ub, info = discrim_ub(w, menus, opt)                upper bound of a scenario for any per-OD prices

`ws` are scenario windows (`window_instance`), `p` an n x n price matrix (task jk), `menus[j, k]` the sorted price
levels of task jk.

P step.  With x, W, M of every scenario and period fixed, p keeps them a stable matching (blocking_pairs_sparse's
definition, plus individual rationality) iff for every node i with workers, with H_iab = Σ_{i2: d_i2a <= d_ia} x_i2ab:
    p_jk >= d_ij + c_jk                              i took jk
    p_jk - p_ab >= (d_ij + c_jk) - (d_ia + c_ab)     i took jk, pool ab not filled for i (H_iab < M_ab)
    p_ab <= d_ia + c_ab                              i has an idle worker, pool ab not filled for i
These are difference rows on per-OD menus, so the feasible set is closed under the componentwise min and has a least
element; it minimises the wage bill for any task volumes and is found by raising from the lower bounds.  It is <= p,
so the old operations stay feasible at a wage <= the old one, and an optimal O step never lowers the profit.
ODs with no task (in any scenario and period) appear only on the right of a row: `:keep` holds them at their price;
`:follow` lets them drop with the used ODs, then puts each back as high as the rows allow but <= its price.

Upper bound.  For per-OD prices p on the menus, the optimal operations of a scenario have x_ijk > 0 only if
p_jk >= d_ij + c_jk, so every such worker is paid at least the lowest level of jk's menu >= d_ij + c_jk.  Dropping the
stable matching and paying each worker that level gives a scenario MILP without prices whose value bounds Q_s(p) for
every p on the menus (`discrim_ub`).
"""

up(L, v) = (h = findfirst(>=(v - 1e-9), L)) === nothing ? error("no menu level >= $v") : L[h]
down(L, v) = (h = findlast(<=(v + 1e-9), L)) === nothing ? error("no menu level <= $v") : L[h]
uniform_menus(n, menu) = [sort(Float64.(menu)) for _ in 1:n, _ in 1:n]

"""Values of a solved sparse block as plain arrays; x as (i, j, k, t) => count."""
function snapshot(b)
    val(v) = v === nothing ? nothing : round.(value.(v))
    xv = value.(b.x)
    x = Dict{NTuple{4,Int},Int}()
    for (e, (i, j, k)) in enumerate(b.trips), t in axes(xv, 2)
        a = round(Int, xv[e, t])
        a > 0 && (x[(i, j, k, t)] = a)
    end
    return (x = x, A = val(b.A), U = val(b.U), W = val(b.W), Mrem = val(b.Mrem), Pin = val(b.Pin), Gin = val(b.Gin),
            Ycum = val(b.Ycum), Yod = val(b.Yod), YU = val(b.YU), L = val(b.L), I = val(b.I), rA = val(b.rA),
            β = val(b.β), mhat = val(b.mhat), Mt = val(b.Mt), mtil = val(b.mtil))
end

"""Tasks done per OD over the horizon (n x n) in a snapshot."""
function task_volume(sn)
    mt = sn.mtil
    n = size(mt, 1)
    return [sum(mt[j, k, t] for t in axes(mt, 3)) for j in 1:n, k in 1:n]
end

"""MIP start from a snapshot taken at another price that keeps it stable (the P step's output): every variable of
the block, x by triple, the tier binaries z (direct form) from their rows.  Errors if a used triple is not allowed."""
function start_from!(b, sn)
    for f in (:A, :U, :W, :Mrem, :Pin, :Ycum, :Yod, :YU, :L, :I, :rA, :β, :mhat, :Mt, :mtil)
        v = getfield(b, f)
        v === nothing || set_start_value.(v, getfield(sn, f))
    end
    old = axes(sn.Gin, 2)
    for k in axes(b.Gin, 1), r in axes(b.Gin, 2), t in axes(b.Gin, 3)
        set_start_value(b.Gin[k, r, t], r in old ? sn.Gin[k, r, t] : 0.0)
    end
    allowed = Set(b.trips)
    all(key -> (key[1], key[2], key[3]) in allowed, keys(sn.x)) || error("start uses a task the price does not allow")
    T = size(b.x, 2)
    for (e, (i, j, k)) in enumerate(b.trips), t in 1:T
        set_start_value(b.x[e, t], get(sn.x, (i, j, k, t), 0))
    end
    n = length(b.by_i)
    for t in 1:T
        h = 0
        for i in 1:n
            tiers = sort(unique(b.u[e] for e in b.by_i[i] if b.u[e] > 0))
            for v in tiers
                h += 1
                took = sum((get(sn.x, (b.trips[e]..., t), 0) for e in b.by_i[i] if b.u[e] >= v); init = 0)
                set_start_value(b.zt[t][h], sn.W[i, t] - took > 0.5 ? 1.0 : 0.0)
            end
        end
        h == length(b.zt[t]) || error("tier count mismatch (start_from! assumes tier_form = :direct)")
    end
end

"""O step: every scenario at price p (wage = IR = p, exact tier block), from `starts` (snapshots) or the
post-nothing start.  `opt(s)` is the optimizer of scenario s.  Returns values, snapshots and per-scenario info."""
function ostep(ws, p::AbstractMatrix, opt; starts = nothing)
    vals, snaps, infos = Float64[], Any[], Any[]
    for (s, w) in enumerate(ws)
        m = Model(opt(s))
        tb = @elapsed begin
            b = add_scenario_sparse!(m, w, w.scen[1], p, p; stability = :tier)
            starts === nothing ? nothing_start!(w, b) : start_from!(b, starts[s])
            @objective(m, Max, b.profit)
        end
        v0 = value(v -> something(start_value(v), 0.0), b.profit)
        optimize!(m)
        st = JuMP.termination_status(m)
        primal_status(m) == MOI.FEASIBLE_POINT || error("O step, scenario $s: $st")
        push!(vals, objective_value(m))
        push!(snaps, snapshot(b))
        push!(infos, (status = string(st), optimal = st == MOI.OPTIMAL, start = v0, solve_s = solve_time(m),
                      build_s = tb, bound = objective_bound(m), blocking = length(blocking_pairs_sparse(w, b, p)),
                      ops = op_stats_sparse(w, b, p)))
    end
    return vals, snaps, infos
end

"""Rows of the P step from the operations of every scenario: lb (IR, per OD), ub (idle workers), vol (tasks done),
diff[(j, k, a, b)] = c for p_jk - p_ab >= c (the largest c over scenarios and periods)."""
function price_rows(ws, snaps)
    n, d, c = ws[1].n, ws[1].d, ws[1].c
    thr(i, j, k) = d[i, j] + c[j, k]
    lb, ub, vol = fill(-Inf, n, n), fill(Inf, n, n), zeros(n, n)
    diff = Dict{NTuple{4,Int},Float64}()
    for (w, sn) in zip(ws, snaps)
        T = w.T
        took = [Dict{Int,Vector{NTuple{3,Int}}}() for _ in 1:T]          # t: i => (j, k, count)
        inpool = [Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}() for _ in 1:T]   # t: (j, k) => (d_ij, count)
        for ((i, j, k, t), a) in sn.x
            push!(get!(took[t], i, NTuple{3,Int}[]), (j, k, a))
            push!(get!(inpool[t], (j, k), Tuple{Int,Int}[]), (d[i, j], a))
            vol[j, k] += a
            lb[j, k] = max(lb[j, k], thr(i, j, k))
        end
        for t in 1:T
            posted = [(a, b2, round(Int, sn.Mt[a, b2, t])) for a in 1:n, b2 in 1:n if sn.Mt[a, b2, t] > 0.5]
            for i in 1:n
                Wi = round(Int, sn.W[i, t])
                Wi == 0 && continue
                mine = get(took[t], i, NTuple{3,Int}[])
                idle = Wi > sum((a for (_, _, a) in mine); init = 0)
                for (a, b2, M) in posted
                    H = sum((cnt for (dd, cnt) in get(inpool[t], (a, b2), Tuple{Int,Int}[]) if dd <= d[i, a]); init = 0)
                    H < M || continue
                    idle && (ub[a, b2] = min(ub[a, b2], thr(i, a, b2)))
                    for (j, k, _) in mine
                        (j, k) == (a, b2) && continue
                        key = (j, k, a, b2)
                        diff[key] = max(get(diff, key, -Inf), thr(i, j, k) - thr(i, a, b2))
                    end
                end
            end
        end
    end
    return (lb = lb, ub = ub, vol = vol, diff = diff)
end

"""Does p satisfy the rows (and lie on the menus)?"""
function rows_ok(p, menus, rows; tol = 1e-9)
    used = rows.vol .> 0
    all(any(abs(p[jk] - l) <= tol for l in menus[jk]) for jk in eachindex(p)) || return false
    all(!used[jk] || p[jk] >= rows.lb[jk] - tol for jk in eachindex(p)) || return false
    all(p .<= rows.ub .+ tol) || return false
    return all(p[j, k] - p[a, b2] >= cv - tol for ((j, k, a, b2), cv) in rows.diff)
end

"""P step: the least prices on the menus that satisfy the rows of the current operations (module docstring).
`p` must satisfy them (it is the price the operations were solved at)."""
function least_prices(p::AbstractMatrix, menus, rows; unused::Symbol = :keep)
    unused in (:keep, :follow) || error("unused must be :keep or :follow")
    rows_ok(p, menus, rows) || error("the current price does not satisfy its own rows")
    used = rows.vol .> 0
    q = [used[jk] ? up(menus[jk], max(rows.lb[jk], first(menus[jk]))) :
         unused === :keep ? p[jk] : first(menus[jk]) for jk in CartesianIndices(p)]
    for sweep in 1:length(p) + 1                     # a longest-path pass per sweep; no positive cycle since p fits
        changed = false
        for ((j, k, a, b2), cv) in rows.diff
            q[j, k] >= q[a, b2] + cv - 1e-9 && continue
            used[j, k] || error("a row with an unused OD on the left")
            q[j, k] = up(menus[j, k], q[a, b2] + cv)
            changed = true
        end
        changed || break
        sweep == length(p) + 1 && error("P step did not converge")
    end
    if unused === :follow
        cap = [used[jk] ? q[jk] : min(p[jk], rows.ub[jk]) for jk in CartesianIndices(p)]
        for ((j, k, a, b2), cv) in rows.diff
            used[a, b2] || (cap[a, b2] = min(cap[a, b2], q[j, k] - cv))
        end
        for jk in CartesianIndices(p)
            used[jk] || (q[jk] = down(menus[jk], cap[jk]))
        end
    end
    all(q .<= p .+ 1e-9) && rows_ok(q, menus, rows) || error("P step result fails its checks")
    return q
end

"""Upper bound of scenario w for any per-OD prices on `menus` (module docstring): no stable matching, worker i on jk
paid the lowest level of menus[j, k] >= d_ij + c_jk, tasks allowed up to the top level.  Returns the MIP bound."""
function discrim_ub(w, menus, opt)
    n = w.n
    top = [last(menus[j, k]) for j in 1:n, k in 1:n]
    m = Model(opt)
    b = add_scenario_sparse!(m, w, w.scen[1], zeros(n, n), top; stability = :none)
    nothing_start!(w, b)
    pay = [up(menus[j, k], w.d[i, j] + w.c[j, k]) for (i, j, k) in b.trips]
    paid = AffExpr(0.0)
    for e in eachindex(b.trips), t in 1:w.T
        add_to_expression!(paid, pay[e], b.x[e, t])
    end
    @objective(m, Max, b.profit - paid)
    optimize!(m)
    st = JuMP.termination_status(m)
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    ops = feas ? op_stats_sparse(w, b, zeros(n, n)) : nothing
    return objective_bound(m), (status = string(st), optimal = st == MOI.OPTIMAL, value = feas ? objective_value(m) : NaN,
                                solve_s = solve_time(m), paid = feas ? value(paid) : NaN, ops = ops)
end

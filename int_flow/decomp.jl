"""
decomp.jl  —  TODO #3: decompositions of one fixed-price day (sparse tier block, integer rider split,
int_flow/README.md) built on the #2 result: with x and the binaries fixed, the rest is a network LP whose vertices
are integral (weekly/2026-10-31_11-06/TODO.md "#2 核对").

    r = cg_solve(w, P, opt)              column generation on x, then the restricted MIP (exact)
    r = benders_solve(w, P, opt)         Benders: master (x, β, z) MIP, subproblem = the rest as an LP (exact)

`w` is a one-scenario instance (`window_instance` / `scenario_view`), `P` the fixed price, `opt` an optimizer
factory, `tl` a time limit in seconds for the whole method.  Both return a NamedTuple: model and block holding the
final solution, value, bound, proved, a per-iteration log.

Column generation (cg_solve)
  Restricted problem = the full block with x[e, t] = 0 outside the active set (upper bound 0, every row kept), so it
  is a restriction of the full problem.  Start (`init = :d0`): the triples with d_ij = 0, the worker at the task's
  node (620 of the 4,310 allowed triples per period at price 4 on n = 30).
  LP phase   solve the restricted LP, read the reduced cost of every inactive x, activate those with rc > tol,
             repeat.  When none is left the restricted LP value is the full LP value UB.
  fix step   (`fixstep`) two restrictions of the MIP, each with time limit `fix_tl`; the first that reaches UB is
             optimal, otherwise the best solution starts the next step.
               :all   x, β, z integral in the final LP solution fixed at their value, MIP over the fractional ones;
                      the rest is the network LP of #2
               :zero  only the x at 0 fixed: the MIP over the support of the LP solution, every binary free
               :face  only the x with rc ≠ 0 fixed (at their LP value, a bound), every binary free.  Every solution
                      worth UB is an optimal LP solution, so it has these x at the same bound: when LP = MIP the
                      optimum of this restriction is UB
  MIP phase  solve the restricted MIP: LB, the value of a feasible solution of the full problem.  LB = UB, or
             LB > UB − 1 when the profit is integral, proves optimality.  Otherwise reduced-cost fixing with the
             final LP duals: a solution with x[e, t] ≥ 1 is worth at most UB + rc[e, t], so a solution better than
             LB only uses columns with rc ≥ LB − UB; activate those and solve the MIP again (exact).
  Pricing only reads reduced costs; there is no pricing problem to solve.

Benders (benders_solve)
  Master variables v = (x, β, z), the worker assignments and every binary.  Master = the rows of the block that
  only contain v and the worker flow (W, Gin): worker capacity and flow, the worker tier rows W − C ≤ Q1 z; plus
  η ≤ cuts.  Subproblem Q(v) = the whole block as an LP with v fixed (bikes, task pools, the pool rows of stable
  matching, the service rule with β fixed).  For integral v this is the network LP of #2: Q(v) is the exact value
  of the rest and its simplex vertex is integral, so Benders over v is exact with an LP subproblem.
    optimality cut   η ≤ Q(v̄) + g·(v − v̄),  g = reduced costs of the fixed v
    feasibility cut  y0 + y·v ≤ 0  from the Farkas ray of the infeasible subproblem (y0 = Σ rhs·ray over the
                     other rows and bounds; MOI convention: the ray certifies infeasibility when y0 + y·v̄ > 0)
  phase :lp   master LP until η̄ = Q(v̄) (then η̄ is the LP bound of the full block)
  phase :mip  master MIP, one cut per master solve; LB = best Q(v̄) of an integral v̄, UB = master bound
"""

include(joinpath(@__DIR__, "partial_int.jl"))
include(joinpath(@__DIR__, "lazy_stab.jl"))

using Gurobi        # activate!: the LP basis through the C API (other solvers: plain bound changes)

_proved(lb, ub, integral) = lb >= ub - 1e-6 * max(1.0, abs(ub)) || (integral && lb > ub - 1 + 1e-6)
_integral_objective(m) = all(isinteger, values(objective_function(m).terms)) &&
                         isinteger(objective_function(m).constant)

function _mip!(m, deadline)
    cap_time_limit!(m, deadline)
    optimize!(m)
    st = JuMP.termination_status(m)
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    bnd = try objective_bound(m) catch; Inf end
    return (status = string(st), value = feas ? objective_value(m) : -Inf, bound = bnd, solve_s = solve_time(m),
            nodes = try MOI.get(m, MOI.NodeCount()) catch; -1 end)
end

"""
Set the upper bound of the inactive columns `xs` (0 so far) to `ub` without losing the LP basis.  Gurobi keeps a
column with ub = 0 and rc > 0 nonbasic at its upper bound, so relaxing the bound alone starts the next solve at
x = ub, far from primal feasible (T = 100: 1.7e7 infeasibility, 17k dual simplex iterations to add 37 columns).
With Gurobi the columns are marked nonbasic at the lower bound instead: the old vertex stays primal feasible and the
caller re-solves by primal simplex.
"""
function activate!(m::Model, xs::AbstractVector{VariableRef}, ub::Float64)
    grb = solver_name(m) == "Gurobi" ? unsafe_backend(m) : nothing
    nc, nr = Ref{Cint}(0), Ref{Cint}(0)
    if grb !== nothing
        Gurobi.GRBgetintattr(grb, "NumVars", nc)
        Gurobi.GRBgetintattr(grb, "NumConstrs", nr)
        vb, cb = Vector{Cint}(undef, nc[]), Vector{Cint}(undef, nr[])
        ok = Gurobi.GRBgetintattrarray(grb, "VBasis", 0, nc[], vb) == 0 &&
             Gurobi.GRBgetintattrarray(grb, "CBasis", 0, nr[], cb) == 0
        ok || (grb = nothing)                  # no basis (not an LP solve)
    end
    foreach(x -> set_upper_bound(x, ub), xs)
    grb === nothing && return false
    Gurobi.GRBupdatemodel(grb)
    for x in xs
        vb[Gurobi.c_column(grb, optimizer_index(x))+1] = -1     # nonbasic at lower bound
    end
    return Gurobi.GRBsetintattrarray(grb, "VBasis", 0, nc[], vb) == 0 &&
           Gurobi.GRBsetintattrarray(grb, "CBasis", 0, nr[], cb) == 0
end

"""MIP solve after the groups the lazy callback added in the previous solve became ordinary rows (TODO #9)."""
_mipS!(m, S, deadline) = (S === nothing || flush_pending!(m, S); _mip!(m, deadline))

"""
Column generation on x, then the restricted MIP with reduced-cost fixing (module docstring).
`stab = :lazy` (TODO #9, int_flow/lazy_stab.jl): the stable-matching rows on demand.  LP phase: when no column has
rc > tol, the groups no z ∈ [0, 1] satisfies get their rows (re-solve by dual simplex), until neither columns nor
rows are added; the rows left out have dual 0 and their z reduced cost 0, so the end is the full LP optimum and the
reduced-cost fixing stays valid.  MIP phase: the remaining groups by the lazy callback.  `stab = :defer` (TODO #9,
11-04): the same LP phase, then every remaining group added as ordinary rows before the first MIP solve (no
callback; the z of these groups are not fixed by the fix step: their LP value means nothing).  `lp_only`: stop after the
LP phase (with `init = :all`: row generation on the full LP).  `sep_tol`: tolerance of separate_lp (tests: < −1 adds
every group in the LP phase).
"""
function cg_solve(w::TSInstance, P::AbstractMatrix, opt; init::Symbol = :d0, tol::Float64 = 1e-6,
                  tl::Float64 = Inf, fixstep::Bool = true, fix_tl::Float64 = 600.0, verbose::Bool = true,
                  stab::Symbol = :tier, lp_only::Bool = false, sep_tol::Float64 = 1e-7)
    init in (:d0, :all) || error("init must be :d0 or :all")
    stab in (:tier, :lazy, :defer) || error("stab must be :tier, :lazy or :defer")
    say(a...) = verbose && (println(a...); flush(stdout))
    S = nothing
    tb = @elapsed begin
        m = Model(opt)
        b = add_scenario_sparse!(m, w, w.scen[1], P, P; stability = stab === :tier ? :tier : :none)
        stab === :tier || (S = stab_init!(m, w, b))
        w.vterm > 0 ? continue_start!(w, b, 1, initial_state(w, b)) : nothing_start!(w, b)
        @objective(m, Max, b.profit)
    end
    t0 = time()
    deadline = t0 + tl
    start = value(v -> something(start_value(v), 0.0), b.profit)
    integral = _integral_objective(m)
    E, T = length(b.trips), w.T
    X = [b.x[e, t] for e in 1:E, t in 1:T]          # b.x is a DenseAxisArray
    active = [init === :all || w.d[i, j] == 0 for (i, j, k) in b.trips, _ in 1:T]
    for I in CartesianIndices(active)
        active[I] || set_upper_bound(X[I], 0.0)
    end
    relaxed = relax_ints!(b, :lp)
    S === nothing || relax_z!(S)
    log = NamedTuple[]
    rc = zeros(E, T)
    ub = NaN
    while true
        cap_time_limit!(m, deadline)
        optimize!(m)
        st = JuMP.termination_status(m)
        st == MOI.OPTIMAL || error("cg LP: $st")
        ub = objective_value(m)
        tr = @elapsed rc .= reduced_cost.(X)
        add = [I for I in CartesianIndices(rc) if !active[I] && rc[I] > tol]
        gs = Tuple{Int,Int}[]
        tsep = (S !== nothing && isempty(add)) ? @elapsed(gs = separate_lp(S, value; tol = sep_tol)) : 0.0
        push!(log, (phase = "lp", value = ub, solve_s = solve_time(m), rc_s = tr, active = count(active),
                    added = length(add), max_rc = maximum(rc[I] for I in CartesianIndices(rc) if !active[I]; init = -Inf),
                    sep_s = tsep, groups_added = length(gs), groups_active = S === nothing ? 0 : count(S.active)))
        say(@sprintf("cg lp %2d: %.4f  %.1f s (rc %.1f s)  active %d  +%d", length(log), ub, solve_time(m), tr,
                     count(active), length(add)),
            S === nothing ? "" : @sprintf("  stab groups %d (+%d, sep %.1f s)", count(S.active), length(gs), tsep))
        if !isempty(add)
            foreach(I -> active[I] = true, add)
            activate!(m, X[add], Float64(w.Q1)) && set_attribute(m, "Method", 0)     # primal simplex from the old vertex
        elseif !isempty(gs)
            add_groups!(m, S, gs)
            solver_name(m) == "Gurobi" && set_attribute(m, "Method", 1)   # dual simplex: the old basis stays dual feasible
        else
            break
        end
    end
    lp_s = time() - t0
    if lp_only
        return (model = m, block = b, value = NaN, bound = ub, lp_bound = ub, proved = false, log = log, build_s = tb,
                lp_s = lp_s, method_s = time() - t0, start = start, active = count(active), columns = E * T, stab = S)
    end
    # z of the groups without rows are free: not fixed by the fix step
    keep = vcat(X[active], bin_vars(b), S === nothing ? VariableRef[] : S.z[S.active])
    kv = value.(keep)                          # the LP solution, read before the model changes
    solver_name(m) == "Gurobi" && set_attribute(m, "Method", -1)
    restore_ints!(relaxed)
    if stab === :lazy
        restore_z!(S)
        lazy_stab!(m, S)
    elseif stab === :defer
        restore_z!(S)
        nleft = ngroups(S) - count(S.active)
        ta = @elapsed nr = add_groups!(m, S, [(g, t) for g in eachindex(S.node), t in 1:S.T if !S.active[g, t]])
        push!(log, (phase = "addback", value = NaN, solve_s = 0.0, add_s = ta, groups_added = nleft, rows = nr))
        say(@sprintf("cg addback: %d groups, %d stable-matching rows added in %.1f s", nleft, nr, ta))
    end
    lb = start
    nx = count(active)
    rca = rc[active]                           # reduced costs of the final LP, in the order of keep[1:nx]
    for level in (fixstep ? (:all, :zero, :face) : ())
        # fix step (module docstring); the rest stays integer
        fixv = [(v, round(a)) for (k, (v, a)) in enumerate(zip(keep, kv))
                if level === :all ? abs(a - round(a)) <= 1e-9 :
                   level === :zero ? k <= nx && abs(a) <= 1e-9 :
                   k <= nx && abs(rca[k]) > tol && abs(a - round(a)) <= 1e-9]
        foreach(((v, a),) -> fix(v, a; force = true), fixv)
        rf = _mipS!(m, S, min(deadline, time() + fix_tl))
        lb = max(lb, rf.value)
        push!(log, (phase = "fix_" * string(level), value = rf.value, bound = rf.bound, status = rf.status,
                    solve_s = rf.solve_s, nodes = rf.nodes, active = nx, added = 0, fixed = length(fixv),
                    free = length(keep) - length(fixv)))
        say(@sprintf("cg fix %s: %.4f (%s)  %.1f s  %d of %d x/binaries fixed at their LP value", level, rf.value,
                     rf.status, rf.solve_s, length(fixv), length(keep)))
        if _proved(lb, ub, integral)           # its solution stays in the model
            return (model = m, block = b, value = lb, bound = ub, lp_bound = ub, proved = true, log = log,
                    build_s = tb, lp_s = lp_s, method_s = time() - t0, start = start, active = nx,
                    columns = E * T, stab = S)
        end
        if rf.value > -Inf                     # start of the next step
            xs = all_variables(m)
            xv = value.(xs)
        end
        for (v, _) in fixv
            unfix(v)
            is_binary(v) || (set_lower_bound(v, 0.0); set_upper_bound(v, Float64(w.Q1)))
        end
        rf.value > -Inf && set_start_value.(xs, xv)
        set_time_limit_sec(m, isfinite(deadline) ? max(1.0, deadline - time()) : nothing)   # undo the fix_tl cap
    end
    r = _mipS!(m, S, deadline)
    lb = max(lb, r.value)
    push!(log, (phase = "mip", value = r.value, bound = r.bound, status = r.status, solve_s = r.solve_s,
                nodes = r.nodes, active = count(active), added = 0))
    say(@sprintf("cg mip 1: %.4f (bound %.4f, %s)  %.1f s  active %d", r.value, r.bound, r.status, r.solve_s,
                 count(active)))
    bound = min(ub, r.bound)
    if !_proved(lb, ub, integral)
        # keep the incumbent as the start of the second solve (read before the model changes)
        if r.value > -Inf
            xs = all_variables(m)
            xv = value.(xs)                 # not fused with set_start_value: that marks the model modified
            set_start_value.(xs, xv)
        end
        # reduced-cost fixing: columns that can be positive in a solution better than lb
        add = [I for I in CartesianIndices(rc) if !active[I] && rc[I] >= lb - ub - tol]
        for I in add
            active[I] = true
            set_upper_bound(X[I], Float64(w.Q1))
        end
        r2 = _mipS!(m, S, deadline)
        push!(log, (phase = "mip", value = r2.value, bound = r2.bound, status = r2.status, solve_s = r2.solve_s,
                    nodes = r2.nodes, active = count(active), added = length(add)))
        say(@sprintf("cg mip 2: %.4f (bound %.4f, %s)  %.1f s  active %d (+%d by rc >= %.4f)", r2.value, r2.bound,
                     r2.status, r2.solve_s, count(active), length(add), lb - ub))
        lb = max(lb, r2.value)
        # the second MIP is exact over the columns that can beat lb; the others are worth < lb
        bound = min(ub, max(r2.bound, lb))
    end
    return (model = m, block = b, value = lb, bound = bound, lp_bound = ub, proved = _proved(lb, bound, integral),
            log = log, build_s = tb, lp_s = lp_s, method_s = time() - t0, start = start, active = count(active),
            columns = E * T, stab = S)
end

master_vars(b) = vcat(x_vars(b), bin_vars(b))

"""Benders master: the block built without an optimizer, minus every row and variable outside (x, β, z, W, Gin)."""
function benders_master(w::TSInstance, P::AbstractMatrix)
    mm = Model()
    b = add_scenario_sparse!(mm, w, w.scen[1], P, P; stability = :tier)
    keep = Set(index.(vcat(master_vars(b), vec(collect(b.W)), vec(collect(b.Gin)))))
    drop = ConstraintRef[]
    for (F, S) in list_of_constraint_types(mm)
        F <: GenericAffExpr || continue
        for c in all_constraints(mm, F, S)
            all(v -> index(v) in keep, keys(constraint_object(c).func.terms)) || push!(drop, c)
        end
    end
    delete(mm, drop)
    delete(mm, [v for v in all_variables(mm) if !(index(v) in keep)])
    revenue = sum(sum(w.R .* w.scen[1][t]) for t in 1:w.T)
    η = @variable(mm, upper_bound = revenue, base_name = "eta")
    @objective(mm, Max, η)
    return mm, b, η, master_vars(b)
end

"""Farkas cut of an infeasible subproblem: (y0, y) with y0 + y·v ≤ 0 for every feasible v (module docstring)."""
function farkas_cut(ms::Model, vs::Vector{VariableRef})
    fixed = Set(index.(vs))
    y0 = 0.0
    for (F, S) in list_of_constraint_types(ms)
        S <: Union{MOI.LessThan{Float64},MOI.GreaterThan{Float64},MOI.EqualTo{Float64}} || continue
        for c in all_constraints(ms, F, S)
            F == VariableRef && S == MOI.EqualTo{Float64} && index(constraint_object(c).func) in fixed && continue
            y = dual(c)
            y == 0 || (y0 += MOI.constant(MOI.get(ms, MOI.ConstraintSet(), c)) * y)
        end
    end
    return y0, [dual(FixRef(v)) for v in vs]
end

"""Benders with the LP subproblem (module docstring).  `mopt`: optimizer of the master (default `opt`)."""
function benders_solve(w::TSInstance, P::AbstractMatrix, opt; mopt = opt, tl::Float64 = Inf, maxit::Int = 100_000,
                       tol::Float64 = 1e-6, keep_cuts::Bool = false, verbose::Bool = true)
    say(a...) = verbose && (println(a...); flush(stdout))
    tb = @elapsed begin
        mm, bm, η, vm = benders_master(w, P)
        set_optimizer(mm, mopt)
        ms = Model(opt)
        bs = add_scenario_sparse!(ms, w, w.scen[1], P, P; stability = :tier)
        @objective(ms, Max, bs.profit)
        relax_ints!(bs, :lp)
        vs = master_vars(bs)
        foreach(v -> fix(v, 0.0; force = true), vs)
        set_attribute(ms, "InfUnbdInfo", 1)
        set_attribute(ms, "DualReductions", 0)
    end
    t0 = time()
    deadline = t0 + tl
    integral = _integral_objective(ms)
    nv = length(vm)
    vcur = zeros(nv)                           # current fixed values of vs
    log = NamedTuple[]
    cuts = Any[]
    lb, ub, vbest = -Inf, Inf, nothing
    lp_bound = NaN
    sub_s = master_s = 0.0
    for phase in (:lp, :mip)
        undo = phase === :lp ? relax_integrality(mm) : nothing
        while time() < deadline && length(log) < maxit
            cap_time_limit!(mm, deadline)
            optimize!(mm)
            st = JuMP.termination_status(mm)
            st == MOI.OPTIMAL || (say("master: $st"); break)
            master_s += solve_time(mm)
            ηbar = value(η)
            vbar = value.(vm)
            phase === :mip && (vbar = round.(vbar); ub = min(ub, objective_bound(mm)))
            for k in 1:nv                      # only the ones that changed (most x stay 0)
                vbar[k] == vcur[k] || (fix(vs[k], vbar[k]; force = true); vcur[k] = vbar[k])
            end
            cap_time_limit!(ms, deadline)
            optimize!(ms)
            ss = JuMP.termination_status(ms)
            sub_s += solve_time(ms)
            if ss == MOI.OPTIMAL
                Q = objective_value(ms)
                g = reduced_cost.(vs)
                phase === :mip && Q > lb && (lb = Q; vbest = vbar)
                push!(log, (phase = string(phase), cut = "opt", eta = ηbar, Q = Q, lb = lb, ub = ub,
                            master_s = solve_time(mm), sub_s = solve_time(ms), nnz = count(!iszero, g)))
                done = phase === :lp ? ηbar <= Q + tol * max(1.0, abs(Q)) : _proved(lb, ub, integral)
                say(@sprintf("benders %s %3d: eta %.4f  Q %.4f  lb %.4f ub %.4f  master %.2f s sub %.2f s  nnz %d",
                             phase, length(log), ηbar, Q, lb, ub, solve_time(mm), solve_time(ms), count(!iszero, g)))
                phase === :lp && done && (lp_bound = ηbar)
                done && break
                ex = AffExpr(Q)
                for k in 1:nv
                    g[k] == 0 || add_to_expression!(ex, g[k], vm[k] - vbar[k])
                end
                @constraint(mm, η <= ex)
                keep_cuts && push!(cuts, (kind = :opt, c0 = Q - sum(g .* vbar), coef = g))
            elseif ss == MOI.INFEASIBLE
                y0, y = farkas_cut(ms, vs)
                viol = y0 + sum(y .* vbar)
                push!(log, (phase = string(phase), cut = "feas", eta = ηbar, Q = NaN, lb = lb, ub = ub,
                            master_s = solve_time(mm), sub_s = solve_time(ms), nnz = count(!iszero, y)))
                say(@sprintf("benders %s %3d: eta %.4f  sub infeasible, ray %.3e at v̄  master %.2f s sub %.2f s  nnz %d",
                             phase, length(log), ηbar, viol, solve_time(mm), solve_time(ms), count(!iszero, y)))
                viol > 1e-9 || error("Farkas ray does not cut off v̄ ($viol)")
                ex = AffExpr(y0)
                for k in 1:nv
                    y[k] == 0 || add_to_expression!(ex, y[k], vm[k])
                end
                @constraint(mm, ex <= 0)
                keep_cuts && push!(cuts, (kind = :feas, c0 = y0, coef = y))
            else
                say("subproblem: $ss"); break
            end
        end
        phase === :lp && undo()
    end
    if vbest !== nothing                       # leave the best solution in the subproblem model
        for (v, a) in zip(vs, vbest)
            fix(v, a; force = true)
        end
        set_time_limit_sec(ms, nothing)
        optimize!(ms)
    end
    return (model = ms, block = bs, value = lb, bound = ub, lp_bound = lp_bound, proved = _proved(lb, ub, integral),
            log = log, cuts = cuts, build_s = tb, method_s = time() - t0, master_s = master_s, sub_s = sub_s,
            nmaster = nv, master = mm)
end

"""
partial_int.jl  —  TODO #2: keep the integrality of only some variables of the sparse tier block
(add_scenario_sparse!, stability = :tier) on the integer rider split (int_flow/README.md).

    relaxed = relax_ints!(b, form)      form = :full | :xbin | :x | :lp | :nox | :bin
    cl = cleanup!(m, b)                 after :xbin: fix x and the binaries, re-solve the rest by simplex
    restore_ints!(relaxed); fix x       after :x: the exact model with that x (fixedx_value)
    drop_service_rule!(m, b)            TODO #8: delete the service rule rows (β stays, in no row)
    service_violations(w, b)            (i, t) where riders go unserved while bikes are left: Y < I and Y < D

Variable groups of the block:
  x                 worker assignments x[e, t]
  general integers  Ycum, Yod, YU, L, I, rA, mhat, Mt, mtil
  binaries          β (service rule), z (stability tiers)
:xbin keeps x and the binaries, :x keeps x only, :lp keeps nothing.  TODO #8: :nox keeps all but x (a task may be
split among workers from different nodes, task counts and bikes stay integral); :bin keeps the binaries only (the
bundling of a worker and a bike fully relaxed, the service rule and stability tiers exact).  The derivation in
weekly/2026-10-31_11-06/TODO.md ("#2 核对") says :xbin has the optimum of :full and that, with x and the binaries
fixed, the rest is a network LP with integral vertices; `cleanup!` returns such a vertex (dual simplex, no barrier).
"""

int_vars(b) = reduce(vcat, [vec(collect(v)) for v in (b.Ycum, b.Yod, b.YU, b.L, b.I, b.rA, b.mhat, b.Mt, b.mtil)
                            if v !== nothing])
bin_vars(b) = vcat(vec(collect(b.β)), b.z)
x_vars(b) = vec(collect(b.x))

"""Drop integrality by group; returns what was relaxed, for `restore_ints!`."""
function relax_ints!(b, form::Symbol)
    form in (:full, :xbin, :x, :lp, :nox, :bin) || error("form must be :full, :xbin, :x, :lp, :nox or :bin")
    ints = form in (:full, :nox) ? VariableRef[] : filter(is_integer, int_vars(b))
    bins = form in (:x, :lp) ? filter(is_binary, bin_vars(b)) : VariableRef[]
    xs = form in (:lp, :nox, :bin) ? filter(is_integer, x_vars(b)) : VariableRef[]
    foreach(unset_integer, ints)
    foreach(unset_integer, xs)
    for v in bins
        unset_binary(v)
        set_lower_bound(v, 0.0)
        set_upper_bound(v, 1.0)
    end
    return (ints = ints, bins = bins, xs = xs)
end

function restore_ints!(r)
    foreach(set_integer, r.ints)
    foreach(set_integer, r.xs)
    for v in r.bins
        delete_lower_bound(v)
        delete_upper_bound(v)
        set_binary(v)
    end
end

nfrac(vals; tol = 1e-6) = count(v -> abs(v - round(v)) > tol, vals)

"""
Delete the service rule rows Ycum ≥ ΣI − Q3(1 − β), Ycum ≥ D − Q3 β (every row with a β), so the platform may leave
riders unserved while bikes stand idle or are locked for tasks.  β stays as a free binary in no row (nothing_start!
still sets it).  Returns the number of rows deleted (2 per node and period).
"""
function drop_service_rule!(m::Model, b)
    βs = Set(index.(vec(collect(b.β))))
    drop = ConstraintRef[]
    for (F, S) in list_of_constraint_types(m)
        F <: GenericAffExpr || continue
        for c in all_constraints(m, F, S)
            any(v -> index(v) in βs, keys(constraint_object(c).func.terms)) && push!(drop, c)
        end
    end
    delete(m, drop)
    return length(drop)
end

"""(i, t) of a solution where the service rule fails: some riders unserved (Ycum < D) while bikes are left (Ycum < I)."""
function service_violations(w::TSInstance, b; tol = 1e-6)
    out = Tuple{Int,Int}[]
    for t in 1:w.T, i in 1:w.n
        y, I, D = value(b.Ycum[1, i, t]), value(b.I[1, i, t]), sum(w.scen[1][t][i, :])
        y < I - tol && y < D - tol && push!(out, (i, t))
    end
    return out
end

"""Fix x and the binaries at their rounded values, make the model an LP and re-solve it by simplex
(`lp_attrs`: Gurobi dual simplex; HiGHS takes "solver" => "simplex").
Returns the LP value and the fractional count of the general integers before and after."""
function cleanup!(m::Model, b; lp_attrs = ("Method" => 1,))
    before = nfrac(value.(int_vars(b)))
    keep = vcat(x_vars(b), bin_vars(b))
    vals = round.(value.(keep))
    for (v, a) in zip(keep, vals)
        is_integer(v) && unset_integer(v)
        is_binary(v) && unset_binary(v)
        fix(v, a; force = true)
    end
    foreach(v -> set_start_value(v, nothing), all_variables(m))
    foreach(a -> set_attribute(m, a.first, a.second), lp_attrs)
    optimize!(m)
    st = JuMP.termination_status(m)
    ok = st == MOI.OPTIMAL
    return (status = string(st), value = ok ? objective_value(m) : NaN, solve_s = solve_time(m),
            frac_before = before, frac_after = ok ? nfrac(value.(int_vars(b))) : -1)
end

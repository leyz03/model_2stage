"""
price_bnb.jl  —  baseline solver: branch and bound over the price menu (model_2stage)

Spec: model_2stage/04_algorithm.md

    res = price_bnb(inst, optimizer; inner = :pre, tl = 3600.0)
    res = price_bnb(inst, optimizer; sparse = true, rtol = 1e-4)     # real-data instances

`sparse = true` (src/sparse_scenario.jl): exact values by the utility-tier MILP and relaxations with
x only on allowed triples; MIP gaps may be positive, so every exact value keeps its MIP bound too
(a point's bound is the mean of its scenario bounds) and relaxations use their MIP bounds.
`rtol` prunes a box when its bound is within rtol * |best| of the best value.

Outer  best-first branch and bound over boxes of candidate prices, one interval per group.
       Bound of a box = mean over scenarios of the price-box relaxation (no stable matching,
       individual rationality at the box top, wages at the box bottom), which is ≥ the
       stable-matching value for every price in the box.
Inner  exact SAA value of a single price vector: one MILP per scenario with
       `stability = inner` (:full / :pre / :lazy), cached by price vector.
Menu   candidate prices per group (`candidate_prices`): every price below the group's
       lowest acceptance threshold min(d_ij + c_jk) gives the same outcome (nobody accepts),
       so only the lowest menu price is kept from that range; with a single group the
       optimum is at the lowest price or at a threshold d_ij + c_jk (between thresholds the
       profit only falls with the wage), so only those are kept.
"""

include(joinpath(@__DIR__, "sparse_scenario.jl"))

using Printf

"""Acceptance thresholds d_ij + c_jk of each price group, sorted."""
group_thresholds(inst::TSInstance) =
    [sort(unique(Float64(inst.d[i, j] + inst.c[j, k]) for i in 1:inst.n, j in 1:inst.n, k in 1:inst.n
                 if inst.group[j, k] == g)) for g in 1:inst.ngroups]

function candidate_prices(inst::TSInstance)
    menu = sort(inst.menu)
    out = Vector{Vector{Float64}}()
    for thr in group_thresholds(inst)
        keep = inst.ngroups == 1 ? [p for p in menu if p in thr] : [p for p in menu if p >= thr[1]]
        push!(out, sort(unique(vcat(menu[1], keep))))
    end
    return out
end

struct TimeUp <: Exception end

mutable struct BnBStats
    nodes::Int
    exact_points::Int
    exact_milps::Int
    exact_s::Float64           # wall time inside exact evaluations (build + solve)
    exact_solver_s::Float64    # solver time only
    relax_points::Int
    relax_milps::Int
    relax_s::Float64
    relax_solver_s::Float64
    lazy_cuts::Int
    max_nbin_exact::Int
    max_nbin_relax::Int
    exact_not_optimal::Int     # scenario MILPs stopped by their time limit (value is a lower bound)
    relax_not_optimal::Int     # relaxations stopped by their time limit (best bound used)
end
BnBStats() = BnBStats(0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0)

"""
    price_bnb(inst, optimizer; inner = :pre, tl = 3600.0, tol = 1e-6, log = true)

Returns a NamedTuple: best value and price vector, final upper bound, status
(:optimal or :time_limit), wall time, candidate list sizes and BnBStats.
"""
function price_bnb(inst::TSInstance, optimizer; inner::Symbol = :pre, tl::Float64 = 3600.0,
                   tol::Float64 = 1e-6, log::Bool = true, sparse::Bool = false, rtol::Float64 = 0.0)
    t0 = time()
    cands = candidate_prices(inst)
    thr = group_thresholds(inst)
    G, S = inst.ngroups, length(inst.scen)
    st = BnBStats()
    pmat(pv) = [pv[inst.group[j, k]] for j in 1:inst.n, k in 1:inst.n]
    deadline = t0 + tl
    # an evaluation that cannot finish by the deadline is abandoned (TimeUp), never half-counted
    checktime() = time() > deadline && throw(TimeUp())
    exact_cache = Dict{Vector{Float64},Float64}()
    exact_ub = Dict{Vector{Float64},Float64}()          # mean MIP bound of an exact value (= value when gap 0)
    point_ub = -Inf                                     # bounds of point boxes taken off the queue
    relax_cache = Dict{Tuple{Vector{Float64},Vector{Float64}},Float64}()

    function exact(pv)
        haskey(exact_cache, pv) && return exact_cache[pv]
        tw = time(); tot = 0.0; totub = 0.0
        P = pmat(pv)
        for s in 1:S
            checktime()
            if sparse
                val, _, _, info = solve_tier_sparse(inst, s, P, optimizer; deadline)
                info = merge(info, (ncuts = 0,))
                totub += info.bound
            else
                val, _, _, info = solve_scenario(inst, s, P, optimizer; stability = inner, deadline)
                totub += val
            end
            sparse && GC.gc()                   # free the Gurobi model (C memory) before the next one
            checktime()
            tot += val
            st.exact_solver_s += info.solve_s
            st.lazy_cuts += info.ncuts
            st.max_nbin_exact = max(st.max_nbin_exact, info.nbin)
            info.optimal || (st.exact_not_optimal += 1)
        end
        st.exact_points += 1; st.exact_milps += S; st.exact_s += time() - tw
        exact_ub[pv] = totub / S
        return exact_cache[pv] = tot / S
    end
    function relax(a, b)
        key = (a, b)
        haskey(relax_cache, key) && return relax_cache[key]
        tw = time(); tot = 0.0
        A, B = pmat(a), pmat(b)
        for s in 1:S
            checktime()
            val, info = (sparse ? solve_relax_sparse : solve_relax)(inst, s, A, B, optimizer; deadline)
            sparse && GC.gc()
            checktime()
            tot += val
            st.relax_solver_s += info.solve_s
            st.max_nbin_relax = max(st.max_nbin_relax, info.nbin)
            info.optimal || (st.relax_not_optimal += 1)
        end
        st.relax_points += 1; st.relax_milps += S; st.relax_s += time() - tw
        return relax_cache[key] = tot / S
    end
    # heuristic point of a box: per group the highest threshold inside [a_g, b_g]
    # (same acceptances as b_g at the lowest price that keeps them), else a_g
    function heuristic(lo, hi)
        [begin
             a, b = cands[g][lo[g]], cands[g][hi[g]]
             inside = [p for p in cands[g] if a <= p <= b && p in thr[g]]
             isempty(inside) ? a : maximum(inside)
         end for g in 1:G]
    end

    best, best_pv = -Inf, Float64[]
    consider(pv) = (v = exact(pv); v > best + tol && ((best, best_pv) = (v, pv)); v)
    slack() = rtol > 0 && isfinite(best) ? max(tol, rtol * abs(best)) : tol   # best = -Inf before the first point
    # `known` receives the box's upper bound as soon as it is computed, so a TimeUp in the
    # heuristic evaluation that follows still leaves a valid bound for the box
    function bound(lo, hi, known::Ref{Float64})
        st.nodes += 1
        if lo == hi
            pv = [cands[g][lo[g]] for g in 1:G]
            consider(pv)
            return exact_ub[pv]
        end
        ub = relax([cands[g][lo[g]] for g in 1:G], [cands[g][hi[g]] for g in 1:G])
        known[] = ub
        ub > best + slack() && consider(heuristic(lo, hi))
        return ub
    end

    lo0, hi0 = ones(Int, G), length.(cands)
    queue = Tuple{Float64,Vector{Int},Vector{Int}}[]
    status = :optimal
    pending = (Ref(Inf), lo0, hi0)                      # box being processed when time runs out
    try
        push!(queue, (bound(lo0, hi0, pending[1]), lo0, hi0))
        while !isempty(queue)
            k = argmax(first.(queue))
            ub, lo, hi = queue[k]
            deleteat!(queue, k)
            ub <= best + slack() && (empty!(queue); break)   # best-first: nothing left can win
            lo == hi && (point_ub = max(point_ub, ub); continue)   # a point: its bound is its exact bound
            pending = (Ref(ub), lo, hi)
            checktime()
            g = argmax(hi .- lo)
            mid = (lo[g] + hi[g]) ÷ 2
            for (l, h) in ((lo[g], mid), (mid + 1, hi[g]))
                clo, chi = copy(lo), copy(hi)
                clo[g], chi[g] = l, h
                cub = bound(clo, chi, Ref(Inf))
                cub > best + slack() && push!(queue, (cub, clo, chi))
            end
            log && @info @sprintf("bnb  nodes %d  open %d  best %.4f  ub %.4f  %.0f s",
                                  st.nodes, length(queue), best,
                                  isempty(queue) ? best : maximum(first.(queue)), time() - t0)
        end
    catch e
        e isa TimeUp || rethrow()
        status = :time_limit
        push!(queue, (pending[1][], pending[2], pending[3]))   # the unfinished box keeps its bound (Inf if none yet)
    end
    final_ub = max(best, point_ub, isempty(queue) ? -Inf : maximum(first.(queue)))
    return (value = best, prices = best_pv, ub = final_ub, status = status, seconds = time() - t0,
            candidates = length.(cands), menu = length(inst.menu), stats = st)
end

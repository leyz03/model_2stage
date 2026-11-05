"""
lag_td.jl  —  TODO #10 (2026-11-04): one evaluation of the Lagrangian time decomposition of src/lagrangian_td.jl
(`form = :sparse`) on the integer rider split (int_flow), with the blocks solved separately (one job each) and in a
chosen integrality form.

    cuts = lag_cuts(T, B)                         B balanced blocks of 1..T
    lpv, λ, keys = lag_lambda(v, P, cuts, opt)    joint LP of the linked blocks: its value and λ_k (boundary k = after
                                                  block k), with the state keys they belong to
    m, blk = lag_block(v, P, cuts, k, λ, opt; form)
        block k alone, objective profit_k + λ_k·y_k − λ_{k−1}·z_k; integrality as relax_ints!(·, form):
        :full every integer, :xbin x and β, z (same block optimum, TODO #2), :bin only β, z (a relaxation),
        :lp none (also the state copies; with the LP multipliers Σ_k = LP value)

    L(λ) = Σ_k (proven bound of block k)  ≥  the optimum, for every λ, also for a time-limited or a relaxed block.

v is a one-scenario instance (scenario_view / window_instance), so s = 1 everywhere.  The incoming copies z of a block
stay integral in :full, :xbin and :bin (every state of a feasible plan is integral, so this keeps the bound valid).
"""

include(joinpath(@__DIR__, "..", "src", "lagrangian_td.jl"))      # also sparse_scenario.jl, fixed_price.jl
include(joinpath(@__DIR__, "partial_int.jl"))

const LAG_FORMS = (:full, :xbin, :bin, :lp)

"""B balanced blocks of 1..T (lengths differ by at most one)."""
function lag_cuts(T::Int, B::Int)
    1 <= B <= T || error("need 1 ≤ B ≤ T, got B = $B, T = $T")
    lens = [T ÷ B + (k <= T % B ? 1 : 0) for k in 1:B]          # as td_cuts, with B fixed instead of the length
    ends = cumsum(lens)
    return [(e - l + 1, e) for (l, e) in zip(lens, ends)]
end

"""Joint LP of the linked blocks: (LP value, λ per boundary, state keys per boundary, solve seconds)."""
function lag_lambda(v::TSInstance, P::AbstractMatrix, cuts, optimizer)
    t0 = time()
    lpv, duals, m, blks = td_lp_multipliers(v, 1, P, cuts, optimizer; form = :sparse)
    for k in 1:length(cuts)-1
        blks[k].ykeys == blks[k+1].zkeys || error("boundary $k: outgoing and incoming keys differ")
    end
    return lpv, duals, [blks[k].ykeys for k in 1:length(cuts)-1], time() - t0
end

"""Block k of `cuts` in its own model with the Lagrangian objective at λ, integrality by `form`."""
function lag_block(v::TSInstance, P::AbstractMatrix, cuts, k::Int, λ, optimizer; form::Symbol = :full)
    form in LAG_FORMS || error("form must be one of $LAG_FORMS")
    B = length(cuts)
    m = Model(optimizer)
    ta, tb = cuts[k]
    blk = add_td_block!(m, v, 1, P, ta, tb; form = :sparse)
    k < B && length(λ[k]) != length(blk.y) && error("block $k: $(length(blk.y)) outgoing states, λ has $(length(λ[k]))")
    k > 1 && length(λ[k-1]) != length(blk.z) && error("block $k: $(length(blk.z)) incoming states, λ has $(length(λ[k-1]))")
    obj = AffExpr(0.0)
    add_to_expression!(obj, blk.b.profit)
    k < B && for (x, l) in zip(blk.y, λ[k])
        add_to_expression!(obj, l, x)
    end
    k > 1 && for (x, l) in zip(blk.z, λ[k-1])
        add_to_expression!(obj, -l, x)
    end
    @objective(m, Max, obj)
    if form === :lp
        relax_integrality(m)
    elseif form !== :full
        relax_ints!(blk.b, form)
    end
    return m, blk
end

# λ and its keys as JSON-friendly data, and back (the keys are checked against the block that uses them)
lag_keys_json(keys) = [Dict("f" => string(k.f), "idx" => collect(k.idx)) for k in keys]
lag_same_keys(keys, js) = length(keys) == length(js) &&
    all(string(k.f) == j["f"] && collect(k.idx) == j["idx"] for (k, j) in zip(keys, js))

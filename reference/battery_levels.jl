"""
battery_levels.jl  —  LIME-OPR v2 with K discrete battery levels (EXP-BATTLVL)

Spec: experiments/EXP-BATTLVL-battery-levels/FORMULATION.md

Levels ℓ = 1..K, 1 = fullest.  Levels 1..K-1 can be rented; level K is the old
"unavailable" state and can only leave it through a swap task.  K = 2 is exactly
model_v2 (one usable level A, one dead level U); the size sweep checks that the
generic builder reproduces model_v2.jl's state bits and variable/constraint counts
at K = 2 before reading anything into K ≥ 3.

Behavioural rules added on top of model_v2 (all in FORMULATION.md):
  * riders take the fullest usable bike first (H-first):  for every ℓ < K
        Y^{≤ℓ}_i = min{ Σ_{ℓ'≤ℓ} I^{ℓ'}_i , D_i },   one β per (ℓ, i);
  * the platform still chooses which ODs the level-ℓ riders go to (integer, as D3);
  * a ride i→j turns a level-ℓ bike into level ℓ' with scaled count q[ℓ,ℓ',i,j]
    (Σ_ℓ' q = Γ, q = 0 for ℓ' < ℓ); Γ = lcm of the denominators, as SC-1;
  * swaps take a level-K bike to level 1;
  * moves (j ≠ k):  move = :top  only level-1 bikes are moved (pools/pipelines as v2);
                    move = :any  any usable level, level kept, pools and the
                                 task pipeline carry the level tag.

Like model_v2.jl this is "standard" SDDiP: binary expansion of every live state
component, no A2 slacks.
"""

# frozen copy (2026-10-03) from lime-opr/experiments/EXP-BATTLVL-battery-levels/; only this include path changed
include(joinpath(@__DIR__, "model_v2.jl"))

using Random, Distributions

struct BLInstance
    name::String
    n::Int
    T::Int
    K::Int
    move::Symbol                       # :top or :any
    Γ::Int
    q::Array{Int,4}                    # q[ℓ, ℓ2, i, j], ℓ = 1..K-1
    R::Matrix{Float64}
    p::Matrix{Float64}
    Cp::Float64
    d::Matrix{Int}
    c::Matrix{Int}
    Cmax::Int
    Hmax::Int
    Q1::Int
    Q2::Float64
    Q3::Int
    A0::Matrix{Int}                    # A0[ℓ, i], ℓ = 1..K-1
    U0::Vector{Int}
    W0::Vector{Int}
    Ω::Vector{Vector{Matrix{Int}}}
    prob::Vector{Vector{Float64}}
    comps::Vector{StateComponent}      # live components only
    θ::Vector{Float64}
end

n_state_bits(inst::BLInstance) = sum(c.bits for c in inst.comps)

# ── level transition probabilities ───────────────────────────────────────────

"""
    level_transition(Δ, θL, K) -> Matrix{Rational{BigInt}}  (K-1) × K

State of charge uniform within each band.  Band K (dead) is [0, θL); the usable
range [θL, 1] is cut into K-1 equal bands, band 1 on top.  A ride consumes Δ
(fraction of a full battery).  Entry [ℓ, ℓ2] is the probability that a bike that
left in band ℓ arrives in band ℓ2: the share of [a-Δ, a+w-Δ] inside band ℓ2,
where [a, a+w] is band ℓ.  With Δ ≤ w this is 1-Δ/w on the diagonal and Δ/w one
band down, the derivation recorded for the experiment.
"""
function level_transition(Δ::Rational, θL::Rational, K::Int)
    w = (1 - θL) // (K - 1)
    lo(ℓ) = ℓ == K ? big(-10) // 1 : θL + (K - 1 - ℓ) * w   # band K extends below 0
    hi(ℓ) = ℓ == K ? θL : θL + (K - ℓ) * w
    P = zeros(Rational{BigInt}, K - 1, K)
    for ℓ in 1:K-1, ℓ2 in 1:K
        a, b = lo(ℓ) - Δ, hi(ℓ) - Δ
        P[ℓ, ℓ2] = max(0, min(b, hi(ℓ2)) - max(a, lo(ℓ2))) // w
    end
    all(sum(P[ℓ, :]) == 1 for ℓ in 1:K-1) || error("transition rows do not sum to 1")
    return P
end

"""
Round every row of P to the 1/grid lattice (largest remainder, row sum kept at 1),
then Γ = lcm of the denominators that survive (SC-1: the smallest exact scale).
Exact rationals already on the lattice are left untouched.
"""
function round_to_grid(P::Matrix{Rational{BigInt}}, grid::Int)
    out = similar(P)
    for ℓ in axes(P, 1)
        raw = P[ℓ, :] .* grid
        fl = floor.(BigInt, raw)
        short = grid - sum(fl)
        order = sortperm(raw .- fl; rev = true)
        for m in 1:short
            fl[order[m]] += 1
        end
        out[ℓ, :] = fl .// grid
    end
    return out
end

"""
    phi01_matched(φ, w) -> Matrix{Float64}

A 0/1 failure matrix with (close to) the same flow-weighted failure rate as φ: walk the
ODs in decreasing φ_ij (longest rides first; ties by smaller weight) and switch an OD
to φ = 1 whenever that brings the switched weight share closer to Σ w φ / Σ w.
"""
function phi01_matched(φ::AbstractMatrix, w::AbstractMatrix)
    target = sum(w .* φ)
    order = sort(vec(collect(CartesianIndices(φ))); by = I -> (-φ[I], w[I]))
    out = zeros(size(φ))
    acc = 0.0
    for I in order
        (w[I] > 0 && abs(acc + w[I] - target) < abs(acc - target)) || continue
        out[I] = 1.0
        acc += w[I]
    end
    return out
end

"""Expected ride volume per OD over the horizon (Σ_t λ_ij^t), the weight for phi01_matched."""
toy_od_weight(path, variant) =
    sum(_mat(l, Float64) for l in JSON.parsefile(path)["variants"][variant]["lambda_ijt"])

# ── state components and their reachability bounds (ST-6′ generalised) ───────

bits_for(ub) = ub <= 0 ? 0 : ceil(Int, log2(ub + 1))

function derive_components(n, T, K, Γ, q, c, d, Cmax, Hmax, Q1, Q3, U0, Ω, move)
    N = 1:n
    dbar = [maximum(ω[i, j] for ω in Ω[t]) for t in 1:T, i in N, j in N]
    db(t, i, j) = 1 <= t <= T ? dbar[t, i, j] : 0
    # largest scaled inflow into level ℓ2 per ride i→j, over the departure level
    qmax = [maximum(q[ℓ, ℓ2, i, j] for ℓ in 1:K-1) for ℓ2 in 1:K, i in N, j in N]
    B = Γ * Q3
    τ = [d[i, j] + c[j, k] for i in N, j in N, k in N]
    comps = StateComponent[]
    add!(g, idx, ub) = (b = bits_for(ub); b > 0 && push!(comps, StateComponent(g, idx, b)))

    for ℓ in 1:K-1, j in N
        add!(:A, (ℓ, j), B)
    end
    for j in N
        ub = maximum(min(B, Γ * U0[j] + sum(qmax[K, i, j] * db(s, i, j) for s in 1:t-1, i in N; init = 0))
                     for t in 1:T)
        add!(:U, (j,), ub)
    end
    for ℓ2 in 1:K, j in N, r in 2:Cmax
        ub = maximum(min(B, sum(qmax[ℓ2, i, j] * db(t - m, i, j)
                                for m in 0:Cmax-r for i in N if c[i, j] == r + m; init = 0))
                     for t in 1:T)
        add!(:P, (ℓ2, j, r), ub)
    end
    for k in N
        add!(:W, (k,), Q1)
    end
    tags = move === :any ? (1:K-1) : (1:1)
    for ℓ in tags, k in N, r in 2:Hmax
        # tag ℓ > 1 only comes from moves (j ≠ k); swaps always finish at level 1
        reach = any(τ[i, j, k] >= r for i in N, j in N if ℓ == 1 || j != k)
        add!(:G, move === :any ? (ℓ, k, r) : (k, r), reach ? Q1 : 0)
    end
    for ℓ in tags, j in N, k in N
        add!(:Mrem, move === :any ? (ℓ, j, k) : (j, k), (ℓ == 1 || j != k) ? Q3 : 0)
    end
    return comps
end

"""(IC-2): -Σ_{s≥t} rev^max(s), the per-stage revenue capped by B_total rides."""
function initial_cut_bounds(R, Ω, B_total, T)
    n = size(R, 1)
    θ = zeros(T + 1)
    for t in T:-1:1
        caps = sort([(R[i, j], maximum(ω[i, j] for ω in Ω[t])) for i in 1:n, j in 1:n][:]; rev = true)
        room, rev = B_total, 0.0
        for (r, cap) in caps
            take = min(cap, room)
            rev += r * take
            room -= take
            room <= 0 && break
        end
        θ[t] = θ[t+1] - rev
    end
    return θ
end

"""
Assemble a BLInstance.  `Δ[i,j]` is the energy a ride i→j consumes (fraction of a
full battery), `θL` the swap threshold.  Initially every usable bike sits on level 1
(the v2 convention that the day starts fully charged).
"""
function make_bl_instance(; name, K, move = :top, Δ, θL, grid, R, p, Cp, d, c,
                          A0, U0, W0, Ω, prob)
    n, T = size(R, 1), length(Ω)
    N = 1:n
    Pq = [round_to_grid(level_transition(Δ[i, j], θL, K), grid) for i in N, j in N]
    Γ = Int(foldl(lcm, (denominator(x) for P in Pq for x in P); init = big(1)))
    q = zeros(Int, K - 1, K, n, n)
    for i in N, j in N, ℓ in 1:K-1, ℓ2 in 1:K
        v = Γ * Pq[i, j][ℓ, ℓ2]
        isinteger(v) || error("Γ does not clear the denominators")
        q[ℓ, ℓ2, i, j] = Int(v)
    end
    Cmax = maximum(c)
    Hmax = maximum(d[i, j] + c[j, k] for i in N, j in N, k in N)
    Q1, Q3 = sum(W0), sum(A0) + sum(U0)
    Q2 = maximum(p) + maximum(d) + maximum(c)
    all(maximum(sum(ω[i, :]) for ω in Ω[t]) <= Q3 for t in 1:T, i in N) ||
        error("(SC-5) violated: some node demand exceeds Q3")
    A0lev = zeros(Int, K - 1, n)
    A0lev[1, :] = A0
    comps = derive_components(n, T, K, Γ, q, c, d, Cmax, Hmax, Q1, Q3, U0, Ω, move)
    return BLInstance(name, n, T, K, move, Γ, q, Float64.(R), Float64.(p), Float64(Cp),
                      d, c, Cmax, Hmax, Q1, Float64(Q2), Q3, A0lev, U0, W0, Ω, prob, comps,
                      initial_cut_bounds(R, Ω, Q3, T))
end

"""
Toy (model_v2/toy/toy_instance.json): φ = 0.15 is read as Δ/0.8 with θL = 0.2.
`phi` overrides the instance's φ matrix (used for the φ ∈ {0,1} comparison at K = 2).
"""
function load_toy_bl(path, variant; K, move = :top, θL = 1 // 5, grid = 20, phi = nothing, tag = "")
    raw = JSON.parsefile(path)
    det, var = raw["deterministic"], raw["variants"][variant]
    φ = phi === nothing ? _mat(det["phi_ij"], Float64) : Float64.(phi)
    Δ = [rationalize(BigInt, x; tol = 1e-12) * (1 - θL) for x in φ]
    Ω = [[_mat(a, Int) for a in st] for st in var["Omega"]]
    return make_bl_instance(; name = "$(variant)$(tag)/K$(K)/$(move)", K, move, Δ, θL, grid,
        R = _mat(det["R_ij"], Float64), p = _mat(det["p_jk"], Float64), Cp = det["C_p"],
        d = _mat(det["d_ij"], Int), c = _mat(det["c_ij"], Int),
        A0 = Int.(det["A0"]), U0 = Int.(det["U0"]), W0 = Int.(det["W0"]),
        Ω, prob = [Float64.(p) for p in var["probabilities"]])
end

"""
Paris rung (sddip/instances/real/*.json) in v2 form: a frozen tree with a
deterministic stage 1 (round half up of λ¹) and `Nb` Poisson atoms per later stage.
φ_ij is read as Δ_ij/(1-θL) and every transition probability is rounded to 1/grid.
"""
function load_real_bl(path; K, move = :top, θL = 1 // 5, grid = 20, Nb = 20, seed = 20260912,
                      phi01 = false)
    raw = JSON.parsefile(path)
    n, T = raw["dims"]["n"], raw["dims"]["T"]
    φ = _mat(raw["phi_ij"], Float64)
    if phi01
        φ = phi01_matched(φ, sum(_mat(l, Float64) for l in raw["lambda_ijt"]))
    end
    Δ = [rationalize(BigInt, x; tol = 1e-9) * (1 - θL) for x in φ]
    λ = [_mat(raw["lambda_ijt"][t], Float64) for t in 1:T]
    rng = MersenneTwister(seed)
    Ω = [[floor.(Int, λ[1] .+ 0.5)]]
    for t in 2:T
        push!(Ω, [[rand(rng, Poisson(λ[t][i, j])) for i in 1:n, j in 1:n] for _ in 1:Nb])
    end
    prob = vcat([[1.0]], [fill(1 / Nb, Nb) for _ in 2:T])
    tag = splitext(basename(path))[1][1:3] * (phi01 ? "-phi01" : "")
    return make_bl_instance(; name = "$(tag)/K$(K)/$(move)", K, move, Δ, θL, grid,
        R = _mat(raw["R_ij"], Float64), p = _mat(raw["p_jk"], Float64), Cp = raw["econ"]["C_p"],
        d = _mat(raw["d_ij"], Int), c = _mat(raw["c_ij"], Int),
        A0 = Int.(raw["init"]["A0"]), U0 = Int.(raw["init"]["U0"]), W0 = Int.(raw["init"]["W0"]),
        Ω, prob)
end

function initial_component_value(inst::BLInstance, c::StateComponent)
    c.group === :A && return inst.Γ * inst.A0[c.index[1], c.index[2]]
    c.group === :U && return inst.Γ * inst.U0[c.index[1]]
    c.group === :W && return inst.W0[c.index[1]]
    return 0
end

function initial_bits(inst::BLInstance)
    out = Int[]
    for c in inst.comps
        v = initial_component_value(inst, c)
        v < 2^c.bits || error("initial value $v of $(c.group)$(c.index) needs > $(c.bits) bits")
        append!(out, [(v >> l) & 1 for l in 0:c.bits-1])
    end
    return out
end

# ── the policy graph ─────────────────────────────────────────────────────────

function build_bl_model(inst::BLInstance; optimizer = nothing)
    n, T, K, Γ = inst.n, inst.T, inst.K, inst.Γ
    N, Lu = 1:n, 1:inst.K-1                       # nodes, usable levels
    Q1, Q2, Q3 = inst.Q1, inst.Q2, inst.Q3
    Cmax, Hmax, q, d, c = inst.Cmax, inst.Hmax, inst.q, inst.d, inst.c
    anymove = inst.move === :any
    tags = anymove ? Lu : (1:1)
    τ = [d[i, j] + c[j, k] for i in N, j in N, k in N]
    u = [inst.p[j, k] - d[i, j] - c[j, k] for i in N, j in N, k in N]
    # (j,k,r) triples a started task can enter: some worker origin i has τ_ijk = r
    jkr = [(j, k, r) for j in N, k in N, r in 1:Hmax if any(τ[i, j, k] == r for i in N)][:]
    tagok(ℓ, j, k) = ℓ == 1 || j != k              # swaps carry tag 1 only

    ranges = UnitRange{Int}[]
    lookup = Dict{Tuple{Symbol,Tuple},Int}()
    pos = 0
    for (a, comp) in enumerate(inst.comps)
        push!(ranges, pos+1:pos+comp.bits)
        lookup[(comp.group, comp.index)] = a
        pos += comp.bits
    end
    nbits = pos
    bits0 = initial_bits(inst)

    return SDDP.LinearPolicyGraph(;
        stages = T, sense = :Min, lower_bound = inst.θ[2], optimizer = optimizer,
    ) do sp, t
        @variable(sp, λ[k = 1:nbits], Bin, SDDP.State, initial_value = bits0[k])
        @constraint(sp, [k = 1:nbits], λ[k].in >= 0)
        @constraint(sp, [k = 1:nbits], λ[k].in <= 1)

        function dec(group, idx, side)
            a = get(lookup, (group, idx), 0)
            a == 0 && return AffExpr(0.0)
            vars = side === :in ? [λ[k].in for k in ranges[a]] : [λ[k].out for k in ranges[a]]
            return @expression(sp, sum(2^(l - 1) * vars[l] for l in eachindex(vars)))
        end
        inA(ℓ, j) = dec(:A, (ℓ, j), :in)
        inU(j) = dec(:U, (j,), :in)
        inW(k) = dec(:W, (k,), :in)
        inP(ℓ2, j, r) = r > Cmax ? AffExpr(0.0) : dec(:P, (ℓ2, j, r), :in)
        gkey(ℓ, k, r) = anymove ? (ℓ, k, r) : (k, r)
        mkey(ℓ, j, k) = anymove ? (ℓ, j, k) : (j, k)
        inG(ℓ, k, r) = r > Hmax ? AffExpr(0.0) : dec(:G, gkey(ℓ, k, r), :in)
        inMrem(ℓ, j, k) = dec(:Mrem, mkey(ℓ, j, k), :in)

        @variables(sp, begin
            0 <= Ycum[Lu, N] <= Q3, Int            # Y^{≤ℓ}_i: riders served by levels 1..ℓ
            0 <= Yod[Lu, N, N] <= Q3, Int          # level-ℓ rides i→j
            0 <= L[N] <= Q3, Int
            0 <= I[Lu, N] <= Q3, Int
            0 <= rA[Lu, N] <= Γ - 1, Int
            β[Lu, N], Bin
            0 <= mhat[ℓ in tags, j in N, k in N; tagok(ℓ, j, k)] <= Q3, Int
            0 <= Mt[ℓ in tags, j in N, k in N; tagok(ℓ, j, k)] <= Q3, Int
            0 <= mtil[N, N] <= Q3, Int
            0 <= x[N, N, N] <= Q1, Int
            0 <= v[N, N, N] <= Q3
            0 <= s[N] <= Q2
            δ[N, N, N], Bin
            η[N, N, N], Bin
            ζ[N], Bin
        end)
        Ylev(ℓ, i) = ℓ == 1 ? 1.0 * Ycum[1, i] : Ycum[ℓ, i] - Ycum[ℓ-1, i]
        Mjk(j, k) = sum(Mt[ℓ, j, k] for ℓ in tags if tagok(ℓ, j, k))

        # ── H-first service: Y^{≤ℓ} = min{Σ_{ℓ'≤ℓ} I^{ℓ'}, D} for every ℓ ──────
        @constraint(sp, [ℓ in Lu, i in N], inA(ℓ, i) == Γ * I[ℓ, i] + rA[ℓ, i])   # (D1) per level
        @constraint(sp, [ℓ in Lu, i in N], Ycum[ℓ, i] <= sum(I[l2, i] for l2 in 1:ℓ))
        @constraint(sp, cYD[ℓ in Lu, i in N], Ycum[ℓ, i] <= 0)
        @constraint(sp, [ℓ in Lu, i in N], Ycum[ℓ, i] >= sum(I[l2, i] for l2 in 1:ℓ) - Q3 * (1 - β[ℓ, i]))
        @constraint(sp, cYDb[ℓ in Lu, i in N], Ycum[ℓ, i] + Q3 * β[ℓ, i] >= 0)
        @constraint(sp, [ℓ in Lu, i in N], sum(Yod[ℓ, i, j] for j in N) == Ylev(ℓ, i))
        @constraint(sp, cYod[i in N, j in N], sum(Yod[ℓ, i, j] for ℓ in Lu) <= 0)
        @constraint(sp, cL[i in N], L[i] + Ycum[K-1, i] == 0)

        # ── posting: swaps lock dead bikes, moves lock usable bikes of their tag ──
        @constraint(sp, [j in N], Γ * mhat[1, j, j] <= inU(j))
        @constraint(sp, [ℓ in tags, j in N],
            Γ * sum(mhat[ℓ, j, k] for k in N if k != j) <= inA(ℓ, j) - Γ * Ylev(ℓ, j))

        # ── pools and matching capacity ──────────────────────────────────────
        @constraint(sp, [ℓ in tags, j in N, k in N; tagok(ℓ, j, k)],
            Mt[ℓ, j, k] == inMrem(ℓ, j, k) + mhat[ℓ, j, k])
        @constraint(sp, [j in N, k in N], mtil[j, k] == sum(x[i, j, k] for i in N))
        @constraint(sp, [j in N, k in N], mtil[j, k] <= Mjk(j, k))
        @constraint(sp, [i in N], sum(x[i, j, k] for j in N, k in N) <= inW(i))
        if anymove
            # which tag a started task carries (optimistic, like the matching itself)
            @variable(sp, 0 <= mtl[ℓ in tags, j in N, k in N; tagok(ℓ, j, k)] <= Q3, Int)
            @variable(sp, 0 <= ytag[ℓ in tags, e in eachindex(jkr); tagok(ℓ, jkr[e][1], jkr[e][2])] <= Q1, Int)
            @constraint(sp, [ℓ in tags, j in N, k in N; tagok(ℓ, j, k)], mtl[ℓ, j, k] <= Mt[ℓ, j, k])
            @constraint(sp, [j in N, k in N], sum(mtl[ℓ, j, k] for ℓ in tags if tagok(ℓ, j, k)) == mtil[j, k])
            @constraint(sp, [e in eachindex(jkr)],
                sum(ytag[ℓ, e] for ℓ in tags if tagok(ℓ, jkr[e][1], jkr[e][2])) ==
                sum(x[i, jkr[e][1], jkr[e][2]] for i in N if τ[i, jkr[e][1], jkr[e][2]] == jkr[e][3]))
            @constraint(sp, [ℓ in tags, j in N, k in N; tagok(ℓ, j, k)],
                sum(ytag[ℓ, e] for e in eachindex(jkr) if jkr[e][1] == j && jkr[e][2] == k) == mtl[ℓ, j, k])
        end
        started(ℓ, j, k) = anymove ? mtl[ℓ, j, k] : mtil[j, k]

        # ── stable matching (S1)-(S6) on the aggregate pool ──────────────────
        for i in N, j in N, k in N
            H = sum(x[i2, j, k] for i2 in N if d[i2, j] <= d[i, j])
            @constraint(sp, v[i, j, k] <= H)
            @constraint(sp, v[i, j, k] <= Mjk(j, k))
            @constraint(sp, v[i, j, k] <= Q3 * δ[i, j, k])
            @constraint(sp, v[i, j, k] >= Mjk(j, k) - Q3 * (1 - δ[i, j, k]))
            @constraint(sp, x[i, j, k] <= Q1 * η[i, j, k])
            @constraint(sp, u[i, j, k] >= s[i] - Q2 * (1 - η[i, j, k]))
            @constraint(sp, s[i] >= u[i, j, k] - Q2 * δ[i, j, k])
        end
        @constraint(sp, [i in N], sum(x[i, j, k] for j in N, k in N) >= inW(i) - Q1 * (1 - ζ[i]))
        @constraint(sp, [i in N], s[i] <= Q2 * ζ[i])

        # ── outgoing pipelines ───────────────────────────────────────────────
        function pipe(inflow, carry)
            e = copy(carry)
            for (coef, var) in inflow
                coef == 0 || add_to_expression!(e, coef, var)
            end
            return e
        end
        P = Dict((ℓ2, j, r) => pipe([(q[ℓ, ℓ2, i, j], Yod[ℓ, i, j]) for ℓ in Lu for i in N if c[i, j] == r],
                                    inP(ℓ2, j, r + 1)) for ℓ2 in 1:K, j in N, r in 1:Cmax)
        G = if anymove
            Dict((ℓ, k, r) => pipe([(1, ytag[ℓ, e]) for e in eachindex(jkr)
                                    if jkr[e][2] == k && jkr[e][3] == r && tagok(ℓ, jkr[e][1], k)],
                                   inG(ℓ, k, r + 1)) for ℓ in tags, k in N, r in 1:Hmax)
        else
            Dict((1, k, r) => pipe([(1, x[i, j, k]) for i in N, j in N if τ[i, j, k] == r],
                                   inG(1, k, r + 1)) for k in N, r in 1:Hmax)
        end

        # ── linking-state transitions ────────────────────────────────────────
        if t < T
            for ℓ in Lu, j in N
                moved = ℓ in tags ? Γ * sum(mhat[ℓ, j, k] for k in N if k != j) : AffExpr(0.0)
                back = ℓ in tags ? Γ * G[(ℓ, j, 1)] : AffExpr(0.0)
                @constraint(sp, dec(:A, (ℓ, j), :out) ==
                    inA(ℓ, j) - Γ * Ylev(ℓ, j) - moved + P[(ℓ, j, 1)] + back)
            end
            @constraint(sp, [j in N], dec(:U, (j,), :out) == inU(j) - Γ * mhat[1, j, j] + P[(K, j, 1)])
            @constraint(sp, [k in N], dec(:W, (k,), :out) ==
                inW(k) - sum(x[k, i, j] for i in N, j in N) + sum(G[(ℓ, k, 1)] for ℓ in tags))
            @constraint(sp, [ℓ in tags, j in N, k in N; tagok(ℓ, j, k)],
                dec(:Mrem, mkey(ℓ, j, k), :out) == Mt[ℓ, j, k] - started(ℓ, j, k))
            @constraint(sp, [ℓ2 in 1:K, j in N, r in 2:Cmax], dec(:P, (ℓ2, j, r), :out) == P[(ℓ2, j, r)])
            @constraint(sp, [ℓ in tags, k in N, r in 2:Hmax], dec(:G, gkey(ℓ, k, r), :out) == G[(ℓ, k, r)])
        else
            @constraint(sp, [k = 1:nbits], λ[k].out == 0)
        end

        @stageobjective(sp,
            -sum(inst.R[i, j] * Yod[ℓ, i, j] for ℓ in Lu, i in N, j in N) +
            inst.Cp * sum(L[i] for i in N) +
            sum(inst.p[j, k] * mtil[j, k] for j in N, k in N))

        SDDP.parameterize(sp, inst.Ω[t], inst.prob[t]) do ω
            for i in N
                Di = sum(ω[i, j] for j in N)
                for ℓ in Lu
                    set_normalized_rhs(cYD[ℓ, i], Di)
                    set_normalized_rhs(cYDb[ℓ, i], Di)
                end
                set_normalized_rhs(cL[i], Di)
                for j in N
                    set_normalized_rhs(cYod[i, j], ω[i, j])
                end
            end
        end
    end
end

"""Decode a λ vector into physical components, keyed by (group, index)."""
function decode_state(inst::BLInstance, bits::AbstractVector{<:Real})
    out = Dict{Tuple{Symbol,Tuple},Int}()
    pos = 0
    for comp in inst.comps
        val = 0
        for l in 1:comp.bits
            val += round(Int, bits[pos+l]) * 2^(l - 1)
        end
        out[(comp.group, comp.index)] = val
        pos += comp.bits
    end
    return out
end

"""
Conservation at the linking point (CONS-B / CONS-W restated on the state):
  Γ B_total = Σ A + Σ U + Σ_{r≥2} P + Γ Σ Mrem + Γ Σ_{r≥2} G,   W_total = Σ W + Σ_{r≥2} G.
Returns (bike residual in scaled units, worker residual).
"""
function conservation_residual(inst::BLInstance, st::Dict)
    bikes = workers = 0
    for ((g, _), v) in st
        g in (:A, :U, :P) && (bikes += v)
        g === :Mrem && (bikes += inst.Γ * v)
        g === :G && (bikes += inst.Γ * v; workers += v)
        g === :W && (workers += v)
    end
    return bikes - inst.Γ * inst.Q3, workers - inst.Q1
end

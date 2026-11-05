"""
two_stage_bl.jl  —  two-stage LIME-OPR with K battery levels (model_2stage)

Spec:  model_2stage/01_model.md
Data:  model_2stage/toy/two_stage_toy.json  (+ reference/model_v2_toy/toy_instance.json for the
       deterministic block)

First stage   task prices p_jk from a discrete menu, chosen before any demand is seen.
Second stage  per scenario xi = (d^1..d^T): one deterministic T-period MILP with the
              EXP-BATTLVL physics (K levels, H-first riders, floor per level, moves carry
              level-1 bikes only, swaps K -> 1) and stable matching under the chosen prices.

Nothing of the SDDiP interface is built: no binary state expansion, no local copies,
no A2 slacks, no initial cuts, no policy graph.  Inside a scenario the period-to-period
state is a set of plain continuous variables tied by equalities (integrality follows
from the integer decisions, exactly as in model_v2 §3.4).  Γ survives only because the
floor rule I = ⌊A⌋ is written as the Euclidean split Ã = Γ I + r.

Prices enter the second stage in two places:
  * wages  Σ_t p_jk m̃_jk^t  -- bilinear, linearised exactly through the menu:
           p_jk = Σ_l P_l z_{g(jk),l},  m̃_jk^t = Σ_l w_jkl^t,  0 ≤ w_jkl^t ≤ Q3 z_{g(jk),l};
  * worker utilities u_ijk = p_jk − d_ij − c_jk in (S3)/(S6), which stay linear in z.
"""

include(joinpath(@__DIR__, "..", "reference", "battery_levels.jl"))

using JuMP

struct TSInstance
    name::String
    n::Int
    T::Int
    K::Int                             # battery levels; 1..K-1 rentable, K dead
    Γ::Int
    q::Array{Int,4}                    # q[ℓ, ℓ2, i, j], ℓ = 1..K-1
    R::Matrix{Float64}
    Cp::Float64
    d::Matrix{Int}
    c::Matrix{Int}
    Cmax::Int
    Hmax::Int
    Q1::Int
    Q3::Int
    A0::Matrix{Int}                    # A0[ℓ, i]; the day starts with every bike on level 1
    U0::Vector{Int}
    W0::Vector{Int}
    menu::Vector{Float64}
    structure::Symbol                  # :uniform, :type or :od
    group::Matrix{Int}                 # price group of task (j, k)
    ngroups::Int
    scen::Vector{Vector{Matrix{Int}}}  # scen[s][t][i, j]
    vterm::Float64                     # cost per battery level unit the day ends below its morning level (0 = none)
    # integer rider split (model_2stage/int_flow): scenU[s][t][i, j] of the riders scen[s][t][i, j] end with a dead
    # bike; empty = expected-flow split through q.  A0s / U0s: morning state of each scenario; empty = A0 / U0 for all.
    scenU::Vector{Vector{Matrix{Int}}}
    A0s::Vector{Matrix{Int}}
    U0s::Vector{Vector{Int}}
end
TSInstance(args::Vararg{Any,22}) = TSInstance(args..., 0.0)
TSInstance(args::Vararg{Any,23}) = TSInstance(args..., Vector{Vector{Matrix{Int}}}(), Matrix{Int}[], Vector{Int}[])

function price_groups(n, structure)
    structure === :uniform && return ones(Int, n, n), 1
    structure === :type && return [j == k ? 2 : 1 for j in 1:n, k in 1:n], 2
    structure === :od && return [(j - 1) * n + k for j in 1:n, k in 1:n], n * n
    error("unknown price structure $structure")
end

"""
    load_ts_instance(path, variant; levels = 3, pool = "train", nscen = nothing,
                     structure = :type, θL = 1//5, grid = 20, menu = nothing)

`nscen` takes the first `nscen` trajectories of the pool (pools are prefix-nested).
`phi` (a number) overrides every φ_ij -- for diagnostics only (phi = 0 gives Γ = 1).
φ is read as Δ/(1-θL) and level transitions follow EXP-BATTLVL FORMULATION §2.

A pool with `scenarios_U` (model_2stage/int_flow/build_int_instance.py) has the integer rider split: no battery levels
(`levels = 2`: rentable + dead), Γ = 1, the riders of each OD split into those whose bike ends dead (scenU) and the
rest, and its `A0` / `U0` give each scenario's morning state.  Its blocks are built on `scenario_view(inst, s)`.
"""
function load_ts_instance(path, variant; levels = 3, pool = "train", nscen = nothing,
                          structure = :type, θL = 1 // 5, grid = 20, menu = nothing, phi = nothing,
                          phi_scale = 1.0, vterm = 0.0)
    raw = JSON.parsefile(path)
    # synthetic instances carry their own deterministic block; the toy reads the v2 toy's
    det = haskey(raw, "deterministic") ? raw["deterministic"] :
        JSON.parsefile(joinpath(dirname(path), "..", "reference", "model_v2_toy", "toy_instance.json"))["deterministic"]
    n, T, K = det["N"], det["T"], levels
    N = 1:n
    pl = raw["variants"][variant]["pools"][pool]
    split = haskey(pl, "scenarios_U")
    split && K != 2 && error("integer rider split: no battery levels, load with levels = 2")
    split && (phi !== nothing || phi_scale != 1.0) && error("integer rider split: φ is frozen in the split, rebuild the instance")
    if split
        Γ, q = 1, zeros(Int, K - 1, K, n, n)            # q is unused: the split is data
    else
        φ = phi === nothing ? _mat(det["phi_ij"], Float64) : fill(Float64(phi), n, n)   # `phi` only for diagnostics
        φ = φ .* phi_scale                              # `phi_scale`: faster / slower battery drain than calibrated
        Δ = [rationalize(BigInt, x; tol = 1e-12) * (1 - θL) for x in φ]
        Pq = [round_to_grid(level_transition(Δ[i, j], θL, K), grid) for i in N, j in N]
        Γ = Int(foldl(lcm, (denominator(x) for P in Pq for x in P); init = big(1)))
        q = zeros(Int, K - 1, K, n, n)
        for i in N, j in N, ℓ in 1:K-1, ℓ2 in 1:K
            q[ℓ, ℓ2, i, j] = Int(Γ * Pq[i, j][ℓ, ℓ2])
        end
    end
    d, c = _mat(det["d_ij"], Int), _mat(det["c_ij"], Int)
    A0 = zeros(Int, K - 1, n)
    A0[1, :] = Int.(det["A0"])
    ids = nscen === nothing ? eachindex(pl["scenarios"]) : (1:nscen)
    scen = [[_mat(m, Int) for m in pl["scenarios"][s]] for s in ids]
    scenU, A0s, U0s = Vector{Vector{Matrix{Int}}}(), Matrix{Int}[], Vector{Int}[]
    if split
        scenU = [[_mat(m, Int) for m in pl["scenarios_U"][s]] for s in ids]
        all(all(0 .<= scenU[s][t] .<= scen[s][t]) for s in eachindex(scen), t in 1:T) ||
            error("scenarios_U must lie between 0 and the demand")
        A0s = [reshape(Int.(pl["A0"][s]), 1, n) for s in ids]
        U0s = [Int.(pl["U0"][s]) for s in ids]
    end
    fleet = split ? [sum(A0s[s]) + sum(U0s[s]) for s in eachindex(scen)] : fill(sum(A0) + sum(Int.(det["U0"])), length(scen))
    Q3 = maximum(fleet)                                 # a scenario's view has its own fleet (scenario_view)
    all(sum(scen[s][t][i, :]) <= fleet[s] for s in eachindex(scen), t in 1:T, i in N) ||
        error("(SC-5) violated: some node demand exceeds Q3")
    mn = menu === nothing ? Float64.(raw["pricing"]["menu"]) : Float64.(menu)
    group, ngroups = price_groups(n, structure)
    return TSInstance("$(variant)/$(pool)/K$(K)/$(structure)" * (split ? "/int" : ""), n, T, K, Γ, q,
        _mat(det["R_ij"], Float64), Float64(det["C_p"]), d, c, maximum(c),
        maximum(d[i, j] + c[j, k] for i in N, j in N, k in N),
        sum(Int.(det["W0"])), Q3, A0, Int.(det["U0"]), Int.(det["W0"]),
        mn, structure, group, ngroups, scen, Float64(vterm), scenU, A0s, U0s)
end

"""
Scenario `s` alone, with its own morning state and fleet (Q3): the instance on which a per-scenario instance (integer
rider split, daily morning state) builds its blocks.  For any other instance the same as `inst` restricted to `s`.
"""
function scenario_view(inst::TSInstance, s::Int)
    A0 = isempty(inst.A0s) ? inst.A0 : inst.A0s[s]
    U0 = isempty(inst.U0s) ? inst.U0 : inst.U0s[s]
    Q3 = isempty(inst.A0s) ? inst.Q3 : sum(A0) + sum(U0)
    return TSInstance(inst.name, inst.n, inst.T, inst.K, inst.Γ, inst.q, inst.R, inst.Cp, inst.d, inst.c,
        inst.Cmax, inst.Hmax, inst.Q1, Q3, A0, U0, inst.W0, inst.menu, inst.structure, inst.group, inst.ngroups,
        [inst.scen[s]], inst.vterm, isempty(inst.scenU) ? inst.scenU : [inst.scenU[s]],
        isempty(inst.A0s) ? inst.A0s : [A0], isempty(inst.U0s) ? inst.U0s : [U0])
end

"""
Riders of ξ whose bike ends dead under the integer rider split, or `nothing` (expected-flow split through q).
A per-scenario instance only builds blocks on `scenario_view(inst, s)` and its scenario 1.
"""
function dead_riders(inst::TSInstance, ξ)
    isempty(inst.scenU) && isempty(inst.A0s) && return nothing
    length(inst.scen) == 1 && ξ === inst.scen[1] ||
        error("per-scenario instance: build blocks on scenario_view(inst, s) and its scenario 1")
    return isempty(inst.scenU) ? nothing : inst.scenU[1]
end

"""Largest Γ-scaled number of a rider's bikes that reach level ℓ2 (Γ under the integer rider split)."""
qmax(inst::TSInstance, ℓ2, i, j) = isempty(inst.scenU) ? maximum(inst.q[ℓ, ℓ2, i, j] for ℓ in 1:inst.K-1) : inst.Γ

"""Price matrix p[j,k] for menu indices `choice[g]`, one per price group."""
price_matrix(inst::TSInstance, choice::AbstractVector{<:Integer}) =
    [inst.menu[choice[inst.group[j, k]]] for j in 1:inst.n, k in 1:inst.n]

stability_Q2(inst::TSInstance, pmax) = pmax + maximum(inst.d) + maximum(inst.c)

# ── first stage ──────────────────────────────────────────────────────────────

"""Menu-choice binaries z[g, l] (one price per group) and the price expressions p[j,k]."""
function add_prices!(m::Model, inst::TSInstance)
    G, Lm = 1:inst.ngroups, eachindex(inst.menu)
    z = @variable(m, [G, Lm], Bin, base_name = "z")
    @constraint(m, [g in G], sum(z[g, l] for l in Lm) == 1)
    p = [@expression(m, sum(inst.menu[l] * z[inst.group[j, k], l] for l in Lm))
         for j in 1:inst.n, k in 1:inst.n]
    return (z = z, p = p)
end

# ── second stage: one scenario = one T-period block ─────────────────────────

"""
    add_scenario!(m, inst, ξ, price; tag = "", stability = :full) -> NamedTuple

`price` is one of
  * a fixed Matrix p[j,k]                       (evaluation; `stability` picks the formulation)
  * the NamedTuple returned by `add_prices!`    (extensive form; always `:full`)
  * a NamedTuple (wage = a, ir = b)             (price-box relaxation, see below)

`stability` for a fixed price:
  :full  (S1)-(S6) for every (i,j,k), global Big-M Q2 (the original formulation)
  :pre   same feasible set, reduced with the known utilities u = p - d - c:
         u < 0 -> x = 0 (individual rationality) and no η/δ/v;  u = 0 -> η only;
         u > 0 -> η and δ;  Big-Ms per pair: (S3) smax_i - u, (S6) u, (S5) smax_i,
         where smax_i = max(0, max_jk u_ijk) bounds s_i
  :lazy  as :pre, but (S1)/(S6) of the u > 0 pairs are left out and added as lazy
         constraints when an integer solution has that blocking pair (see lazy_stability!)

Price-box relaxation (wage = a, ir = b, a ≤ p ≤ b elementwise): no stable matching at all,
a worker may take (j,k) only if b_jk ≥ d_ij + c_jk, wages are paid at a.  Its value is an
upper bound on the stable-matching value for every price in the box.

Time blocks (fixed_price/LAGRANGIAN_TD.md): `init = :free` leaves the incoming state of
period 1 unfixed (the caller bounds it), `carry = true` also builds the state after period T
(index T + 1 of A, U, W, Mrem, Pin, Gin).  The defaults give the whole-day block.

Integer rider split (`dead_riders`): YU[i,j,t] ≤ scenU of the riders served from i to j end with a dead bike, the
other Yod − YU ≤ ξ − scenU arrive rentable; both are whole bikes (Γ = 1, so (D1) is I = A).  `nothing` otherwise.

Returns the scenario profit expression and the containers needed for diagnostics and
for the lazy callback.
"""
function add_scenario!(m::Model, inst::TSInstance, ξ::Vector{Matrix{Int}}, price; tag = "",
                       stability::Symbol = :full, init::Symbol = :fixed, carry::Bool = false)
    n, T, K, Γ = inst.n, inst.T, inst.K, inst.Γ
    inst.vterm > 0 && error("vterm (end-of-day battery penalty) needs the sparse formulation")
    N, Lu, Tt = 1:n, 1:K-1, 1:T
    Q1, Q3 = inst.Q1, inst.Q3
    Cmax, Hmax, q, d, c = inst.Cmax, inst.Hmax, inst.q, inst.d, inst.c
    fixed = price isa AbstractMatrix
    relax = price isa NamedTuple && haskey(price, :wage)
    fixed || relax || stability === :full || error("price decisions only support stability = :full")
    Q2 = stability_Q2(inst, fixed ? max(maximum(price), maximum(inst.menu)) : maximum(inst.menu))
    pr(j, k) = fixed ? price[j, k] : relax ? price.wage[j, k] : price.p[j, k]
    ufix = fixed ? [price[j, k] - d[i, j] - c[j, k] for i in N, j in N, k in N] : nothing
    smax = fixed ? [max(0.0, maximum(ufix[i, :, :])) for i in N] : nothing
    cand = Tuple{Int,Int,Int,Int}[]           # (i,j,k,t) whose (S1)/(S6) are lazy
    τ = [d[i, j] + c[j, k] for i in N, j in N, k in N]
    nm(s) = string(s, tag)
    ξU = dead_riders(inst, ξ)

    # incoming state of each period (t = 1 is the morning state, fixed below)
    init in (:fixed, :free) || error("init must be :fixed or :free")
    Ts = carry ? (1:T+1) : Tt
    A    = @variable(m, [Lu, N, Ts], lower_bound = 0, base_name = nm("A"))
    U    = @variable(m, [N, Ts], lower_bound = 0, base_name = nm("U"))
    W    = @variable(m, [N, Ts], lower_bound = 0, base_name = nm("W"))
    Mrem = @variable(m, [N, N, Ts], lower_bound = 0, base_name = nm("Mrem"))
    Pin  = @variable(m, [1:K, N, 2:max(Cmax, 2), Ts], lower_bound = 0, base_name = nm("Pin"))
    Gin  = @variable(m, [N, 2:max(Hmax, 2), Ts], lower_bound = 0, base_name = nm("Gin"))
    inP(ℓ2, j, r, t) = r > Cmax ? AffExpr(0.0) : 1.0 * Pin[ℓ2, j, r, t]
    inG(k, r, t) = r > Hmax ? AffExpr(0.0) : 1.0 * Gin[k, r, t]

    # (I1)-(I2): full fleet on level 1, workers at W0, no open jobs, empty pipelines
    if init === :fixed
        for i in N
            for ℓ in Lu
                fix(A[ℓ, i, 1], Γ * inst.A0[ℓ, i]; force = true)
            end
            fix(U[i, 1], Γ * inst.U0[i]; force = true)
            fix(W[i, 1], inst.W0[i]; force = true)
            for k in N
                fix(Mrem[i, k, 1], 0; force = true)
            end
        end
        fix.(Pin[:, :, :, 1], 0; force = true)
        fix.(Gin[:, :, 1], 0; force = true)
    end

    # period decisions (model_v2 §4 / EXP-BATTLVL with move = :top)
    @variables(m, begin
        0 <= Ycum[Lu, N, Tt] <= Q3, Int
        0 <= Yod[Lu, N, N, Tt] <= Q3, Int
        0 <= L[N, Tt] <= Q3, Int
        0 <= I[Lu, N, Tt] <= Q3, Int
        0 <= rA[Lu, N, Tt] <= Γ - 1, Int
        β[Lu, N, Tt], Bin
        0 <= mhat[N, N, Tt] <= Q3, Int
        0 <= Mt[N, N, Tt] <= Q3, Int
        0 <= mtil[N, N, Tt] <= Q3, Int
        0 <= x[N, N, N, Tt] <= Q1, Int
        0 <= v[N, N, N, Tt] <= Q3
        0 <= s[N, Tt] <= Q2
        δ[N, N, N, Tt], Bin
        η[N, N, N, Tt], Bin
        ζ[N, Tt], Bin
    end)
    # anonymise so that several scenarios can live in one model
    for obj in (Ycum, Yod, L, I, rA, β, mhat, Mt, mtil, x, v, s, δ, η, ζ)
        for var in obj
            set_name(var, string(name(var), tag))
        end
    end
    for key in (:Ycum, :Yod, :L, :I, :rA, :β, :mhat, :Mt, :mtil, :x, :v, :s, :δ, :η, :ζ)
        unregister(m, key)
    end
    YU = ξU === nothing ? nothing : @variable(m, [i in N, j in N, t in Tt], Int, lower_bound = 0,
                                              upper_bound = ξU[t][i, j], base_name = nm("YU"))

    Ylev(ℓ, i, t) = ℓ == 1 ? 1.0 * Ycum[1, i, t] : Ycum[ℓ, i, t] - Ycum[ℓ-1, i, t]
    profit = AffExpr(0.0)
    pprofit = AffExpr[]                       # profit of each period (time decomposition)

    for t in Tt
        D = [sum(ξ[t][i, j] for j in N) for i in N]

        # ── H-first service with the floor rule per level ────────────────────
        @constraint(m, [ℓ in Lu, i in N], A[ℓ, i, t] == Γ * I[ℓ, i, t] + rA[ℓ, i, t])          # (D1)
        @constraint(m, [ℓ in Lu, i in N], Ycum[ℓ, i, t] <= sum(I[l2, i, t] for l2 in 1:ℓ))
        @constraint(m, [ℓ in Lu, i in N], Ycum[ℓ, i, t] <= D[i])
        @constraint(m, [ℓ in Lu, i in N],
            Ycum[ℓ, i, t] >= sum(I[l2, i, t] for l2 in 1:ℓ) - Q3 * (1 - β[ℓ, i, t]))
        @constraint(m, [ℓ in Lu, i in N], Ycum[ℓ, i, t] >= D[i] - Q3 * β[ℓ, i, t])
        @constraint(m, [ℓ in Lu, i in N], sum(Yod[ℓ, i, j, t] for j in N) == Ylev(ℓ, i, t))
        @constraint(m, [i in N, j in N], sum(Yod[ℓ, i, j, t] for ℓ in Lu) <= ξ[t][i, j])
        YU === nothing || @constraint(m, [i in N, j in N], Yod[1, i, j, t] - YU[i, j, t] <= ξ[t][i, j] - ξU[t][i, j])
        @constraint(m, [i in N], L[i, t] == D[i] - Ycum[K-1, i, t])

        # ── posting locks bikes: swaps lock dead bikes, moves lock level-1 bikes ──
        @constraint(m, [j in N], Γ * mhat[j, j, t] <= U[j, t])
        @constraint(m, [j in N],
            Γ * sum(mhat[j, k, t] for k in N if k != j) <= A[1, j, t] - Γ * Ylev(1, j, t))

        # ── pool and matching capacity ───────────────────────────────────────
        @constraint(m, [j in N, k in N], Mt[j, k, t] == Mrem[j, k, t] + mhat[j, k, t])
        @constraint(m, [j in N, k in N], mtil[j, k, t] == sum(x[i, j, k, t] for i in N))
        @constraint(m, [j in N, k in N], mtil[j, k, t] <= Mt[j, k, t])
        @constraint(m, [i in N], sum(x[i, j, k, t] for j in N, k in N) <= W[i, t])

        # ── stable matching (S1)-(S6) with u_ijk = p_jk − d_ij − c_jk ─────────
        if relax
            # no stability: only individual rationality at the top of the price box
            for i in N, j in N, k in N
                price.ir[j, k] < d[i, j] + c[j, k] && set_upper_bound(x[i, j, k, t], 0)
                fix(δ[i, j, k, t], 0; force = true); fix(η[i, j, k, t], 0; force = true)
                fix(v[i, j, k, t], 0; force = true)
            end
            for i in N
                fix(ζ[i, t], 0; force = true); fix(s[i, t], 0; force = true)
            end
        elseif stability === :full
            for i in N, j in N, k in N
                H = sum(x[i2, j, k, t] for i2 in N if d[i2, j] <= d[i, j])
                u = pr(j, k) - d[i, j] - c[j, k]
                @constraint(m, v[i, j, k, t] <= H)
                @constraint(m, v[i, j, k, t] <= Mt[j, k, t])
                @constraint(m, v[i, j, k, t] <= Q3 * δ[i, j, k, t])
                @constraint(m, v[i, j, k, t] >= Mt[j, k, t] - Q3 * (1 - δ[i, j, k, t]))
                @constraint(m, x[i, j, k, t] <= Q1 * η[i, j, k, t])                        # (S2)
                @constraint(m, u >= s[i, t] - Q2 * (1 - η[i, j, k, t]))                     # (S3)
                @constraint(m, s[i, t] >= u - Q2 * δ[i, j, k, t])                           # (S6)
            end
            @constraint(m, [i in N], sum(x[i, j, k, t] for j in N, k in N) >= W[i, t] - Q1 * (1 - ζ[i, t]))
            @constraint(m, [i in N], s[i, t] <= Q2 * ζ[i, t])                               # (S5)
        else                                    # :pre / :lazy, utilities known
            for i in N
                set_upper_bound(s[i, t], smax[i])
            end
            for i in N, j in N, k in N
                u = ufix[i, j, k]
                if u < 0                                                                    # IR: never taken
                    set_upper_bound(x[i, j, k, t], 0)
                    fix(η[i, j, k, t], 0; force = true)
                else
                    @constraint(m, x[i, j, k, t] <= Q1 * η[i, j, k, t])                    # (S2)
                    @constraint(m, u >= s[i, t] - (smax[i] - u) * (1 - η[i, j, k, t]))      # (S3)
                end
                if u > 0
                    if stability === :lazy
                        push!(cand, (i, j, k, t))
                    else
                        H = sum(x[i2, j, k, t] for i2 in N if d[i2, j] <= d[i, j])
                        @constraint(m, v[i, j, k, t] <= H)
                        @constraint(m, v[i, j, k, t] <= Mt[j, k, t])
                        @constraint(m, v[i, j, k, t] <= Q3 * δ[i, j, k, t])
                        @constraint(m, v[i, j, k, t] >= Mt[j, k, t] - Q3 * (1 - δ[i, j, k, t]))
                        @constraint(m, s[i, t] >= u - u * δ[i, j, k, t])                   # (S6)
                    end
                else                                                                        # (S6) holds: s ≥ 0 ≥ u
                    fix(δ[i, j, k, t], 0; force = true)
                    fix(v[i, j, k, t], 0; force = true)
                end
            end
            @constraint(m, [i in N], sum(x[i, j, k, t] for j in N, k in N) >= W[i, t] - Q1 * (1 - ζ[i, t]))
            @constraint(m, [i in N], s[i, t] <= smax[i] * ζ[i, t])                          # (S5)
        end

        # ── wages: Σ p_jk m̃_jk, exact through the menu when prices are decisions ──
        wage = AffExpr(0.0)
        if fixed || relax
            for j in N, k in N
                add_to_expression!(wage, pr(j, k), mtil[j, k, t])
            end
        else
            Lm = eachindex(inst.menu)
            w = @variable(m, [N, N, Lm], lower_bound = 0, base_name = nm("w$t"))
            for j in N, k in N
                g = inst.group[j, k]
                @constraint(m, sum(w[j, k, l] for l in Lm) == mtil[j, k, t])
                @constraint(m, [l in Lm], w[j, k, l] <= Q3 * price.z[g, l])
                for l in Lm
                    add_to_expression!(wage, inst.menu[l], w[j, k, l])
                end
            end
        end
        pt = sum(inst.R[i, j] * Yod[ℓ, i, j, t] for ℓ in Lu, i in N, j in N) - inst.Cp * sum(L[i, t] for i in N) - wage
        push!(pprofit, pt)
        add_to_expression!(profit, pt)

        t == T && !carry && break # overnight reset: no state after the last period

        # ── outgoing pipelines and the next period's incoming state ──────────
        Pout(ℓ2, j, r) = inP(ℓ2, j, r + 1, t) +
            sum((ride_to(ℓ2, ℓ -> Yod[ℓ, i, j, t], YU === nothing ? nothing : YU[i, j, t], q, Lu, K, i, j)
                 for i in N if c[i, j] == r); init = AffExpr(0.0))
        Gout(k, r) = inG(k, r + 1, t) +
            sum((1.0 * x[i, j, k, t] for i in N, j in N if τ[i, j, k] == r); init = AffExpr(0.0))
        for ℓ in Lu, j in N
            moved = ℓ == 1 ? Γ * sum(mhat[j, k, t] for k in N if k != j) : AffExpr(0.0)
            back = ℓ == 1 ? Γ * Gout(j, 1) : AffExpr(0.0)
            @constraint(m, A[ℓ, j, t+1] == A[ℓ, j, t] - Γ * Ylev(ℓ, j, t) - moved + Pout(ℓ, j, 1) + back)
        end
        @constraint(m, [j in N], U[j, t+1] == U[j, t] - Γ * mhat[j, j, t] + Pout(K, j, 1))
        @constraint(m, [k in N], W[k, t+1] == W[k, t] - sum(x[k, i, j, t] for i in N, j in N) + Gout(k, 1))
        @constraint(m, [j in N, k in N], Mrem[j, k, t+1] == Mt[j, k, t] - mtil[j, k, t])
        for ℓ2 in 1:K, j in N, r in 2:max(Cmax, 2)
            @constraint(m, Pin[ℓ2, j, r, t+1] == (r <= Cmax ? Pout(ℓ2, j, r) : AffExpr(0.0)))
        end
        for k in N, r in 2:max(Hmax, 2)
            @constraint(m, Gin[k, r, t+1] == (r <= Hmax ? Gout(k, r) : AffExpr(0.0)))
        end
    end

    return (profit = profit, A = A, U = U, W = W, Mrem = Mrem, Pin = Pin, Gin = Gin,
            Yod = Yod, YU = YU, L = L, mtil = mtil, mhat = mhat, x = x,
            Mt = Mt, s = s, δ = δ, v = v, u = ufix, cand = cand,
            pprofit = pprofit, Ycum = Ycum, I = I, rA = rA, β = β, η = η, ζ = ζ, ξU = ξU)
end

"""
Γ-scaled bikes of the riders served from i to j that arrive on level ℓ2, with `yod(ℓ)` the riders served from level ℓ
and `yu` those whose bike ends dead: Σ_ℓ q[ℓ, ℓ2, i, j] yod(ℓ), or under the integer rider split (`yu !== nothing`,
K = 2) yod(1) − yu rentable and yu dead.  Works on variables (model rows) and on numbers (MIP starts).
"""
function ride_to(ℓ2, yod, yu, q, Lu, K, i, j)
    yu === nothing && return sum(q[ℓ, ℓ2, i, j] * yod(ℓ) for ℓ in Lu)
    return ℓ2 == 1 ? yod(1) - yu : ℓ2 == K ? 1 * yu : 0 * yu
end

# ── models ───────────────────────────────────────────────────────────────────

"""Extensive form: prices + every scenario in `ids`, equal weights.  Maximises profit."""
function build_ef(inst::TSInstance, optimizer; ids = eachindex(inst.scen))
    m = Model(optimizer)
    price = add_prices!(m, inst)
    blocks = [add_scenario!(m, inst, inst.scen[s], price; tag = "_s$s") for s in ids]
    @objective(m, Max, sum(b.profit for b in blocks) / length(ids))
    return m, price, blocks
end

"""
lazy_stability!(cb, m, inst, b, added, ncuts, strategy)

Separation of the aggregate blocking-pair condition at an integer solution (the logic
of the old MILP line's seperation_solver.py).  Pair (i, jk) in period t blocks when
  * worker side: node i has workers and either one of them is idle (s_i must be 0) or
    the worst task they took has utility below u_ijk, and
  * job side:    the pool M_jk is not filled by workers at distance ≤ d_ij.
For a blocking pair its (S1)/(S6) rows are submitted as lazy constraints.
`strategy = :first` submits one pair per call (EXP-MILP-CUTSTRATEGY found this fastest),
`:all` every blocking pair of the solution.
"""
function lazy_stability!(cb, m, inst::TSInstance, b, added, ncuts, strategy)
    callback_node_status(cb, m) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
    n, d, Q3 = inst.n, inst.d, inst.Q3
    xv = callback_value.(Ref(cb), b.x)
    umin = Dict{Tuple{Int,Int},Float64}()
    function worst_u(i, t)
        get!(umin, (i, t)) do
            Wv = callback_value(cb, b.W[i, t])
            Wv < 0.5 && return Inf                              # nobody there: cannot block
            taken = sum(xv[i, :, :, t])
            Wv - taken >= 0.5 && return 0.0                     # an idle worker
            minimum(b.u[i, j, k] for j in 1:n, k in 1:n if xv[i, j, k, t] > 0.5)
        end
    end
    for (idx, (i, j, k, t)) in enumerate(b.cand)
        added[idx] && continue
        b.u[i, j, k] > worst_u(i, t) + 1e-6 || continue
        H = sum(xv[i2, j, k, t] for i2 in 1:n if d[i2, j] <= d[i, j])
        H < callback_value(cb, b.Mt[j, k, t]) - 0.5 || continue
        u = b.u[i, j, k]
        Hx = sum(b.x[i2, j, k, t] for i2 in 1:n if d[i2, j] <= d[i, j])
        v, δ, Mt, s = b.v[i, j, k, t], b.δ[i, j, k, t], b.Mt[j, k, t], b.s[i, t]
        for con in (@build_constraint(v <= Hx), @build_constraint(v <= Mt),
                    @build_constraint(v <= Q3 * δ), @build_constraint(v >= Mt - Q3 * (1 - δ)),
                    @build_constraint(s >= u - u * δ))
            MOI.submit(m, MOI.LazyConstraint(cb), con)
        end
        added[idx] = true
        ncuts[] += 1
        strategy === :first && return
    end
end

"""
Profit of scenario `s` under the fixed price matrix `p` (one small MILP).
Returns (value, model, block, stats) with stats = (solve_s, ncuts, nbin).
`:lazy` needs a solver with lazy-constraint callbacks (Gurobi).
"""
function solve_scenario(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer;
                        stability::Symbol = :full, strategy::Symbol = :first, deadline::Float64 = Inf)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    b = add_scenario!(m, inst, inst.scen[s], p; stability)
    @objective(m, Max, b.profit)
    ncuts = Ref(0)
    if stability === :lazy
        set_attribute(m, "LazyConstraints", 1)
        added = falses(length(b.cand))
        MOI.set(m, MOI.LazyConstraintCallback(), cb -> lazy_stability!(cb, m, inst, b, added, ncuts, strategy))
    end
    optimize!(m)
    st = JuMP.termination_status(m)
    ok = st == MOI.OPTIMAL
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    ok || st == MOI.TIME_LIMIT || error("scenario $s: $st")
    # at a time limit the incumbent is only a lower bound (-Inf if none was found);
    # callers see optimal = false / feasible = false
    return (feas ? objective_value(m) : -Inf), m, b,
           (solve_s = solve_time(m), ncuts = ncuts[], nbin = count_free_binaries(m), optimal = ok, feasible = feas)
end

"""Price-box relaxation of scenario `s`: wages at `a`, individual rationality at `b` (upper bound)."""
function solve_relax(inst::TSInstance, s::Int, a::AbstractMatrix, b::AbstractMatrix, optimizer;
                     deadline::Float64 = Inf)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    blk = add_scenario!(m, inst, inst.scen[s], (wage = a, ir = b))
    @objective(m, Max, blk.profit)
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("relaxation, scenario $s: $st")
    # the MIP best bound is a valid upper bound even when the time limit stops the solve
    return (st == MOI.OPTIMAL ? objective_value(m) : objective_bound(m)),
           (solve_s = solve_time(m), nbin = count_free_binaries(m), optimal = st == MOI.OPTIMAL)
end

"""Lower the model's time limit so that the solve ends by `deadline` (a `time()` value)."""
function cap_time_limit!(m::Model, deadline::Float64)
    isfinite(deadline) || return
    cur = time_limit_sec(m)
    set_time_limit_sec(m, max(1.0, min(cur === nothing ? Inf : cur, deadline - time())))
end

"""Binary variables that are not fixed (the ones the MIP actually branches on before presolve)."""
count_free_binaries(m::Model) = count(v -> is_binary(v) && !is_fixed(v), all_variables(m))

"""Exact SAA value of a fixed price matrix on the scenarios `ids`: mean and per-scenario profits."""
function evaluate_prices(inst::TSInstance, p::AbstractMatrix, optimizer; ids = eachindex(inst.scen),
                         stability::Symbol = :full)
    vals = [solve_scenario(inst, s, p, optimizer; stability)[1] for s in ids]
    return sum(vals) / length(vals), vals
end

"""
Bike / worker conservation on every incoming state t = 1..T of a solved block
(CONS-B / CONS-W restated at the linking point, as EXP-BATTLVL §5).
Returns the largest absolute residual over the periods.
"""
function conservation_residual(inst::TSInstance, b)
    Γ, n, K, T = inst.Γ, inst.n, inst.K, inst.T
    worst = 0.0
    for t in 1:T
        bikes = sum(value(b.A[ℓ, j, t]) for ℓ in 1:K-1, j in 1:n) + sum(value(b.U[j, t]) for j in 1:n) +
                sum(value.(b.Pin[:, :, :, t])) + Γ * sum(value.(b.Mrem[:, :, t])) +
                Γ * sum(value.(b.Gin[:, :, t]))
        workers = sum(value(b.W[k, t]) for k in 1:n) + sum(value.(b.Gin[:, :, t]))
        worst = max(worst, abs(bikes - Γ * inst.Q3), abs(workers - inst.Q1))
    end
    return worst
end

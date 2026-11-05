"""
sparse_scenario.jl  —  one scenario block whose worker assignments exist only where individual
rationality allows them (real-data experiment, weekly/2026-10-10_10-16/文档/model_2stage/realdata/README.md)

    b = add_scenario_sparse!(m, inst, ξ, wage, ir; stability = :none | :tier)
    solve_tier_sparse(inst, s, p, optimizer)      exact fixed-price value (stable matching)
    solve_r1_sparse(inst, s, p, optimizer)        R1: no stable matching, workers refuse u < 0
    solve_relax_sparse(inst, s, a, b, optimizer)  price-box relaxation (wages at a, IR at b)

Same physics, objective and feasible set as `add_scenario!` with a fixed price / price box
(two_stage_bl.jl); only the building differs:
  * x[e, t] exists only for the allowed triples e = (i, j, k) with ir_jk >= d_ij + c_jk.  The dense
    builder creates x, v, δ, η for all n^3 triples per period and fixes most of them (n = 50, T = 20:
    10 million variables);
  * the worker pipeline only goes up to the longest allowed task;
  * stable matching (fixed price, `stability = :tier`) is the utility-tier formulation of
    fixed_price/TIER_COMPARE.md.  `tier_form = :direct` (default) writes its rows as they are on the
    sparse x; `tier_form = :compact` writes them with running sums, so it has O(allowed triples)
    nonzeros instead of O(triples x nodes).  For worker node i with distinct positive utilities v_h
    and task jk with distance levels δ (the distinct d_ij of its positive-utility workers):
        C_iht  = Σ_{jk: u_ijk >= v_h} x_ijk          running sum over the tiers of i, top down
        W_it - C_iht <= Q1 z_iht                       z binary
        S_jkδt = Σ_{i: d_ij <= δ, u_ijk > 0} x_ijk     running sum over the distance levels of jk
        M_jkt - S_jkδt <= Q3 (1 - f_jkδt)              f in [0, 1]
        z_{i,h(u_ijk),t} <= f_{jk,d_ij,t}              every positive-utility triple
    Every i2 with d_i2j <= d_ij has u_i2jk >= u_ijk, so S at level d_ij is the H_ijk of
    add_tier_stability!.  z = 1 forces f = 1 and hence its row M - H <= 0; z = 0 allows f = 0.
    Same feasible set in (x, M, W) and the same LP bound as add_tier_stability!.

Integer rider split (`dead_riders`, model_2stage/int_flow): YU[i, j, t] ≤ scenU of the riders served end with a dead
bike, as in `add_scenario!`; the MIP starts serve the riders whose bike stays rentable first.

Time blocks (relax-and-fix `form = :sparse`, lagrangian_td.jl `form = :sparse`): `init = :free` and `carry = true`
work as in `add_scenario!`; `b.zt[t]` are the tier binaries of period t (`b.z` all of them).  The worker pipeline Gin
only has the durations of allowed tasks (2:b.Hmax); a longer one cannot carry a worker, so its state is zero.
"""

include(joinpath(@__DIR__, "fixed_price.jl"))

"""Allowed triples (i, j, k) with ir_jk >= d_ij + c_jk and their index lists."""
function allowed_triples(inst::TSInstance, ir::AbstractMatrix)
    n, d, c = inst.n, inst.d, inst.c
    trips = [(i, j, k) for j in 1:n for k in 1:n for i in 1:n if ir[j, k] >= d[i, j] + c[j, k]]
    by_i = [Int[] for _ in 1:n]
    by_jk = [Int[] for _ in 1:n, _ in 1:n]
    for (e, (i, j, k)) in enumerate(trips)
        push!(by_i[i], e)
        push!(by_jk[j, k], e)
    end
    τ = [d[i, j] + c[j, k] for (i, j, k) in trips]
    return trips, by_i, by_jk, τ
end

function add_scenario_sparse!(m::Model, inst::TSInstance, ξ::Vector{Matrix{Int}}, wage::AbstractMatrix,
                              ir::AbstractMatrix; stability::Symbol = :none, tag = "", tier_form::Symbol = :direct,
                              init::Symbol = :fixed, carry::Bool = false)
    stability in (:none, :tier) || error("stability must be :none or :tier")
    carry = carry || inst.vterm > 0           # the end-of-day penalty needs the state after period T
    stability === :tier && wage != ir && error("tier stability needs a fixed price (wage == ir)")
    init in (:fixed, :free) || error("init must be :fixed or :free")
    n, T, K, Γ = inst.n, inst.T, inst.K, inst.Γ
    N, Lu, Tt = 1:n, 1:K-1, 1:T
    Q1, Q3, Cmax, q, d, c = inst.Q1, inst.Q3, inst.Cmax, inst.q, inst.d, inst.c
    trips, by_i, by_jk, τ = allowed_triples(inst, ir)
    E = length(trips)
    Hmax = max(2, maximum(τ; init = 1))
    by_kτ = Dict{Tuple{Int,Int},Vector{Int}}()
    for (e, (_, _, k)) in enumerate(trips)
        push!(get!(by_kτ, (k, τ[e]), Int[]), e)
    end
    nm(s) = string(s, tag)
    ξU = dead_riders(inst, ξ)

    Ts = carry ? (1:T+1) : Tt
    A    = @variable(m, [Lu, N, Ts], lower_bound = 0, base_name = nm("A"))
    U    = @variable(m, [N, Ts], lower_bound = 0, base_name = nm("U"))
    W    = @variable(m, [N, Ts], lower_bound = 0, base_name = nm("W"))
    Mrem = @variable(m, [N, N, Ts], lower_bound = 0, base_name = nm("Mrem"))
    Pin  = @variable(m, [1:K, N, 2:max(Cmax, 2), Ts], lower_bound = 0, base_name = nm("Pin"))
    Gin  = @variable(m, [N, 2:Hmax, Ts], lower_bound = 0, base_name = nm("Gin"))
    inP(ℓ2, j, r, t) = r > Cmax ? AffExpr(0.0) : 1.0 * Pin[ℓ2, j, r, t]
    inG(k, r, t) = r > Hmax ? AffExpr(0.0) : 1.0 * Gin[k, r, t]
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

    Ycum = @variable(m, [Lu, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("Ycum"))
    Yod  = @variable(m, [Lu, N, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("Yod"))
    L    = @variable(m, [N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("L"))
    I    = @variable(m, [Lu, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("I"))
    rA   = @variable(m, [Lu, N, Tt], Int, lower_bound = 0, upper_bound = Γ - 1, base_name = nm("rA"))
    β    = @variable(m, [Lu, N, Tt], Bin, base_name = nm("beta"))
    mhat = @variable(m, [N, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("mhat"))
    Mt   = @variable(m, [N, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("Mt"))
    mtil = @variable(m, [N, N, Tt], Int, lower_bound = 0, upper_bound = Q3, base_name = nm("mtil"))
    x    = @variable(m, [1:E, Tt], Int, lower_bound = 0, upper_bound = Q1, base_name = nm("x"))
    YU   = ξU === nothing ? nothing : @variable(m, [i in N, j in N, t in Tt], Int, lower_bound = 0,
                                                upper_bound = ξU[t][i, j], base_name = nm("YU"))
    xsum(es, t) = sum((x[e, t] for e in es); init = AffExpr(0.0))

    Ylev(ℓ, i, t) = ℓ == 1 ? 1.0 * Ycum[1, i, t] : Ycum[ℓ, i, t] - Ycum[ℓ-1, i, t]
    profit = AffExpr(0.0)
    pprofit = AffExpr[]
    for t in Tt
        D = [sum(ξ[t][i, j] for j in N) for i in N]
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
        @constraint(m, [j in N], Γ * mhat[j, j, t] <= U[j, t])
        @constraint(m, [j in N],
            Γ * sum(mhat[j, k, t] for k in N if k != j) <= A[1, j, t] - Γ * Ylev(1, j, t))
        @constraint(m, [j in N, k in N], Mt[j, k, t] == Mrem[j, k, t] + mhat[j, k, t])
        @constraint(m, [j in N, k in N], mtil[j, k, t] == xsum(by_jk[j, k], t))
        @constraint(m, [j in N, k in N], mtil[j, k, t] <= Mt[j, k, t])
        @constraint(m, [i in N], xsum(by_i[i], t) <= W[i, t])
        wage_t = sum(wage[j, k] * mtil[j, k, t] for j in N, k in N)
        pt = sum(inst.R[i, j] * Yod[ℓ, i, j, t] for ℓ in Lu, i in N, j in N) - inst.Cp * sum(L[i, t] for i in N) - wage_t
        push!(pprofit, pt)
        add_to_expression!(profit, pt)
        t == T && !carry && break
        Pout(ℓ2, j, r) = inP(ℓ2, j, r + 1, t) +
            sum((ride_to(ℓ2, ℓ -> Yod[ℓ, i, j, t], YU === nothing ? nothing : YU[i, j, t], q, Lu, K, i, j)
                 for i in N if c[i, j] == r); init = AffExpr(0.0))
        Gout(k, r) = inG(k, r + 1, t) + xsum(get(by_kτ, (k, r), Int[]), t)
        for ℓ in Lu, j in N
            moved = ℓ == 1 ? Γ * sum(mhat[j, k, t] for k in N if k != j) : AffExpr(0.0)
            back = ℓ == 1 ? Γ * Gout(j, 1) : AffExpr(0.0)
            @constraint(m, A[ℓ, j, t+1] == A[ℓ, j, t] - Γ * Ylev(ℓ, j, t) - moved + Pout(ℓ, j, 1) + back)
        end
        @constraint(m, [j in N], U[j, t+1] == U[j, t] - Γ * mhat[j, j, t] + Pout(K, j, 1))
        @constraint(m, [k in N], W[k, t+1] == W[k, t] - xsum(by_i[k], t) + Gout(k, 1))
        @constraint(m, [j in N, k in N], Mrem[j, k, t+1] == Mt[j, k, t] - mtil[j, k, t])
        for ℓ2 in 1:K, j in N, r in 2:max(Cmax, 2)
            @constraint(m, Pin[ℓ2, j, r, t+1] == (r <= Cmax ? Pout(ℓ2, j, r) : AffExpr(0.0)))
        end
        for k in N, r in 2:Hmax
            @constraint(m, Gin[k, r, t+1] == Gout(k, r))
        end
    end

    term = inst.vterm > 0 ? add_terminal_sparse!(m, inst, profit, A, Pin, Gin, Mrem, Hmax, nm) : nothing
    u = [wage[j, k] - d[i, j] - c[j, k] for (i, j, k) in trips]
    zt = stability !== :tier ? [VariableRef[] for _ in Tt] :
         tier_form === :compact ? add_tier_sparse!(m, inst, x, W, Mt, trips, u, tag) :
         add_tier_direct!(m, inst, x, W, Mt, trips, by_i, by_jk, u, tag)
    z = reduce(vcat, zt; init = VariableRef[])
    return (profit = profit, pprofit = pprofit, A = A, U = U, W = W, Mrem = Mrem, Pin = Pin, Gin = Gin,
            Yod = Yod, YU = YU, L = L, mtil = mtil, mhat = mhat, Mt = Mt, x = x, trips = trips, by_i = by_i,
            by_jk = by_jk, u = u, Ycum = Ycum, I = I, rA = rA, β = β, z = z, zt = zt, ntier = length(z), ξ = ξ,
            ξU = ξU, wage = wage, Hmax = Hmax, term = term)
end

"""
End-of-day battery: E = battery left after period T in level units (a bike on level ℓ counts K − ℓ: H 2, M 1, dead 0;
bikes still riding count at their arrival level, task bikes under way and bikes held by open moves count as H).
E0 = the morning battery (`inst.A0`, which window instances keep).  Adds Edef ≥ E0 − E, Edef ≥ 0 and charges
`inst.vterm` per unit to `profit`: the battery used today and not swapped back has to be replaced before tomorrow.
"""
function add_terminal_sparse!(m::Model, inst::TSInstance, profit, A, Pin, Gin, Mrem, Hmax, nm)
    n, K, T, Γ, Cmax = inst.n, inst.K, inst.T, inst.Γ, inst.Cmax
    wl(ℓ) = K - ℓ
    E0 = sum(wl(ℓ) * inst.A0[ℓ, i] for ℓ in 1:K-1, i in 1:n)
    E = AffExpr(0.0)
    for ℓ in 1:K-1, i in 1:n
        add_to_expression!(E, wl(ℓ) / Γ, A[ℓ, i, T+1])
    end
    for ℓ in 1:K-1, j in 1:n, r in 2:max(Cmax, 2)
        add_to_expression!(E, wl(ℓ) / Γ, Pin[ℓ, j, r, T+1])
    end
    for k in 1:n, r in 2:Hmax
        add_to_expression!(E, wl(1), Gin[k, r, T+1])
    end
    for j in 1:n, k in 1:n
        j == k || add_to_expression!(E, wl(1), Mrem[j, k, T+1])
    end
    δ = @variable(m, lower_bound = 0, base_name = nm("Edef"))
    @constraint(m, δ >= E0 - E)
    add_to_expression!(profit, -inst.vterm, δ)
    return (δ = δ, E = E, E0 = E0)
end

"""Start value of the end-of-day deficit from the start values of the state after period T; returns the penalty."""
function terminal_start!(inst::TSInstance, b)
    b.term === nothing && return 0.0
    def = max(0.0, b.term.E0 - value(v -> something(start_value(v), 0.0), b.term.E))
    set_start_value(b.term.δ, def)
    return inst.vterm * def
end

"""Utility-tier stable matching with running sums (module docstring).  Returns the z binaries per period."""
function add_tier_sparse!(m::Model, inst::TSInstance, x, W, Mt, trips, u, tag)
    n, T, d, Q1, Q3 = inst.n, inst.T, inst.d, inst.Q1, inst.Q3
    pos = [e for e in eachindex(trips) if u[e] > 0]
    # worker side: tiers of i, highest utility first
    tiers = [sort(unique(u[e] for e in pos if trips[e][1] == i); rev = true) for i in 1:n]
    tier_of = [Dict(v => h for (h, v) in enumerate(tiers[i])) for i in 1:n]
    at_tier = [[Int[] for _ in tiers[i]] for i in 1:n]
    for e in pos
        i = trips[e][1]
        push!(at_tier[i][tier_of[i][u[e]]], e)
    end
    # job side: distance levels of jk, nearest first
    levels = Dict{Tuple{Int,Int},Vector{Int}}()
    for e in pos
        i, j, k = trips[e]
        push!(get!(levels, (j, k), Int[]), d[i, j])
    end
    for key in keys(levels)
        levels[key] = sort(unique(levels[key]))
    end
    at_level = Dict(key => [Int[] for _ in lv] for (key, lv) in levels)
    for e in pos
        i, j, k = trips[e]
        push!(at_level[(j, k)][searchsortedfirst(levels[(j, k)], d[i, j])], e)
    end
    zs = [VariableRef[] for _ in 1:T]
    for t in 1:T
        z = Dict{Tuple{Int,Int},VariableRef}()
        for i in 1:n
            prev = nothing
            for h in eachindex(tiers[i])
                C = @variable(m, lower_bound = 0, base_name = string("C_", i, "_", h, "_", t, tag))
                @constraint(m, C == (prev === nothing ? AffExpr(0.0) : 1.0 * prev) + sum(x[e, t] for e in at_tier[i][h]))
                zz = @variable(m, binary = true, base_name = string("z_", i, "_", h, "_", t, tag))
                @constraint(m, W[i, t] - C <= Q1 * zz)
                z[(i, h)] = zz
                prev = C
                push!(zs[t], zz)
            end
        end
        for ((j, k), lv) in levels
            prev = nothing
            for (l, δ) in enumerate(lv)
                S = @variable(m, lower_bound = 0, base_name = string("S_", j, "_", k, "_", δ, "_", t, tag))
                @constraint(m, S == (prev === nothing ? AffExpr(0.0) : 1.0 * prev) + sum(x[e, t] for e in at_level[(j, k)][l]))
                f = @variable(m, lower_bound = 0, upper_bound = 1, base_name = string("f_", j, "_", k, "_", δ, "_", t, tag))
                @constraint(m, Mt[j, k, t] - S <= Q3 * (1 - f))
                for e in at_level[(j, k)][l]
                    i = trips[e][1]
                    @constraint(m, z[(i, tier_of[i][u[e]])] <= f)
                end
                prev = S
            end
        end
    end
    return zs
end

"""The rows of add_tier_stability! on the sparse x (one row per positive-utility triple, H written out).
Returns the z binaries per period."""
function add_tier_direct!(m::Model, inst::TSInstance, x, W, Mt, trips, by_i, by_jk, u, tag)
    n, T, d, Q1, Q3 = inst.n, inst.T, inst.d, inst.Q1, inst.Q3
    tiers = [sort(unique(u[e] for e in by_i[i] if u[e] > 0)) for i in 1:n]
    tier_of = [Dict(v => h for (h, v) in enumerate(tiers[i])) for i in 1:n]
    zs = [VariableRef[] for _ in 1:T]
    for t in 1:T
        z = Dict{Tuple{Int,Int},VariableRef}()
        for i in 1:n, (h, v) in enumerate(tiers[i])
            zz = @variable(m, binary = true, base_name = string("z_", i, "_", h, "_", t, tag))
            @constraint(m, W[i, t] - sum(x[e, t] for e in by_i[i] if u[e] >= v) <= Q1 * zz)
            z[(i, h)] = zz
            push!(zs[t], zz)
        end
        for (e, (i, j, k)) in enumerate(trips)
            u[e] > 0 || continue
            H = sum(x[e2, t] for e2 in by_jk[j, k] if d[trips[e2][1], j] <= d[i, j])
            @constraint(m, Mt[j, k, t] - H <= Q3 * (1 - z[(i, tier_of[i][u[e]])]))
        end
    end
    return zs
end

"""
MIP start "post no task": simulate the block's physics with no task posted (H-first service with the
floor rule, riders split over destinations in index order) and give every variable its value.  Feasible
for every price, with or without stability (no posted job, so no blocking pair).  Gurobi's heuristics
often find no incumbent at all on the real-data instances (T = 40, n = 10: none within 600 s).
Variables of the compact tier rows (C, S, f) are left to the solver.
"""
function nothing_start!(inst::TSInstance, b)
    n, T, K, Γ, Cmax, q, c, ξ = inst.n, inst.T, inst.K, inst.Γ, inst.Cmax, inst.q, inst.c, b.ξ
    Lu = 1:K-1
    A = [Γ * inst.A0[ℓ, i] for ℓ in Lu, i in 1:n]
    U = Γ .* copy(inst.U0)
    Pin = zeros(Int, K, n, max(Cmax, 2))                    # Pin[ℓ2, j, r], r = 2..Cmax used
    ss(v, x) = set_start_value(v, x)
    for t in 1:T
        for i in 1:n
            for ℓ in Lu
                ss(b.A[ℓ, i, t], A[ℓ, i])
            end
            ss(b.U[i, t], U[i]); ss(b.W[i, t], inst.W0[i])
            for k in 1:n
                ss(b.Mrem[i, k, t], 0); ss(b.mhat[i, k, t], 0); ss(b.Mt[i, k, t], 0); ss(b.mtil[i, k, t], 0)
            end
        end
        for ℓ2 in 1:K, j in 1:n, r in 2:max(Cmax, 2)
            ss(b.Pin[ℓ2, j, r, t], Pin[ℓ2, j, r])
        end
        ss.(b.Gin[:, :, t], 0)
        ss.(b.x[:, t], 0)
        Yod = zeros(Int, K - 1, n, n)
        for i in 1:n
            D = sum(ξ[t][i, :])
            I = [A[ℓ, i] ÷ Γ for ℓ in Lu]
            cap = ξ[t][i, :]
            prev = 0
            for ℓ in Lu
                cumI = sum(I[1:ℓ])
                Yc = min(cumI, D)
                ss(b.I[ℓ, i, t], I[ℓ]); ss(b.rA[ℓ, i, t], A[ℓ, i] - Γ * I[ℓ])
                ss(b.Ycum[ℓ, i, t], Yc); ss(b.β[ℓ, i, t], cumI <= D ? 1 : 0)
                need = Yc - prev
                for j in 1:n
                    take = min(cap[j], need)
                    Yod[ℓ, i, j] = take; cap[j] -= take; need -= take
                end
                need == 0 || error("nothing_start!: riders could not be split")
                for j in 1:n
                    ss(b.Yod[ℓ, i, j, t], Yod[ℓ, i, j])
                end
                A[ℓ, i] -= Γ * (Yc - prev)
                prev = Yc
            end
            ss(b.L[i, t], D - prev)
        end
        YUv = dead_start!(b, Yod, ξ, t)
        t == T && break
        # riders end their trips: pipelines advance by one period
        Pout = zeros(Int, K, n, max(Cmax, 2) + 1)
        for ℓ2 in 1:K, j in 1:n, r in 1:max(Cmax, 2)
            Pout[ℓ2, j, r] = (r + 1 <= Cmax ? Pin[ℓ2, j, r + 1] : 0) +
                sum((ride_to(ℓ2, ℓ -> Yod[ℓ, i, j], YUv === nothing ? nothing : YUv[i, j], q, Lu, K, i, j)
                     for i in 1:n if c[i, j] == r); init = 0)
        end
        for j in 1:n
            for ℓ in Lu
                A[ℓ, j] += Pout[ℓ, j, 1]
            end
            U[j] += Pout[K, j, 1]
        end
        for ℓ2 in 1:K, j in 1:n, r in 2:max(Cmax, 2)
            Pin[ℓ2, j, r] = r <= Cmax ? Pout[ℓ2, j, r] : 0
        end
    end
    for z in b.z
        set_start_value(z, 1)
    end
    return b
end

"""
Integer rider split: start values of YU[:, :, t] for the riders `Yod` of a MIP start, serving the riders whose bike
stays rentable first (YU = what is served beyond ξ − scenU); returns them, or `nothing` without the split.
"""
function dead_start!(b, Yod, ξ, t)
    b.ξU === nothing && return nothing
    n = size(Yod, 2)
    YUv = [max(0, Yod[1, i, j] - (ξ[t][i, j] - b.ξU[t][i, j])) for i in 1:n, j in 1:n]
    for i in 1:n, j in 1:n
        set_start_value(b.YU[i, j, t], YUv[i, j])
    end
    return YUv
end

"""State of a sparse block at its first period when the block starts from the instance's morning state."""
initial_state(inst::TSInstance, b) =
    (A = inst.Γ .* inst.A0, U = inst.Γ .* inst.U0, W = copy(inst.W0), Mrem = zeros(Int, inst.n, inst.n),
     Pin = zeros(Int, inst.K, inst.n, max(inst.Cmax, 2)), Gin = zeros(Int, inst.n, b.Hmax))

"""Rounded state of a solved sparse block at period t (same layout as `initial_state`); `val` reads a variable."""
function read_state(inst::TSInstance, b, t::Int; val = value)
    n, K, Cm = inst.n, inst.K, max(inst.Cmax, 2)
    Pin = zeros(Int, K, n, Cm)
    for ℓ2 in 1:K, j in 1:n, r in 2:Cm
        Pin[ℓ2, j, r] = round(Int, val(b.Pin[ℓ2, j, r, t]))
    end
    Gin = zeros(Int, n, b.Hmax)
    for k in 1:n, r in 2:b.Hmax
        Gin[k, r] = round(Int, val(b.Gin[k, r, t]))
    end
    # element by element: the containers may be DenseAxisArrays, the state must be plain Arrays
    return (A = [round(Int, val(b.A[ℓ, i, t])) for ℓ in 1:K-1, i in 1:n], U = [round(Int, val(b.U[i, t])) for i in 1:n],
            W = [round(Int, val(b.W[i, t])) for i in 1:n], Mrem = [round(Int, val(b.Mrem[j, k, t])) for j in 1:n, k in 1:n],
            Pin = Pin, Gin = Gin)
end

"""
    continue_start!(inst, b, t0, state) -> profit of periods t0..T

MIP start that continues a sparse block (direct tier rows or none) from its state at period t0 with the "post no new
task" policy: riders as in `nothing_start!` (H-first, floor, destinations in index order); nothing is posted; the
tasks still open are matched by worker-proposing deferred acceptance (each worker proposes to the open pools in
decreasing utility, u > 0 only; a pool keeps its nearest workers), which satisfies the tier rows; workers and bikes
move through the pipelines as in the model.  `state` has the layout of `initial_state` (A, U, Pin Γ-scaled).  Sets the
start of every variable of periods t0..T (and of the outgoing state T+1 when the block carries it); the caller keeps
the starts of the earlier periods consistent (relax_and_fix: the fixed values).  Feasible whenever `state` is the
block's actual state at t0, so relax-and-fix windows and forward blocks always have an incumbent.
"""
function continue_start!(inst::TSInstance, b, t0::Int, state)
    n, K, Γ, Cmax, q, c, d, ξ = inst.n, inst.K, inst.Γ, inst.Cmax, inst.q, inst.c, inst.d, b.ξ
    T, Hm, Cm, Lu = length(ξ), b.Hmax, max(Cmax, 2), 1:K-1
    carry = size(b.A, 3) > T
    trips, u = b.trips, b.u
    A, U, W, Mrem = copy(state.A), copy(state.U), copy(state.W), copy(state.Mrem)
    Pin, Gin = copy(state.Pin), copy(state.Gin)
    # worker preferences: positive-utility trips of node i, best first
    pref = [sort([e for e in b.by_i[i] if u[e] > 0]; by = e -> (-u[e], e)) for i in 1:n]
    tiers = [sort(unique(u[e] for e in b.by_i[i] if u[e] > 0)) for i in 1:n]   # order of add_tier_direct!
    ss(v, x) = set_start_value(v, x)
    function set_state!(t)
        for ℓ in Lu, i in 1:n
            ss(b.A[ℓ, i, t], A[ℓ, i])
        end
        for i in 1:n
            ss(b.U[i, t], U[i]); ss(b.W[i, t], W[i])
            for k in 1:n
                ss(b.Mrem[i, k, t], Mrem[i, k])
            end
        end
        for ℓ2 in 1:K, j in 1:n, r in 2:Cm
            ss(b.Pin[ℓ2, j, r, t], Pin[ℓ2, j, r])
        end
        for k in 1:n, r in 2:Hm
            ss(b.Gin[k, r, t], Gin[k, r])
        end
    end
    profit = 0.0
    for t in t0:T
        set_state!(t)
        # riders
        Yod = zeros(Int, K - 1, n, n)
        for i in 1:n
            D = sum(ξ[t][i, :])
            I = [A[ℓ, i] ÷ Γ for ℓ in Lu]
            cap = ξ[t][i, :]
            prev = 0
            for ℓ in Lu
                cumI = sum(I[1:ℓ])
                Yc = min(cumI, D)
                ss(b.I[ℓ, i, t], I[ℓ]); ss(b.rA[ℓ, i, t], A[ℓ, i] - Γ * I[ℓ])
                ss(b.Ycum[ℓ, i, t], Yc); ss(b.β[ℓ, i, t], cumI <= D ? 1 : 0)
                need = Yc - prev
                for j in 1:n
                    take = min(cap[j], need)
                    Yod[ℓ, i, j] = take; cap[j] -= take; need -= take
                end
                need == 0 || error("continue_start!: riders could not be split")
                for j in 1:n
                    ss(b.Yod[ℓ, i, j, t], Yod[ℓ, i, j])
                end
                A[ℓ, i] -= Γ * (Yc - prev)
                prev = Yc
            end
            ss(b.L[i, t], D - prev)
            profit += sum(inst.R[i, j] * Yod[ℓ, i, j] for ℓ in Lu, j in 1:n) - inst.Cp * (D - prev)
        end
        YUv = dead_start!(b, Yod, ξ, t)
        # open tasks: worker-proposing deferred acceptance, pools keep the nearest workers
        x = zeros(Int, length(trips))
        held = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()          # pool -> (distance, trip) per held worker
        queue = [(i, 0) for i in 1:n for _ in 1:W[i]]                   # (node, last preference index tried)
        while !isempty(queue)
            i, p = pop!(queue)
            p += 1
            while p <= length(pref[i]) && Mrem[trips[pref[i][p]][2], trips[pref[i][p]][3]] == 0
                p += 1
            end
            p > length(pref[i]) && continue                             # no open pool it wants: idle
            e = pref[i][p]
            _, j, k = trips[e]
            h = get!(held, (j, k), Tuple{Int,Int}[])
            push!(h, (d[i, j], e))
            if length(h) > Mrem[j, k]
                worst = argmax(first.(h))                               # farthest (ties: the earliest)
                _, er = h[worst]
                deleteat!(h, worst)
                push!(queue, (trips[er][1], findfirst(==(er), pref[trips[er][1]])))
            end
        end
        for h in values(held), (_, e) in h
            x[e] += 1
        end
        mtil = zeros(Int, n, n)
        for (e, (i, j, k)) in enumerate(trips)
            mtil[j, k] += x[e]
            ss(b.x[e, t], x[e])
        end
        for j in 1:n, k in 1:n
            ss(b.mhat[j, k, t], 0); ss(b.Mt[j, k, t], Mrem[j, k]); ss(b.mtil[j, k, t], mtil[j, k])
            profit -= b.wage[j, k] * mtil[j, k]
        end
        if !isempty(b.zt[t])
            zs = b.zt[t]
            pos = 0
            for i in 1:n
                busy = [u[e] for e in b.by_i[i] for _ in 1:x[e]]
                for v in tiers[i]
                    pos += 1
                    ss(zs[pos], W[i] - count(>=(v), busy) > 0 ? 1 : 0)
                end
            end
            pos == length(zs) || error("continue_start!: tier binaries out of order (compact tier rows?)")
        end
        t == T && !carry && break
        # pipelines and the next state
        arr = zeros(Int, n, Hm)                                         # workers dispatched now, by end node and duration
        for (e, (i, j, k)) in enumerate(trips)
            x[e] > 0 && (arr[k, d[i, j] + c[j, k]] += x[e])
        end
        Gout(k, r) = (r + 1 <= Hm ? Gin[k, r+1] : 0) + arr[k, r]
        Pout(ℓ2, j, r) = (r + 1 <= Cmax ? Pin[ℓ2, j, r+1] : 0) +
                         sum((ride_to(ℓ2, ℓ -> Yod[ℓ, i, j], YUv === nothing ? nothing : YUv[i, j], q, Lu, K, i, j)
                              for i in 1:n if c[i, j] == r); init = 0)
        G1 = [Gout(k, 1) for k in 1:n]
        P1 = [Pout(ℓ2, j, 1) for ℓ2 in 1:K, j in 1:n]
        Pnext = zeros(Int, K, n, Cm)
        for ℓ2 in 1:K, j in 1:n, r in 2:Cm
            Pnext[ℓ2, j, r] = r <= Cmax ? Pout(ℓ2, j, r) : 0
        end
        Gnext = zeros(Int, n, Hm)
        for k in 1:n, r in 2:Hm
            Gnext[k, r] = Gout(k, r)
        end
        for j in 1:n
            for ℓ in Lu
                A[ℓ, j] += P1[ℓ, j] + (ℓ == 1 ? Γ * G1[j] : 0)
            end
            U[j] += P1[K, j]
        end
        for i in 1:n
            W[i] += G1[i] - sum((x[e] for e in b.by_i[i]); init = 0)
        end
        Mrem .-= mtil
        Pin, Gin = Pnext, Gnext
        t == T && set_state!(T + 1)
    end
    return profit - terminal_start!(inst, b)
end

_sizes(m) = (nvar = num_variables(m), ncon = num_constraints(m; count_variable_in_set_constraints = false),
             nbin = count_free_binaries(m), nint = count(v -> is_integer(v) && !is_fixed(v), all_variables(m)))

function _solve_sparse(inst, s, wage, ir, optimizer, stability, deadline, what; tier_form = :direct, start = true)
    m = Model(optimizer)
    cap_time_limit!(m, deadline)
    tb = @elapsed begin
        b = add_scenario_sparse!(m, inst, inst.scen[s], wage, ir; stability, tier_form)
        if start          # with the end-of-day penalty the start must also set the state after period T
            inst.vterm > 0 ? continue_start!(inst, b, 1, initial_state(inst, b)) : nothing_start!(inst, b)
        end
        @objective(m, Max, b.profit)
    end
    v0 = start ? value(v -> something(start_value(v), 0.0), b.profit) : NaN   # profit of posting no task
    optimize!(m)
    st = JuMP.termination_status(m)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("$what, scenario $s: $st")
    feas = primal_status(m) == MOI.FEASIBLE_POINT
    bnd = try objective_bound(m) catch; Inf end
    return (feas ? objective_value(m) : -Inf), m, b,
           merge((solve_s = solve_time(m), build_s = tb, bound = bnd, optimal = st == MOI.OPTIMAL, feasible = feas,
                  ntrip = length(b.trips), ntier = b.ntier, nothing_value = v0), _sizes(m))
end

"""Exact fixed-price value of scenario `s` (stable matching by utility tiers).  `bound` is the MIP bound."""
solve_tier_sparse(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; deadline::Float64 = Inf,
                  tier_form::Symbol = :direct, start::Bool = true) =
    _solve_sparse(inst, s, p, p, optimizer, :tier, deadline, "tier"; tier_form, start)

"""R1 of scenario `s`: no stable matching, workers refuse u < 0."""
solve_r1_sparse(inst::TSInstance, s::Int, p::AbstractMatrix, optimizer; deadline::Float64 = Inf, start::Bool = true) =
    _solve_sparse(inst, s, p, p, optimizer, :none, deadline, "R1"; start)

"""Price-box relaxation of scenario `s` (wages at `a`, IR at `b`).  Returns (upper bound, info): the MIP bound,
which is valid also when the solve stops at a positive MIP gap or the time limit."""
function solve_relax_sparse(inst::TSInstance, s::Int, a::AbstractMatrix, b::AbstractMatrix, optimizer;
                            deadline::Float64 = Inf)
    val, _, _, info = _solve_sparse(inst, s, a, b, optimizer, :none, deadline, "relaxation")
    return info.bound, info
end

"""Blocking pairs of a solved sparse block at price `p` (same definition as `blocking_pairs`; O(n^3 T), tests only)."""
function blocking_pairs_sparse(inst::TSInstance, b, p::AbstractMatrix)
    n, T, d, c = inst.n, inst.T, inst.d, inst.c
    xv = zeros(Int, n, n, n, T)
    xval = round.(Int, value.(b.x))
    for (e, (i, j, k)) in enumerate(b.trips), t in 1:T
        xv[i, j, k, t] = xval[e, t]
    end
    Wv, Mv = round.(Int, value.(b.W)), round.(Int, value.(b.Mt))
    out = NamedTuple[]
    for t in 1:T, i in 1:n
        Wv[i, t] == 0 && continue
        idle = Wv[i, t] > sum(xv[i, :, :, t])
        worst = idle ? 0.0 : minimum(p[j, k] - d[i, j] - c[j, k] for j in 1:n, k in 1:n if xv[i, j, k, t] > 0)
        for j in 1:n, k in 1:n
            u = p[j, k] - d[i, j] - c[j, k]
            u > worst + 1e-9 || continue
            sum(xv[i2, j, k, t] for i2 in 1:n if d[i2, j] <= d[i, j]) < Mv[j, k, t] || continue
            push!(out, (i = i, j = j, k = k, t = t, u = u))
        end
    end
    return out
end

"""What a solved sparse block does over the horizon (counts are bikes / tasks / worker-periods)."""
function op_stats_sparse(inst::TSInstance, b, wage::AbstractMatrix)
    n, T = inst.n, inst.T
    mt, M, W, L = value.(b.mtil), value.(b.Mt), value.(b.W), value.(b.L)
    xv = value.(b.x)
    sw = sum(mt[j, j, t] for j in 1:n, t in 1:T)
    τ = [inst.d[i, j] + inst.c[j, k] for (i, j, k) in b.trips]
    busy = sum(xv[e, t] * min(τ[e], T - t + 1) for e in eachindex(b.trips), t in 1:T; init = 0.0)
    return (served = sum(value.(b.Yod)), dead = b.YU === nothing ? NaN : sum(value.(b.YU)), lost = sum(L),
            moves = sum(mt) - sw, swaps = sw,
            wages = sum(wage[j, k] * mt[j, k, t] for j in 1:n, k in 1:n, t in 1:T),
            idle_worker_periods = sum(W[i, t] - sum(xv[e, t] for e in b.by_i[i]; init = 0.0) for i in 1:n, t in 1:T),
            busy_worker_periods = busy, open_left = sum(M .- mt))
end

"""
model_v2.jl  —  LIME-OPR v2 formulation as a standard SDDP.jl policy graph

Spec:  model_v2/01_problem_definition.md  (business model)
       model_v2/02_scenario_tree.md       (frozen Poisson tree, stage 1 = root)
       model_v2/03_sddip_interface.md     (binary state encoding)

"Standard" SDDiP means: every linking-state component is binary-expanded with
the per-component bit counts exported by model_v2/toy/build_toy_tree.py, cuts come
from SDDP.jl's native duality handlers, and nothing else is added.  In
particular the A2 bounded-elastic slacks and code-overflow slacks of
03_sddip_interface.md §3 are NOT built here.  Reachable states always have a
feasible stage problem (a stable matching always exists), so the forward pass
never needs them.

Two SDDP.jl facts this file works around (SDDP.jl 1.13.1, duality_handlers.jl):
  * StrengthenedConicDuality unfixes `x.in` without restoring any bounds, so the
    local-copy box 0 <= z <= 1 is added as explicit rows (03 §2.5).
  * LagrangianDuality restores bounds *and integrality* of Bin states, so its
    relaxation keeps z binary rather than Zou's continuous [0,1].  Cuts stay
    valid on the binary cube; this is recorded, not changed.
"""

using JSON, JuMP, SDDP

struct StateComponent
    group::Symbol          # :A :U :W :Mrem :PA :PU :G
    index::Tuple           # (j,) / (j,k) / (j,r)
    bits::Int
end

struct V2Instance
    name::String
    n::Int
    T::Int
    Γ::Int
    qφ::Matrix{Int}
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
    A0::Vector{Int}
    U0::Vector{Int}
    W0::Vector{Int}
    Ω::Vector{Vector{Matrix{Int}}}     # Ω[t][ℓ][i,j]
    prob::Vector{Vector{Float64}}
    comps::Vector{StateComponent}      # live components only (bits > 0)
    θ::Vector{Float64}                 # (IC-2), θ[t] for t = 1..T+1
end

_mat(x, T::Type) = T[x[i][j] for i in eachindex(x), j in eachindex(x[1])]

"""Load one variant (e.g. "toy-A") of model_v2/toy/toy_instance.json."""
function load_v2_instance(path::AbstractString, variant::AbstractString)
    raw = JSON.parsefile(path)
    det = raw["deterministic"]
    var = raw["variants"][variant]
    der = var["derived"]
    n, T = det["N"], det["T"]
    comps = StateComponent[]
    for c in der["state"]["components"]
        c["bits"] > 0 || continue
        push!(comps, StateComponent(Symbol(c["group"]), Tuple(Int.(c["index"])), c["bits"]))
    end
    Ω = [[_mat(atom, Int) for atom in stage] for stage in var["Omega"]]
    return V2Instance(
        variant, n, T, raw["gamma"], _mat(der["q_phi"], Int),
        _mat(det["R_ij"], Float64), _mat(det["p_jk"], Float64), Float64(det["C_p"]),
        _mat(det["d_ij"], Int), _mat(det["c_ij"], Int),
        der["C_max"], der["H_max"], der["Q1"], Float64(der["Q2"]), der["Q3"],
        Int.(det["A0"]), Int.(det["U0"]), Int.(det["W0"]),
        Ω, [Float64.(p) for p in var["probabilities"]],
        comps, Float64.(der["theta_tightened"]),
    )
end

n_state_bits(inst::V2Instance) = sum(c.bits for c in inst.comps)

"""Physical value of component `c` in the initial state S_1 (I1)-(I2)."""
function initial_component_value(inst::V2Instance, c::StateComponent)
    c.group === :A && return inst.Γ * inst.A0[c.index[1]]
    c.group === :U && return inst.Γ * inst.U0[c.index[1]]
    c.group === :W && return inst.W0[c.index[1]]
    return 0                                  # open jobs and pipeline tails empty
end

"""Standard binary expansion of every live component, concatenated."""
function initial_bits(inst::V2Instance)
    out = Int[]
    for c in inst.comps
        v = initial_component_value(inst, c)
        v < 2^c.bits || error("initial value $v of $(c.group)$(c.index) needs > $(c.bits) bits")
        append!(out, [(v >> l) & 1 for l in 0:c.bits-1])
    end
    return out
end

"""
    build_v2_model(inst; optimizer) -> SDDP.PolicyGraph

One node per physical stage.  Stage 1 has a single realization, so its node is
the deterministic root of Algorithm 2 and `SDDP.calculate_bound` is the stage-1
value under the current cuts.
"""
function build_v2_model(inst::V2Instance; optimizer)
    n, T, Γ = inst.n, inst.T, inst.Γ
    N = 1:n
    Q1, Q2, Q3 = inst.Q1, inst.Q2, inst.Q3
    Cmax, Hmax = inst.Cmax, inst.Hmax
    qφ, d, c = inst.qφ, inst.d, inst.c
    τ = [d[i, j] + c[j, k] for i in N, j in N, k in N]
    u = [inst.p[j, k] - d[i, j] - c[j, k] for i in N, j in N, k in N]

    # bit ranges per component, and lookup (group, index) -> component id
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

    # (IC-2): SDDP.jl takes one bound for every node's cost-to-go; the loosest
    # one that matters is the cost-to-go of stage 1, i.e. θ_2.
    lower_bound = inst.θ[2]

    return SDDP.LinearPolicyGraph(;
        stages = T, sense = :Min, lower_bound = lower_bound, optimizer = optimizer,
    ) do sp, t
        @variable(sp, λ[k = 1:nbits], Bin, SDDP.State, initial_value = bits0[k])
        # local-copy box as explicit rows (StrengthenedConicDuality drops bounds)
        @constraint(sp, [k = 1:nbits], λ[k].in >= 0)
        @constraint(sp, [k = 1:nbits], λ[k].in <= 1)

        function dec(group, idx, side)
            a = get(lookup, (group, idx), 0)
            a == 0 && return AffExpr(0.0)       # provably-zero component
            vars = side === :in ? [λ[k].in for k in ranges[a]] : [λ[k].out for k in ranges[a]]
            return @expression(sp, sum(2^(l - 1) * vars[l] for l in eachindex(vars)))
        end
        inA(j) = dec(:A, (j,), :in)
        inU(j) = dec(:U, (j,), :in)
        inW(k) = dec(:W, (k,), :in)
        inMrem(j, k) = dec(:Mrem, (j, k), :in)
        inPA(j, r) = r > Cmax ? AffExpr(0.0) : dec(:PA, (j, r), :in)
        inPU(j, r) = r > Cmax ? AffExpr(0.0) : dec(:PU, (j, r), :in)
        inG(k, r) = r > Hmax ? AffExpr(0.0) : dec(:G, (k, r), :in)

        # ── local variables ────────────────────────────────────────────────
        @variables(sp, begin
            0 <= Y[N] <= Q3, Int
            0 <= Yod[N, N] <= Q3, Int
            0 <= L[N] <= Q3, Int
            0 <= I[N] <= Q3, Int
            0 <= rA[N] <= Γ - 1, Int
            β[N], Bin
            0 <= mhat[N, N] <= Q3, Int
            0 <= M[N, N] <= Q3, Int
            0 <= mtil[N, N] <= Q3, Int
            0 <= x[N, N, N] <= Q1, Int
            0 <= v[N, N, N] <= Q3
            0 <= s[N] <= Q2
            δ[N, N, N], Bin
            η[N, N, N], Bin
            ζ[N], Bin
        end)

        # ── customer service (D1)-(D4); RHS set per realization ────────────
        @constraint(sp, [i in N], inA(i) == Γ * I[i] + rA[i])                 # (D1)
        @constraint(sp, [i in N], Y[i] <= I[i])                               # (D2)
        @constraint(sp, cYD[i in N], Y[i] <= 0)                               # Y <= D
        @constraint(sp, [i in N], Y[i] >= I[i] - Q3 * (1 - β[i]))
        @constraint(sp, cYDb[i in N], Y[i] + Q3 * β[i] >= 0)                  # Y >= D - Q3 β
        @constraint(sp, [i in N], sum(Yod[i, j] for j in N) == Y[i])          # (D3)
        @constraint(sp, cYod[i in N, j in N], Yod[i, j] <= 0)                 # Y_ij <= d_ij
        @constraint(sp, cL[i in N], L[i] + Y[i] == 0)                         # (D4)

        # ── task posting (P1)-(P2) ─────────────────────────────────────────
        @constraint(sp, [j in N], Γ * mhat[j, j] <= inU(j))
        @constraint(sp, [j in N], Γ * sum(mhat[j, k] for k in N if k != j) <= inA(j) - Γ * Y[j])

        # ── open jobs, matching capacity (M1)-(M3), (W1) ───────────────────
        @constraint(sp, [j in N, k in N], M[j, k] == inMrem(j, k) + mhat[j, k])
        @constraint(sp, [j in N, k in N], mtil[j, k] == sum(x[i, j, k] for i in N))
        @constraint(sp, [j in N, k in N], mtil[j, k] <= M[j, k])
        @constraint(sp, [i in N], sum(x[i, j, k] for j in N, k in N) <= inW(i))

        # ── stable matching (S1)-(S6) ──────────────────────────────────────
        for i in N, j in N, k in N
            H = sum(x[i2, j, k] for i2 in N if d[i2, j] <= d[i, j])
            @constraint(sp, v[i, j, k] <= H)
            @constraint(sp, v[i, j, k] <= M[j, k])
            @constraint(sp, v[i, j, k] <= Q3 * δ[i, j, k])
            @constraint(sp, v[i, j, k] >= M[j, k] - Q3 * (1 - δ[i, j, k]))
            @constraint(sp, x[i, j, k] <= Q1 * η[i, j, k])                   # (S2)
            @constraint(sp, u[i, j, k] >= s[i] - Q2 * (1 - η[i, j, k]))       # (S3)
            @constraint(sp, s[i] >= u[i, j, k] - Q2 * δ[i, j, k])             # (S6)
        end
        @constraint(sp, [i in N], sum(x[i, j, k] for j in N, k in N) >= inW(i) - Q1 * (1 - ζ[i]))  # (S4)
        @constraint(sp, [i in N], s[i] <= Q2 * ζ[i])                          # (S5)

        # ── outgoing pipelines of stage t (D7)-(D9), (G1)-(G2) ─────────────
        function pipe_in(inflow, carry)
            e = copy(carry)
            for (coef, var) in inflow
                add_to_expression!(e, coef, var)
            end
            return e
        end
        PA = Dict((j, r) => pipe_in([(Γ - qφ[i, j], Yod[i, j]) for i in N if c[i, j] == r],
                                    inPA(j, r + 1)) for j in N, r in 1:Cmax)
        PU = Dict((j, r) => pipe_in([(qφ[i, j], Yod[i, j]) for i in N if c[i, j] == r],
                                    inPU(j, r + 1)) for j in N, r in 1:Cmax)
        G = Dict((k, r) => pipe_in([(1, x[i, j, k]) for i in N, j in N if τ[i, j, k] == r],
                                   inG(k, r + 1)) for k in N, r in 1:Hmax)

        # ── linking-state transitions ──────────────────────────────────────
        if t < T
            @constraint(sp, [j in N], dec(:A, (j,), :out) ==
                inA(j) - Γ * Y[j] - Γ * sum(mhat[j, k] for k in N if k != j) + PA[(j, 1)] + Γ * G[(j, 1)])
            @constraint(sp, [j in N], dec(:U, (j,), :out) == inU(j) - Γ * mhat[j, j] + PU[(j, 1)])
            @constraint(sp, [k in N], dec(:W, (k,), :out) ==
                inW(k) - sum(x[k, i, j] for i in N, j in N) + G[(k, 1)])
            @constraint(sp, [j in N, k in N], dec(:Mrem, (j, k), :out) == M[j, k] - mtil[j, k])
            # dropped (provably zero) tails get `0 == expr`: a wrong bound derivation
            # shows up as infeasibility instead of silently losing bikes
            @constraint(sp, [j in N, r in 2:Cmax], dec(:PA, (j, r), :out) == PA[(j, r)])
            @constraint(sp, [j in N, r in 2:Cmax], dec(:PU, (j, r), :out) == PU[(j, r)])
            @constraint(sp, [k in N, r in 2:Hmax], dec(:G, (k, r), :out) == G[(k, r)])
        else
            # no stage T+1 state; the overnight reset lives outside the model
            @constraint(sp, [k = 1:nbits], λ[k].out == 0)
        end

        @stageobjective(sp,
            -sum(inst.R[i, j] * Yod[i, j] for i in N, j in N) +
            inst.Cp * sum(L[i] for i in N) +
            sum(inst.p[j, k] * mtil[j, k] for j in N, k in N))

        SDDP.parameterize(sp, inst.Ω[t], inst.prob[t]) do ω
            for i in N
                Di = sum(ω[i, j] for j in N)
                set_normalized_rhs(cYD[i], Di)
                set_normalized_rhs(cYDb[i], Di)
                set_normalized_rhs(cL[i], Di)
                for j in N
                    set_normalized_rhs(cYod[i, j], ω[i, j])
                end
            end
        end
    end
end

"""Decode a named λ state vector (Vector of values) into physical components."""
function decode_state(inst::V2Instance, bits::AbstractVector{<:Real})
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

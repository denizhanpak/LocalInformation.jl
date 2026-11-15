# Partial information decomposition with the pointwise redundancy functions ISx and IPM,
# and per-sample atoms (src/pid.jl).
import JSON3
using CUDA
using LinearAlgebra: istriu

pidest(r; base = 2, d = CodifyVariables(UniqueElements())) =
    JointProbabilities(PartialInformationDecomposition(r; base), d)
localmi(xs, t; base = 2) = local_association(
    JointProbabilities(MIShannon(; base), CodifyVariables(UniqueElements())), StateSpaceSet(xs...), t)
nodes_of(res) = [first(p) for p in atoms(res)]

# Samples whose empirical distribution is `states` weighted by `weights`.
function repeat_states(states, weights)
    rows = reduce(vcat, [fill(k, w) for (k, w) in enumerate(weights)])
    nv = length(states[1])
    return rows, [[Int(states[k][v]) for k in rows] for v in 1:nv]
end

@testset "Exact values, two sources" begin
    l2(x) = log2(x)
    bits = [(a, b) for a in 0:1 for b in 0:1]
    x1, x2 = first.(bits), last.(bits)
    HT_and = 2 - 0.75l2(3)      # H(X1 ∧ X2)
    I1_and = 1.5 - 0.75l2(3)    # I(X1; X1 ∧ X2)
    red_and = 0.5l2(32 / 27)
    # (target, ISx atoms, IPM atoms), atoms as (redundancy, unique 1, unique 2, synergy).
    cases = Dict(
        "xor" => (x1 .⊻ x2, (l2(2 / 3), l2(3 / 2), l2(3 / 2), l2(4 / 3)), (0, 0, 0, 1)),
        "and" => (x1 .& x2, (red_and, I1_and - red_and, I1_and - red_and, HT_and - 2I1_and + red_and),
                  (1.75 - 0.75l2(3), -0.25, -0.25, 0.75)),
        "copy" => (2x1 .+ x2, (l2(4 / 3), l2(3 / 2), l2(3 / 2), l2(4 / 3)), (1, 0, 0, 1)),
    )
    for (name, (t, isx, ipm)) in cases, (r, expected) in ((ISx(), isx), (IPM(), ipm))
        est = pidest(r)
        res = association(est, x1, x2, t)
        got = (redundancy(res), unique_information(res, 1), unique_information(res, 2), synergy(res))
        @test all(isapprox.(got, expected; atol = 1e-12))
        l = local_association(est, x1, x2, t)
        @test all(isapprox.(mean.((redundancy(l), unique_information(l, 1),
            unique_information(l, 2), synergy(l))), expected; atol = 1e-12))
    end
    # Two-bit copy with the target given as a 2-column StateSpaceSet.
    l = local_association(pidest(ISx()), x1, x2, StateSpaceSet(x1, x2))
    @test mean(synergy(l)) ≈ log2(4 / 3)
    # Redundant source: one bit of redundancy, nothing else.
    for r in (ISx(), IPM())
        res = association(pidest(r), [0, 1], [0, 1], [0, 1])
        @test redundancy(res) ≈ 1
        @test all(abs(last(a)) < 1e-12 for a in atoms(res)[2:end])
    end
end

@testset "Cross-check against syntropy" begin
    fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "syntropy_pid_fixtures.json"), String))
    @test fx.meta.unit == "bits"
    for (name, case) in pairs(fx.cases)
        rows, vars = repeat_states(case.states, case.weights)
        n = length(vars) - 1
        for (key, r) in ((:isx, ISx()), (:ipm, IPM()))
            est = pidest(r)
            res = association(est, vars...)
            l = local_association(est, vars...)
            avg = case[key].avg
            @test length(avg) == length(res.atoms)
            @test all(isapprox(atom(res, a), v; atol = 1e-10) for (a, v) in avg)
            # Pointwise atoms of every sample equal syntropy's for its state.
            ok = true
            for (k, ptw) in enumerate(case[key].ptw)
                i = findfirst(==(k), rows)
                ok &= all(isapprox(atom(l, a)[i], v; atol = 1e-10) for (a, v) in ptw)
            end
            @test ok
        end
    end
end

@testset "Invariant, $n sources, base $b" for n in 2:4, b in BASES
    rng = Xoshiro(10 + n)
    N = 600
    xs = [randn(rng, N) for _ in 1:n]
    t = xs[1] .+ xs[end] .* randn(rng, N) .+ 0.5randn(rng, N)
    d = CodifyVariables(ValueBinning(RectangularBinning(3, true)))
    xi = [rand(rng, 0:2, N) for _ in 1:n]
    ti = mod.(xi[1] .+ (n > 2 ? xi[2] .* xi[3] : xi[2]) .+ rand(rng, 0:1, N), 3)
    for r in (ISx(), IPM())
        for (inputs, dd) in (((xs..., t), d), ((xi..., ti), CodifyVariables(UniqueElements())))
            est = pidest(r; base = b, d = dd)
            res = association(est, inputs...)
            l = local_association(est, inputs...)
            @test l isa LocalPIDResult{<:PartialInformationDecomposition,Float64}
            @test size(l.atoms) == (N, length(res.atoms))
            # Columns are in lattice order, bottom first, like `atoms(::PIDResult)`.
            @test nodes_of(l) == nodes_of(res)
            @test all(isapprox(mean(v), atom(res, a); atol = 1e-12) for (a, v) in atoms(l))
            # A single-source node is the plug-in mutual information (Associations' encoding).
            mi = association(JointProbabilities(MIShannon(; base = b), dd), inputs[1], inputs[end])
            @test res.redundancies[res.lattice.index[[UInt(1)]]] ≈ mi atol = 1e-12
            # The atoms sum to the local mutual information of all sources and the target.
            if dd.outcome_spaces[1] isa UniqueElements
                lmi = localmi(inputs[1:n], inputs[end]; base = b)
                @test vec(sum(l.atoms; dims = 2)) ≈ lmi atol = 1e-10
                @test sum(res.atoms) ≈ mean(lmi) atol = 1e-12
            end
            # A selection gives the same columns as `atoms = :all`.
            sel = nodes_of(res)[[end, 1, 2]]
            ls = local_association(est, inputs...; atoms = sel)
            @test ls.atoms ≈ hcat((atom(l, a) for a in sel)...) atol = 1e-12
            @test nodes_of(ls) == sel
        end
    end
    # A multi-column source.
    est = pidest(ISx(); base = b, d = d)
    S = StateSpaceSet(xs[1], xs[2])
    @test all(isapprox(mean(v), atom(association(est, S, xs[2:n]..., t), a); atol = 1e-12)
              for (a, v) in atoms(local_association(est, S, xs[2:n]..., t)))
end

@testset "Möbius function" begin
    for n in 3:4
        L = Associations.redundancy_lattice(n)
        m = length(L.nodes)
        # μ(i, j) is atom j of the inversion of the unit redundancy vector e_i.
        Minv = zeros(Int, m, m)
        for i in 1:m
            e = zeros(m); e[i] = 1
            Minv[i, :] .= round.(Int, Associations.mobius_inversion(L, e))
        end
        ok = true
        for j in 1:m
            idx, val = LI.mobius_column(L, j)
            col = zeros(Int, m); col[idx] .= val
            ok &= col == Minv[:, j]
        end
        @test ok
        Z = LI.pid_lattice_data(n).zeta
        @test istriu(Z) && all(Z[k, k] for k in 1:m)
    end
end

@testset "Five sources (selected atoms)" begin
    rng = Xoshiro(55)
    N = 2000
    xs = [rand(rng, 0:1, N) for _ in 1:5]
    t = xs[1] .⊻ xs[2] .| (xs[3] .& xs[4] .& xs[5])
    for r in (ISx(), IPM())
        est = pidest(r)
        sel = [[collect(1:5)], [[i] for i in 1:5], [[3]], [[1, 2], [3, 4, 5]]]
        l = local_association(est, xs..., t; atoms = sel)
        @test size(l.atoms) == (N, 4)
        res = association(est, xs..., t)
        @test mean(synergy(l)) ≈ synergy(res) atol = 1e-12
        @test mean(redundancy(l)) ≈ redundancy(res) atol = 1e-12
        @test mean(unique_information(l, 3)) ≈ unique_information(res, 3) atol = 1e-12
        @test mean(atom(l, [[3, 4, 5], [1, 2]])) ≈ atom(res, [[1, 2], [3, 4, 5]]) atol = 1e-12
        @test sum(res.atoms) ≈ mean(localmi(xs, t)) atol = 1e-10
    end
    # `atoms = :all` for 5 sources and many samples is refused before allocating.
    e = try local_association(pidest(ISx()), xs..., t; memory_limit = 2^20) catch err; err end
    @test e isa ArgumentError && occursin("memory_limit", e.msg) && occursin("atoms = ", e.msg)
end

@testset "Accessors, show and errors" begin
    rng = Xoshiro(7)
    x, y, z = rand(rng, 0:1, 200), rand(rng, 0:1, 200), rand(rng, 0:1, 200)
    t = x .⊻ y
    est = pidest(IPM())
    l = local_association(est, x, y, z, t; atoms = [[[1, 2, 3]]])
    @test synergy(l) == l.atoms[:, 1]
    # PIDAtom: per-sample values of one atom, whose mean is `association`.
    for a in (:synergy, :redundancy, :unique => 2, [[1], [2, 3]])
        e = JointProbabilities(PIDAtom(PartialInformationDecomposition(IPM()), a), CodifyVariables(UniqueElements()))
        @test invariant_holds(e, x, y, z, t)
    end
    msg(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $e" end
    @test occursin("was not computed", msg(() -> redundancy(l)))
    @test occursin("not a node", msg(() -> local_association(est, x, y, z, t; atoms = [[[1], [1, 2]]])))
    @test occursin("wrap it", msg(() -> local_association(est, x, y, z, t; atoms = [[1], [2]])))
    @test occursin("between 1 and 3", msg(() -> atom(l, [[4]])))
    @test_throws ArgumentError local_association(est, x, y, z, t; atoms = :synergy)
    s = sprint(show, MIME"text/plain"(), local_association(est, x, y, t))
    @test occursin("LocalPIDResult: 2 sources, redundancy function IPM", s) && occursin("{12}", s)
    l32 = local_association(est, x, y, t; precision = Float32)
    @test eltype(l32.atoms) === Float32
    @test l32.atoms ≈ local_association(est, x, y, t).atoms rtol = 1e-5
    # IMin and IMMI have no per-sample form.
    for r in (IMin(), IMMI())
        m = msg(() -> local_association(pidest(r), x, y, t))
        @test occursin(string(nameof(typeof(r))), m) && occursin("ISx()", m)
        @test occursin("not an average of per-sample terms", m)
    end
    @test occursin("ISx", msg(() -> local_association(
        GaussianDecomposition(PartialInformationDecomposition(IMMI())), randn(50), randn(50), randn(50))))
    # ISx and IPM are discrete-only, for `association` and `local_association`.
    for r in (ISx(), IPM()), est2 in (GaussianDecomposition(PartialInformationDecomposition(r)),
                                       EntropyDecomposition(PartialInformationDecomposition(r), Kraskov()))
        a, b, c = randn(rng, 50), randn(rng, 50), randn(rng, 50)
        m = msg(() -> association(est2, a, b, c))
        @test occursin("discrete data only", m) && occursin(string(nameof(typeof(r))), m)
        @test occursin("discrete data only", msg(() -> local_association(est2, a, b, c)))
    end
    @test occursin("CodifyVariables", msg(() -> association(
        JointProbabilities(PartialInformationDecomposition(ISx()), CodifyPoints(UniqueElementsEncoding(x))), x, y, t)))
end

@testset "GPU Möbius inversion" begin
    if CUDA.functional()
        gpu = CUDABackend()
        rng = Xoshiro(77)
        N = 3000
        for n in (3, 4), r in (ISx(), IPM())
            xs = [rand(rng, 0:3, N) for _ in 1:n]
            t = mod.(xs[1] .+ xs[2] .* xs[end] .+ rand(rng, 0:1, N), 4)
            est = pidest(r)
            lc = local_association(est, xs..., t)
            lg64 = local_association(est, xs..., t; backend = gpu, precision = Float64)
            @test lg64.atoms ≈ lc.atoms rtol = 1e-12 atol = 1e-12
            lg = local_association(est, xs..., t; backend = gpu)
            @test eltype(lg.atoms) === Float32
            @test maximum(abs, lg.atoms .- lc.atoms) ≤ 1e-4 * max(1, maximum(abs, lc.atoms))
        end
        xs = [rand(rng, 0:2, N) for _ in 1:5]
        t = mod.(xs[1] .+ xs[2] .* xs[3] .+ xs[4], 3)
        sel = [[collect(1:5)], [[i] for i in 1:5], [[1, 2], [3]]]
        est = pidest(ISx())
        lc = local_association(est, xs..., t; atoms = sel)
        @test local_association(est, xs..., t; atoms = sel, backend = gpu, precision = Float64).atoms ≈ lc.atoms rtol = 1e-12 atol = 1e-12
        lg = local_association(est, xs..., t; atoms = sel, backend = gpu)
        @test maximum(abs, lg.atoms .- lc.atoms) ≤ 1e-4 * max(1, maximum(abs, lc.atoms))
    else
        @info "CUDA not functional; skipping GPU PID tests."
        @test_skip false
    end
end

# High-cardinality sources take the hash-table branch of `joint_labels` (taken when the
# product of the two state counts exceeds max(4N, 2^16)).
@testset "Many states: hash-table joint labels" begin
    LI = LocalInformation
    g = Xoshiro(21)
    N = 1000
    a = rand(g, 1:400, N); b = rand(g, 1:400, N)
    la, ka = LI.relabel(a); lb, kb = LI.relabel(b)
    @test widemul(ka, kb) > max(4N, 1 << 16)   # the branch under test
    labels, k = LI.joint_labels(la, ka, lb, kb)
    ref, kref = LI.relabel(collect(zip(la, lb)))
    @test labels == ref && k == kref
    # The dense branch gives the same labels on small inputs.
    sa, ska = LI.relabel(rand(g, 1:5, 200)); sb, skb = LI.relabel(rand(g, 1:5, 200))
    @test first(LI.joint_labels(sa, ska, sb, skb)) == first(LI.relabel(collect(zip(sa, sb))))

    # End to end: with high-cardinality sources, single-source redundancies still equal
    # Associations' mutual information, local atoms still average to the PIDResult atoms,
    # and per-sample atoms still sum to the local MI.
    x1 = rand(g, 1:400, N); x2 = rand(g, 1:400, N); t = mod.(x1 .+ x2, 3)
    d = CodifyVariables(UniqueElements())
    for r in (ISx(), IPM())
        est = pidest(r)
        res = association(est, x1, x2, t)
        @test unique_information(res, 1) + redundancy(res) ≈
              association(JointProbabilities(MIShannon(; base = 2), d), x1, t) atol = 1e-10
        lres = local_association(est, x1, x2, t)
        for (node, _) in atoms(res)
            @test mean(atom(lres, node)) ≈ atom(res, node) atol = 1e-10
        end
        @test vec(sum(lres.atoms; dims = 2)) ≈ localmi((x1, x2), t) atol = 1e-9
    end
end

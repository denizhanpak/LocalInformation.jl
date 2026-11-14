import JSON3

rng = Xoshiro(11)
N = 500
x = randn(rng, N); y = x .+ randn(rng, N); z = y .+ randn(rng, N)
W = StateSpaceSet(randn(rng, N, 2) .+ x)
V = StateSpaceSet(randn(rng, N, 3) .+ y)

@testset "KSG1, KSG2 invariant, base $b" for b in BASES
    for E in (KSG1, KSG2)
        for est in (E(MIShannon(; base = b); k = 3), E(MIShannon(; base = b); k = 2, w = 3),
                    E(MIShannon(; base = b); k = 4, metric_marginals = LI.Euclidean()))
            for (a, c) in ((x, y), (W, y), (x, V), (W, V))
                @test invariant_holds(est, a, c)
            end
        end
    end
end

@testset "FPVP invariant, base $b" for b in BASES
    for est in (FPVP(CMIShannon(; base = b); k = 3), FPVP(CMIShannon(; base = b); k = 2, w = 2))
        for (a, c, d) in ((x, z, y), (x, z, W), (W, V, z), (x, y, V))
            @test invariant_holds(est, a, c, d)
        end
    end
end

@testset "Counts against brute force" begin
    # Independent O(N²) implementation of the conventions in src/ksg.jl.
    n = 150
    a, c, d = StateSpaceSet(randn(rng, n, 2)), randn(rng, n), randn(rng, n)
    cheb(P, i, j) = maximum(abs, P[:, i] .- P[:, j])
    A, C, Dm = LI.point_matrix(a), LI.point_matrix(c), LI.point_matrix(d)
    function kth(P, i, k, w)
        cand = [j for j in 1:n if abs(i - j) > w]
        o = sortperm([cheb(P, i, j) for j in cand])
        return cheb(P, i, cand[o[k]]), cand[o[k]]
    end
    within(P, i, r) = count(j -> cheb(P, i, j) ≤ r, 1:n)
    for (k, w) in ((1, 0), (3, 0), (2, 4))
        J = vcat(A, C)
        ε = [kth(J, i, k, w)[1] for i in 1:n]
        @test LI.ksg_counts(KSG1(; k, w), a, c) == [[within(P, i, ε[i]) for i in 1:n] for P in (A, C)]
        js = [kth(J, i, k, w)[2] for i in 1:n]
        @test LI.ksg_counts(KSG2(; k, w), a, c) == [[within(P, i, cheb(P, i, js[i])) - 1 for i in 1:n] for P in (A, C)]
        J3 = vcat(A, C, Dm)
        ε3 = [kth(J3, i, k, w)[1] for i in 1:n]
        @test LI.ksg_counts(FPVP(; k, w), a, c, d) ==
              [[within(P, i, ε3[i]) for i in 1:n] for P in (vcat(A, Dm), vcat(C, Dm), Dm)]
    end
end

@testset "Precision, errors" begin
    est = KSG1(; k = 3)
    l32 = local_association(est, x, y; precision = Float32)
    @test eltype(l32) === Float32
    @test l32 ≈ local_association(est, x, y) rtol = 1e-6
    @test_throws ArgumentError local_association(KSG1(; k = 3), x[1:3], y[1:3])
    @test_throws ArgumentError local_association(KSG2(; k = 2, w = 2), x[1:6], y[1:6])
    @test_throws ArgumentError local_association(FPVP(; k = 3, w = 10), x[1:20], y[1:20], z[1:20])
    @test_throws ArgumentError local_association(KSG1(), x, y, z) # MIShannon takes 2 inputs
    @test_throws ArgumentError local_association(KSG1(), x, y[1:10])
end

@testset "Cross-check against syntropy" begin
    # syntropy (fixtures/generate_syntropy_ksg_fixtures.py) uses the same k-th neighbour
    # (Chebyshev, self excluded) but different count conventions; every difference below is
    # predicted exactly from our counts.
    fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "syntropy_ksg_fixtures.json"), String))
    data = [Float64.(v) for v in fx.data]
    k = Int(fx.k)
    getvar(idx) = length(idx) == 1 ? data[idx[1]+1] : StateSpaceSet((data[i+1] for i in idx)...)
    X, Y = getvar(fx.mi_x), getvar(fx.mi_y)
    syn1, syn2, syn2k1 = Float64.(fx.ksg1), Float64.(fx.ksg2), Float64.(fx.ksg2_k1)

    # KSG1. syntropy counts strictly inside ε (paper), we count `≤ ε` (Associations), so we
    # also count the k-th joint neighbour in the marginal m* where it attains ε. That term
    # changes from ψ(n - 1) to ψ(n): syntropy - ours = 1/(n_{m*} - 1).
    est = KSG1(MIShannon(; base = ℯ); k)
    l = local_association(est, X, Y)
    n = LI.ksg_counts(est, X, Y)
    @test all(t -> any(m -> isapprox(syn1[t] - l[t], 1 / (n[m][t] - 1); atol = 1e-10), 1:2), eachindex(l))

    # KSG2 with k = 1: syntropy's ε_m (maximum over the k neighbours) and constant 1/k equal
    # Associations' (the k-th neighbour, constant 1): identical local values.
    @test local_association(KSG2(MIShannon(; base = ℯ); k = 1), X, Y) ≈ syn2k1 rtol = 1e-12

    # KSG2 with k > 1: syntropy's ε_m ≥ ours, so its counts are ≥ ours, and its constant is
    # larger by 1 - 1/k. Equality holds where the k-th neighbour is the farthest in both marginals.
    l2 = local_association(KSG2(MIShannon(; base = ℯ); k), X, Y)
    Δ = syn2 .- (l2 .+ (1 - 1 / k))
    @test all(≤(1e-10), Δ)
    @test count(δ -> abs(δ) < 1e-10, Δ) > length(Δ) ÷ 4

    # FPVP. Same `<` vs `≤` difference: where ε is attained in x (y), n_xz (n_yz) gains the
    # neighbour; where it is attained in z, n_xz, n_yz and n_z all do.
    Xc, Yc, Zc = getvar(fx.cmi_x), getvar(fx.cmi_y), getvar(fx.cmi_z)
    est = FPVP(CMIShannon(; base = ℯ); k)
    lc = local_association(est, Xc, Yc, Zc)
    nxz, nyz, nz = LI.ksg_counts(est, Xc, Yc, Zc)
    sync = Float64.(fx.cmi)
    predicted(t) = (1 / (nxz[t] - 1), 1 / (nyz[t] - 1), 1 / (nxz[t] - 1) + 1 / (nyz[t] - 1) - 1 / (nz[t] - 1))
    @test all(t -> any(p -> isapprox(sync[t] - lc[t], p; atol = 1e-10), predicted(t)), eachindex(lc))
end

rng = Xoshiro(99)
N = 800
x = randn(rng, N); y = x .+ randn(rng, N); z = y .+ randn(rng, N)
W = StateSpaceSet(randn(rng, N, 2) .+ z)
higher_order = (TotalCorrelation, DualTotalCorrelation, OInformation, SInformation)

@testset "Invariant, base $b" for b in BASES
    for M in higher_order
        d = CodifyVariables(ValueBinning(RectangularBinning(3)))
        @test invariant_holds(JointProbabilities(M(; base = b), d), x, y, z)
        @test invariant_holds(JointProbabilities(M(; base = b), d), x, y, W)
        @test invariant_holds(EntropyDecomposition(M(; base = b), PlugIn(Shannon()), d), x, y, z, W)
        @test invariant_holds(EntropyDecomposition(M(; base = b), Kraskov(; k = 3)), x, y, W)
        @test invariant_holds(EntropyDecomposition(M(; base = b), KozachenkoLeonenko()), x, y, z)
    end
end

# Exactly balanced binary data, so that every empirical probability is exact.
balanced_bits(rng, n) = shuffle(rng, repeat([0, 1], n ÷ 2))
function balanced_pairs(rng, n)
    p = shuffle(rng, repeat([(0, 0), (0, 1), (1, 0), (1, 1)], n ÷ 4))
    return first.(p), last.(p)
end
u = CodifyVariables(UniqueElements())

@testset "COPY: every local MI is 1 bit" begin
    a = balanced_bits(rng, 400)
    @test local_association(JointProbabilities(MIShannon(), u), a, a) ≈ ones(400)
    @test local_association(JointProbabilities(MIShannon(; base = ℯ), u), a, copy(a)) ≈ fill(log(2), 400)
end

@testset "XOR" begin
    x1, x2 = balanced_pairs(rng, 400)
    t = x1 .⊻ x2
    @test local_association(JointProbabilities(MIShannon(), u), x1, x2) ≈ zeros(400) atol = 1e-12
    @test local_association(JointProbabilities(MIShannon(), u), x1, t) ≈ zeros(400) atol = 1e-12
    @test local_association(JointProbabilities(MIShannon(), u), StateSpaceSet(x1, x2), t) ≈ ones(400)
    # Pure synergy: Ω = 2 - n = -1 bit, and locally -1 bit at every sample.
    lo = local_association(JointProbabilities(OInformation(), u), x1, x2, t)
    @test lo ≈ fill(-1.0, 400)
    @test mean(lo) ≈ association(JointProbabilities(OInformation(), u), x1, x2, t) ≈ -1
    # Same with the decomposition estimator.
    lo2 = local_association(EntropyDecomposition(OInformation(), PlugIn(Shannon()), u), x1, x2, t)
    @test lo2 ≈ lo
end

@testset "Redundant Gaussian copies: Ω > 0" begin
    s = randn(rng, N)
    copies = [s .+ 0.1 .* randn(rng, N) for _ in 1:3]
    lo = local_association(GaussianDecomposition(OInformation()), copies...)
    @test mean(lo) > 0
    @test mean(lo) ≈ association(GaussianDecomposition(OInformation()), copies...)
    @test count(>(0), lo) > N ÷ 2
end

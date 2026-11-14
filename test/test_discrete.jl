rng = Xoshiro(1234)
N = 1000
x = randn(rng, N); y = x .+ randn(rng, N); z = y .+ randn(rng, N)
W = StateSpaceSet(randn(rng, N, 2) .+ x)          # multi-column input
xi, yi, zi = rand(rng, 1:3, N), rand(rng, 1:4, N), rand(rng, 1:2, N)
yi .= mod1.(xi .+ yi, 4)

discretizations = (
    CodifyVariables(ValueBinning(RectangularBinning(4))),
    CodifyVariables(OrdinalPatterns{3}()),
)

@testset "JointProbabilities MI/CMI, base $b" for b in BASES
    for d in discretizations
        @test invariant_holds(JointProbabilities(MIShannon(; base = b), d), x, y)
        @test invariant_holds(JointProbabilities(CMIShannon(; base = b), d), x, z, y)
    end
    d = CodifyVariables(ValueBinning(RectangularBinning(3)))
    @test invariant_holds(JointProbabilities(MIShannon(; base = b), d), W, y)
    @test invariant_holds(JointProbabilities(CMIShannon(; base = b), d), W, z, y)
    u = CodifyVariables(UniqueElements())
    @test invariant_holds(JointProbabilities(MIShannon(; base = b), u), xi, yi)
    @test invariant_holds(JointProbabilities(CMIShannon(; base = b), u), xi, yi, zi)
    @test invariant_holds(JointProbabilities(MIShannon(; base = b), u), StateSpaceSet(xi, zi), yi)
    # One outcome space per input.
    d2 = CodifyVariables((OrdinalPatterns{3}(), OrdinalPatterns{3}()))
    @test invariant_holds(JointProbabilities(MIShannon(; base = b), d2), x, y)
end

@testset "EntropyDecomposition with PlugIn, base $b" for b in BASES
    for d in (CodifyVariables(ValueBinning(RectangularBinning(4))), CodifyVariables(UniqueElements()),
              CodifyVariables(OrdinalPatterns{3}()))
        @test invariant_holds(EntropyDecomposition(MIShannon(; base = b), PlugIn(Shannon()), d), x, y)
        @test invariant_holds(EntropyDecomposition(CMIShannon(; base = b), PlugIn(Shannon()), d), x, z, y)
    end
    d = CodifyVariables(ValueBinning(RectangularBinning(3)))
    @test invariant_holds(EntropyDecomposition(MIShannon(; base = b), PlugIn(Shannon()), d), W, y)
end

@testset "local_entropy with PlugIn, base $b" for b in BASES
    est = PlugIn(Shannon(; base = b))
    for (o, data) in ((ValueBinning(RectangularBinning(5)), x), (ValueBinning(RectangularBinning(3, true)), W),
                      (OrdinalPatterns{3}(), x), (UniqueElements(), xi))
        l = local_entropy(est, o, data)
        @test mean(l) ≈ information(est, o, data)
        @test all(≥(0), l)
    end
end
# With `precise = false`, one point of W falls outside the binning; `information` drops it.
@test any(==(-1), codify(ValueBinning(RectangularBinning(3)), W))
@test_throws ArgumentError local_entropy(ValueBinning(RectangularBinning(3)), W)
@test local_entropy(UniqueElements(), xi) ≈ local_entropy(PlugIn(Shannon()), UniqueElements(), xi)

# With an embedding outcome space there is one local value per symbol.
@test length(local_association(JointProbabilities(MIShannon(), CodifyVariables(OrdinalPatterns{3}())), x, y)) == N - 2

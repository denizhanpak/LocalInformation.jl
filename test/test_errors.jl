rng = Xoshiro(3)
x, y = randn(rng, 100), randn(rng, 100)
d = CodifyVariables(ValueBinning(RectangularBinning(3)))

msg(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $e" end

for hest in (MillerMadow(), ChaoShen(), Jackknife(), Schuermann())
    m = msg(() -> local_association(EntropyDecomposition(MIShannon(), hest, d), x, y))
    @test occursin(string(nameof(typeof(hest))), m)
    @test occursin("PlugIn(Shannon())", m)
    m = msg(() -> local_entropy(hest, ValueBinning(3), x))
    @test occursin(string(nameof(typeof(hest))), m)
end
@test occursin("Lord", msg(() -> local_entropy(Lord(), x)))
@test occursin("Lord", msg(() -> local_association(EntropyDecomposition(MIShannon(), Lord()), x, y)))
@test occursin("JointEntropyShannon", msg(() -> local_association(JointProbabilities(JointEntropyShannon(), d), x, y)))
@test occursin("Supported estimators", msg(() -> local_association(JointProbabilities(JointEntropyShannon(), d), x, y)))
@test occursin("CodifyPoints", msg(() -> local_association(
    JointProbabilities(MIShannon(), CodifyPoints(OrdinalPatternEncoding(2))), StateSpaceSet(x, y), StateSpaceSet(y, x))))
@test_throws ArgumentError local_association(GaussianMI(), x, y; backend = :gpu)
@test_throws ArgumentError local_association(GaussianMI(), x, y; precision = Int)

# Outcome spaces whose probabilities are not relative counts of the per-sample outcomes
# that `codify` returns: a local plug-in entropy could not average to `information`.
@testset "local_entropy: outcome spaces without per-sample counts" begin
    z = randn(Xoshiro(31), 500)
    # Not counting-based at all.
    for o in (WeightedOrdinalPatterns{3}(), AmplitudeAwareOrdinalPatterns{3}(), PowerSpectrum())
        @test occursin("counting-based", msg(() -> local_entropy(PlugIn(Shannon()), o, z)))
    end
    # Counting-based, but `codify` (one symbol per sample) and `probabilities` (dispersion
    # patterns) describe different outcomes. Before this check, the local mean was ~1.58
    # bits against `information` ≈ 3.16 bits on this data.
    @test occursin("relative counts", msg(() -> local_entropy(PlugIn(Shannon()), Dispersion(), z)))
    # Count-based spaces still work and satisfy the invariant.
    for (o, data) in ((OrdinalPatterns{3}(), z), (OrdinalPatterns{4}(), z),
                      (UniqueElements(), rand(Xoshiro(32), 1:6, 500)),
                      (ValueBinning(RectangularBinning(5, true)), z))
        @test mean(local_entropy(PlugIn(Shannon()), o, data)) ≈ information(PlugIn(Shannon()), o, data) rtol = 1e-12
    end
end

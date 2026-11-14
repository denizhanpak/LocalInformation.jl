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

# GPU tests run only when CUDA.jl is installed and functional.
using CUDA

if CUDA.functional()
    gpu = CUDABackend()
    rng = Xoshiro(5)
    N = 5000
    G = randn(rng, N, 5) * randn(rng, 5, 5)
    x, y, z = G[:, 1], G[:, 2], G[:, 3]
    W = StateSpaceSet(G[:, 4:5])
    # Float32 tolerance: relative 1e-4 on means, elementwise absolute 1e-4 × the largest
    # local magnitude.
    function close32(a, b)
        scale = max(1.0, maximum(abs, b))
        return eltype(a) === Float32 && isapprox(mean(a), mean(b); rtol = 1e-4, atol = 1e-5 * scale) &&
            maximum(abs, a .- b) ≤ 1e-4 * scale
    end
    ests = Any[GaussianMI(), GaussianCMI(CMIShannon(; base = ℯ))]
    for M in (TotalCorrelation, DualTotalCorrelation, OInformation, SInformation), c in (false, true)
        push!(ests, GaussianDecomposition(M(); copula = c))
    end
    for est in ests
        inputs = est isa GaussianMI ? (x, W) : est isa GaussianCMI ? (x, y, W) : (x, y, z, W)
        lc = local_association(est, inputs...)
        lg = local_association(est, inputs...; backend = gpu)
        @test close32(lg, lc)
        @test mean(lg) ≈ association(est, inputs...) rtol = 1e-4 atol = 1e-5
        lg64 = local_association(est, inputs...; backend = gpu, precision = Float64)
        @test eltype(lg64) === Float64
        @test lg64 ≈ lc rtol = 1e-10
    end
    # Discrete estimators count on the CPU and return the requested precision.
    est = JointProbabilities(MIShannon(), CodifyVariables(ValueBinning(RectangularBinning(4))))
    l = local_association(est, x, y; backend = gpu)
    @test eltype(l) === Float32
    @test l ≈ local_association(est, x, y) rtol = 1e-6
    # No GPU k-NN yet (Phase 3 adds it).
    @test_throws ArgumentError local_association(EntropyDecomposition(MIShannon(), Kraskov()), x, y; backend = gpu)
else
    @info "CUDA not functional; skipping GPU tests."
    @test_skip false
end

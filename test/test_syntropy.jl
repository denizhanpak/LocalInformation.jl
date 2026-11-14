# Cross-check against syntropy (fixtures/generate_syntropy_fixtures.py). syntropy works in
# nats, takes data as (n_variables, n_samples), and uses the maximum-likelihood covariance
# (ddof = 0); its k-NN entropy is -ψ(k) + ψ(N) + d log(2ρ_i).
import JSON3

fx = JSON3.read(read(joinpath(@__DIR__, "fixtures", "syntropy_fixtures.json"), String))

@testset "Gaussian" begin
    g = fx.gaussian
    vars = [Float64.(v) for v in g.data]
    N = length(vars[1])
    H = LI.local_subset_entropies(GaussianDecomposition(OInformation(; base = ℯ)), vars...)
    # Local entropies differ by the constant (d/2) log(N/(N-1)): we use the N-1 covariance
    # for log det (as Associations), syntropy the ML covariance (see src/gaussian.jl).
    for (key, S) in ((:h_all, [1, 2, 3, 4]), (:h_01, [1, 2]), (:h_2, [3]))
        shift = (length(S) / 2) * log(N / (N - 1))
        @test H(S) ≈ Float64.(g[key]) .+ shift rtol = 1e-10
    end
    # The constants cancel in MI and higher-order measures: elementwise equality.
    lmi = local_association(GaussianMI(MIShannon(; base = ℯ)), vars[1], StateSpaceSet(vars[2], vars[3]))
    @test lmi ≈ Float64.(g.mi_0_12) rtol = 1e-9
    for (key, M) in ((:tc, TotalCorrelation), (:dtc, DualTotalCorrelation),
                     (:oinfo, OInformation), (:sinfo, SInformation))
        l = local_association(GaussianDecomposition(M(; base = ℯ)), vars...)
        @test l ≈ Float64.(g[key]) rtol = 1e-9
    end
end

@testset "k-NN differential entropy" begin
    kd = fx.knn
    X = StateSpaceSet([Float64.(v) for v in kd.data]...)
    k = Int(kd.k)
    d = 2
    # p = ∞: identical to our Kraskov form with the Chebyshev metric (unit-ball volume 2^d).
    linf = LI.local_knn_entropies(X, k, 0, CPU(), Float64; metric = LI.Chebyshev())
    @test linf ≈ Float64.(kd.h_pinf) rtol = 1e-10
    @test mean(linf) ≈ kd.mean_pinf rtol = 1e-10
    # p = 2: syntropy keeps the 2^d (Chebyshev) volume with Euclidean distances, so it
    # differs from ComplexityMeasures' `Kraskov` by the constant d log 2 - log V_d.
    l2 = local_entropy(Kraskov(Shannon(; base = ℯ); k), X)
    @test Float64.(kd.h_p2) .- l2 ≈ fill(d * log(2) - log(π), length(l2)) rtol = 1e-10
end

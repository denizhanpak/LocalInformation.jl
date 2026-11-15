rng = Xoshiro(7)
N = 600
x = randn(rng, N); y = x .+ randn(rng, N); z = y .+ randn(rng, N)
W = StateSpaceSet(randn(rng, N, 2) .+ x)

@testset "local_entropy, base $b" for b in BASES
    for est in (Kraskov(Shannon(; base = b); k = 3), Kraskov(Shannon(; base = b); k = 2, w = 3),
                KozachenkoLeonenko(Shannon(; base = b)), KozachenkoLeonenko(Shannon(; base = b); w = 2))
        for data in (x, W)
            @test mean(local_entropy(est, data)) ≈ information(est, data) rtol = 1e-10
        end
    end
end

@testset "EntropyDecomposition MI/CMI, base $b" for b in BASES
    for hest in (Kraskov(; k = 4), Kraskov(; k = 3, w = 2), KozachenkoLeonenko())
        @test invariant_holds(EntropyDecomposition(MIShannon(; base = b), hest), x, y)
        @test invariant_holds(EntropyDecomposition(MIShannon(; base = b), hest), W, y)
        @test invariant_holds(EntropyDecomposition(CMIShannon(; base = b), hest), x, z, y)
        @test invariant_holds(EntropyDecomposition(CMIShannon(; base = b), hest), x, z, W)
    end
end

# Kraskov with k = 1 and KozachenkoLeonenko differ only in the constants
# -ψ(1) + ψ(N) = H_{N-1} vs γ + log(N - 1).
lk = local_entropy(Kraskov(Shannon(; base = ℯ); k = 1), W)
lkl = local_entropy(KozachenkoLeonenko(Shannon(; base = ℯ)), W)
@test lk .- lkl ≈ fill(sum(1 ./ (1:N-1)) - MathConstants.eulergamma - log(N - 1), N)

# Unit-ball volumes for the metrics `local_knn_entropies` accepts. Only Euclidean is used
# by the public Kraskov/KozachenkoLeonenko path, so the others are checked directly.
@testset "Unit-ball volumes" begin
    LI = LocalInformation
    # Closed forms: L1 ball in D dimensions has volume 2^D / D!.
    @test exp(LI.log_ball_volume(LI.Cityblock(), 1)) ≈ 2
    @test exp(LI.log_ball_volume(LI.Cityblock(), 2)) ≈ 2
    @test exp(LI.log_ball_volume(LI.Cityblock(), 3)) ≈ 4 / 3
    @test exp(LI.log_ball_volume(LI.Euclidean(), 2)) ≈ π
    @test exp(LI.log_ball_volume(LI.Euclidean(), 3)) ≈ 4π / 3
    @test exp(LI.log_ball_volume(LI.Chebyshev(), 3)) ≈ 8
    for D in 1:6
        # Minkowski reduces to Cityblock (p = 1) and Euclidean (p = 2), and tends to the
        # Chebyshev cube as p grows.
        @test LI.log_ball_volume(LI.Minkowski(1), D) ≈ LI.log_ball_volume(LI.Cityblock(), D)
        @test LI.log_ball_volume(LI.Minkowski(2), D) ≈ LI.log_ball_volume(LI.Euclidean(), D)
        @test LI.log_ball_volume(LI.Minkowski(1000), D) ≈ LI.log_ball_volume(LI.Chebyshev(), D) rtol = 1e-2
    end
    # Monte Carlo check of the L3 ball volume in 3 dimensions.
    mc = Xoshiro(11)
    P = 2 .* rand(mc, 3, 400_000) .- 1
    frac = mean(sum(abs.(P) .^ 3; dims = 1) .≤ 1)
    @test exp(LI.log_ball_volume(LI.Minkowski(3), 3)) ≈ 8frac rtol = 1e-2
end

# The k-NN entropy is consistent for any norm once the matching ball volume is used.
@testset "k-NN entropy with Cityblock and Minkowski metrics" begin
    LI = LocalInformation
    g = Xoshiro(12)
    X = StateSpaceSet(randn(g, 4000, 2))
    htrue = log(2π * ℯ)  # entropy of a standard 2-D Gaussian, nats
    for metric in (LI.Euclidean(), LI.Chebyshev(), LI.Cityblock(), LI.Minkowski(3))
        h = LI.local_knn_entropies(X, 4, 0, CPU(), Float64; metric)
        @test length(h) == length(X)
        @test mean(h) ≈ htrue atol = 0.06
    end
end

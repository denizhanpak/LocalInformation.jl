using LinearAlgebra: cholesky, logdet, Symmetric

rng = Xoshiro(42)
N = 800
A = randn(rng, 5, 5)
G = randn(rng, N, 5) * A .+ randn(rng, 1, 5)
x, y, z = G[:, 1], G[:, 2], G[:, 3]
W = StateSpaceSet(G[:, 4:5])
higher_order = (TotalCorrelation, DualTotalCorrelation, OInformation, SInformation)

@testset "GaussianDecomposition, base $b, copula $c" for b in BASES, c in (false, true)
    for M in higher_order
        @test invariant_holds(GaussianDecomposition(M(; base = b); copula = c), x, y, z)
        @test invariant_holds(GaussianDecomposition(M(; base = b); copula = c), x, W, z)
    end
end

@testset "GaussianMI / GaussianCMI, base $b" for b in BASES
    for normalize in (false, true)
        @test invariant_holds(GaussianMI(MIShannon(; base = b); normalize), x, y)
        @test invariant_holds(GaussianMI(MIShannon(; base = b); normalize), W, y)
        @test invariant_holds(GaussianCMI(CMIShannon(; base = b); normalize), x, z, y)
        @test invariant_holds(GaussianCMI(CMIShannon(; base = b); normalize), x, y, W)
    end
end

@testset "Closed form: -log 𝒩(x; μ, Σ̂) + (d/2) log(N/(N-1))" begin
    # Joint local entropy of (x, W) by hand, with the maximum-likelihood covariance.
    M = hcat(x, Matrix(W))
    d = size(M, 2)
    μ = vec(mean(M; dims = 1))
    Σ̂ = (M .- μ')' * (M .- μ') / N
    Σ̂inv = inv(Σ̂)
    neglogpdf = [0.5 * (d * log(2π) + logdet(Σ̂) + (r - μ)' * Σ̂inv * (r - μ)) for r in eachrow(M)]
    H = LI.local_subset_entropies(GaussianDecomposition(OInformation(; base = ℯ)), x, W)
    @test H((1, 2)) ≈ neglogpdf .+ (d / 2) * log(N / (N - 1)) rtol = 1e-10
    # Base 2 is the same in bits.
    H2 = LI.local_subset_entropies(GaussianDecomposition(OInformation()), x, W)
    @test H2((1, 2)) ≈ H((1, 2)) ./ log(2)
    # The local MI of a bivariate Gaussian is the log density ratio (constants cancel).
    Σ̂xy = Σ̂[1:2, 1:2]
    lmi = map(eachrow(M)) do r
        u = r[1:2] - μ[1:2]
        lpxy = -0.5 * (2 * log(2π) + logdet(Σ̂xy) + u' * inv(Σ̂xy) * u)
        lpx = -0.5 * (log(2π) + log(Σ̂xy[1, 1]) + u[1]^2 / Σ̂xy[1, 1])
        lpy = -0.5 * (log(2π) + log(Σ̂xy[2, 2]) + u[2]^2 / Σ̂xy[2, 2])
        lpxy - lpx - lpy
    end
    @test local_association(GaussianMI(MIShannon(; base = ℯ)), x, W[:, 1]) ≈ lmi rtol = 1e-9
end

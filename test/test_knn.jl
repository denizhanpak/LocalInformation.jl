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

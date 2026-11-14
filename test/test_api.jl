using LocalInformation: LocalSubsetEntropies, resolve_precision, default_precision

@test resolve_precision(CPU(), nothing) === Float64
@test resolve_precision(CPU(), Float32) === Float32
@test_throws ArgumentError resolve_precision(:gpu, nothing)
@test_throws ArgumentError resolve_precision(CPU(), Int)

rng = Xoshiro(1)
x, y, z = randn(rng, 200), randn(rng, 200), randn(rng, 200)
H = LI.local_subset_entropies(GaussianDecomposition(OInformation()), x, y, z)
@test H isa LocalSubsetEntropies{Float64}
@test H(()) == zeros(200)
@test H((2, 1)) === H([1, 2])                 # order-insensitive and cached
@test_throws ArgumentError H((1, 1))
# `entropy_combination` calls H(∅) for fewer than two variables.
@test Associations.entropy_combination(OInformation(), H, [1]) == zeros(200)
@test mean(H((1, 2, 3))) ≈ Associations.subset_entropies(GaussianDecomposition(OInformation()), x, y, z)((1, 2, 3))

# Float32 on the CPU.
l32 = local_association(GaussianDecomposition(TotalCorrelation()), x, y, z; precision = Float32)
@test eltype(l32) === Float32
@test mean(l32) ≈ association(GaussianDecomposition(TotalCorrelation()), x, y, z) rtol = 1e-4

@test_throws ArgumentError local_association(GaussianMI(), x, y[1:100])

# knn_distances against brute force, with a Theiler window and two metrics.
P = randn(rng, 2, 60)
for (metric, dist) in ((LI.Euclidean(), (a, b) -> sqrt(sum(abs2, a - b))),
                       (LI.Chebyshev(), (a, b) -> maximum(abs, a - b)))
    for (k, w) in ((1, 0), (3, 0), (2, 4))
        brute = map(1:60) do i
            d = [dist(P[:, i], P[:, j]) for j in 1:60 if abs(i - j) > w]
            sort(d)[k]
        end
        @test LI.knn_distances(P, k, w; metric) ≈ brute
    end
end
@test LI.knn_distances(StateSpaceSet(permutedims(P)), 2, 1) ≈ LI.knn_distances(P, 2, 1)
@test LI.knn_distances(P[1, :], 2) ≈ LI.knn_distances(P[1:1, :], 2)
@test_throws ArgumentError LI.knn_distances(P, 0)
@test_throws ArgumentError LI.knn_distances(P, 30, 15)

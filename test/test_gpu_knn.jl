# GPU k-NN kernel, k-NN entropies and KSG/FPVP on the GPU. Run only when CUDA is functional.
using CUDA

# The KernelAbstractions kernels also run on the CPU backend, so they are checked against
# the k-d tree even without a GPU: edge sizes around the 256-point tile, small N, Theiler
# windows. Distances and indices must match exactly in Float64 on continuous data.
@testset "k-NN and range-count kernels on the KernelAbstractions CPU backend" begin
    krng = Xoshiro(41)
    for N in (5, 7, 37, 255, 256, 257, 600), D in (1, 3),
            metric in (LI.Chebyshev(), LI.Cityblock(), LI.Euclidean()), k in (1, 3), w in (0, 1, 2)
        N - (2w + 1) ≥ k || continue
        P = randn(krng, D, N)
        dc, ic = LI.kth_neighbours(CPU(), P, k, w, metric, Float64)
        dg, ig = LI.gpu_knn(CPU(), P, k, w, metric, Float64)
        @test dg ≈ dc rtol = 1e-12
        @test ig == ic
        r = dc .* 1.5
        @test LI.gpu_range_counts(CPU(), P, r, metric, Float64) == LI.range_counts(CPU(), P, r, metric, Float64)
    end
end


if CUDA.functional()
    gpu = CUDABackend()
    rng = Xoshiro(17)

    @testset "knn_distances and neighbour indices" begin
        N = 2000
        for D in (1, 3), metric in (LI.Chebyshev(), LI.Cityblock(), LI.Euclidean()), (k, w) in ((1, 0), (5, 0), (3, 4))
            P = randn(rng, D, N)
            dc, ic = LI.kth_neighbours(CPU(), P, k, w, metric, Float64)
            dg, ig = LI.gpu_knn(gpu, P, k, w, metric, Float64)
            @test ig == ic
            if metric isa LI.Euclidean && D > 1
                # Σ Δ² is contracted to fused multiply-adds differently by the CPU's SIMD
                # code and by the GPU compiler: last-bit differences only.
                @test dg ≈ dc rtol = 1e-14
            else
                @test dg == dc
            end
            d32 = LI.knn_distances(P, k, w; metric, backend = gpu)
            @test eltype(d32) === Float32
            @test d32 ≈ dc rtol = 1e-5
            # Range counts with radii away from the data's distances: exact.
            r = dc .* 1.5
            @test LI.gpu_range_counts(gpu, P, r, metric, Float64) == LI.range_counts(CPU(), P, r, metric, Float64)
        end
        # More queries than one launch chunk, N not a multiple of the workgroup size.
        P = randn(rng, 1, LI.GPU_QUERY_CHUNK + 1234)
        @test LI.knn_distances(P, 2, 1; metric = LI.Chebyshev(), backend = gpu, precision = Float64) ==
              LI.knn_distances(P, 2, 1; metric = LI.Chebyshev())
        @test_throws ArgumentError LI.knn_distances(P[:, 1:100], 2; metric = LI.Minkowski(3), backend = gpu)
    end

    N = 4000
    G = randn(rng, N, 5) * randn(rng, 5, 5)
    x, y, z = G[:, 1], G[:, 2], G[:, 3]
    W = StateSpaceSet(G[:, 4:5])
    V = StateSpaceSet(G[:, 3:5])
    # Float32 checks: means within rtol 1e-3, and elementwise agreement (1e-3 × the largest
    # local magnitude) on at least 99% of the samples; the rest are samples where a Float32
    # distance rounds across a radius and a count shifts by one.
    function mostly_close(a, b)
        scale = max(1.0, maximum(abs, b))
        return eltype(a) === Float32 && isapprox(mean(a), mean(b); rtol = 1e-3, atol = 1e-4 * scale) &&
            count(abs.(a .- b) .≤ 1e-3 * scale) ≥ 0.99 * length(b)
    end

    @testset "k-NN entropies on the GPU" begin
        for est in (Kraskov(Shannon(; base = 2); k = 3), Kraskov(Shannon(; base = ℯ); k = 2, w = 3),
                    KozachenkoLeonenko(Shannon()))
            for data in (x, W)
                lc = local_entropy(est, data)
                lg = local_entropy(est, data; backend = gpu)
                @test mostly_close(lg, lc)
                @test mean(lg) ≈ information(est, data) rtol = 1e-3
                lg64 = local_entropy(est, data; backend = gpu, precision = Float64)
                @test lg64 ≈ lc rtol = 1e-12
            end
        end
        for hest in (Kraskov(; k = 4), KozachenkoLeonenko())
            est = EntropyDecomposition(MIShannon(), hest)
            @test mostly_close(local_association(est, x, W; backend = gpu), local_association(est, x, W))
            est = EntropyDecomposition(CMIShannon(), hest)
            @test local_association(est, x, y, W; backend = gpu, precision = Float64) ≈
                  local_association(est, x, y, W) rtol = 1e-10
        end
    end

    @testset "KSG1, KSG2, FPVP on the GPU" begin
        cases = Any[]
        for E in (KSG1, KSG2)
            push!(cases, (E(; k = 3), (x, y)), (E(MIShannon(; base = ℯ); k = 2, w = 3), (W, V)),
                (E(; k = 4, metric_marginals = LI.Euclidean()), (x, V)))
        end
        push!(cases, (FPVP(; k = 3), (x, z, y)), (FPVP(CMIShannon(; base = ℯ); k = 2, w = 2), (x, W, V)))
        for (est, inputs) in cases
            nc = LI.ksg_counts(est, inputs...)
            @test LI.ksg_counts(est, inputs...; backend = gpu, precision = Float64) == nc
            lc = local_association(est, inputs...)
            @test local_association(est, inputs...; backend = gpu, precision = Float64) ≈ lc rtol = 1e-12
            lg = local_association(est, inputs...; backend = gpu)
            @test mostly_close(lg, lc)
            # atol: some of these (conditional) MIs are ≈ 0.
            @test mean(lg) ≈ association(est, inputs...) rtol = 1e-3 atol = 1e-3
        end
        @test_throws ArgumentError local_association(KSG1(; metric_marginals = LI.Minkowski(3)), x, y; backend = gpu)
    end
else
    @info "CUDA not functional; skipping GPU k-NN tests."
    @test_skip false
end

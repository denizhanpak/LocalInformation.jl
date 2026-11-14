using LocalInformation
using Associations
using Random: Xoshiro, shuffle
using Statistics: mean
using StateSpaceSets: StateSpaceSet
using Test

const LI = LocalInformation

# mean(local values) ≈ estimate, for an `association` estimator.
function invariant_holds(est, x...; atol = 1e-10, kwargs...)
    l = local_association(est, x...; kwargs...)
    return length(l) > 0 && isapprox(mean(l), association(est, x...); atol, rtol = 1e-10)
end

const BASES = (2, ℯ)

@testset "LocalInformation.jl" begin
    @testset "Backends and subset-entropy callable" begin include("test_api.jl") end
    @testset "Discrete plug-in" begin include("test_discrete.jl") end
    @testset "Gaussian" begin include("test_gaussian.jl") end
    @testset "k-NN" begin include("test_knn.jl") end
    @testset "KSG1, KSG2, FPVP" begin include("test_ksg.jl") end
    @testset "GPU k-NN (CUDA)" begin include("test_gpu_knn.jl") end
    @testset "Higher-order and exact values" begin include("test_higher_order.jl") end
    @testset "Cross-check against syntropy" begin include("test_syntropy.jl") end
    @testset "Partial information decomposition" begin include("test_pid.jl") end
    @testset "Unsupported estimators" begin include("test_errors.jl") end
    @testset "GPU (CUDA)" begin include("test_gpu.jl") end
end

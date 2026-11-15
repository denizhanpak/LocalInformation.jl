# GPU

`backend` takes a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend. CUDA support is a package extension, loaded by `using CUDA`:

```julia
using Associations, LocalInformation, CUDA

X = [randn(10^6) for _ in 1:6]
est = GaussianDecomposition(OInformation())
ω = local_association(est, X...; backend = CUDABackend())                       # Vector{Float32}
ω64 = local_association(est, X...; backend = CUDABackend(), precision = Float64)
```

`precision = nothing` (the default) computes in `Float64` on the CPU and `Float32` on a GPU.
The result is always a host `Vector`. On the GPU, the Gaussian path computes the covariance,
its Cholesky factors and log-determinants on the CPU in `Float64`, and the triangular solves
and reductions over samples on the device. Discrete estimators count on the CPU whatever
the backend. k-NN searches and range counts on a GPU use an exact brute-force
KernelAbstractions kernel (O(N²) work, O(N) memory; metrics `Chebyshev`, `Euclidean`,
`Cityblock`). In `Float64` the neighbour counts equal the CPU's on continuous data; in
`Float32` a distance that rounds across a radius can shift a count by one (≲ 0.1% of samples
in the tests), which changes that sample's local value by up to `ψ(n+1) - ψ(n) = 1/n`.

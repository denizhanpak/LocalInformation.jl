# CUDA support. The Gaussian path is backend-generic (KernelAbstractions allocation,
# `ldiv!` on a `LowerTriangular` CuMatrix, which CUDA.jl maps to CUBLAS trsm, and GPUArrays
# reductions), and so are the k-NN kernels (KernelAbstractions, `src/gpu_knn.jl`), so this
# extension only validates the backend.
module LocalInformationCUDAExt

using CUDA: CUDA, CUDABackend
using LocalInformation: LocalInformation

function LocalInformation.check_backend(::CUDABackend)
    if !CUDA.functional()
        throw(ArgumentError(
            "`backend = CUDABackend()` was requested, but CUDA is not functional on this " *
            "machine (`CUDA.functional() == false`). Use `backend = CPU()`."
        ))
    end
    return nothing
end

end # module

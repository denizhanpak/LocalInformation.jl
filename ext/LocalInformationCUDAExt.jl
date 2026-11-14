# CUDA support. The Gaussian path is backend-generic (KernelAbstractions allocation,
# `ldiv!` on a `LowerTriangular` CuMatrix, which CUDA.jl maps to CUBLAS trsm, and GPUArrays
# reductions), so this extension only validates the backend. Phase 3 adds the GPU k-NN
# kernel (a `LocalInformation.knn_distances(::KernelAbstractions.GPU, ...)` method).
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

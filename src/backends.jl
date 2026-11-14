# Compute backends and precision.
#
# Every public function takes `backend` (a KernelAbstractions backend, default `CPU()`)
# and `precision` (`nothing` or a floating-point type). Internal code calls
# `resolve_precision(backend, precision)` once at its entry point; it validates the
# backend and returns the element type `T` to compute in. Arrays are moved between host
# and device with `to_device` and `to_host`. Results returned to the user are always host
# `Vector{T}`s.

"""
    LocalInformation.default_precision(backend) → Type{<:AbstractFloat}

The element type used when `precision = nothing`: `Float64` on `CPU()`, `Float32` on any
GPU backend (consumer GPUs run `Float64` at a small fraction of the `Float32` rate).
"""
default_precision(::KA.CPU) = Float64
default_precision(::KA.GPU) = Float32

"""
    LocalInformation.check_backend(backend) → nothing

Throw an `ArgumentError` if `backend` cannot be used. Package extensions add methods for
their backend types (the CUDA extension checks `CUDA.functional()`).
"""
check_backend(::KA.Backend) = nothing
function check_backend(backend)
    throw(ArgumentError(
        "`backend` must be a KernelAbstractions backend, e.g. `CPU()` or `CUDABackend()` " *
        "(after `using CUDA`). Got $(repr(backend))."
    ))
end

"""
    LocalInformation.resolve_precision(backend, precision) → T

Validate `backend` (with [`LocalInformation.check_backend`](@ref)) and return the element
type to compute in: `precision` if it is a floating-point type, and
[`LocalInformation.default_precision`](@ref)`(backend)` if it is `nothing`.
"""
function resolve_precision(backend, precision)
    check_backend(backend)
    return _precision(backend, precision)
end
_precision(backend, ::Nothing) = default_precision(backend)
_precision(backend, ::Type{T}) where {T<:AbstractFloat} = T
function _precision(backend, precision)
    throw(ArgumentError(
        "`precision` must be `nothing` or a floating-point type such as `Float32` or " *
        "`Float64`. Got $(repr(precision))."
    ))
end

"""
    LocalInformation.to_device(backend, A::AbstractArray, T) → array on `backend`

Copy `A`, converted to element type `T`, to `backend` (`KernelAbstractions.allocate` +
`copyto!`). On the CPU an `Array{T}` is returned as is, without copying.
"""
function to_device(backend, A::AbstractArray, ::Type{T}) where {T}
    host = convert(Array{T}, A)
    backend isa KA.CPU && return host
    dev = KA.allocate(backend, T, size(A)...)
    copyto!(dev, host)
    return dev
end

"""
    LocalInformation.to_host(A) → Array

Copy a device array to host memory (`Array(A)`; a no-op for an `Array`).
"""
to_host(A::Array) = A
to_host(A::AbstractArray) = Array(A)

# Multiply local values in nats by the factor that converts them to the unit `base`
# (`convert_logunit` is linear), keeping the element type.
function nats_to_unit!(h::AbstractVector{T}, base) where {T}
    base == ℯ && return h
    h .*= T(convert_logunit(1.0, ℯ, base))
    return h
end

# ----------------------------------------------------------------------------------------
# The local subset-entropy callable.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.LocalSubsetEntropies{T}

The callable returned by [`LocalInformation.local_subset_entropies`](@ref). `H(S)` returns
the `Vector{T}` of local joint entropies of the inputs with indices `S`, computed by
`H.h(key)` for the sorted subset `key::Vector{Int}` and cached. `H(())` (empty subset)
returns `zeros(T, H.N)`, the additive zero required by `Associations.entropy_combination`.
"""
struct LocalSubsetEntropies{T,F}
    h::F
    N::Int
    cache::Dict{Vector{Int},Vector{T}}
end
LocalSubsetEntropies{T}(h, N::Integer) where {T} =
    LocalSubsetEntropies{T,typeof(h)}(h, Int(N), Dict{Vector{Int},Vector{T}}())

function (H::LocalSubsetEntropies{T})(S) where {T}
    key = sort!(collect(Int, S))
    isempty(key) && return zeros(T, H.N)
    allunique(key) || throw(ArgumentError("Subset indices must be distinct. Got $(collect(S))."))
    return get!(() -> H.h(key)::Vector{T}, H.cache, key)
end

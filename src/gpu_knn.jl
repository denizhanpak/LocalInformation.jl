# Exact brute-force k-nearest-neighbour search and per-point range counts for
# KernelAbstractions GPU backends.
#
# One work-item per query point; the reference points are streamed through workgroup-local
# memory in tiles of `G` points (one point per work-item and tile), so nothing of size N × N
# is ever stored. Each work-item keeps its `k` best (distance, index) pairs in a private,
# sorted buffer. Queries are launched in chunks of `GPU_QUERY_CHUNK` points to keep single
# kernel launches short.
#
# Distances are evaluated as NearestNeighbors.jl's `KDTree` evaluates them, so that results
# in `Float64` are bit-identical to the CPU path on continuous data:
#
#   Chebyshev: max_d |Δ_d|          Cityblock: Σ_d |Δ_d|          Euclidean: Σ_d Δ_d²
#
# summed in dimension order, with the square root taken only for reported Euclidean
# distances, and range queries comparing `Σ Δ² ≤ r²` (`r^2` is `KDTree`'s `eval_pow`).
# Range counts use `≤ r` and include the query point itself, as `NearestNeighbors.inrangecount`.

using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, @Const

const GPU_METRICS = Union{Euclidean,Chebyshev,Cityblock}
const GPU_QUERY_CHUNK = 1 << 15
const GPU_LOCALMEM_BYTES = 16 * 1024

@inline knn_op(::Euclidean, a, b) = abs2(a - b)
@inline knn_op(::Union{Chebyshev,Cityblock}, a, b) = abs(a - b)
@inline knn_reduce(::Chebyshev, s, t) = max(s, t)
@inline knn_reduce(::Union{Euclidean,Cityblock}, s, t) = s + t
@inline knn_finish(::Euclidean, s) = sqrt(s)
@inline knn_finish(::Union{Chebyshev,Cityblock}, s) = s
@inline knn_pow(::Euclidean, r) = abs2(r)
@inline knn_pow(::Union{Chebyshev,Cityblock}, r) = r

# Reduced (pre-`knn_finish`) distance between the query `xi` and column `t` of `tile`.
@inline function reduced_distance(metric, xi::NTuple{D}, tile, t) where {D}
    s = knn_op(metric, xi[1], @inbounds tile[1, t])
    for d in 2:D
        s = knn_reduce(metric, s, knn_op(metric, xi[d], @inbounds tile[d, t]))
    end
    return s
end

@kernel unsafe_indices = true function knn_kernel!(dist, idx, @Const(P), i0, nq, w, metric,
        ::Val{D}, ::Val{K}, ::Val{G}) where {D,K,G}
    T = eltype(P)
    N = size(P, 2)
    l = @index(Local, Linear)
    q = (@index(Group, Linear) - 1) * G + l
    i = i0 + q
    active = q ≤ nq
    tile = @localmem T (D, G)
    bd = @private T (K,)
    bi = @private Int32 (K,)
    for s in 1:K
        @inbounds bd[s] = typemax(T)
        @inbounds bi[s] = Int32(0)
    end
    xi = ntuple(d -> active ? (@inbounds P[d, i]) : zero(T), Val(D))
    for j0 in 0:G:N-1
        j = j0 + l
        if j ≤ N
            for d in 1:D
                @inbounds tile[d, l] = P[d, j]
            end
        end
        @synchronize
        if active
            for t in 1:min(G, N - j0)
                s = reduced_distance(metric, xi, tile, t)
                jj = j0 + t
                # Strict `<`: among equal distances the lower index is kept first.
                if s < @inbounds(bd[K]) && abs(i - jj) > w
                    p = K
                    while p > 1 && s < @inbounds(bd[p-1])
                        @inbounds bd[p] = bd[p-1]
                        @inbounds bi[p] = bi[p-1]
                        p -= 1
                    end
                    @inbounds bd[p] = s
                    @inbounds bi[p] = jj % Int32
                end
            end
        end
        @synchronize
    end
    if active
        @inbounds dist[i] = knn_finish(metric, bd[K])
        @inbounds idx[i] = bi[K]
    end
end

@kernel unsafe_indices = true function range_count_kernel!(counts, @Const(P), @Const(r), i0, nq,
        metric, ::Val{D}, ::Val{G}) where {D,G}
    T = eltype(P)
    N = size(P, 2)
    l = @index(Local, Linear)
    q = (@index(Group, Linear) - 1) * G + l
    i = i0 + q
    active = q ≤ nq
    tile = @localmem T (D, G)
    xi = ntuple(d -> active ? (@inbounds P[d, i]) : zero(T), Val(D))
    ri = active ? knn_pow(metric, @inbounds r[i]) : zero(T)
    c = Int32(0)
    for j0 in 0:G:N-1
        j = j0 + l
        if j ≤ N
            for d in 1:D
                @inbounds tile[d, l] = P[d, j]
            end
        end
        @synchronize
        if active
            for t in 1:min(G, N - j0)
                c += ifelse(reduced_distance(metric, xi, tile, t) ≤ ri, Int32(1), Int32(0))
            end
        end
        @synchronize
    end
    if active
        @inbounds counts[i] = c
    end
end

function check_gpu_metric(metric)
    metric isa GPU_METRICS && return nothing
    throw(ArgumentError(
        "GPU k-nearest-neighbour searches support the metrics `Euclidean()`, `Chebyshev()` " *
        "and `Cityblock()`. Got $(repr(metric)). Use `backend = CPU()` for other metrics."
    ))
end

# Tile width (= workgroup size): 256 points, fewer if a tile would exceed 16 KiB.
function gpu_tile_width(D, ::Type{T}) where {T}
    G = min(256, prevpow(2, max(1, GPU_LOCALMEM_BYTES ÷ (D * sizeof(T)))))
    G ≥ 32 || throw(ArgumentError("Dimension D = $D is too large for the GPU k-NN kernel."))
    return G
end

"""
    LocalInformation.gpu_knn(backend::KernelAbstractions.Backend, P::AbstractMatrix, k, w, metric, T) → (dists::Vector{T}, idxs::Vector{Int})

Exact brute-force search on `backend`, in precision `T`: for every column `i` of the
`D × N` matrix `P`, the distance to and the index of its `k`-th nearest neighbour among the
columns `j` with `abs(i - j) > w`. Ties in distance are broken by the lower index.
`metric` is `Euclidean()`, `Chebyshev()` or `Cityblock()`.
"""
function gpu_knn(backend::KA.Backend, P::AbstractMatrix, k::Int, w::Int, metric, ::Type{T}) where {T}
    check_gpu_metric(metric)
    D, N = size(P)
    G = gpu_tile_width(D, T)
    Pd = to_device(backend, P, T)
    dist = KA.allocate(backend, T, N)
    idx = KA.allocate(backend, Int32, N)
    kernel! = knn_kernel!(backend, G)
    for i0 in 0:GPU_QUERY_CHUNK:N-1
        nq = min(GPU_QUERY_CHUNK, N - i0)
        kernel!(dist, idx, Pd, i0, nq, w, metric, Val(D), Val(k), Val(G);
            ndrange = cld(nq, G) * G)
    end
    KA.synchronize(backend)
    return to_host(dist), Vector{Int}(to_host(idx))
end

"""
    LocalInformation.gpu_range_counts(backend::KernelAbstractions.Backend, P::AbstractMatrix, r::AbstractVector, metric, T) → Vector{Int}

For every column `i` of the `D × N` matrix `P`, the number of columns `j` (including `i`
itself, and with no Theiler window) at distance `≤ r[i]` from it, computed on `backend` in
precision `T`. Same semantics as `NearestNeighbors.inrangecount(KDTree(P, metric), P[:, i], r[i])`.
"""
function gpu_range_counts(backend::KA.Backend, P::AbstractMatrix, r::AbstractVector, metric, ::Type{T}) where {T}
    check_gpu_metric(metric)
    D, N = size(P)
    length(r) == N || throw(DimensionMismatch("Need one radius per point."))
    G = gpu_tile_width(D, T)
    Pd = to_device(backend, P, T)
    rd = to_device(backend, r, T)
    counts = KA.allocate(backend, Int32, N)
    kernel! = range_count_kernel!(backend, G)
    for i0 in 0:GPU_QUERY_CHUNK:N-1
        nq = min(GPU_QUERY_CHUNK, N - i0)
        kernel!(counts, Pd, rd, i0, nq, metric, Val(D), Val(G); ndrange = cld(nq, G) * G)
    end
    KA.synchronize(backend)
    return Vector{Int}(to_host(counts))
end

function knn_distances(backend::KA.Backend, P::AbstractMatrix, k::Int, w::Int, metric, ::Type{T}) where {T}
    return first(gpu_knn(backend, P, k, w, metric, T))
end

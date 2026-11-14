# k-nearest-neighbour local entropies (Kozachenko-Leonenko family).
#
# ComplexityMeasures' estimators are means of per-sample terms (see
# `differential_info_estimators/nearest_neighbors/{Kraskov,KozachenkoLeonenko}.jl`):
#
#   Kraskov:            h_i = -ψ(k) + ψ(N) + log V_D + D log ρ_i
#   KozachenkoLeonenko: h_i = D log ρ_i + log V_D + γ + log(N - 1)        (k = 1)
#
# where ρ_i is the Euclidean distance from point i to its k-th nearest neighbour among the
# points j with |i - j| > w (Theiler window w; w = 0 excludes only the point itself), and
# V_D is the volume of the D-dimensional Euclidean unit ball.
#
# The neighbour step is `knn_distances(backend, points, k, w, metric, T)`: the CPU method
# uses NearestNeighbors.jl; other (GPU) backends use the brute-force kernel in `gpu_knn.jl`.

"""
    knn_distances(points, k::Integer, w::Integer = 0; metric = Euclidean(), backend = CPU(), precision = nothing) → Vector{T}
    knn_distances(backend, P::AbstractMatrix, k::Int, w::Int, metric, T) → Vector{T}

Distance from every point to its `k`-th nearest neighbour, excluding the neighbours `j` of
point `i` with `abs(i - j) ≤ w` (a Theiler window; `w = 0` excludes only the point itself).
Returned as a host `Vector{T}` in the order of the points.

`points` is a `StateSpaceSet`, an `AbstractVector` (a 1-D variable) or a `D × N` matrix
whose columns are the points. `metric` is a Distances.jl metric (`Euclidean()`,
`Chebyshev()`, ...).

The second, positional form is the backend-dispatched implementation called by the first,
with `P` a `D × N` host matrix (`Float64`, one column per point) and
`T = LocalInformation.resolve_precision(backend, precision)`. The `CPU` method uses a
`NearestNeighbors.KDTree` (in `Float64`, converted to `T`); any other KernelAbstractions
backend (a GPU) uses the exact brute-force kernel [`LocalInformation.gpu_knn`](@ref), in
precision `T` and for the metrics `Euclidean()`, `Chebyshev()` and `Cityblock()`.
"""
function knn_distances(points, k::Integer, w::Integer = 0;
        metric = Euclidean(), backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    P = point_matrix(points)
    N = size(P, 2)
    k ≥ 1 || throw(ArgumentError("`k` must be ≥ 1. Got k = $k."))
    w ≥ 0 || throw(ArgumentError("The Theiler window `w` must be ≥ 0. Got w = $w."))
    if N - (2w + 1) < k
        throw(ArgumentError(
            "Too few points ($N) for k = $k neighbours outside a Theiler window w = $w."
        ))
    end
    return knn_distances(backend, P, Int(k), Int(w), metric, T)::Vector{T}
end

# D × N Float64 host matrix, one column per point.
point_matrix(X::AbstractStateSpaceSet) = permutedims(Matrix{Float64}(Matrix(X)))
point_matrix(x::AbstractVector{<:Real}) = Matrix{Float64}(reshape(x, 1, :))
point_matrix(P::AbstractMatrix{<:Real}) = Matrix{Float64}(P)

function knn_distances(::KA.CPU, P::AbstractMatrix, k::Int, w::Int, metric, ::Type{T}) where {T}
    tree = KDTree(P, metric)
    N = size(P, 2)
    if w == 0
        # The point itself is at distance 0 ≤ every other distance, so the (k+1)-th
        # smallest distance including it is the k-th smallest excluding it (also with
        # duplicate points).
        _, dists = knn(tree, P, k + 1, true)
        return T[d[k+1] for d in dists]
    end
    out = Vector{T}(undef, N)
    for i in 1:N
        _, d = knn(tree, view(P, :, i), k, true, j -> abs(i - j) ≤ w)
        out[i] = d[k]
    end
    return out
end

"""
    LocalInformation.log_ball_volume(metric, D) → Float64

Natural logarithm of the volume of the `D`-dimensional unit ball of `metric`.
"""
log_ball_volume(::Euclidean, D::Integer) = (D / 2) * log(π) - loggamma(D / 2 + 1)
log_ball_volume(::Chebyshev, D::Integer) = D * log(2)
log_ball_volume(::Cityblock, D::Integer) = D * log(2) - loggamma(D + 1)
function log_ball_volume(m::Minkowski, D::Integer)
    p = m.p
    return D * (log(2) + loggamma(1 + 1 / p)) - loggamma(1 + D / p)
end

"""
    LocalInformation.local_knn_entropies(X, k, w, backend, T; metric = Euclidean(), estimator = :kraskov) → Vector{T}

Local k-NN entropies, in nats, of the points `X` (`StateSpaceSet`, vector or `D × N`
matrix): `-ψ(k) + ψ(N) + log V_D + D log ρ_i` for `estimator = :kraskov`, and
`D log ρ_i + log V_D + γ + log(N - 1)` (with `k = 1`) for
`estimator = :kozachenkoleonenko`, where `V_D` is the unit-ball volume of `metric`.
`Kraskov` and `KozachenkoLeonenko` (ComplexityMeasures) always use `Euclidean()`.
"""
function local_knn_entropies(X, k::Int, w::Int, backend, ::Type{T};
        metric = Euclidean(), estimator::Symbol = :kraskov) where {T}
    P = point_matrix(X)
    D, N = size(P)
    ρ = knn_distances(P, k, w; metric, backend, precision = T)
    c = if estimator === :kraskov
        -digamma(k) + digamma(N) + log_ball_volume(metric, D)
    elseif estimator === :kozachenkoleonenko
        k == 1 || throw(ArgumentError("KozachenkoLeonenko uses k = 1."))
        log_ball_volume(metric, D) + MathConstants.eulergamma + log(N - 1)
    else
        throw(ArgumentError("Unknown k-NN entropy estimator $(repr(estimator))."))
    end
    return T(c) .+ T(D) .* log.(ρ)
end

const KNNShannon = Union{Kraskov{<:Shannon},KozachenkoLeonenko{<:Shannon}}

knn_parameters(est::Kraskov) = (est.k, est.w, :kraskov)
knn_parameters(est::KozachenkoLeonenko) = (1, est.w, :kozachenkoleonenko)

function local_knn_entropies(est::KNNShannon, X, backend, ::Type{T}) where {T}
    k, w, estimator = knn_parameters(est)
    return local_knn_entropies(X, k, w, backend, T; metric = Euclidean(), estimator)
end

function local_entropy(est::KNNShannon, x; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    return nats_to_unit!(local_knn_entropies(est, StateSpaceSet(x), backend, T), est.definition.base)
end

function local_entropy(est::DifferentialInfoEstimator, x; backend = CPU(), precision = nothing)
    unsupported_entropy_estimator(est)
end

function local_subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure,<:KNNShannon},
        x...; backend = CPU(), precision = nothing,
    )
    T = resolve_precision(backend, precision)
    N = check_input_lengths(x...)
    X = StateSpaceSet.(x)
    base = est.definition.base
    h(S) = nats_to_unit!(local_knn_entropies(est.est, StateSpaceSet(X[S]...), backend, T), base)
    return LocalSubsetEntropies{T}(h, N)
end

function local_subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure,<:DifferentialInfoEstimator},
        x...; backend = CPU(), precision = nothing,
    )
    unsupported_entropy_estimator(est)
end

# Gaussian local entropies.
#
# Associations' `GaussianDecomposition` uses the covariance matrix Σ = cov(M) of all input
# columns, with the unbiased (N - 1) normalisation, and H(S) = ½(d log(2πe) + log det Σ_SS).
# The local entropy of sample i is
#
#     h_i(S) = ½ (d log(2π) + log det Σ_SS + N/(N-1) · z_i'z_i),   z_i = L_S \ (x_i - μ)_S,
#
# with L_S the Cholesky factor of Σ_SS. Since mean_i z_i'z_i = tr(Σ_SS⁻¹ Σ̂_SS) = d(N-1)/N for
# the maximum-likelihood covariance Σ̂ = (N-1)/N Σ, the factor N/(N-1) makes the mean of the
# quadratic term exactly d, so mean_i h_i(S) = H(S) exactly. Equivalently,
#
#     h_i(S) = -log 𝒩(x_i; μ, Σ̂_SS) + (d/2) log(N/(N-1)),
#
# the negative log-density under the maximum-likelihood Gaussian plus a constant. The
# constant cancels in every mutual information and higher-order measure (their entropy
# coefficients, weighted by dimension, sum to zero), so local MI and O-information values
# equal those computed with -log 𝒩(x_i; μ, Σ̂) (e.g. by syntropy, which uses Σ̂).
#
# Σ, its Cholesky factors and log-determinants are computed on the CPU in Float64 (they are
# small), as is the centering. The centered N × D data live on the backend in precision T;
# each subset copies its columns, does one triangular solve from the right,
# Z = Xc_S / L_S' (`rdiv!`, CUBLAS trsm on CUDA), and one row reduction z_i'z_i there.

struct GaussianLocalModel{A<:AbstractMatrix}
    Xc::A                           # N × D centered data on the backend
    Σ::Matrix{Float64}              # D × D covariance, `cov(M)` as in Associations
    columns::Vector{UnitRange{Int}} # columns of each input in Σ
end

# N × D Float64 matrix of all input columns (a fresh copy), as
# `reduce(hcat, Matrix.(StateSpaceSet.(x)))` in Associations' `subset_entropies`.
function data_matrix(x::Tuple)
    N = check_input_lengths(x...)
    blocks = map(xk -> xk isa AbstractStateSpaceSet ? Matrix(xk) : reshape(xk, :, 1), x)
    widths = collect(size.(blocks, 2))
    M = Matrix{Float64}(undef, N, sum(widths))
    col = 0
    for B in blocks
        M[:, (col + 1):(col + size(B, 2))] .= B
        col += size(B, 2)
    end
    ends = cumsum(widths)
    columns = [(e - w + 1):e for (e, w) in zip(ends, widths)]
    return M, columns
end

function GaussianLocalModel(x::Tuple, copula::Bool, backend, ::Type{T}) where {T}
    M, columns = data_matrix(x)
    if copula
        M = Associations.gaussian_copula_transform(M)
    end
    size(M, 1) ≥ 2 || throw(ArgumentError("Gaussian local entropies need at least 2 samples."))
    Σ = cov(M)
    M .-= mean(M; dims = 1)
    return GaussianLocalModel(to_device(backend, M, T), Matrix{Float64}(Σ), columns)
end

"""
    LocalInformation.local_gaussian_entropies(model, S::Vector{Int}, backend, T) → Vector{T}

Local Gaussian entropies, in nats, of the inputs with indices `S`, for a model built by
`GaussianLocalModel(x, copula, backend, T)`. See the comment at the top of `gaussian.jl`
for the formula and its normalisation.
"""
function local_gaussian_entropies(model::GaussianLocalModel, S::Vector{Int}, backend, ::Type{T}) where {T}
    idxs = reduce(vcat, collect.(model.columns[S]))
    d = length(idxs)
    N = size(model.Xc, 1)
    C = cholesky(Symmetric(model.Σ[idxs, idxs]))
    U = to_device(backend, Matrix(C.U), T)     # U = L'
    Z = model.Xc[:, idxs]                      # N × d copy on the backend
    rdiv!(Z, UpperTriangular(U))               # rows z_i' = (x_i - μ)' L⁻ᵀ
    q = vec(sum(abs2, Z; dims = 2))
    c0 = T(0.5 * (d * log(2π) + logdet(C)))
    c1 = T(0.5 * N / (N - 1))
    return to_host(c0 .+ c1 .* q)
end

function gaussian_subset_entropies(x::Tuple, copula::Bool, base, backend, precision)
    T = resolve_precision(backend, precision)
    model = GaussianLocalModel(x, copula, backend, T)
    h(S) = nats_to_unit!(local_gaussian_entropies(model, S, backend, T), base)
    return LocalSubsetEntropies{T}(h, size(model.Xc, 1))
end

function local_subset_entropies(est::GaussianDecomposition, x...; backend = CPU(), precision = nothing)
    return gaussian_subset_entropies(x, est.copula, est.definition.base, backend, precision)
end

# `GaussianMI` and `GaussianCMI` compute the mutual information from a correlation matrix.
# Mutual informations are invariant to the per-column scaling (and to the N vs N - 1
# normalisation), so the local entropies above, without copula, give their local values.
function local_subset_entropies(est::Union{GaussianMI,GaussianCMI}, x...; backend = CPU(), precision = nothing)
    return gaussian_subset_entropies(x, false, est.definition.base, backend, precision)
end

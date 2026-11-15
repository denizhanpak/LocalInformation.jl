# Discrete plug-in local entropies: h_i = -log(n(ω_i) / N), where ω_i is the joint outcome
# of the i-th sample of the selected inputs, n(ω) its count and N the number of samples.
# The mean over samples is the plug-in Shannon entropy of the empirical distribution.
#
# Counting is done with a hash table on the CPU, whatever the backend; the result is a
# host `Vector{T}`.

# Local plug-in entropies, in nats, of the joint outcomes of the code vectors in `codes`.
function local_plugin_entropies(::Type{T}, codes::AbstractVector...) where {T}
    N = length(first(codes))
    keys = collect(zip(codes...))
    cnt = Dict{eltype(keys),Int}()
    sizehint!(cnt, min(N, 1 << 16))
    for k in keys
        cnt[k] = get(cnt, k, 0) + 1
    end
    logN = log(N)
    return T[logN - log(cnt[k]) for k in keys]
end

# Plug-in local subset entropies from one code vector per input.
function plugin_subset_entropies(::Type{T}, codes, base) where {T}
    N = check_input_lengths(codes...)
    h(S) = nats_to_unit!(local_plugin_entropies(T, codes[S]...), base)
    return LocalSubsetEntropies{T}(h, N)
end

# Per-sample outcomes of each input, exactly as `Associations.counts(d, x...)` encodes them
# for `JointProbabilities` (see `counts_and_probs/counts.jl` in Associations.jl).
function joint_codes(d::CodifyVariables{1,<:UniqueElements}, x...)
    return [collect(xk) for xk in x] # `counts(UniqueElements(), x...)` counts raw elements/points.
end
function joint_codes(d::CodifyVariables{1}, x...)
    o = first(d.outcome_spaces)
    return [codify(o, xk isa AbstractStateSpaceSet{1} ? Associations.as_vec(xk) : xk) for xk in x]
end
function joint_codes(d::CodifyVariables{N}, x::Vararg{Any,N}) where {N}
    return collect(Associations.codify(d, x...))
end

# Per-sample outcomes for `EntropyDecomposition`, as in `Associations.subset_entropies`:
# multi-column encodings become `StateSpaceSet`s whose points are the outcomes.
function decomposition_codes(discretization, x...)
    return [collect(StateSpaceSet(e)) for e in Associations.codified_marginals(discretization, x...)]
end

const PlugInShannon = PlugIn{<:Shannon}

function local_subset_entropies(
        est::JointProbabilities{<:MultivariateInformationMeasure,<:CodifyVariables,<:RelativeAmount},
        x...; backend = CPU(), precision = nothing,
    )
    T = resolve_precision(backend, precision)
    return plugin_subset_entropies(T, joint_codes(est.discretization, x...), est.definition.base)
end

function local_subset_entropies(est::JointProbabilities, x...; backend = CPU(), precision = nothing)
    throw(ArgumentError(
        "Local values for `JointProbabilities` need a `CodifyVariables` discretization and " *
        "the default `RelativeAmount` probabilities estimator (plug-in probabilities), e.g. " *
        "`JointProbabilities(def, CodifyVariables(ValueBinning(RectangularBinning(4))))`. " *
        "Got discretization $(nameof(typeof(est.discretization))) and probabilities " *
        "estimator $(nameof(typeof(est.pest)))."
    ))
end

function local_subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure,<:PlugInShannon},
        x...; backend = CPU(), precision = nothing,
    )
    T = resolve_precision(backend, precision)
    if !(est.pest isa RelativeAmount)
        throw(ArgumentError(
            "Local values for `EntropyDecomposition` with `PlugIn` need the default " *
            "`RelativeAmount` probabilities estimator. Got $(nameof(typeof(est.pest)))."
        ))
    end
    codes = decomposition_codes(est.discretization, x...)
    return plugin_subset_entropies(T, codes, est.definition.base)
end

# Every other discrete entropy estimator (bias-corrected ones) is not a mean of local terms.
function local_subset_entropies(
        est::EntropyDecomposition{<:MultivariateInformationMeasure,<:DiscreteInfoEstimator},
        x...; backend = CPU(), precision = nothing,
    )
    unsupported_entropy_estimator(est)
end

# ----------------------------------------------------------------------------------------
# Single-variable local entropy.
# ----------------------------------------------------------------------------------------
# Per-sample outcomes of one variable, as counted by `probabilities(o, x)`.
single_codes(::UniqueElements, x) = collect(x)
function single_codes(o::OutcomeSpace, x)
    if !is_counting_based(o)
        throw(ArgumentError(
            "Local plug-in entropy needs an outcome space whose probabilities are relative " *
            "counts of per-sample outcomes, and $(nameof(typeof(o))) is not counting-based " *
            "(`is_counting_based(o) == false`). Use e.g. `OrdinalPatterns`, `ValueBinning` " *
            "or `UniqueElements`."
        ))
    end
    c = codify(o, x)
    check_count_based(o, x, c)
    return c
end

# The local plug-in entropy -log(n/N) only averages to `information(est, o, x)` when
# `probabilities(o, x)` are the relative counts of the outcomes that `codify(o, x)` returns.
# Weighted outcome spaces (`WeightedOrdinalPatterns`, `AmplitudeAwareOrdinalPatterns`) are
# not counting-based at all. Some counting-based spaces still fail this: `Dispersion`'s
# `codify` gives one Gaussian-CDF symbol per sample, but `probabilities` counts dispersion
# patterns (embedding vectors of symbols), so the two describe different outcomes.
function check_count_based(o::OutcomeSpace, x, c)
    p = sort!(filter(>(0), collect(probabilities(o, x).p)))
    counts = Dict{eltype(c),Int}()
    for ci in c
        counts[ci] = get(counts, ci, 0) + 1
    end
    q = sort!([n / length(c) for n in values(counts)])
    if length(p) != length(q) || !isapprox(p, q; rtol = 1e-10)
        throw(ArgumentError(
            "The probabilities of $(nameof(typeof(o))) are not the relative counts of the " *
            "per-sample outcomes that `codify` returns, so local plug-in entropies would not " *
            "average to `information`. Use e.g. `OrdinalPatterns`, `ValueBinning` or " *
            "`UniqueElements`."
        ))
    end
    return nothing
end
# `ValueBinning` encodes points outside the binning as -1, and `probabilities(o, x)` drops
# them (with `precise = false`, the maximum point can fall outside). Such samples have no
# local entropy that is consistent with `information(est, o, x)`.
function single_codes(o::ValueBinning, x)
    c = codify(o, x)
    n = count(==(-1), c)
    if n > 0
        throw(ArgumentError(
            "$n of $(length(c)) points fall outside the binning (encoded as -1), and " *
            "`information` drops such points, so they have no local entropy. Use a binning " *
            "that covers every point, e.g. `ValueBinning(RectangularBinning(n, true))`."
        ))
    end
    return c
end

function local_entropy(est::PlugInShannon, o::OutcomeSpace, x; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    return nats_to_unit!(local_plugin_entropies(T, single_codes(o, x)), est.definition.base)
end
local_entropy(o::OutcomeSpace, x; kwargs...) = local_entropy(PlugIn(Shannon()), o, x; kwargs...)
function local_entropy(est::DiscreteInfoEstimator, o::OutcomeSpace, x; backend = CPU(), precision = nothing)
    unsupported_entropy_estimator(est)
end

# Local mutual information and conditional mutual information from local subset entropies:
#
#   i(x; y)_t     = h(x)_t + h(y)_t - h(x, y)_t
#   i(x; y | z)_t = h(x, z)_t + h(y, z)_t - h(x, y, z)_t - h(z)_t
#
# These are the same decompositions Associations uses for `EntropyDecomposition`; for
# `JointProbabilities` (plug-in) and the Gaussian estimators the means equal Associations'
# direct formulas exactly.

const LOCAL_MI_ESTIMATORS = Union{
    JointProbabilities{<:MIShannon},
    EntropyDecomposition{<:MIShannon},
    GaussianMI{<:MIShannon},
}

const LOCAL_CMI_ESTIMATORS = Union{
    JointProbabilities{<:CMIShannon},
    EntropyDecomposition{<:CMIShannon},
    GaussianCMI{<:CMIShannon},
}

function local_association(est::LOCAL_MI_ESTIMATORS, x, y; backend = CPU(), precision = nothing)
    H = local_subset_entropies(est, x, y; backend, precision)
    return H((1,)) + H((2,)) - H((1, 2))
end

function local_association(est::LOCAL_CMI_ESTIMATORS, x, y, z; backend = CPU(), precision = nothing)
    H = local_subset_entropies(est, x, y, z; backend, precision)
    return H((1, 3)) + H((2, 3)) - H((1, 2, 3)) - H((3,))
end

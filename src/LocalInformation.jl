"""
    LocalInformation

Local (per-sample, pointwise) information measures, as a companion to Associations.jl.

For an estimator `est` of an Associations.jl measure, [`local_association`](@ref)`(est, x...)`
returns one value per sample, and `mean(local_association(est, x...)) ≈ association(est, x...)`.
[`local_entropy`](@ref) does the same for single-variable Shannon entropy estimators.
"""
module LocalInformation

using Associations
using Associations: JointProbabilities, EntropyDecomposition, GaussianDecomposition,
    GaussianMI, GaussianCMI, MIShannon, CMIShannon, HigherOrderInformationMeasure,
    CodifyVariables, MultivariateInformationMeasure
using ComplexityMeasures: ComplexityMeasures, Shannon, PlugIn, RelativeAmount, UniqueElements,
    OutcomeSpace, ValueBinning, Kraskov, KozachenkoLeonenko, DiscreteInfoEstimator,
    DifferentialInfoEstimator, InformationMeasureEstimator, codify, convert_logunit
import KernelAbstractions as KA
using KernelAbstractions: CPU
using LinearAlgebra: Symmetric, UpperTriangular, cholesky, logdet, rdiv!
using NearestNeighbors: KDTree, knn, Euclidean, Chebyshev, Cityblock, Minkowski
using SpecialFunctions: digamma, loggamma
using StateSpaceSets: StateSpaceSet, AbstractStateSpaceSet, dimension
using Statistics: cov, mean

export local_association, local_entropy
export CPU # re-exported from KernelAbstractions, the default `backend`

# Backend/precision helpers and the `LocalSubsetEntropies` callable.
include("backends.jl")
# Public entry points, generic fallbacks and error messages.
include("api.jl")

# Local subset entropies, one file per estimator family.
include("discrete.jl")
include("gaussian.jl")
include("knn.jl")

# Phase 3 (dedicated KSG/k-NN estimators and the GPU k-NN kernel) goes here:
include("ksg.jl")

# Local measures built from local subset entropies.
include("bivariate.jl")
include("higher_order.jl")

# Phase 5 (local PID redundancy functions) goes here:
include("pid.jl")

# Integrated information decomposition (ΦID); uses the PID helpers above.
include("phiid.jl")
include("phiid_pointwise.jl")
include("phiid_local_mmi.jl")

end # module

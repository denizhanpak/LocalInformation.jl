# Local higher-order measures (TotalCorrelation, DualTotalCorrelation, OInformation,
# SInformation). Associations' `entropy_combination` only adds, subtracts, scales by
# integers and sums the values of its entropy callable, so the vector-valued local subset
# entropies give the local measure elementwise.

const LOCAL_HIGHER_ORDER_ESTIMATORS = Union{
    JointProbabilities{<:HigherOrderInformationMeasure},
    EntropyDecomposition{<:HigherOrderInformationMeasure},
    GaussianDecomposition{<:HigherOrderInformationMeasure},
}

function local_association(est::LOCAL_HIGHER_ORDER_ESTIMATORS, x...; backend = CPU(), precision = nothing)
    Associations.verify_number_of_inputs_vars(est.definition, length(x))
    H = local_subset_entropies(est, x...; backend, precision)
    return Associations.entropy_combination(est.definition, H, collect(1:length(x)))
end

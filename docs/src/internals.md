# Internals

These are not exported and may change. Every local measure is built from
[`LocalInformation.local_subset_entropies`](@ref).

## Local subset entropies

```@docs
LocalInformation.local_subset_entropies
LocalInformation.LocalSubsetEntropies
LocalInformation.local_gaussian_entropies
LocalInformation.local_knn_entropies
```

## Nearest neighbours

```@docs
LocalInformation.knn_distances
LocalInformation.log_ball_volume
LocalInformation.kth_neighbours
LocalInformation.range_counts
LocalInformation.ksg_counts
LocalInformation.gpu_knn
LocalInformation.gpu_range_counts
```

## Backends and precision

```@docs
LocalInformation.default_precision
LocalInformation.check_backend
LocalInformation.resolve_precision
LocalInformation.to_device
LocalInformation.to_host
```

## Subset counts (PID and ΦID)

```@docs
LocalInformation.SubsetCounts
LocalInformation.pid_counts
LocalInformation.PIDLatticeData
```

## ΦID

The product lattice and the average atoms:

```@docs
LocalInformation.PhiIDLattice
LocalInformation.phiid_lattice
LocalInformation.phiid_mobius
LocalInformation.PHIID_INFORMATION_DYNAMICS
LocalInformation.phiid_prepare
LocalInformation.phiid_series
LocalInformation.lag_series
LocalInformation.phiid_from_past_future
LocalInformation.phiid_redundancies
```

Per-sample atoms with `ISx` and `IPM`, and the inversion helper:

```@docs
LocalInformation.local_phiid_from_past_future
LocalInformation.PhiIDPointwiseData
LocalInformation.phiid_local_atoms
LocalInformation.local_phiid_result
```

Per-sample atoms with `IMMI`:

```@docs
LocalInformation.local_phiid_mmi
LocalInformation.local_pair_mutual_informations
LocalInformation.mmi_pair_selection
LocalInformation.mmi_selection_weights
LocalInformation.PHIID_MMI_TIE_RTOL
```

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

## PID

```@docs
LocalInformation.PIDCounts
LocalInformation.PIDLatticeData
```

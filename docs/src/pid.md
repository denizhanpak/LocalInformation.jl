# Partial information decomposition

Two redundancy functions with a pointwise form plug into Associations' PID
(`PartialInformationDecomposition`, through its `Associations.pid_redundancies` hook), for
discrete data (`JointProbabilities` with `CodifyVariables`, plug-in probabilities). For a
sample with target outcome `t` and joint outcome `a_j` of each source set `A_j` of a lattice
node `α = {A_1, …, A_k}`:

- `ISx()`, the shared-exclusion redundancy of Makkeh, Gutknecht & Wibral (2021):
  `i_sx(α; t) = log P(t ∩ (a_1 ∪ … ∪ a_k)) / (P(t) P(a_1 ∪ … ∪ a_k))`. Union probabilities
  are computed exactly from counts by inclusion–exclusion over subsets of `α`.
- `IPM()`, the `i_±` of Finn & Lizier (2018):
  `i_±(α; t) = min_j h(a_j) - min_j h(a_j | t)`.

The redundancy of a node is the sample mean of its local redundancies, so `association`
returns an ordinary `PIDResult`, and `local_association` returns per-sample atoms (a
`LocalPIDResult`) whose means are the atoms of `association`:

```julia
using Associations, LocalInformation, Statistics

x, y = rand(0:1, 10_000), rand(0:1, 10_000); t = x .⊻ y
est = JointProbabilities(PartialInformationDecomposition(ISx()), CodifyVariables(UniqueElements()))
res = association(est, x, y, t)           # PIDResult: synergy(res) ≈ log2(4/3)
loc = local_association(est, x, y, t)     # LocalPIDResult, N × 4 atoms
mean(synergy(loc)) ≈ synergy(res)         # true; also redundancy, unique_information, atom, atoms
sum(loc.atoms; dims = 2)                  # local mutual information i(x, y; t) of each sample
```

Per-sample atoms are the Möbius inversion of the local redundancies, done for all samples
at once as a triangular solve `A = R / Z` with the zeta matrix `Z` of the lattice (on the
`backend`, e.g. CUBLAS with `backend = CUDABackend()`). Local redundancies depend on a
sample only through its joint state, so `R` has one row per distinct state, computed on the
CPU. For 5 sources (7579 atoms) select the atoms you need, e.g.
`atoms = [[[1, 2, 3, 4, 5]], [[1], [2], [3], [4], [5]]]`: then only the needed columns of
`Z⁻¹` (the lattice's Möbius function, computed exactly) and the redundancies of the few
nodes where it is nonzero are computed. `atoms = :all` throws an `ArgumentError` before
allocating if it would need more than `memory_limit` bytes (default 2 GiB). A `PIDAtom`
estimator gives the per-sample values of one atom as a `Vector`.

`IMin` and `IMMI` have no per-sample form (their redundancies are minima of averages), so
`local_association` throws an `ArgumentError` for them. `test/fixtures/syntropy_pid_fixtures.json`
(from `test/fixtures/generate_syntropy_pid_fixtures.py`) holds syntropy's average and
pointwise `isx`/`ipm` atoms, which the tests match to 1e-10.

```@docs
ISx
IPM
LocalPIDResult
```

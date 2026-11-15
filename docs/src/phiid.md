# Integrated information decomposition (ΦID)

The integrated information decomposition (ΦID) of Mediano, Rosas et al. splits the
time-delayed mutual information ``I(\mathbf{X}_t; \mathbf{X}_{t+\tau})`` between the past
and the future of a system of ``n`` variables into *atoms*. Each atom is information that is
redundant, unique or synergistic among the past variables *and* among the future
variables. LocalInformation.jl supports ``n = 2`` (16 atoms) and ``n = 3`` (324 atoms).
For both, `association` returns the average atoms and [`local_association`](@ref) the atoms
of every sample.

```@example phiid
using LocalInformation, Associations, Statistics, Random

rng = Xoshiro(1)
N = 20_000
x, y = zeros(Int, N), zeros(Int, N)
for t in 2:N
    x[t] = rand(rng) < 0.9 ? x[t-1] : 1 - x[t-1]      # x keeps its value with probability 0.9
    y[t] = rand(rng) < 0.9 ? x[t-1] : rand(rng, 0:1)  # y copies x, with noise
end

est = JointProbabilities(IntegratedInformationDecomposition(), CodifyVariables(UniqueElements()))
res = association(est, x, y)
```

```@example phiid
information_dynamics(res)
```

## The product lattice

The PID redundancy lattice of ``n`` sources has one node per antichain of variable sets
(Williams & Beer). For ``n = 2`` it has four nodes, ``\{1\}\{2\} \preceq \{1\}, \{2\}
\preceq \{12\}``, and for ``n = 3`` it has 18. ΦID uses one copy of this lattice for the past
variables and one for the future variables. Each atom ``\Pi(\alpha \to \beta)`` belongs to
a pair of nodes, ``\alpha`` (past) and ``\beta`` (future). The pairs are ordered by the
product order: ``(\alpha', \beta') \preceq (\alpha, \beta)`` if and only if
``\alpha' \preceq \alpha`` and ``\beta' \preceq \beta``. A double redundancy
``I_\cap^{\alpha \to \beta}`` is defined for every pair, and the atoms are its Möbius
inversion:

```math
I_\cap^{\alpha \to \beta} = \sum_{\alpha' \preceq \alpha,\; \beta' \preceq \beta} \Pi(\alpha' \to \beta'),
\qquad \Pi = Z^{-\top} R\, Z^{-1},
```

with ``R_{\alpha\beta} = I_\cap^{\alpha \to \beta}`` and ``Z`` the zeta matrix of the PID
lattice. The atoms sum to ``I(\mathbf{X}_t; \mathbf{X}_{t+\tau})``. The atoms of one past
node, summed over the future nodes, give the PID atom of that node for the information
that the past variables carry about the joint future, in the PID with the same redundancy
function. The column sums give the reverse decomposition.

For two variables the atoms have names, as in `phyid`. The first letter is the past node,
`t` separates the two, and the last letter is the future node. The letters are `r` =
``\{1\}\{2\}`` (redundancy), `x` = ``\{1\}``, `y` = ``\{2\}`` and `s` = ``\{12\}``
(synergy). Variable 1 is the first input, `x`, and variable 2 the second, `y`:

| past ↓ future → | `r` = {1}{2} | `x` = {1} | `y` = {2} | `s` = {12} |
|:--|:--|:--|:--|:--|
| `r` = {1}{2} | `rtr` | `rtx` | `rty` | `rts` |
| `x` = {1} | `xtr` | `xtx` | `xty` | `xts` |
| `y` = {2} | `ytr` | `ytx` | `yty` | `yts` |
| `s` = {12} | `str` | `stx` | `sty` | `sts` |

`rtr` is the bottom of the product lattice and `sts` the top. Use `atom(res, :xty)`, or for
any ``n`` the pair of nodes as collections of variable sets,
`atom(res, [[1]] => [[2]])`. `atoms(res)` lists every atom from the bottom to the top.
`res.atoms` and `res.redundancies` are the ``m \times m`` matrices (rows = past node,
columns = future node, nodes in the order of `res.lattice.lattice.nodes`).

```@example phiid
atom(res, :xty) == atom(res, [[1]] => [[2]])
```

With three variables only the `α => β` form works. `atoms(res)` returns
`(α => β) => value` pairs, here sorted to show the largest atoms:

```@example phiid
z = rand(rng, 0:1, N)
z[2:end] .= ifelse.(rand(rng, N - 1) .< 0.8, x[1:end-1] .⊻ y[1:end-1], z[2:end])  # z ≈ x ⊻ y, lagged
res3 = association(est, x, y, z)
first(sort(atoms(res3); by = p -> -abs(last(p))), 4)
```

```@example phiid
atom(res3, [[1]] => [[1], [2]]), length(atoms(res3))
```

## Information dynamics

[`information_dynamics`](@ref) groups the 16 atoms of two variables into the six modes of
Mediano et al. (arXiv:2109.13186, "A new taxonomy for information dynamics in complex
systems"; PNAS 2025, doi:10.1073/pnas.2423297122, "An Extended Taxonomy for Information
Dynamics in Complex Systems"):

| Mode | Atoms |
|:--|:--|
| `storage` | `rtr + xtx + yty + sts` |
| `copy` | `xtr + ytr` |
| `transfer` | `xty + ytx` |
| `erasure` | `rtx + rty` |
| `downward_causation` | `str + stx + sty` |
| `upward_causation` | `rts + xts + yts` |

The modes are disjoint and cover all 16 atoms, so they sum to
``I(\mathbf{X}_t; \mathbf{X}_{t+\tau})``. Synergistic storage `sts` is part of `storage`, as
in both papers. For a [`LocalPhiIDResult`](@ref) every mode is a per-sample vector. The
definitions live in one place, `LocalInformation.PHIID_INFORMATION_DYNAMICS`.

```@example phiid
sum(information_dynamics(res)) ≈ sum(res.atoms)
```

## Double redundancies and estimators

The redundancy function is the first argument of
`IntegratedInformationDecomposition(redundancy = IMMI(); τ = 1, base = 2)`.

- `IMMI()` (minimum mutual information, the default) is the double redundancy of `phyid`
  and of the MATLAB ΦID toolbox:
  ``I_\cap^{\alpha \to \beta} = \min_{A \in \alpha, B \in \beta} I(\mathbf{X}^A_t; \mathbf{X}^B_{t+\tau})``.
  It works with every estimator family whose joint entropies Associations can compute.
- [`ISx`](@ref)`()` is Varley's shared-exclusion double redundancy ``i^{\tau sx}_\cap``
  (Varley 2023): ``h_{sx}(\alpha) + h_{sx}(\beta) - h_{sx}(\alpha \cap \beta)``. Here
  ``h_{sx}(\alpha)`` is minus the log-probability that at least one set of ``\alpha`` takes
  its value in the sample, ``h_{sx}(\beta)`` the same for ``\beta``, and
  ``h_{sx}(\alpha \cap \beta)`` that of both events at once.
- [`IPM`](@ref)`()` is the ``i_\pm`` of Finn & Lizier (2018) with past and future source sets:
  ``\min_k h(a_k) + \min_l h(b_l) - \min_{k,l} h(a_k, b_l)``, where ``a_k`` and ``b_l``
  are the sample's values of the sets ``A_k \in \alpha`` and ``B_l \in \beta``, and
  ``h = -\log P``.

`ISx` and `IPM` are pointwise. The local double redundancy of a sample depends only on its
joint state, and the average double redundancy is the mean of the local ones. They are
computed from plug-in probabilities of the joint states, so they need discrete data. The
docstring of [`local_association`](@ref) for these two gives the formulas in full. `IMin`
has no double-redundancy form and throws an `ArgumentError`.

With `def = IntegratedInformationDecomposition(r; τ, base)`:

| Estimator | `IMMI()` average | `IMMI()` per sample | `ISx()`, `IPM()` average | `ISx()`, `IPM()` per sample |
|:--|:-:|:-:|:-:|:-:|
| `JointProbabilities(def, CodifyVariables(o))` (plug-in) | yes | yes | yes | yes |
| `EntropyDecomposition(def, PlugIn(Shannon()), CodifyVariables(o))` | yes | yes | no | no |
| `EntropyDecomposition(def, e, CodifyVariables(o))`, `e` another discrete Shannon estimator (e.g. `MillerMadow()`) | yes | no | no | no |
| `EntropyDecomposition(def, Kraskov(; k))` | yes | yes, CPU or GPU | no | no |
| `EntropyDecomposition(def, KozachenkoLeonenko())` | yes | yes, CPU or GPU | no | no |
| `EntropyDecomposition(def, e)`, `e` another differential Shannon estimator | yes | no | no | no |
| `GaussianDecomposition(def; copula)` | yes | yes, CPU or GPU | no | no |

"Average" is `association(est, x...)`, which returns a [`PhiIDResult`](@ref). "Per sample"
is `local_association(est, x...)`, which returns a [`LocalPhiIDResult`](@ref). Per-sample
atoms need entropies that are means of per-sample terms. The bias-corrected discrete
estimators, and the differential estimators other than `Kraskov` and
`KozachenkoLeonenko`, therefore give average atoms only. Every "no" throws an `ArgumentError`
that lists the supported estimators. Values are in the unit `base` (bits by default).

```@example phiid
est_sx = JointProbabilities(IntegratedInformationDecomposition(ISx()), CodifyVariables(UniqueElements()))
information_dynamics(association(est_sx, x, y))
```

The same data give different atoms under different redundancy functions, and atoms can be
negative. MMI also has a known artefact: for two independent copied bits it gives
`rts = str = 1` and `sts = 2` bits, compensated by `xts = yts = stx = sty = -1`, because
``\min(I(x_t; x_{t+1}y_{t+1}), I(y_t; x_{t+1}y_{t+1})) = 1`` bit counts as redundant.

## Lagging and encoding

The inputs are 2 or 3 variables. Each one is an `AbstractVector` or a `StateSpaceSet` (a
multivariate variable), and all have the same length ``N > \tau``.

- The lagged system has the rows ``t = 1, \ldots, N - \tau``. The past is
  ``\mathbf{X}_t = (x_1[t], \ldots, x_n[t])`` and the future is
  ``\mathbf{X}_{t+\tau} = (x_1[t+\tau], \ldots, x_n[t+\tau])``. Row `t` of a
  `LocalPhiIDResult` is the sample with its past at time `t`. `phyid` and the syntropy
  fixtures lag the same way.
- Discrete estimators encode each series once, over its full length, and then lag the
  codes. The past and the future of a variable therefore share one encoding, e.g. the same
  bins of a `ValueBinning`. The encoding is the one that `association` uses with the same
  `JointProbabilities` or `EntropyDecomposition` estimator.
- With an outcome space that embeds the data (e.g. `OrdinalPatterns`), ``N`` and ``\tau``
  count encoded symbols.
- Continuous estimators (k-NN, Gaussian) lag the raw data.

To give the past and the future yourself, with no lagging and no shared encoding, use the
internal entry points `LocalInformation.phiid_from_past_future(est, past, future)` (average
atoms), `LocalInformation.local_phiid_mmi` (per-sample, `IMMI`) and
`LocalInformation.local_phiid_from_past_future` (per-sample, `ISx` and `IPM`). Each takes
`past` and `future` as tuples of `n` variables (see [Internals](@ref)).

## Per-sample atoms

For every supported estimator, the mean of each local atom is the average atom, and the
local atoms of a sample sum to its local ``I(\mathbf{X}_t; \mathbf{X}_{t+\tau})``.

```@example phiid
loc = local_association(est, x, y)
size(loc.atoms), mean(atom(loc, :xty)) ≈ atom(res, :xty)
```

The accessors are those of a `PhiIDResult` and return per-sample vectors. `loc.atoms` is
an ``(N - \tau) \times m^2`` matrix: row `t` is sample `t`, and column `i + (j - 1) m`
holds the atom of past node `i` and future node `j`.

`ISx` and `IPM` compute the local double redundancies once per distinct joint state, from
counts of every subset of the ``2n`` lagged variables. They then invert all states at once
(one matrix product with ``Z^{-1} \otimes Z^{-1}``) and copy the atoms to the samples.

### The local MMI convention

MMI is a minimum of averages, so its local form is a convention. LocalInformation.jl uses
the one of `phyid` and of the MATLAB ΦID toolbox, extended to three variables. Every pair
of nonempty variable sets, ``A`` of the past and ``B`` of the future, has the local mutual
information

```math
i_t(A; B) = h_t(\mathbf{X}^A_t) + h_t(\mathbf{X}^B_{t+\tau}) - h_t(\mathbf{X}^A_t, \mathbf{X}^B_{t+\tau}),
```

computed from the local joint entropies of the estimator. Its mean ``\bar I(A; B)`` is the
mutual information that `association` uses. For each atom ``(\alpha \to \beta)``, the local
double redundancy of sample ``t`` is ``i_t(A^*; B^*)``, where ``(A^*, B^*)`` is the pair in
``\alpha \times \beta`` with the smallest mean. The local atoms are the Möbius inversion of
these local double redundancies. As a result:

- the mean of every local atom is the average atom;
- identities between atoms and single-set double redundancies hold for every sample; for
  example the local transfer entropy equals `xtr + xty + str + sty` (see the recipes
  below).

The selected pair depends on the whole sample through the means. A sample's local MMI
double redundancy is therefore not a function of its joint state alone, unlike `ISx` and
`IPM`. For this reason the PID's `local_association` refuses `IMMI`. For ΦID this
convention is the established one.

**Ties.** The candidate pairs ``(A, B)`` are ordered by the bitmask of ``A``, then by that
of ``B``. The variable sets therefore come in the order ``\{1\}, \{2\}, \{1,2\}, \{3\},
\{1,3\}, \{2,3\}, \{1,2,3\}``. The first pair whose mean is within
`LocalInformation.PHIID_MMI_TIE_RTOL * max(1, maximum(abs, Ī))` of the smallest mean is
selected. The tolerance is `1e-12`; it absorbs summation-order rounding, so that exact ties
in the data resolve the same way every time.

- For `rtr` the candidates are ``I(x_t; x_{t+\tau})``, ``I(x_t; y_{t+\tau})``,
  ``I(y_t; x_{t+\tau})`` and ``I(y_t; y_{t+\tau})``, in this order, and the first of a tie
  wins. `phyid` does the same (`np.argmin`).
- For the single-target redundancies (e.g. ``\{1\}\{2\} \to \{1\}``), `phyid` keeps the
  *second* pair of an exact tie, and LocalInformation.jl the first.
- Ties change local values, never their means. `phyid` decides ties on floating-point
  means, so in exactly symmetric systems it can resolve them either way.

**Gaussian data and `ddof`.** The Gaussian local entropies of LocalInformation.jl are the
maximum-likelihood ones (see [Supported estimators](@ref)). `phyid` evaluates its local
Gaussian densities with the unbiased covariance (`np.cov`, `ddof = 1`). The means, i.e. the
average atoms, agree exactly. The local values differ by a factor on their deviation from
the mean. With ``N' = N - \tau`` lagged samples,

```math
\text{local}_\text{LocalInformation} = \text{average} + \frac{N'}{N' - 1}\,(\text{local}_\text{phyid} - \text{average}),
```

a relative difference of about ``1/N'``. `phyid`'s discrete MMI supports binary data
only: it binarises each lagged vector as `v > mean(v)`. LocalInformation.jl accepts any
alphabet. For 0/1 series that contain both values the two agree.

### GPU and memory

```julia
using LocalInformation, Associations, CUDA
u, v = randn(100_000), randn(100_000)
est = GaussianDecomposition(IntegratedInformationDecomposition())
loc = local_association(est, u, v; backend = CUDABackend())                       # Float32 atoms
loc64 = local_association(est, u, v; backend = CUDABackend(), precision = Float64)
```

- With `IMMI`, the k-NN and Gaussian estimators compute the local entropies on the
  `backend`. The selection and inversion are one matrix product, `Lmi * W`, also on the
  `backend`.
- Discrete estimators count on the CPU. With `ISx` and `IPM` the local double
  redundancies are also computed on the CPU, in `Float64`, and only the inversion runs on
  the `backend`.
- The result is always a host `Matrix`. `precision` defaults to `Float64` on the CPU and
  `Float32` on a GPU.
- `memory_limit` is in bytes (default `2^31`, i.e. 2 GiB). The estimated peak memory, host
  and device, is checked before any large allocation, and an `ArgumentError` is thrown if
  it exceeds the limit.
- The result alone takes ``(N - \tau) \cdot m^2 \cdot`` `sizeof(T)` bytes, with ``m^2 = 16``
  for two variables and 324 for three. For example, ``10^6`` samples of three variables in
  `Float64` need 2.6 GB. Use `precision = Float32`, fewer samples, or a larger
  `memory_limit`.

## Recipes

Several classical measures are sums of atoms. Write ``x_t, y_t`` for the past and
``x_{t+\tau}, y_{t+\tau}`` for the future of the two variables:

| Measure | Sum of atoms |
|:--|:--|
| transfer entropy ``TE_{x \to y} = I(x_t; y_{t+\tau} \mid y_t)`` | `xtr + xty + str + sty` |
| transfer entropy ``TE_{y \to x} = I(y_t; x_{t+\tau} \mid x_t)`` | `ytr + ytx + str + stx` |
| active information storage ``I(x_t; x_{t+\tau})`` | `rtr + rtx + xtr + xtx` |
| active information storage ``I(y_t; y_{t+\tau})`` | `rtr + rty + ytr + yty` |
| time-delayed mutual information ``I(x_t, y_t; x_{t+\tau}, y_{t+\tau})`` | all 16 atoms |
| ``\Phi_{WMS}`` + `rtr`, with ``\Phi_{WMS} = I(\text{past}; \text{future}) - I(x_t; x_{t+\tau}) - I(y_t; y_{t+\tau})`` | `xty + ytx + rts + xts + yts + str + stx + sty + sts` |

The transfer entropies here use one past sample of each variable. `str` appears in both,
which is why their sum (the causal density) counts it twice. These identities hold for
every redundancy function, on average and, for the local atoms, per sample.

Mediano et al. define ``\Phi_R = \Phi_{WMS} + \min_{i,j} I(X^i_t; X^j_{t+\tau})``. The added
term is the MMI double redundancy `rtr`. So ``\Phi_R`` equals the 9-atom sum above only with
`IMMI`. With `ISx` or `IPM`, ``\Phi_{WMS}`` + `rtr` still equals the 9-atom sum, but it is
not the paper's ``\Phi_R``.

The identities checked against Associations' Gaussian estimators on the lagged series:

```@example phiid
xg, yg = zeros(N), zeros(N)
for t in 2:N
    xg[t] = 0.6xg[t-1] + randn(rng)
    yg[t] = 0.4yg[t-1] + 0.5xg[t-1] + randn(rng)          # x drives y
end
resg = association(GaussianDecomposition(IntegratedInformationDecomposition()), xg, yg)
a(s) = atom(resg, s)
xp, yp, xf, yf = xg[1:end-1], yg[1:end-1], xg[2:end], yg[2:end]   # τ = 1
mi(u, v) = association(GaussianMI(), u, v)
tdmi = mi(StateSpaceSet(xp, yp), StateSpaceSet(xf, yf))
Φwms = tdmi - mi(xp, xf) - mi(yp, yf)
[
    "TE x → y" => (a(:xtr) + a(:xty) + a(:str) + a(:sty), association(GaussianCMI(), xp, yf, yp)),
    "TE y → x" => (a(:ytr) + a(:ytx) + a(:str) + a(:stx), association(GaussianCMI(), yp, xf, xp)),
    "AIS of x" => (a(:rtr) + a(:rtx) + a(:xtr) + a(:xtx), mi(xp, xf)),
    "I(past; future)" => (sum(resg.atoms), tdmi),
    "Φ_WMS + rtr" => (sum(a, (:xty, :ytx, :rts, :xts, :yts, :str, :stx, :sty, :sts)), Φwms + a(:rtr)),
    "Φ_R (MMI)" => (Φwms + a(:rtr), Φwms + minimum(mi(p, f) for p in (xp, yp), f in (xf, yf))),
]
```

The same holds per sample, against local transfer entropy:

```@example phiid
locg = local_association(GaussianDecomposition(IntegratedInformationDecomposition()), xg, yg)
lte = local_association(GaussianCMI(), xp, yf, yp)
maximum(abs, atom(locg, :xtr) .+ atom(locg, :xty) .+ atom(locg, :str) .+ atom(locg, :sty) .- lte)
```

## Validation

The test suite compares the atoms with two Python reference implementations. The fixtures
are in bits. `test/fixtures/phiid_reference_notes.md` records both packages' conventions
and the checks behind the statements on this page.

- **syntropy** (commit `f6ca750`): `test/fixtures/syntropy_phiid_fixtures.json`, generated by
  `generate_syntropy_phiid_fixtures.py`. It has 12 cases: 7 exact discrete distributions
  (5 with ``n = 2``, 2 with ``n = 3``), 3 discrete series (``\tau = 1, 2``) and 2 Gaussian
  VAR(1) series. All cases have average MMI atoms, and the discrete ones also have average
  and pointwise `isx` and `ipm` atoms. The tests match them to `1e-10` (Gaussian:
  `1e-8`).
- **phyid** (commit `6c5f2e9`): `test/fixtures/phyid_fixtures.json`, generated by
  `generate_phyid_fixtures.py`. It has 5 cases: 3 Gaussian and 2 binary series, with
  average and local MMI atoms. The local atoms match to `1e-10` for discrete data and
  `1e-8` for Gaussian data, after the `ddof` rescaling above.

Further tests, across the estimator families, check:

- the transfer-entropy, active-information-storage and total identities of the recipes,
  on average and per sample;
- the double redundancies of single variable sets (mutual informations) and the PID row
  and column sums;
- that the mean of the local atoms equals the average atoms;
- exact discrete systems: copy, swap, parity and full redundancy.

## References

- Mediano, P. A. M., Rosas, F., Carhart-Harris, R. L., Seth, A. K., & Barrett, A. B.
  (2019). Beyond integrated information: A taxonomy of information dynamics phenomena.
  arXiv:1909.02297.
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D.,
  Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information
  dynamics via Integrated Information Decomposition. arXiv:2109.13186.
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D.,
  Seth, A. K., & Barrett, A. B. (2025). Toward a unified taxonomy of information dynamics
  via Integrated Information Decomposition. PNAS, 122(39), e2423297122.
  doi:10.1073/pnas.2423297122.
- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., et al. (2022). A synergistic core for
  human brain evolution and cognition. Nature Neuroscience, 25, 771–782 (`phyid`:
  <https://github.com/Imperial-MIND-lab/integrated-info-decomp>).
- Varley, T. F. (2023). Decomposing past and future: Integrated information decomposition
  based on shared probability mass exclusions. PLOS ONE, 18(3), e0282950.
  doi:10.1371/journal.pone.0282950.
- Finn, C., & Lizier, J. T. (2018). Pointwise partial information decomposition using the
  specificity and ambiguity lattices. Entropy, 20(4), 297.

## API

```@docs
IntegratedInformationDecomposition
PhiIDResult
LocalPhiIDResult
information_dynamics
```

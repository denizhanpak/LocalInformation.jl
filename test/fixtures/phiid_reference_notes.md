# ΦID reference notes (syntropy and phyid conventions)

Sources:

- syntropy: commit `f6ca750` (`f6ca7508ada126ead728f5be7a67ca20d6e96a7e`). Paths below are relative to the `syntropy/` package directory.
- phyid: https://github.com/Imperial-MIND-lab/integrated-info-decomp, commit `6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44` (2026-03-14). phyid is not on PyPI (`pypi.org/pypi/phyid` returns 404) and the repo has no tags. Paths are relative to `phyid/`.
- MATLAB reference: phyid's git submodule `matlab` = https://github.com/pmediano/PhiID, commit `a633cc1`. I read it but did not run it. phyid's own tests compare its local and average atoms with MATLAB output stored on OSF; I did not run those tests.

Fixtures:

| file | cases | generator |
|---|---|---|
| `syntropy_phiid_fixtures.json` | 12: 7 `discrete` (5 with n = 2, 2 with n = 3), 3 `discrete_series`, 2 `gaussian` | `generate_syntropy_phiid_fixtures.py` |
| `phyid_fixtures.json` | 5: 3 `gaussian`, 2 `discrete_series` (binary) | `generate_phyid_fixtures.py` |

All values in both files are in bits. syntropy's and phyid's Gaussian values are natively in nats; the generators divide them by ln 2.

The files add a few keys beyond the agreed schema:

- syntropy, n = 2 cases: `atom_names` (phyid-style names in `atom_order`).
- phyid, every case: `redundancy`.
- Both files: some extra `meta` keys.

## 0. Atom naming (n = 2)

An antichain over two variables gets one letter: r = {1}{2}, x = {1}, y = {2}, s = {12}. The atom name is the past letter, then `t`, then the future letter. Past variable 1 is x_t and past variable 2 is y_t. Future variable 1 is x_{t+τ} and future variable 2 is y_{t+τ}.

syntropy keys atoms as `(past_antichain, future_antichain)`. Each antichain is a tuple of tuples of 0-based positions within `inputs` or `target`. For example, `(((0,),), ((1,),))` is xty.

The fixture `atom_order` is the node order of syntropy's pickled lattice, written with 1-based indices:

| index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| name | xtx | xtr | xty | xts | rtx | rtr | rty | rts | ytx | ytr | yty | yts | stx | str | sty | sts |

phyid's order is different (`utils.py:4-9`): rtr, rtx, rty, rts, xtr, xtx, xty, xts, ytr, ytx, yty, yts, str, stx, sty, sts.

phyid's internal names: x = source past, y = target past, a = source future (`t1`), b = target future (`t2`).

## 1. Double-redundancy functions in syntropy

Notation:

- Past antichain α = {A_1..A_k} and future antichain β = {B_1..B_l}. Each A_i or B_j is a set of variables.
- A sample joint state s has outcomes a_i = s[A_i] and b_j = s[B_j].
- p is the joint pmf. h(·) = −log2 p(·) is the local entropy of a marginal outcome (`discrete/decompositions.py:16-52`).

**MMI** (`discrete/decompositions.py:214-223`; Gaussian `gaussian/decompositions.py:160-172`):
R(α→β) = min_{i,j} I(A_i; B_j).

- Average only. `_pid` returns `(None, avg)` for discrete data (`discrete/decompositions.py:479-488`) and only `avg` for Gaussian data (`gaussian/decompositions.py:321-330`).

**isx**, Varley's shared-exclusion i_tsx (`discrete/decompositions.py:392-411`, h_sx at `98-152`):
r(α→β; s) = h_sx(α) + h_sx(β) − h_sx(α×β), where

- h_sx(α) = −log2 P(∪_i {A_i = a_i});
- h_sx(β) = −log2 P(∪_j {B_j = b_j});
- h_sx(α×β) = −log2 P(∪_{i,j} {A_i = a_i and B_j = b_j}).

The joint term is the union over all k·l pairs, each pair being the source A_i ∪ B_j. The docstring writes this term as "h_sx(α∩β)" (`integrated_information_decomposition`, line 609). That is the same quantity: by distributivity, (∪_i {A_i = a_i}) ∩ (∪_j {B_j = b_j}) = ∪_{i,j} {A_i = a_i and B_j = b_j}, and the intersection form is how Varley (2023) defines it.

**ipm** (`discrete/decompositions.py:296-316`):
r(α→β; s) = min_i h(a_i) + min_j h(b_j) − min_{i,j} h(a_i, b_j).

- The pair entropy is looked up as `sources[state][tuple(set(s1+s2))]`. This relies on CPython iterating a set of small ints in ascending order, which matches the sorted powerset keys. It holds for ≤ 8 variables.

Pointwise and average results:

- isx and ipm are evaluated per joint state, and the Möbius inversion is applied per state.
- Pointwise atoms are keyed by `(input_state, target_state)` (`discrete/decompositions.py:500-511`).
- Averages are Σ_s p(s)·π(·; s) (`513-521`).
- Units: bits. Discrete MI uses `math.log2` (`discrete/shannon.py:161`). Gaussian entropies use the natural log via `slogdet` (`gaussian/shannon.py:56-59`), so Gaussian results are in nats.
- Gaussian covariance: `check_cov(None, data)` = `np.cov(data, ddof=0)` (`gaussian/utils.py:34`). `data` is channels × samples. Our fixture passes the 4 × (N−τ) lagged matrix with rows (x_t, y_t, x_{t+τ}, y_{t+τ}). ddof has no effect on MI.

I checked these formulas by writing my own implementation from the formulas above, using a zeta-matrix solve over the order in §2. It reproduces every discrete fixture case to ≤ 6e-15 bits:

- averages for mmi, isx and ipm;
- pointwise values for isx and ipm;
- n = 2 and n = 3.

## 2. Lattice order and Möbius inversion

- Williams–Beer order on antichains: α′ ⪯ α iff every A ∈ α contains some A′ ∈ α′.
- Product order on atoms: (α′→β′) ⪯ (α→β) iff α′ ⪯ α and β′ ⪯ β (`lattices/product_lattice.py:65`; PNAS 2025 Eq. 7).
- Bottom: rtr = ({1}{2}→{1}{2}). Top: sts = ({12}→{12}).
- Lattice sizes: 16 nodes for 2×2 and 18² = 324 for 3×3.
- Lattices are loaded from pickles `lattices/pi_lattice_{n_in}{n_out}.pickle` (`lattices/utils.py:8-40`). Available: 22, 23, 32, 33.
- Möbius inversion (`lattices/mobius_inversion.py:10-62`): π(α→β) = R(α→β) − Σ_{(α′→β′) ≺ (α→β)} π(α′→β′). Equivalently, R(α→β) = Σ_{(α′→β′) ⪯ (α→β)} π(α′→β′).
- Nodes are visited in decreasing `distance_from_top`. `distance_from_top` is the shortest-path length from the top node (`product_lattice.py:85-88`). The code also relies on reversing a Python set of small ints (`mobius_inversion.py:44-46`).

The generator asserts two things for both the 22 and 33 pickles:

- each node's stored `descendants` set equals the strict down-set under the product of Williams–Beer orders;
- every descendant has a strictly larger `distance_from_top`, so the visiting order is valid.

phyid solves the same system as a fixed 16×16 matrix (`calculate.py:122-149`; MATLAB `PhiIDFull.m:188-208`). Each row maps one "known" quantity to a lattice node, and the row is the zeta row of that node:

| phyid known | lattice node | quantity |
|---|---|---|
| rtr | {1}{2}→{1}{2} | double redundancy |
| R_xyta | {1}{2}→{1} | red(x_t, y_t → x_{t+τ}) |
| R_xytb | {1}{2}→{2} | red(x_t, y_t → y_{t+τ}) |
| R_xytab | {1}{2}→{12} | red(x_t, y_t → x_{t+τ}y_{t+τ}) |
| R_abtx | {1}→{1}{2} | backward PID: x_t onto the future pair |
| R_abty | {2}→{1}{2} | backward PID: y_t onto the future pair |
| R_abtxy | {12}→{1}{2} | backward PID: (x_t, y_t) onto the future pair |
| I_xta, I_xtb, I_yta, I_ytb, I_xyta, I_xytb, I_xtab, I_ytab, I_xytab | single-source, single-target nodes | MIs |

Under MMI every phyid known equals syntropy's min over the pairs of that node. The average atoms are therefore identical. Max |phyid − syntropy| was ≤ 3e-15 bits on all 4 syntropy series cases and all 5 phyid cases (Gaussian and binary).

## 3. Information-dynamics modes (Mediano et al.)

I checked this against two sources:

- arXiv:2109.13186 v1 (27 Sep 2021), p. 5, section "A new taxonomy for information dynamics in complex systems", read from the PDF text;
- PNAS 122(39) e2423297122 (2025), section "An Extended Taxonomy for Information Dynamics in Complex Systems", read from the PMC full text PMC12501198.

Both list **six disjoint modes**:

| mode | atoms |
|---|---|
| Storage | rtr + xtx + yty + **sts** |
| Copy | xtr + ytr |
| Transfer | xty + ytx |
| Erasure | rtx + rty |
| Downward causation | str + stx + sty |
| Upward causation | rts + xts + yts |

Your table is **confirmed**. sts sits **inside storage**. The papers call it "synergistic storage (Syn → Syn)" and do not list it separately. The word "decoupling" appears in neither paper's text.

"Causal decoupling" comes from Rosas, Mediano et al. 2020 ("Reconciling emergences", PLOS Comput Biol): "collective properties have irreducible causal power over other collective properties". That page's equations did not render, so I could not check the exact atom expression there.

Double-redundancy functions used in the papers: MMI and CCS. The arXiv version uses CCS in its main text and MMI in its SM. According to the PMC text, the PNAS version reverses this: MMI in the main text and CCS in the SI.

## 4. Identities for tests

TE_{X→Y} = I(X_t; Y_{t+τ} | Y_t).

| identity | source |
|---|---|
| TE_{X→Y} = xtr + xty + str + sty | arXiv Eq. 4; PNAS Eq. 4: TE(1→2) = Syn→Red + Syn→Un2 + Un1→Red + Un1→Un2 |
| TE_{Y→X} = ytr + ytx + str + stx | by symmetry; str is shared, which is why causal density double-counts it |
| I(X_t; X_{t+τ}) = rtr + rtx + xtr + xtx | AIS, Eq. 2 |
| I(Y_t; Y_{t+τ}) = rtr + rty + ytr + yty | AIS of Y |
| I(X_t, Y_t; Y_{t+τ}) = rtr + rty + xtr + xty + ytr + yty + str + sty | — |
| I(past; future) = Σ of all 16 atoms | TDMI, PNAS Eq. 11 |
| Φ_WMS = I(past; future) − I(X_t; X_{t+τ}) − I(Y_t; Y_{t+τ}) | Eq. 3 |
| Φ_WMS + rtr = xty + ytx + rts + xts + yts + str + stx + sty + sts | algebra of the rows above |

On Φ_R: the PNAS paper defines Φ_R = I(X_t; X_{t+1}) − Σ_i I(X^i_t; X^i_{t+1}) + min_{i,j} I(X^i_t; X^j_{t+1}). The added term is the MMI double redundancy. So "Φ_R = 9-atom sum" holds exactly with MMI atoms. With isx or ipm atoms, Φ_WMS + rtr still equals the 9-atom sum, but it is not the paper's Φ_R.

I checked every row numerically. Max |diff| is ≤ 3e-14 in each check:

- syntropy `markov_binary_tau1`: averages for mmi, isx and ipm, against plug-in MI/CMI.
- The same case, pointwise for isx and ipm, against local MI/CMI per state. The identities hold pointwise because single-source, single-target isx and ipm reduce to local MI.
- syntropy `var1_tau1`: Gaussian MMI against closed-form MI/CMI from `np.cov(ddof=0)`.
- phyid `var1_tau1`: local rows against local Gaussian MI/CMI computed with phyid's ddof = 1 covariance.

## 5. phyid's local MMI convention

Call: `calc_PhiID(src, trg, tau, kind="gaussian"|"discrete", redundancy="MMI"|"CCS")` (`calculate.py:154-215`). It returns only local atoms, one array of length N−τ per name. The average is the sample mean of those arrays; phyid's test and MATLAB `PhiIDFull.m:55` both compute it this way.

How the local atoms are built:

- **Every redundancy is a local MI.** Each redundancy (6 single-target and 1 double) is the local MI time series of one source/target pair. The chosen pair is the one with the smallest **sample mean** of its local MI (`measures.py:56-91`; MATLAB `PhiIDFull.m:250-256`, `private/DoubleRedundancyMMI.m:106-121`).
- **Single-target ties.** `mi_1 if mean(mi_1) < mean(mi_2) else mi_2`, so a tie picks the second pair (`measures.py:73`).
- **Double-redundancy ties.** `np.argmin` over [I(x_t;x_{t+τ}), I(x_t;y_{t+τ}), I(y_t;x_{t+τ}), I(y_t;y_{t+τ})], so a tie picks the first in that order (`measures.py:91`; MATLAB uses a strict `<` in the same loop order).
- **Tie caveat.** Ties are decided on floating-point means, so exactly symmetric systems can resolve either way.
- **Atoms.** Local atoms are the per-sample solve of the 16×16 system (`calculate.py:149`). They sum to the local I(past; future) of each sample, which the generator asserts.

### Gaussian preprocessing (`calculate.py:184-190`, `16-21`; `measures.py:10-28`)

- **Lagging.** Past = `x[:-tau], y[:-tau]` and future = `x[tau:], y[tau:]`. This uses samples t = 1..N−τ (1-based) and drops the last τ past samples. The same lagging is used for the syntropy series fixtures.
- **Scaling.** Each of the 4 lagged rows is divided by its own sample std (ddof = 1) and is not centred. This has no effect on MI.
- **Local entropy.** h_t(S) = −ln 𝒩(X_S,t; mean(X_S), `np.cov(X)`[S,S]). `np.cov` uses its default ddof = 1. The log is natural (`-np.log(pdf)`), so values are in nats.
- **Mean of the local MI.** With the sample mean and any ddof, the mean of the local MI equals the closed-form Gaussian MI exactly. phyid averages therefore equal syntropy's Gaussian MMI averages.
- **Local values depend on ddof.** Write N′ = N − τ. With the ML covariance (ddof = 0), local MI − I is larger by the factor N′/(N′−1). The same holds for the atoms, because the pair selection does not depend on ddof:
  `local_ML = average + N′/(N′−1) · (local_phyid − average)`.
  The generator checks this by rerunning phyid with `np.cov(·, ddof=0)` patched in. Max deviation: 1.4e-14 nats.

### Discrete data

phyid supports discrete MMI, but only for binary data:

- Each lagged vector is binarised as `v > mean(v)` (`utils.py:26-27`). Past and future copies are thresholded separately.
- Local entropies are −log2 of empirical frequencies over the N−τ samples, in bits (`measures.py:31-53`). The alphabet is hard-coded to {0,1}.
- For 0/1 series, the result is the plug-in local MI on the empirical lagged distribution.
- phyid prints a harmless `divide by zero in log2` warning for unobserved combinations, which are never looked up.

### CCS

CCS is in the code (`measures.py:94-113`; double CCS built at `calculate.py:96-108`), although its docstrings say "To be implemented". phyid's tests compare its CCS output with MATLAB output; I did not run those tests. The fixtures contain MMI only.

## 6. Convention differences

| | syntropy | phyid | LocalInformation.jl |
|---|---|---|---|
| units | discrete bits, Gaussian **nats** | discrete bits, Gaussian **nats** | `base` (fixtures: bits) |
| MMI | average only, no local | local = local MI of the pair with the smallest mean MI | — |
| isx / ipm | pointwise and average (discrete only) | not available | — |
| Gaussian covariance | `np.cov(ddof=0)`, average only | `np.cov` ddof = 1 for local densities | `cov` (N−1) for log det, plus N/(N−1) on the quadratic term (`src/gaussian.jl:1-18`): local values are the ML ones |
| lagging | user builds the joint (fixtures: t = 1..N−τ) | t = 1..N−τ | — |
| discrete alphabet | any | binary only (mean-binarised) | any |

Points to watch:

1. **Gaussian local MMI.** Average atoms agree across all three. phyid's local Gaussian atoms do **not** equal ML-covariance local atoms. Before comparing phyid with LocalInformation.jl's Gaussian local entropies, apply `local_ML = avg + N′/(N′−1)(local_phyid − avg)`, with N′ = number of rows. This is a difference of about 1/N′ relative to the deviation from the mean, roughly 1e-3 for these cases.
2. **Units in syntropy.** syntropy's discrete and Gaussian ΦID use different units (bits vs nats). Its return types also differ: discrete returns `(None, avg)` for mmi and `(ptw, avg)` otherwise; Gaussian returns `avg` for mmi and `(ptw, avg)` for ipm.
3. **Pointwise rows.** syntropy's pointwise values are per distinct joint state. For series cases, map each lagged sample to its row in `states`.
4. **MMI on the copy case.** On `copy`, syntropy MMI gives xtx = yty = 1. It also gives rts = str = 1, sts = 2, and xts = yts = stx = sty = −1. This is correct MMI behaviour: red(x_t, y_t → x_{t+1}y_{t+1}) = min(1, 1) = 1 for independent sources. phyid's equations give the same.
5. **Nothing looked wrong numerically** in either package for these cases. The only issue found is the CPython set-ordering dependence in §1 and §2.

## Regeneration

```sh
# syntropy (checkout at commit f6ca750; package dir contains syntropy/)
cd test/fixtures
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 14,15 \
  uv run --no-project --with numpy --with scipy --with networkx \
  python -I generate_syntropy_phiid_fixtures.py /path/to/syntropy f6ca750 syntropy_phiid_fixtures.json

# phyid
D=$(mktemp -d) && git clone https://github.com/Imperial-MIND-lab/integrated-info-decomp.git "$D/repo" \
  && git -C "$D/repo" checkout 6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44
cd test/fixtures
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 14,15 \
  uv run --no-project --with numpy --with scipy \
  python -I generate_phyid_fixtures.py "$D/repo" 6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44 phyid_fixtures.json
```

The JSON files were generated with Python 3.12.16 and numpy 2.5.4. Seeds are fixed; random streams come from `np.random.default_rng`.

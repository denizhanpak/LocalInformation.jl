# Per-sample ΦID atoms with the MMI double redundancy (src/phiid_local_mmi.jl).
# Helpers are prefixed `lmmi_` (test files share one module).
import JSON3
using CUDA

const LMMI_IID = IntegratedInformationDecomposition
const LMMI_UE = CodifyVariables(UniqueElements())
const LMMI_BINS = ValueBinning(RectangularBinning(3, true))
const LMMI_FAMILIES = (:discrete, :binned, :kraskov, :kl, :gaussian)

lmmi_msg(f) = try f(); "no error" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $e" end
lmmi_lag(v, τ) = (v[1:(end - τ)], v[(1 + τ):end])

# A coupled VAR(1): x drives y, z is driven by both.
function lmmi_var(rng, N; n = 2)
    X = zeros(N, 3)
    for t in 2:N
        X[t, 1] = 0.6X[t - 1, 1] + randn(rng)
        X[t, 2] = 0.4X[t - 1, 2] + 0.5X[t - 1, 1] + randn(rng)
        X[t, 3] = 0.3X[t - 1, 3] + 0.3X[t - 1, 1] - 0.4X[t - 1, 2] + randn(rng)
    end
    return Tuple(X[:, k] for k in 1:n)
end

# The ΦID estimator of a family and the MI and CMI estimators of the same family for the
# series that the ΦID lags (integer codes, for the discrete families).
function lmmi_estimators(family, τ, base)
    def = LMMI_IID(; τ, base)
    mi, cmi = MIShannon(; base), CMIShannon(; base)
    if family === :discrete
        return JointProbabilities(def, CodifyVariables(LMMI_BINS)), JointProbabilities(mi, LMMI_UE),
            JointProbabilities(cmi, LMMI_UE)
    elseif family === :binned
        p = PlugIn(Shannon())
        return EntropyDecomposition(def, p, CodifyVariables(LMMI_BINS)), EntropyDecomposition(mi, p, LMMI_UE),
            EntropyDecomposition(cmi, p, LMMI_UE)
    elseif family === :kraskov
        k = Kraskov(; k = 4)
        return EntropyDecomposition(def, k), EntropyDecomposition(mi, k), EntropyDecomposition(cmi, k)
    elseif family === :kl
        k = KozachenkoLeonenko()
        return EntropyDecomposition(def, k), EntropyDecomposition(mi, k), EntropyDecomposition(cmi, k)
    else
        return GaussianDecomposition(def), GaussianMI(mi), GaussianCMI(cmi)
    end
end

lmmi_samples(family) = family in (:kraskov, :kl) ? 600 : 1500

# Max |mean of the local atoms - average atoms|.
function lmmi_mean_gap(est, x...)
    loc = local_association(est, x...)
    avg = association(est, x...)
    ok = loc isa LocalPhiIDResult && size(loc.atoms) == (length(x[1]) - est.definition.τ, length(avg.atoms))
    return ok ? maximum(abs, vec(mean(loc.atoms; dims = 1)) .- vec(avg.atoms)) : Inf
end

# Per-sample identities. Returns the failed checks (empty if all hold).
function lmmi_identities(est, mi, cmi, x; atol = 1e-10)
    res = local_association(est, x...)
    τ = est.definition.τ
    s = last(LI.phiid_series(est, x))  # the series as the ΦID lags them (codes if discrete)
    lagged = [lmmi_lag(v, τ) for v in s]
    past, fut = first.(lagged), last.(lagged)
    out = Dict{Symbol,Bool}()
    rows = vec(sum(res.atoms; dims = 2))
    tdmi = local_association(mi, StateSpaceSet(past...), StateSpaceSet(fut...))
    out[:row_sums] = length(rows) == length(tdmi) && maximum(abs, rows .- tdmi) ≤ atol
    if length(x) == 2
        (xp, yp), (xf, yf) = past, fut
        a(sym) = atom(res, sym)
        te_xy = local_association(cmi, xp, yf, yp)
        te_yx = local_association(cmi, yp, xf, xp)
        out[:te_xy] = maximum(abs, (a(:xtr) .+ a(:xty) .+ a(:str) .+ a(:sty)) .- te_xy) ≤ atol
        out[:te_yx] = maximum(abs, (a(:ytr) .+ a(:ytx) .+ a(:str) .+ a(:stx)) .- te_yx) ≤ atol
        # Active information storage of y: a single-set double redundancy.
        ais_y = local_association(mi, yp, yf)
        out[:ais_y] = maximum(abs, (a(:rtr) .+ a(:rty) .+ a(:ytr) .+ a(:yty)) .- ais_y) ≤ atol
        out[:modes] = maximum(abs, sum(information_dynamics(res)) .- rows) ≤ atol
    end
    failed = [k for (k, v) in out if !v]
    isempty(failed) || @warn "Local ΦID identities failed" est failed
    return failed
end

@testset "Mean of the local atoms = average atoms, $family" for family in LMMI_FAMILIES
    rng = Xoshiro(hash(family) % 1000)
    N = lmmi_samples(family)
    for n in (2, 3), τ in (1, 2), base in (2, ℯ)
        est = first(lmmi_estimators(family, τ, base))
        @test lmmi_mean_gap(est, lmmi_var(rng, N; n)...) ≤ 1e-10
    end
    # A multivariate (StateSpaceSet) variable and a third variable.
    x1, x2, x3 = lmmi_var(rng, N; n = 3)
    est = first(lmmi_estimators(family, 1, ℯ))
    @test lmmi_mean_gap(est, StateSpaceSet(x1, randn(rng, N) .+ x2), x3) ≤ 1e-10
end

@testset "Per-sample identities, $family" for family in LMMI_FAMILIES
    rng = Xoshiro(7 + hash(family) % 1000)
    N = lmmi_samples(family)
    est, mi, cmi = lmmi_estimators(family, 1, 2)
    @test isempty(lmmi_identities(est, mi, cmi, lmmi_var(rng, N)))
    est, mi, cmi = lmmi_estimators(family, 2, ℯ)
    @test isempty(lmmi_identities(est, mi, cmi, lmmi_var(rng, N; n = 3)))
    x1, x2, x3 = lmmi_var(rng, N; n = 3)
    @test isempty(lmmi_identities(est, mi, cmi, (StateSpaceSet(x1, randn(rng, N) .+ x3), x2)))
end

@testset "Pair selection and ties" begin
    for n in (2, 3)
        L = LI.phiid_lattice(n)
        m, nm = length(L.lattice.nodes), 2^n - 1
        rng = Xoshiro(40 + n)
        μ = rand(rng, nm^2)
        sel = LI.mmi_pair_selection(L, μ)
        # The selected mean is the MMI double redundancy min_{A ∈ α, B ∈ β} μ(A, B).
        col(a, b) = Int(a) + (Int(b) - 1) * nm
        @test all(μ[sel[i + (j - 1) * m]] == minimum(μ[col(a, b)] for a in L.lattice.nodes[i], b in L.lattice.nodes[j])
            for i in 1:m, j in 1:m)
        # W = S * kron with the 0/1 selection S.
        S = zeros(Int, nm^2, m^2)
        for (c, p) in enumerate(sel)
            S[p, c] = 1
        end
        @test LI.mmi_selection_weights(L, sel, nm^2) == S * L.kron
    end
    # Ties: the first candidate in (bitmask of A, bitmask of B) order, also when rounding
    # makes a later candidate smaller by less than the tolerance.
    L = LI.phiid_lattice(2)
    ir = L.lattice.index[UInt[1, 2]]     # node {1}{2}
    rtr = ir + (ir - 1) * 4              # atom column of {1}{2} → {1}{2}
    μ = fill(1.0, 9)                     # μ[a + 3(b - 1)], a, b ∈ {1, 2, 3} = {x}, {y}, {xy}
    μ[1 + 3 * 1] = 0.25                  # I(x_t; y_{t+τ})
    μ[2 + 3 * 0] = 0.25 - 1e-15          # I(y_t; x_{t+τ}), tied up to rounding
    @test LI.mmi_pair_selection(L, μ)[rtr] == 1 + 3 * 1
    μ[2 + 3 * 0] = 0.25 - 1e-9           # a real difference
    @test LI.mmi_pair_selection(L, μ)[rtr] == 2 + 3 * 0
    μ[2 + 3 * 0] = NaN
    @test LI.mmi_pair_selection(L, μ)[rtr] == 2 + 3 * 0

    # An exact tie in data: two noisy copy channels x → x', y → y', made symmetric under
    # x ↔ y, so that I(x; y') = I(y; x') (the smallest) and I(x; x') = I(y; y') exactly,
    # while the local values of the tied pairs differ. Rows shuffled so the sums of the tied
    # columns run in different orders. rtr = local I(x_t; y_{t+τ}), the first of the tie.
    rng = Xoshiro(11)
    K = 3000
    xp, yp = rand(rng, 0:1, K), rand(rng, 0:1, K)
    noisy(v) = [rand(rng) < 0.85 ? b : rand(rng, 0:1) for b in v]
    xf, yf = noisy(xp), noisy(yp)
    perm = shuffle(rng, 1:(2K))
    sym(a, b) = vcat(a, b)[perm]
    past, fut = (sym(xp, yp), sym(yp, xp)), (sym(xf, yf), sym(yf, xf))
    est = JointProbabilities(LMMI_IID(), LMMI_UE)
    res = LI.local_phiid_mmi(est, past, fut)
    lmi(a, b) = local_association(JointProbabilities(MIShannon(), LMMI_UE), a, b)
    i_xy, i_yx = lmi(past[1], fut[2]), lmi(past[2], fut[1])
    @test mean(i_xy) ≈ mean(i_yx) atol = 1e-14
    @test maximum(abs, i_xy .- i_yx) > 1e-3
    @test atom(res, :rtr) ≈ i_xy atol = 1e-12
    @test maximum(abs, atom(res, :rtr) .- i_yx) > 1e-3
    @test vec(mean(res.atoms; dims = 1)) ≈ vec(LI.phiid_from_past_future(est, past, fut).atoms) atol = 1e-12
end

@testset "Cross-check against phyid (local atoms)" begin
    path = joinpath(@__DIR__, "fixtures", "phyid_fixtures.json")
    if isfile(path)
        fx = JSON3.read(read(path, String))
        @test fx.meta.unit == "bits"
        kinds = Set{String}()
        for (name, case) in pairs(fx.cases)
            τ = case.tau
            rows = reduce(vcat, [permutedims(Float64.(r)) for r in case.local])
            avg = Float64.(case.average)
            Nr = size(rows, 1)
            @test case.local_sample_indices == 1:Nr
            if case.kind == "gaussian"
                x, y = (Float64.(v) for v in case.series)
                res = local_association(GaussianDecomposition(LMMI_IID(; τ)), x, y)
                # phyid's local Gaussian densities use the ddof = 1 covariance; ours are the
                # maximum-likelihood ones (notes §5).
                ref = avg' .+ Nr / (Nr - 1) .* (rows .- avg')
                atol = 1e-8
            elseif case.kind == "discrete_series"
                x, y = (Int.(v) for v in case.series)
                # phyid binarises each lagged vector as v > mean(v): the identity for 0/1
                # vectors with both values present.
                @test all(Set(v) == Set([0, 1]) for s in (x, y) for v in lmmi_lag(s, τ))
                res = local_association(JointProbabilities(LMMI_IID(; τ), LMMI_UE), x, y)
                ref = rows
                atol = 1e-10
            else
                @warn "Unknown fixture kind $(case.kind) in case $name"
                continue
            end
            push!(kinds, case.kind)
            @test size(res.atoms) == (length(x) - τ, 16) == size(ref)
            gap = maximum(maximum(abs, atom(res, Symbol(s)) .- ref[:, k]) for (k, s) in enumerate(case.atom_names))
            gap ≤ atol || @warn "Local ΦID fixture case $name differs from phyid" gap
            @test gap ≤ atol
            # Means in the `vec` layout (column i + 4(j - 1): past node i, future node j).
            names = [string(LI.pair_symbol(res.lattice, c)) for c in vec(CartesianIndices((4, 4)))]
            @test vec(mean(res.atoms; dims = 1)) ≈ [avg[findfirst(==(nm), case.atom_names)] for nm in names] atol = atol
        end
        @test kinds == Set(["gaussian", "discrete_series"])
    else
        @info "phyid fixtures not found; skipping."
        @test_skip false
    end
end

@testset "Precision, show and errors" begin
    rng = Xoshiro(23)
    x, y, z = lmmi_var(rng, 800; n = 3)
    est = GaussianDecomposition(LMMI_IID())
    l64 = local_association(est, x, y)
    l32 = local_association(est, x, y; precision = Float32)
    @test l64.atoms isa Matrix{Float64} && l32.atoms isa Matrix{Float32}
    @test maximum(abs, l32.atoms .- l64.atoms) ≤ 1e-4 * max(1, maximum(abs, l64.atoms))
    s = sprint(show, MIME"text/plain"(), l64)
    @test occursin("LocalPhiIDResult: 2 variables, τ = 1, redundancy function IMMI", s) && occursin("799 samples × 16 atoms", s)

    # Estimators without local entropies.
    o = CodifyVariables(LMMI_BINS)
    m = lmmi_msg(() -> local_association(EntropyDecomposition(LMMI_IID(), MillerMadow(), o), x, y))
    @test occursin("bias-corrected", m) && occursin("MillerMadow", m) && occursin("association(est, x...)", m)
    m = lmmi_msg(() -> local_association(EntropyDecomposition(LMMI_IID(), ZhuSingh()), x, y))
    @test occursin("only `Kraskov` and `KozachenkoLeonenko`", m) && occursin("ZhuSingh", m)
    m = lmmi_msg(() -> local_association(JointProbabilities(LMMI_IID(), CodifyPoints(UniqueElementsEncoding(x))), x, y))
    @test occursin("CodifyPoints", m)
    # Inputs, as for the average atoms.
    @test occursin("2 to 3 variables", lmmi_msg(() -> local_association(est, x)))
    @test occursin("more than τ", lmmi_msg(() -> local_association(GaussianDecomposition(LMMI_IID(; τ = 5)), x[1:5], y[1:5])))
    @test occursin("same number of samples", lmmi_msg(() -> local_association(est, x, y[1:10])))
    # Memory.
    m = lmmi_msg(() -> local_association(est, x, y, z; memory_limit = 2^20))
    @test occursin("memory_limit", m) && occursin("324 atoms of 3 variables", m)
    @test LI.check_local_mmi_memory(CPU(), Float64, 10^6, 3, 324, Inf) > 2^31
    @test local_association(est, x, y, z; memory_limit = 2^24) isa LocalPhiIDResult
    @test_throws ArgumentError local_association(est, x, y; precision = Int)
    # The past/future entry point.
    @test occursin("same number of variables", lmmi_msg(() -> LI.local_phiid_mmi(est, (x, y), (z,))))
    # The PID keeps refusing per-sample IMMI atoms.
    @test_throws ArgumentError local_association(JointProbabilities(PartialInformationDecomposition(IMMI()), LMMI_UE),
        rand(rng, 0:1, 50), rand(rng, 0:1, 50), rand(rng, 0:1, 50))
end

@testset "GPU (CUDA)" begin
    if CUDA.functional()
        gpu = CUDABackend()
        rng = Xoshiro(31)
        # Float32 tolerance relative to the largest local magnitude.
        lmmi_close32(a, b) = eltype(a) === Float32 && maximum(abs, a .- b) ≤ 1e-4 * max(1, maximum(abs, b))
        for (est, x) in ((GaussianDecomposition(LMMI_IID()), lmmi_var(rng, 4000)),
                         (GaussianDecomposition(LMMI_IID(; τ = 2, base = ℯ)), lmmi_var(rng, 4000; n = 3)),
                         (EntropyDecomposition(LMMI_IID(), Kraskov(; k = 4)), lmmi_var(rng, 1500)),
                         (EntropyDecomposition(LMMI_IID(; τ = 2), Kraskov(; k = 4)), lmmi_var(rng, 1000; n = 3)))
            lc = local_association(est, x...)
            lg = local_association(est, x...; backend = gpu)
            @test lmmi_close32(lg.atoms, lc.atoms)
            lg64 = local_association(est, x...; backend = gpu, precision = Float64)
            @test lg64.atoms isa Matrix{Float64}
            @test maximum(abs, lg64.atoms .- lc.atoms) ≤ 1e-9 * max(1, maximum(abs, lc.atoms))
        end
        # Discrete estimators count on the CPU; the inversion runs on the GPU.
        x, y = lmmi_var(rng, 2000)
        est = JointProbabilities(LMMI_IID(), CodifyVariables(LMMI_BINS))
        lg = local_association(est, x, y; backend = gpu)
        @test lg.atoms isa Matrix{Float32} && maximum(abs, lg.atoms .- local_association(est, x, y).atoms) ≤ 1e-4
        m = lmmi_msg(() -> local_association(GaussianDecomposition(LMMI_IID()), lmmi_var(rng, 2000; n = 3)...;
            backend = gpu, memory_limit = 2^20))
        @test occursin("memory_limit", m)
    else
        @info "CUDA not functional; skipping GPU local ΦID tests."
        @test_skip false
    end
end

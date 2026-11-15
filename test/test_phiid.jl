# Integrated information decomposition (src/phiid.jl).
import JSON3
import LinearAlgebra
using CUDA
using LinearAlgebra: istriu

const IIDec = IntegratedInformationDecomposition
const UE = CodifyVariables(UniqueElements())
const PHIID_NAMES = [:rtr, :rtx, :rty, :rts, :xtr, :xtx, :xty, :xts,
                     :ytr, :ytx, :yty, :yts, :str, :stx, :sty, :sts]

# ΦID of an exact joint distribution: `states` are (past..., future...) tuples, each
# repeated `weights[k]` times, through the past/future entry point.
function phiid_of_states(states, weights = ones(Int, length(states)); n = length(states[1]) ÷ 2, base = 2)
    rows = reduce(vcat, [fill(k, w) for (k, w) in enumerate(weights)])
    col(v) = [Int(states[k][v]) for k in rows]
    est = JointProbabilities(IIDec(; base), UE)
    return LI.phiid_from_past_future(est, Tuple(col(v) for v in 1:n), Tuple(col(v) for v in (n + 1):(2n)))
end

# Lagged copies, as ΦID lags its inputs.
phiid_lag(v, τ) = (v[1:(end - τ)], v[(1 + τ):end])

# Zeta matrix in node order: Zn[i, j] = 1 iff node i ≼ node j.
zeta_nodes(L) = Matrix{Int}(L.lattice.below) + LinearAlgebra.I

# Σ over α' ≼ α, β' ≼ β of the atoms: the double redundancy rebuilt from the atoms.
function rebuilt_redundancy(res, i, j)
    Zn = zeta_nodes(res.lattice)
    return sum(res.atoms[a, b] for a in axes(Zn, 1), b in axes(Zn, 1) if Zn[a, i] == 1 && Zn[b, j] == 1)
end

msg_of(f) = try f(); "no error" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $e" end

@testset "Product lattice" begin
    for (n, m) in ((2, 4), (3, 18))
        L = LI.phiid_lattice(n)
        @test LI.phiid_lattice(n) === L                       # cached
        @test length(L.lattice.nodes) == m && length(L.pairs) == m^2
        @test length(unique(L.pairs)) == m^2
        @test istriu(L.zeta) && all(L.zeta[a, a] for a in 1:m)
        Zn = zeta_nodes(L)
        @test Zn * L.mobius == LinearAlgebra.I               # exact Möbius matrix
        @test L.kron == kron(L.mobius, L.mobius)
        # `pairs` goes bottom to top: no pair is below a pair listed before it.
        below(p, q) = p != q && Zn[p[1], q[1]] == 1 && Zn[p[2], q[2]] == 1
        @test all(!below(L.pairs[b], L.pairs[a]) for a in 1:m^2 for b in (a + 1):m^2)
        # Round trip R = Zᵀ Π Z ↔ Π = Z⁻ᵀ R Z⁻¹, exact in integers and in Float64.
        rng = Xoshiro(n)
        Π = rand(rng, -5:5, m, m)
        R = Zn' * Π * Zn
        @test LI.phiid_mobius(L, R) == Π
        @test LI.phiid_mobius(L, Float64.(R)) == Π
        # The same in vec form, one row per sample.
        @test vec(R)' * L.kron == vec(Π)'
        @test_throws ArgumentError LI.phiid_mobius(L, zeros(m, m + 1))
    end
    @test [LI.pair_symbol(LI.phiid_lattice(2), c) for c in LI.phiid_lattice(2).pairs] == PHIID_NAMES
    @test_throws ArgumentError LI.phiid_lattice(4)
end

@testset "Per-sample inversion helper" begin
    # Linearity: the mean of the local atoms of random rows is the inversion of the mean
    # local redundancy.
    function check_linearity(n, backend, T; tol)
        L = LI.phiid_lattice(n)
        m = length(L.lattice.nodes)
        rng = Xoshiro(30 + n)
        K, N = 25, 400
        R = randn(rng, K, m^2)
        rows = rand(rng, 1:K, N)
        A = LI.phiid_local_atoms(L, R, rows; backend, precision = T)
        Rbar = reshape(vec(mean(R[rows, :]; dims = 1)), m, m)
        Πbar = LI.phiid_mobius(L, Rbar)
        ok = A isa Matrix{T} && size(A) == (N, m^2)
        ok &= maximum(abs, vec(mean(A; dims = 1)) .- vec(Πbar)) ≤ tol
        # Without a row map the rows of R are the samples.
        A1 = LI.phiid_local_atoms(L, R; backend, precision = T)
        ok &= size(A1) == (K, m^2) && maximum(abs, A1[rows, :] .- A) ≤ tol
        # Every row is the inversion of its own redundancies.
        ok &= maximum(abs, A1[3, :] .- vec(LI.phiid_mobius(L, reshape(R[3, :], m, m)))) ≤ tol * 10
        return ok
    end
    for n in (2, 3)
        @test check_linearity(n, CPU(), Float64; tol = 1e-12)
        @test check_linearity(n, CPU(), Float32; tol = 1e-3)
    end
    if CUDA.functional()
        for n in (2, 3)
            @test check_linearity(n, CUDABackend(), Float64; tol = 1e-12)
            @test check_linearity(n, CUDABackend(), Float32; tol = 1e-3)
        end
        L = LI.phiid_lattice(2)
        @test eltype(LI.phiid_local_atoms(L, randn(5, 16); backend = CUDABackend())) === Float32
    else
        @info "CUDA not functional; skipping GPU ΦID inversion tests."
        @test_skip false
    end
    L = LI.phiid_lattice(2)
    @test occursin("16 columns", msg_of(() -> LI.phiid_local_atoms(L, zeros(3, 15))))
    @test occursin("`rows`", msg_of(() -> LI.phiid_local_atoms(L, zeros(3, 16), [1, 4])))
    m = msg_of(() -> LI.phiid_local_atoms(LI.phiid_lattice(3), zeros(10, 324), ones(Int, 10^6); memory_limit = 2^20))
    @test occursin("memory_limit", m)
end

@testset "Exact discrete systems (MMI)" begin
    # Two variables x, y with uniform bits; past (x, y), future (x', y'). With MMI,
    # R[α, β] = min_{A ∈ α, B ∈ β} I(A; B) over the 9 mutual informations between
    # A ∈ {x, y, xy} and B ∈ {x', y', x'y'}. Rows/columns below are r = {1}{2}, x = {1},
    # y = {2}, s = {12}; the atoms are Π = Z⁻ᵀ R Z⁻¹, i.e. first the columns
    # (c_r, c_x - c_r, c_y - c_r, c_s - c_x - c_y + c_r), then the same on the rows.
    bits = [(a, b) for a in 0:1 for b in 0:1]
    function check(res, expected)
        return all(isapprox(atom(res, s), get(expected, s, 0.0); atol = 1e-12) for s in PHIID_NAMES)
    end

    # Copy, x' = x, y' = y: I(x;x') = I(y;y') = 1, I(x;y') = I(y;x') = 0, I(xy;x') =
    # I(xy;y') = I(x;x'y') = I(y;x'y') = 1, I(xy;x'y') = 2. So
    #        r  x  y  s                      r  x  y  s
    #   R = [0  0  0  1   (r)      Z⁻ᵀRZ⁻¹ = [0  0  0  1
    #        0  1  0  1   (x)                 0  1  0 -1
    #        0  0  1  1   (y)                 0  0  1 -1
    #        1  1  1  2]  (s)                 1 -1 -1  2]
    # MMI's redundancy min(I(x;x'y'), I(y;x'y')) = 1 bit (the two-bit-copy problem of
    # I_MMI) makes rts = str = 1 and sts = 2, compensated by -1 atoms; xtx = yty = 1.
    copyres = phiid_of_states([(a, b, a, b) for (a, b) in bits])
    copyatoms = Dict(:rts => 1, :xtx => 1, :xts => -1, :yty => 1, :yts => -1,
                     :str => 1, :stx => -1, :sty => -1, :sts => 2)
    @test check(copyres, copyatoms)
    @test sum(copyres.atoms) ≈ 2
    # Modes: storage = rtr + xtx + yty + sts = 0 + 1 + 1 + 2 = 4, copy = xtr + ytr = 0,
    # transfer = xty + ytx = 0, erasure = rtx + rty = 0, downward causation =
    # str + stx + sty = 1 - 1 - 1 = -1, upward causation = rts + xts + yts = -1; sum 2.
    dyn = information_dynamics(copyres)
    @test keys(dyn) == (:storage, :copy, :transfer, :erasure, :downward_causation, :upward_causation)
    @test values(dyn) == (4.0, 0.0, 0.0, 0.0, -1.0, -1.0)
    @test sum(dyn) ≈ sum(copyres.atoms)

    # Swap, x' = y, y' = x: as the copy with x' and y' exchanged, I(x;y') = I(y;x') = 1,
    # I(x;x') = I(y;y') = 0, so the x and y columns of R (and of the atoms) swap:
    # xty = ytx = 1, and rts = str = 1, sts = 2, xts = yts = stx = sty = -1 as above.
    swapres = phiid_of_states([(a, b, b, a) for (a, b) in bits])
    @test check(swapres, Dict(:rts => 1, :xty => 1, :xts => -1, :ytx => 1, :yts => -1,
                              :str => 1, :stx => -1, :sty => -1, :sts => 2))

    # Parity-preserving, x' uniform and independent of (x, y), y' = x' ⊕ x ⊕ y: every
    # single past or future variable, and (x, y) vs x' or y', or x vs (x', y'), are
    # independent; only I(xy; x'y') = 1 (x' ⊕ y' = x ⊕ y). So R = 0 except R[s, s] = 1,
    # and the only atom is sts = 1.
    parres = phiid_of_states([(a, b, c, c ⊻ a ⊻ b) for (a, b) in bits for c in 0:1])
    @test check(parres, Dict(:sts => 1))

    # Fully redundant, x = y = x' = y' one uniform bit: every I(A; B) = 1, so R = 1
    # everywhere and Π = Z⁻ᵀ 𝟙 Z⁻¹ is 1 at the bottom pair: rtr = 1, the rest 0.
    redres = phiid_of_states([(a, a, a, a) for a in 0:1])
    @test check(redres, Dict(:rtr => 1))

    # Base e: everything scales by ln 2.
    @test phiid_of_states([(a, b, a, b) for (a, b) in bits]; base = ℯ).atoms ≈ copyres.atoms .* log(2)

    # Three variables: three copied bits. Every pair of single-set nodes gives back the
    # mutual information I(past_A; future_B) = |A ∩ B| bits.
    trip = [(a, b, c) for a in 0:1 for b in 0:1 for c in 0:1]
    res3 = phiid_of_states([(t..., t...) for t in trip])
    L3 = res3.lattice
    @test size(res3.atoms) == (18, 18) && sum(res3.atoms) ≈ 3
    ok = true
    for A in 1:7, B in 1:7
        i = L3.lattice.index[[UInt(A)]]
        j = L3.lattice.index[[UInt(B)]]
        ok &= isapprox(rebuilt_redundancy(res3, i, j), count_ones(A & B); atol = 1e-12)
    end
    @test ok
end

# Identities between ΦID atoms and quantities computed by Associations with the same
# estimator family. `est` is the ΦID estimator; `mi`, `cmi`, `pid` the Associations
# estimators of MIShannon, CMIShannon and PartialInformationDecomposition(IMMI()), applied
# to the lagged series (`s`: the series as the ΦID sees them after encoding).
function check_phiid_identities(est, mi, cmi, pid, x, s; atol = 1e-10)
    res = association(est, x...)
    τ = est.definition.τ
    n = length(x)
    lagged = [phiid_lag(v, τ) for v in s]
    past, fut = first.(lagged), last.(lagged)
    joint(vs) = StateSpaceSet(vs...)
    out = Dict{Symbol,Bool}()
    m = length(res.lattice.lattice.nodes)
    out[:type] = res isa PhiIDResult && size(res.atoms) == (m, m)
    # The atoms sum to the time-delayed mutual information.
    out[:total] = isapprox(sum(res.atoms), association(mi, joint(past), joint(fut)); atol)
    # Single-set nodes: Σ_{α' ≼ {A}, β' ≼ {B}} Π = I(past_A; future_B), for all A, B.
    L = res.lattice.lattice
    ok = true
    for A in 1:(2^n - 1), B in 1:(2^n - 1)
        iA, iB = L.index[[UInt(A)]], L.index[[UInt(B)]]
        va, vb = [k for k in 1:n if isodd(A >> (k - 1))], [k for k in 1:n if isodd(B >> (k - 1))]
        ok &= isapprox(rebuilt_redundancy(res, iA, iB), association(mi, joint(past[va]), joint(fut[vb])); atol)
    end
    out[:single_sets] = ok
    # Row sums: the MMI PID of the past variables about the joint future; column sums:
    # the MMI PID of the future variables about the joint past.
    pidf = association(pid, past..., joint(fut))
    pidp = association(pid, fut..., joint(past))
    out[:rows] = all(isapprox(sum(atom(res, α => β) for (β, _) in atoms(pidp)), v; atol) for (α, v) in atoms(pidf))
    out[:cols] = all(isapprox(sum(atom(res, α => β) for (α, _) in atoms(pidf)), v; atol) for (β, v) in atoms(pidp))
    if n == 2
        xp, yp = past
        xf, yf = fut
        a(s) = atom(res, s)
        out[:te_xy] = isapprox(a(:xtr) + a(:xty) + a(:str) + a(:sty), association(cmi, xp, yf, yp); atol)
        out[:te_yx] = isapprox(a(:ytr) + a(:ytx) + a(:str) + a(:stx), association(cmi, yp, xf, xp); atol)
        out[:ais_y] = isapprox(a(:rtr) + a(:rty) + a(:ytr) + a(:yty), association(mi, yp, yf); atol)
        out[:ais_x] = isapprox(a(:rtr) + a(:rtx) + a(:xtr) + a(:xtx), association(mi, xp, xf); atol)
    end
    failed = [k for (k, v) in out if !v]
    isempty(failed) || @warn "ΦID identities failed" est failed
    return isempty(failed)
end

# A coupled VAR(1): x drives y, z (if any) is driven by both.
function phiid_var(rng, N; n = 2)
    X = zeros(N, 3)
    for t in 2:N
        X[t, 1] = 0.6X[t - 1, 1] + randn(rng)
        X[t, 2] = 0.4X[t - 1, 2] + 0.5X[t - 1, 1] + randn(rng)
        X[t, 3] = 0.3X[t - 1, 3] + 0.3X[t - 1, 1] - 0.4X[t - 1, 2] + randn(rng)
    end
    return Tuple(X[:, k] for k in 1:n)
end

@testset "Identities, $family" for family in (:discrete, :gaussian, :kraskov)
    rng = Xoshiro(hash(family) % 1000)
    for (τ, b) in ((1, 2), (2, ℯ))
        if family === :discrete
            o = ValueBinning(RectangularBinning(3, true))
            mk = def -> JointProbabilities(def, UE)
            est = JointProbabilities(IIDec(; τ, base = b), CodifyVariables(o))
            x = phiid_var(rng, 3000)
            s = codify.(Ref(o), x) # encoded once over the full series, then lagged
        elseif family === :gaussian
            mk = def -> def isa MIShannon ? GaussianMI(def) : def isa CMIShannon ? GaussianCMI(def) : GaussianDecomposition(def)
            est = GaussianDecomposition(IIDec(; τ, base = b))
            x = phiid_var(rng, 3000)
            s = x
        else
            mk = def -> EntropyDecomposition(def, Kraskov(; k = 4))
            est = EntropyDecomposition(IIDec(; τ, base = b), Kraskov(; k = 4))
            x = phiid_var(rng, 800)
            s = x
        end
        mi, cmi = mk(MIShannon(; base = b)), mk(CMIShannon(; base = b))
        pid = mk(PartialInformationDecomposition(IMMI(); base = b))
        @test check_phiid_identities(est, mi, cmi, pid, x, s)
        # A multivariate variable (two columns) and a third variable.
        if family === :gaussian
            x3 = phiid_var(rng, 3000; n = 3)
            xs = (StateSpaceSet(x3[1], randn(rng, 3000) .+ x3[2]), x3[3])
            @test check_phiid_identities(est, mi, cmi, pid, xs, xs)
            @test check_phiid_identities(est, mi, cmi, pid, x3, x3; atol = 1e-9)
        elseif family === :discrete && τ == 1
            x3 = phiid_var(rng, 3000; n = 3)
            @test check_phiid_identities(est, mi, cmi, pid, x3, codify.(Ref(o), x3))
        elseif family === :kraskov && τ == 2
            xs = (StateSpaceSet(x[1], x[2]), randn(rng, 800) .+ x[1])
            @test check_phiid_identities(est, mi, cmi, pid, xs, xs)
        end
    end
end

@testset "Encoding once, then lagging" begin
    rng = Xoshiro(91)
    x, y = phiid_var(rng, 2000)
    codes_est(τ) = JointProbabilities(IIDec(; τ), UE)
    # JointProbabilities: each series encoded with its own binning over the full length.
    o = ValueBinning(RectangularBinning(4, true))
    r1 = association(JointProbabilities(IIDec(; τ = 3), CodifyVariables(o)), x, y)
    r2 = association(codes_est(3), codify(o, x), codify(o, y))
    @test r1.atoms ≈ r2.atoms atol = 1e-12
    # EntropyDecomposition: Associations' joint-space binning of all inputs.
    r3 = association(EntropyDecomposition(IIDec(; τ = 2), PlugIn(Shannon()), CodifyVariables(o)), x, y)
    cm = Associations.codified_marginals(o, x, y)
    @test r3.atoms ≈ association(codes_est(2), cm...).atoms atol = 1e-12
    # Another discrete Shannon entropy estimator (averages only): the atoms still sum to
    # Associations' estimate of I(past; future) with the same estimator.
    rmm = association(EntropyDecomposition(IIDec(), MillerMadow(), CodifyVariables(o)), x, y)
    (xp, xf), (yp, yf) = phiid_lag(cm[1], 1), phiid_lag(cm[2], 1)
    @test sum(rmm.atoms) ≈ association(EntropyDecomposition(MIShannon(), MillerMadow(), UE),
        StateSpaceSet(xp, yp), StateSpaceSet(xf, yf)) atol = 1e-10
    # An embedding outcome space: τ counts encoded symbols.
    op = OrdinalPatterns{3}()
    r4 = association(JointProbabilities(IIDec(; τ = 2), CodifyVariables(op)), x, y)
    @test r4.atoms ≈ association(codes_est(2), codify(op, x), codify(op, y)).atoms atol = 1e-12
end

@testset "Accessors and show" begin
    rng = Xoshiro(17)
    x, y = rand(rng, 0:2, 500), rand(rng, 0:1, 500)
    y[2:end] .= mod.(x[1:(end - 1)] .+ y[2:end], 2)
    res = association(JointProbabilities(IIDec(), UE), x, y)
    as = atoms(res)
    @test length(as) == 16
    @test first(as) == (([[1], [2]] => [[1], [2]]) => atom(res, :rtr))
    @test last(as) == (([[1, 2]] => [[1, 2]]) => atom(res, :sts))
    @test [last(p) for p in as] == [atom(res, s) for s in PHIID_NAMES]
    @test all(atom(res, α => β) == v for ((α, β), v) in as)
    @test atom(res, :rts) == atom(res, [[2], [1]] => [(2, 1)])
    @test sum(res.atoms) ≈ association(JointProbabilities(MIShannon(), UE),
        StateSpaceSet(x[1:end-1], y[1:end-1]), StateSpaceSet(x[2:end], y[2:end])) atol = 1e-12
    s = sprint(show, MIME"text/plain"(), res)
    @test occursin("PhiIDResult: 2 variables, τ = 1, redundancy function IMMI, base 2", s)
    @test occursin("rts  {1}{2} → {12}", s) && occursin("total", s)

    # A LocalPhiIDResult built from (here constant) per-sample redundancies.
    N = 40
    L = res.lattice
    lres = LI.local_phiid_result(res.definition, L, reshape(vec(res.redundancies), 1, :), ones(Int, N))
    @test lres isa LocalPhiIDResult{<:IntegratedInformationDecomposition,Float64}
    @test size(lres.atoms) == (N, 16)
    @test atom(lres, :xty) isa SubArray && all(atom(lres, :xty) .≈ atom(res, :xty))
    @test all(all(v .≈ atom(res, p)) for (p, v) in atoms(lres))
    @test [first(p) for p in atoms(lres)] == [first(p) for p in as]
    dl, da = information_dynamics(lres), information_dynamics(res)
    @test keys(dl) == keys(da) && all(dl[k] isa Vector{Float64} && length(dl[k]) == N && all(dl[k] .≈ da[k]) for k in keys(da))
    # The six modes partition the atoms: they sum to I(past; future), on average and per sample.
    @test sum(da) ≈ sum(res.atoms) atol = 1e-12
    lrand = LI.local_phiid_result(res.definition, L, randn(rng, 7, 16), rand(rng, 1:7, N))
    @test sum(information_dynamics(lrand)) ≈ vec(sum(lrand.atoms; dims = 2)) atol = 1e-12
    sl = sprint(show, MIME"text/plain"(), lres)
    @test occursin("LocalPhiIDResult: 2 variables", sl) && occursin("$N samples × 16 atoms", sl)
    l32 = LI.local_phiid_result(res.definition, L, reshape(vec(res.redundancies), 1, :); precision = Float32)
    @test eltype(l32.atoms) === Float32

    # Three variables: no names, no information dynamics; `show` truncates.
    z = rand(rng, 0:1, 500)
    res3 = association(JointProbabilities(IIDec(), UE), x, y, z)
    @test length(atoms(res3)) == 324
    @test atom(res3, [[1], [2], [3]] => [[1, 2, 3]]) == res3.atoms[res3.lattice.lattice.index[UInt[1, 2, 4]], res3.lattice.lattice.index[UInt[7]]]
    s3 = sprint(show, MIME"text/plain"(), res3)
    @test occursin("3 variables", s3) && occursin("304 more atoms", s3)
    @test occursin("2 variables only", msg_of(() -> atom(res3, :rtr)))
    @test occursin("defined for a ΦID of 2 variables", msg_of(() -> information_dynamics(res3)))
    lres3 = LI.local_phiid_result(res3.definition, res3.lattice, reshape(vec(res3.redundancies), 1, :))
    @test_throws ArgumentError information_dynamics(lres3)
    @test occursin("Unknown ΦID atom", msg_of(() -> atom(res, :rtq)))
    @test occursin("`α => β`", msg_of(() -> atom(res, 3)))
    @test occursin("not a node", msg_of(() -> atom(res, [[1], [1, 2]] => [[1]])))
    @test occursin("between 1 and 2", msg_of(() -> atom(res, [[3]] => [[1]])))
end

@testset "Errors" begin
    rng = Xoshiro(3)
    x, y, z, w = (rand(rng, 0:1, 100) for _ in 1:4)
    est = JointProbabilities(IIDec(), UE)
    # Number of inputs.
    @test occursin("2 to 3 variables", msg_of(() -> association(est, x)))
    @test occursin("got 4", msg_of(() -> association(est, x, y, z, w)))
    # τ ≥ 1, N > τ.
    @test occursin("at least 1", msg_of(() -> IIDec(; τ = 0)))
    @test_throws ArgumentError IIDec(IMMI(), -1, 2)
    @test IIDec(; τ = 3, base = ℯ).τ == 3
    @test occursin("more than τ", msg_of(() -> association(JointProbabilities(IIDec(; τ = 5), UE), x[1:5], y[1:5])))
    @test occursin("more than τ", msg_of(() -> association(
        JointProbabilities(IIDec(; τ = 2), CodifyVariables(OrdinalPatterns{3}())), randn(rng, 4), randn(rng, 4))))
    @test occursin("same number of samples", msg_of(() -> association(est, x, y[1:50])))
    @test occursin("must be an `AbstractVector`", msg_of(() -> association(est, x, hcat(y, z))))
    # Redundancy functions without a ΦID double redundancy (ISx/IPM errors: test_phiid_pointwise.jl).
    @test occursin("no double-redundancy form", msg_of(() -> association(JointProbabilities(IIDec(IMin()), UE), x, y)))
    m = msg_of(() -> local_association(JointProbabilities(IIDec(IMin()), UE), x, y))
    @test occursin("IMin has no ΦID double redundancy", m) && !occursin("association(est", m)
    # Unsupported estimators.
    @test occursin("CodifyPoints", msg_of(() -> association(
        JointProbabilities(IIDec(), CodifyPoints(UniqueElementsEncoding(x))), x, y)))
    @test occursin("Supported estimators", msg_of(() -> association(MIDecomposition(IIDec(), KSG1()), randn(rng, 50), randn(rng, 50))))
    # The past/future entry point.
    @test occursin("same number of variables", msg_of(() -> LI.phiid_from_past_future(est, (x, y), (z,))))
    @test occursin("got 1", msg_of(() -> LI.phiid_from_past_future(est, (x,), (z,))))
    # Per-sample ΦID errors: test_phiid_pointwise.jl and test_phiid_local_mmi.jl.
end

@testset "Cross-check against syntropy" begin
    path = joinpath(@__DIR__, "fixtures", "syntropy_phiid_fixtures.json")
    if isfile(path)
        fx = JSON3.read(read(path, String))
        @test fx.meta.unit == "bits"
        kinds = Set{String}()
        for (name, case) in pairs(fx.cases)
            n = case.n
            res = if case.kind == "discrete"
                phiid_of_states(case.states, case.weights; n)
            elseif case.kind == "discrete_series"
                association(JointProbabilities(IIDec(; τ = case.tau), UE), (Int.(v) for v in case.series)...)
            elseif case.kind == "gaussian"
                association(GaussianDecomposition(IIDec(; τ = case.tau)), (Float64.(v) for v in case.series)...)
            else
                @warn "Unknown fixture kind $(case.kind) in case $name"
                continue
            end
            push!(kinds, case.kind)
            atol = case.kind == "gaussian" ? 1e-8 : 1e-10
            ok = length(case.atom_order) == length(res.atoms) && all(
                isapprox(atom(res, a.past => a.future), v; atol)
                for (a, v) in zip(case.atom_order, case.atoms.mmi))
            ok || @warn "ΦID fixture case $name differs from syntropy"
            @test ok
        end
        @test kinds == Set(["discrete", "discrete_series", "gaussian"])
    else
        @info "syntropy ΦID fixtures not found; skipping."
        @test_skip false
    end
end

@testset "Cross-check against phyid (average atoms)" begin
    path = joinpath(@__DIR__, "fixtures", "phyid_fixtures.json")
    if isfile(path)
        fx = JSON3.read(read(path, String))
        @test fx.meta.unit == "bits"
        for (name, case) in pairs(fx.cases)
            res = if case.kind == "discrete_series"
                association(JointProbabilities(IIDec(; τ = case.tau), UE), (Int.(v) for v in case.series)...)
            else
                association(GaussianDecomposition(IIDec(; τ = case.tau)), (Float64.(v) for v in case.series)...)
            end
            atol = case.kind == "gaussian" ? 1e-8 : 1e-10
            @test all(isapprox(atom(res, Symbol(s)), v; atol) for (s, v) in zip(case.atom_names, case.average))
        end
    else
        @info "phyid fixtures not found; skipping."
        @test_skip false
    end
end

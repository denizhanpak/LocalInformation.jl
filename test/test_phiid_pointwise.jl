# ΦID with the pointwise double redundancies ISx and IPM (src/phiid_pointwise.jl), and the
# subset counts shared with the PID (src/pid.jl).
import JSON3
import LinearAlgebra
using CUDA

const PW_IIDec = IntegratedInformationDecomposition
const PW_UE = CodifyVariables(UniqueElements())
const PW_REDUNDANCIES = (ISx(), IPM())

pw_est(r; τ = 1, base = 2, d = PW_UE) = JointProbabilities(PW_IIDec(r; τ, base), d)
pw_msg(f) = try f(); "no error" catch e; e isa ArgumentError ? e.msg : "not an ArgumentError: $e" end
pw_close(u, v; atol = 1e-10) = length(u) == length(v) && maximum(abs, u .- v; init = 0.0) ≤ atol
pw_lag(v, τ) = (v[1:(end - τ)], v[(1 + τ):end])
pw_joint(vs) = StateSpaceSet(vs...)

# Samples of an exact joint distribution: `states` are (past..., future...) tuples, state
# `k` repeated `weights[k]` times. Returns the state of each sample, the past and the future.
function pw_states(states, weights, n)
    rows = reduce(vcat, [fill(k, w) for (k, w) in enumerate(weights)])
    col(v) = [Int(states[k][v]) for k in rows]
    return rows, Tuple(col(v) for v in 1:n), Tuple(col(v) for v in (n + 1):(2n))
end

# A discrete Markov system with `k` symbols: x is persistent, x drives y, z (if any) is
# driven by x and y.
function pw_markov(rng, N; n = 2, k = 2)
    X = zeros(Int, N, 3)
    X[1, :] .= rand(rng, 0:(k - 1), 3)
    for t in 2:N
        X[t, 1] = rand(rng) < 0.8 ? X[t - 1, 1] : rand(rng, 0:(k - 1))
        X[t, 2] = rand(rng) < 0.7 ? X[t - 1, 1] : rand(rng, 0:(k - 1))
        X[t, 3] = rand(rng) < 0.7 ? mod(X[t - 1, 1] + X[t - 1, 2], k) : rand(rng, 0:(k - 1))
    end
    return Tuple(X[:, j] for j in 1:n)
end

@testset "Subset counts" begin
    rng = Xoshiro(5)
    N = 150
    for k in 2:6
        codes = [rand(rng, 0:2, N) for _ in 1:k]
        sc = LI.SubsetCounts(codes)
        K = length(sc.weight)
        @test sc.ninputs == k && sc.N == N && size(sc.counts) == (K, 2^k - 1)
        rep = [findfirst(==(s), sc.state) for s in 1:K]
        @test issorted(rep)                                   # numbered by first appearance
        @test all(sc.weight[s] == count(==(s), sc.state) for s in 1:K)
        @test all((sc.state[i] == sc.state[j]) == all(c[i] == c[j] for c in codes) for i in 1:N, j in 1:N)
        ok = true
        for M in 1:(2^k - 1), s in 1:K
            vars = [v for v in 1:k if isodd(M >> (v - 1))]
            ok &= sc.counts[s, M] == count(t -> all(codes[v][t] == codes[v][rep[s]] for v in vars), 1:N)
        end
        @test ok
        # PID layout: the last input is the target.
        src, srct, tgt = LI.pid_counts(sc)
        nm = 2^(k - 1) - 1
        @test size(src) == size(srct) == (K, nm)
        @test tgt == sc.counts[:, nm + 1]
        @test all(srct[:, M] == sc.counts[:, nm + 1 + M] for M in 1:nm)
    end
    # Multivariate codes (StateSpaceSet points) and the hash-table branch of `joint_labels`.
    S = collect(StateSpaceSet(rand(rng, 0:1, N), rand(rng, 0:1, N)))
    big = rand(rng, 1:400, N)
    sc = LI.SubsetCounts([S, big, rand(rng, 1:400, N)])
    @test sc.counts[:, 1] == [count(==(S[i]), S) for i in [findfirst(==(s), sc.state) for s in 1:length(sc.weight)]]
end

@testset "Union counts, $n variables" for n in (2, 3)
    data = LI.phiid_pointwise_data(n)
    @test LI.phiid_pointwise_data(n) === data                 # cached
    @test data.lattice === LI.phiid_lattice(n)
    nodes = data.lattice.lattice.nodes
    m = length(nodes)
    @test size(data.sx) == (2^(2n) - 1, 2m + m^2) && all(isinteger, data.sx)
    @test data.past == [Int.(node) for node in nodes] && data.future == [Int.(node) .<< n for node in nodes]
    # U = counts * sx are the counts of the unions of events of each state, by brute force.
    rng = Xoshiro(40 + n)
    N = 100
    codes = [rand(rng, 0:1, N) for _ in 1:(2n)]
    sc = LI.SubsetCounts(codes)
    U = sc.counts * data.sx
    agree(t, i, mask) = all(codes[v][t] == codes[v][i] for v in 1:(2n) if isodd(mask >> (v - 1)))
    inunion(t, i, masks) = any(agree(t, i, M) for M in masks)
    ok = true
    for s in eachindex(sc.weight)
        i = findfirst(==(s), sc.state)
        for a in 1:m
            ok &= U[s, a] == count(t -> inunion(t, i, data.past[a]), 1:N)
            ok &= U[s, m + a] == count(t -> inunion(t, i, data.future[a]), 1:N)
        end
        for b in 1:m, a in 1:m
            masks = [A | B for A in data.past[a] for B in data.future[b]]
            ok &= U[s, 2m + a + (b - 1) * m] == count(t -> inunion(t, i, masks), 1:N)
        end
    end
    @test ok
end

@testset "Cross-check against syntropy (ISx, IPM)" begin
    path = joinpath(@__DIR__, "fixtures", "syntropy_phiid_fixtures.json")
    if isfile(path)
        fx = JSON3.read(read(path, String))
        @test fx.meta.unit == "bits"
        kinds, ns = Set{String}(), Set{Int}()
        for (name, case) in pairs(fx.cases), (key, r) in ((:isx, ISx()), (:ipm, IPM()))
            case.kind in ("discrete", "discrete_series") || continue
            n = case.n
            if case.kind == "discrete"
                est = pw_est(r)
                state, past, fut = pw_states(case.states, case.weights, n)
                res = LI.phiid_from_past_future(est, past, fut)
                loc = LI.local_phiid_from_past_future(est, past, fut)
            else
                est = pw_est(r; τ = case.tau)
                series = [Int.(v) for v in case.series]
                res = association(est, series...)
                loc = local_association(est, series...)
                # Each lagged sample t ↔ the row of its state (x_t, …, x_{t+τ}, …).
                τ, N = case.tau, length(series[1])
                row = Dict(Tuple(Int.(s)) => k for (k, s) in enumerate(case.states))
                state = [row[(Tuple(v[t] for v in series)..., Tuple(v[t + τ] for v in series)...)] for t in 1:(N - τ)]
            end
            push!(kinds, case.kind); push!(ns, n)
            pairs_ = [a.past => a.future for a in case.atom_order]
            @test length(pairs_) == length(res.atoms) == size(loc.atoms, 2)
            @test size(loc.atoms, 1) == length(state)
            okavg = all(isapprox(atom(res, p), v; atol = 1e-10) for (p, v) in zip(pairs_, case.atoms[key]))
            # Pointwise atoms: one row per state (copied out of the lazy JSON arrays once).
            ptw = permutedims(reduce(hcat, [Float64.(collect(row)) for row in case.pointwise[key]]))
            okptw = size(ptw, 2) == length(pairs_) &&
                all(pw_close(atom(loc, p), ptw[state, c]) for (c, p) in enumerate(pairs_))
            (okavg && okptw) || @warn "ΦID $key fixture case $name differs from syntropy" okavg okptw
            @test okavg
            @test okptw
            # Average atoms = mean of the local atoms.
            @test all(isapprox(mean(v), atom(res, p); atol = 1e-12) for (p, v) in atoms(loc))
        end
        @test kinds == Set(["discrete", "discrete_series"]) && ns == Set([2, 3])
    else
        @info "syntropy ΦID fixtures not found; skipping."
        @test_skip false
    end
end

@testset "Average = mean of local atoms" begin
    rng = Xoshiro(71)
    N = 1500
    mean_ok(loc, res) = all(isapprox(mean(v), atom(res, p); atol = 1e-12) for (p, v) in atoms(loc))
    for r in PW_REDUNDANCIES
        # Integer series, 2 and 3 variables, τ = 1 and 2, bases 2 and e.
        for n in (2, 3), τ in (1, 2), b in BASES
            x = pw_markov(rng, N; n, k = n == 2 ? 3 : 2)
            est = pw_est(r; τ, base = b)
            res, loc = association(est, x...), local_association(est, x...)
            m = length(res.lattice.lattice.nodes)
            @test res isa PhiIDResult && size(res.atoms) == (m, m)
            @test loc isa LocalPhiIDResult{<:PW_IIDec,Float64} && size(loc.atoms) == (N - τ, m^2)
            @test loc.definition === est.definition && loc.lattice === res.lattice
            @test mean_ok(loc, res)
            @test vec(mean(loc.atoms; dims = 1)) ≈ vec(res.atoms) atol = 1e-12
            if b == ℯ # the unit: everything scales by ln 2
                @test loc.atoms ≈ local_association(pw_est(r; τ), x...).atoms .* log(2) atol = 1e-12
                @test res.atoms ≈ association(pw_est(r; τ), x...).atoms .* log(2) atol = 1e-12
            end
        end
        x, y, z = pw_markov(rng, N; n = 3)
        # A multivariate variable.
        est = pw_est(r)
        S = StateSpaceSet(x, z)
        res, loc = association(est, S, y), local_association(est, S, y)
        @test size(loc.atoms) == (N - 1, 16) && mean_ok(loc, res)
        codes = first(LI.relabel(collect(S)))
        @test loc.atoms ≈ local_association(est, codes, y).atoms atol = 1e-12
        # A binned continuous series, encoded once over the full length.
        u = cumsum(randn(rng, N)) ./ 10
        v = 0.5 .* circshift(u, 1) .+ randn(rng, N)
        o = ValueBinning(RectangularBinning(4, true))
        estb = pw_est(r; τ = 2, d = CodifyVariables(o))
        res, loc = association(estb, u, v), local_association(estb, u, v)
        @test size(loc.atoms) == (N - 2, 16) && mean_ok(loc, res)
        @test loc.atoms ≈ local_association(pw_est(r; τ = 2), codify(o, u), codify(o, v)).atoms atol = 1e-12
        # An embedding outcome space: τ and the rows count encoded symbols.
        op = OrdinalPatterns{3}()
        esto = pw_est(r; τ = 2, d = CodifyVariables(op))
        res, loc = association(esto, u, v), local_association(esto, u, v)
        @test size(loc.atoms) == (N - 2 - 2, 16) && mean_ok(loc, res)
        @test res.atoms ≈ association(pw_est(r; τ = 2), codify(op, u), codify(op, v)).atoms atol = 1e-12
        # The past/future entry point applies the discretization of `est` to each input.
        past, fut = (u[1:(end - 1)], v[1:(end - 1)]), (u[2:end], v[2:end])
        estp = pw_est(r; d = CodifyVariables(o))
        @test LI.phiid_from_past_future(estp, past, fut).atoms ≈
              LI.phiid_from_past_future(pw_est(r), codify.(Ref(o), past), codify.(Ref(o), fut)).atoms atol = 1e-12
    end
end

# Per-sample identities between the local atoms of `est` (on the series `x`, which the ΦID
# sees after encoding as `s`) and local measures computed by LocalInformation.
function pw_identities(est, x, s; atol = 1e-10)
    loc = local_association(est, x...)
    r, b, τ = est.definition.redundancy, est.definition.base, est.definition.τ
    n = length(x)
    lagged = [pw_lag(v, τ) for v in s]
    past, fut = first.(lagged), last.(lagged)
    lmi(p, q) = local_association(JointProbabilities(MIShannon(; base = b), PW_UE), p, q)
    out = Dict{Symbol,Bool}()
    # The atoms of a sample sum to its local I(past; future).
    out[:total] = pw_close(vec(sum(loc.atoms; dims = 2)), lmi(pw_joint(past), pw_joint(fut)); atol)
    # Single-set nodes: the local double redundancy Σ_{α' ≼ {A}, β' ≼ {B}} Π is the local
    # mutual information i(past_A; future_B).
    L = loc.lattice.lattice
    Zn = Matrix{Int}(L.below) + LinearAlgebra.I
    Rloc = loc.atoms * kron(Zn, Zn)
    m = length(L.nodes)
    ok = true
    for A in 1:(2^n - 1), B in 1:(2^n - 1)
        va, vb = [k for k in 1:n if isodd(A >> (k - 1))], [k for k in 1:n if isodd(B >> (k - 1))]
        c = L.index[[UInt(A)]] + (L.index[[UInt(B)]] - 1) * m
        ok &= pw_close(Rloc[:, c], lmi(pw_joint(past[va]), pw_joint(fut[vb])); atol)
    end
    out[:single_sets] = ok
    # Σ_β Π(α → β) = local PID atom α of the past variables about the joint future, and
    # Σ_α Π(α → β) = local PID atom β of the future variables about the joint past.
    pid = JointProbabilities(PartialInformationDecomposition(r; base = b), PW_UE)
    lf = local_association(pid, past..., pw_joint(fut))
    lp = local_association(pid, fut..., pw_joint(past))
    nodes = [first(p) for p in atoms(lf)]
    out[:rows] = all(pw_close(sum(atom(loc, α => β) for β in nodes), v; atol) for (α, v) in atoms(lf))
    out[:cols] = all(pw_close(sum(atom(loc, α => β) for α in nodes), v; atol) for (β, v) in atoms(lp))
    if n == 2
        (xp, yp), (xf, yf) = past, fut
        a(name) = atom(loc, name)
        lcmi(p, q, c) = local_association(JointProbabilities(CMIShannon(; base = b), PW_UE), p, q, c)
        out[:te_xy] = pw_close(a(:xtr) .+ a(:xty) .+ a(:str) .+ a(:sty), lcmi(xp, yf, yp); atol)
        out[:te_yx] = pw_close(a(:ytr) .+ a(:ytx) .+ a(:str) .+ a(:stx), lcmi(yp, xf, xp); atol)
        out[:ais_x] = pw_close(a(:rtr) .+ a(:rtx) .+ a(:xtr) .+ a(:xtx), lmi(xp, xf); atol)
        out[:ais_y] = pw_close(a(:rtr) .+ a(:rty) .+ a(:ytr) .+ a(:yty), lmi(yp, yf); atol)
        out[:dynamics] = pw_close(sum(information_dynamics(loc)), vec(sum(loc.atoms; dims = 2)); atol)
    end
    failed = [k for (k, v) in out if !v]
    isempty(failed) || @warn "Per-sample ΦID identities failed" est failed
    return isempty(failed)
end

@testset "Per-sample identities, $(nameof(typeof(r)))" for r in PW_REDUNDANCIES
    rng = Xoshiro(r isa ISx ? 101 : 102)
    for (n, k, τ, b) in ((2, 3, 1, 2), (2, 2, 2, ℯ), (3, 2, 1, 2), (3, 3, 2, ℯ))
        x = pw_markov(rng, 2000; n, k)
        @test pw_identities(pw_est(r; τ, base = b), x, x)
    end
    # A binned continuous series: the identities hold for the codes.
    o = ValueBinning(RectangularBinning(3, true))
    u = randn(rng, 2000)
    v = 0.7 .* circshift(u, 1) .+ randn(rng, 2000)
    @test pw_identities(pw_est(r; d = CodifyVariables(o)), (u, v), codify.(Ref(o), (u, v)))
end

@testset "GPU inversion (CUDA)" begin
    if CUDA.functional()
        gpu = CUDABackend()
        rng = Xoshiro(81)
        for n in (2, 3), r in PW_REDUNDANCIES
            x = pw_markov(rng, 3000; n, k = 3)
            est = pw_est(r)
            lc = local_association(est, x...)
            lg64 = local_association(est, x...; backend = gpu, precision = Float64)
            @test eltype(lg64.atoms) === Float64
            @test lg64.atoms ≈ lc.atoms rtol = 1e-12 atol = 1e-12
            lg = local_association(est, x...; backend = gpu)
            @test eltype(lg.atoms) === Float32
            @test maximum(abs, lg.atoms .- lc.atoms) ≤ 1e-4 * max(1, maximum(abs, lc.atoms))
        end
    else
        @info "CUDA not functional; skipping GPU pointwise ΦID tests."
        @test_skip false
    end
end

@testset "Show, precision and errors" begin
    rng = Xoshiro(13)
    x, y, z = pw_markov(rng, 400; n = 3)
    est = pw_est(IPM())
    loc = local_association(est, x, y)
    s = sprint(show, MIME"text/plain"(), loc)
    @test occursin("LocalPhiIDResult: 2 variables, τ = 1, redundancy function IPM", s)
    @test occursin("399 samples × 16 atoms", s)
    l32 = local_association(est, x, y; precision = Float32)
    @test eltype(l32.atoms) === Float32
    @test l32.atoms ≈ loc.atoms rtol = 1e-5
    @test occursin("`precision`", pw_msg(() -> local_association(est, x, y; precision = Int)))
    @test occursin("`backend`", pw_msg(() -> local_association(est, x, y; backend = :gpu)))

    # ISx and IPM are discrete-only, for `association` and `local_association`.
    a, b = randn(rng, 200), randn(rng, 200)
    for r in PW_REDUNDANCIES
        name = string(nameof(typeof(r)))
        for e in (GaussianDecomposition(PW_IIDec(r)), EntropyDecomposition(PW_IIDec(r), Kraskov(; k = 3)),
                  EntropyDecomposition(PW_IIDec(r), PlugIn(Shannon()), PW_UE))
            m = pw_msg(() -> association(e, a, b))
            @test occursin("discrete data only", m) && occursin(name, m) && occursin("JointProbabilities", m)
            ml = pw_msg(() -> local_association(e, a, b))
            @test occursin("Per-sample ΦID atoms with $name are defined for discrete data only", ml)
            @test occursin(string(nameof(typeof(e))), ml)
        end
    end
    # A discretization or probabilities estimator other than CodifyVariables / RelativeAmount.
    cp = JointProbabilities(PW_IIDec(ISx()), CodifyPoints(UniqueElementsEncoding(x)))
    @test occursin("CodifyPoints", pw_msg(() -> association(cp, x, y)))
    @test occursin("CodifyPoints", pw_msg(() -> local_association(cp, x, y)))
    ac = JointProbabilities(PW_IIDec(ISx()), PW_UE, AddConstant())
    @test occursin("RelativeAmount", pw_msg(() -> LI.phiid_from_past_future(ac, (x, y), (z, x))))
    @test occursin("RelativeAmount", pw_msg(() -> LI.local_phiid_from_past_future(ac, (x, y), (z, x))))
    # Inputs.
    @test occursin("2 to 3 variables", pw_msg(() -> local_association(est, x)))
    @test occursin("more than τ", pw_msg(() -> local_association(pw_est(ISx(); τ = 3), x[1:3], y[1:3])))
    @test occursin("same number of samples", pw_msg(() -> local_association(est, x, y[1:10])))
    @test occursin("same number of variables", pw_msg(() -> LI.local_phiid_from_past_future(est, (x, y), (z,))))

    # memory_limit: refused before counting when the result alone is too large…
    big = [rand(rng, 0:1, 20_000) for _ in 1:3]
    m = pw_msg(() -> local_association(pw_est(ISx()), big...; memory_limit = 2^20))
    @test occursin("memory_limit", m) && occursin("324 atoms of 3 variables", m) && !occursin("distinct states", m)
    # …and after counting when the per-state redundancies do not fit either.
    need0 = LI.check_phiid_pointwise_memory(CPU(), Float64, ISx(), 3, 20_000 - 1, nothing, Inf)
    m = pw_msg(() -> local_association(pw_est(IPM()), big...; memory_limit = need0 + 1))
    @test occursin("memory_limit", m) && occursin("distinct states", m)
    @test size(local_association(pw_est(IPM()), big...; memory_limit = 2^31).atoms) == (20_000 - 1, 324)
end

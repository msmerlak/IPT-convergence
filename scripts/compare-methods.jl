using DrWatson, Plots
using LinearAlgebra

include(srcdir("hamiltonians.jl"))
q = 8
H = anharmonic_oscillator(1.; dim = 1000, order = q)
X₀ = Matrix{eltype(H)}(I, size(H, 1), 1)

plot(
yaxis = :log, 
xlabel = "iteration",
ylabel = "residual norm",
dpi = 500,
title = " H = H₀ + X^$q"
)

### standard solvers don't work
using KrylovKit, Arpack, ArnoldiMethod

KrylovKit.eigsolve(H, 1, :SR; verbosity = 0, maxiter = 5000)
Arpack.eigs(H; nev = 1, which = :SR, maxiter = 5000)
ArnoldiMethod.partialschur(H; nev = 1, which = SR())


### LOBPCG with diagonal preconditioner

## IterativeSolvers
using IterativeSolvers, Preconditioners

sol_lobpcg_iterativesolvers = lobpcg(H, false, X₀, 1; P = DiagonalPreconditioner(H), maxiter = 50_000, log = true, tol = 1e-10)
plot!(
    [t.residual_norms[1] for t in sol_lobpcg_iterativesolvers.trace[1]],
    label = "LOBPCG with diag. precond."
)

## DFTK
include(srcdir("lobpcg_levitt.jl"))
D = Vector(diag(H))
P = FunctionPreconditioner((y, x) -> y .= x./D)
sol_lobpcg_dftk = LOBPCG(H, X₀, I, P, 1e-10, 1000)
plot!(
    vec(sol_lobpcg_dftk.residual_history),
    label = "LOBPCG (DFTK) with diag. precond."
)

### PRIMME
using PyCall, LinearAlgebra

@pyimport primme
@pyimport scipy.sparse as sp

vals, vecs, info = primme.eigsh(
    sp.csc_matrix(H), 1; 
    v0 = X₀,
    method=:PRIMME_JDQMR, 
    which=:SA, 
    tol = 1e-10,
    OPinv=sp.spdiags(one(eltype(H)) ./ diag(H), [0], size(H)...), return_stats=true, return_history=true, maxiter=1e3,
    convtest = (eval, evec, resNorm) -> resNorm < 1e-10
    )

plot!(
    info["hist"]["resNorm"],
    label = "PRIMME_DYNAMIC"
)


#### IPT

using IterativePerturbationTheory
include(srcdir("relaxed_IPT.jl"))


plot!(
    relaxed_ipt(H, 1; α = .5, trace = true, tol = 1e-5, maxiters = 1000).errors,
        label = "IPT - Simple relaxation (α = 1/2)"
    )

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 1, maxiter = 1000).trace,
    label = "IPT - Anderson (m = 1)"
)

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 5, maxiter = 50_000).trace,
    label = "IPT - Anderson (m = 5)"
)

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 10, maxiter = 50_000).trace,
    label = "IPT - Anderson (m = 10)"
)

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 50, maxiter = 50_000).trace,
    label = "IPT - Anderson (m = 50)"
)

plot!(
    ipt(H, 1; acceleration = :acx, trace = true, tol = 1e-10, maxiter = 50_000).trace,
    label = "IPT - ACX"
)
savefig(plotsdir("anharmonic_traces"))

heatmap(
    relaxed_ipt(anharmonic_oscillator(10.; dim = 10_000), 1; α = .5, trace = true).iterates[:, 1:20],
    xlabel = "component",
    ylabel = "iteration",
    dpi = 500,
    title = " H = H₀ + 10 X⁴"
)
savefig(plotsdir("anharmonic_strong_convergence"))



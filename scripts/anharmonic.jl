using DrWatson, Plots

include(srcdir("hamiltonians.jl"))
H = anharmonic_oscillator(10.; dim = 10_000)

### standard solvers don't work
using KrylovKit, Arpack, ArnoldiMethod
using IterativeSolvers, Preconditioners
KrylovKit.eigsolve(H, 1, :SR; verbosity = 0, maxiter = 5000)
Arpack.eigs(H; nev = 1, which = :SR, maxiter = 5000)
ArnoldiMethod.partialschur(H; nev = 1, which = SR())
lobpcg(H, false, 1; maxiter = 5000)

P = DiagonalPreconditioner(H)
lobpcg(H, false, 1; P = P, maxiter = 5000)

include(srcdir("davidson_herbst.jl"))
davidson(H, H[:, 1:20])

#### IPT

using IterativePerturbationTheory
include(srcdir("relaxed_ipt.jl"))

plot(
    yaxis = :log, 
    xlabel = "iteration",
    ylabel = "residual norm",
    dpi = 500,
    title = " H = H₀ + 10 X⁴"
    )

sol_lobpcg = lobpcg(H, false, 1; P = P, maxiter = 1000, log = true)
plot!(
    [t.residual_norms[1] for t in sol_lobpcg.trace],
    label = "LOBPCG with diag. precond."
)

plot!(
    relaxed_ipt(H, 1; α = .5, trace = true, tol = 6e-5, maxiters = 1000).errors,
        label = "Simple relaxation (α = 1/2)"
    )

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 1, maxiters = 1000).trace,
    label = "Anderson (m = 1)"
)

plot!(
    ipt(H, 1; acceleration = :anderson, trace = true, tol = 1e-10, anderson_memory = 5, maxiters = 1000).trace,
    label = "Anderson (m = 5)"
)

plot!(
    ipt(H, 1; acceleration = :acx, trace = true, tol = 1e-13, maxiters = 1000).trace,
    label = "ACX"
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



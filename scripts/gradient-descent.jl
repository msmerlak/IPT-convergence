using DrWatson
@quickactivate

using DifferentialEquations
using LinearAlgebra

using Plots; gr(dpi = 500)

ρ(H, x) = dot(x, H*x)/dot(x, x)

function GD!(Y, X, p, t) 
    H = p[:H] 
    mul!(Y, H, X, -1, 1)
    axpy!(-dot(X, Y)/dot(X, X), X, Y)
    Y ./= diag(H)
end

include(srcdir("hamiltonians.jl"))

H = anharmonic_oscillator(10., dim = 100, order = 8) 
p = Dict(:H => H)
X₀ = Matrix{Float64}(I, size(H, 1), 1)

pb = ODEProblem(
    GD!, X₀, (0, 10000.), 
    Dict(:H => H);
    callback = TerminateSteadyState()
)

sol = solve(pb, Tsit5())

E₀ = minimum(eigen(Matrix(H)).values)

plot(
    sol.t, 
    [ρ(H, x) - E₀ for x in sol.u];
    yaxis = :log,
    xlabel = "t",
    ylabel = "ρ(xₜ) - E₀",
    label = false,
    title = "preconditioned gradient descent (continuous time)"
)

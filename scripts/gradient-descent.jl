using DrWatson
@quickactivate

using DifferentialEquations
using LinearAlgebra

using Plots; gr(dpi = 500)

include(srcdir("hamiltonians.jl"))
H = anharmonic_oscillator(10., dim = 500, order = 4)

### look at eigenvalues of jacobian
using ForwardDiff

ψ = ipt(H, 1; acceleration = :anderson).vectors[:, 1]
ρ(H, x) = dot(x, H*x)/dot(x, x)
∇ρ(H, x) = (H*x - ρ(H,x)x)./dot(x, x)

eigen(Matrix(
    ForwardDiff.jacobian(x -> ∇ρ(H, x)./diag(H), ψ)
)).values


#### solve ODE
function GD!(Y, X, p, t) 
    H = p[:H] 
    mul!(Y, H, X, -1, 1)
    axpy!(-dot(X, Y)/dot(X, X), X, Y)
    Y ./= diag(H)
end

function IPT!(Y, X, p, t) 
    H = p[:H] 
    mul!(Y, H, X)
    mul!(Y, Diagonal(H), X, -1, 1)
    mul!(Y, X, Diagonal(Y), -1, 1)
    Y ./= H[1, 1] .- diag(H)  
    Y[diagind(Y)] .= 1
    axpy!(-1, X, Y)
end



p = Dict(:H => H)
X₀ = Matrix{Float64}(I, size(H, 1), 1)
E₀ = minimum(eigen(Matrix(H)).values)

plot()
for F! in (GD!, IPT!)
    pb = ODEProblem(
        F!, X₀, (0, 1000.), 
        Dict(:H => H);
        callback = TerminateSteadyState()
    )
    sol = solve(pb, Euler(); dt = .01)
    plot!(
        sol.t, 
        [ρ(H, x) - E₀ for x in sol.u];
        yaxis = :log,
        xlabel = "t",
        ylabel = "ρ(xₜ) - E₀",
        label = String(Symbol(F!))
    )
end
current()

using IterativePerturbationTheory
plot!(
    ipt(H, 1; acceleration = :anderson, anderson_memory = 5, trace = true, tol = 1e-10).trace
)

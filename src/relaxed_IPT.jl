using SparseArrays, LinearMaps, LinearAlgebra

function relaxed_ipt(
    M::Union{Matrix, SparseMatrixCSC, LinearMap},
    k=size(M, 1), # number of eigenpairs requested
    X₀=Matrix{eltype(M)}(I, size(M, 1), k); # initial eigenmatrix
    tol=100 * eps(real(eltype(M))) * norm(M),
    α = 1.,
    trace=false,
    maxiters = 1000
)


    N = size(M, 1)
    T = eltype(M)
    d = view(M, diagind(M))
    D = Diagonal(d)
    G = one(T) ./ (transpose(view(d, 1:k)) .- view(d, :))


    function F!(Y, X)
        mul!(Y, M, X)
        mul!(Y, D, X, -one(T), one(T))
        mul!(Y, X, Diagonal(Y), -one(T), one(T))
        Y .*= G
        Y[diagind(Y)] .= one(T)
        Y .=α * Y + (1-α)*X
    end


    X = copy(X₀)
    Y = similar(X)

    iterations = 0
    error = 1.0

    if trace
        errors = [vec(mapslices(norm, M * X - X * Diagonal(M * X); dims=1))]
        matvecs = [0]
        iterates = [copy(X)]
    end



    while tol < error < Inf && iterations < maxiters

        iterations += 1
        F!(Y, X)
        error = norm(X .- Y)
        X .= Y

        if trace
            push!(matvecs, matvecs[end] + k)
            push!(errors, vec(mapslices(norm, M * X - X * Diagonal(M * X); dims=1)))
            push!(iterates, copy(X))
        end
    end

    converged = error < tol


    if !converged
        return :Failed
    else
        return (
            vectors=X,
            values=diag(M * X),
            matvecs=trace ? matvecs : nothing,
            errors=trace ? reduce(hcat, errors)' : nothing,
            iterates = trace ? reduce(hcat, iterates)' : nothing
        )
    end

end


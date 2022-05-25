using LinearAlgebra

qrortho(X::AbstractArray)   = Array(qr(X).Q)
qrortho(X, Y)       = qrortho(X - Y * Y'X)

function rayleigh_ritz(X::AbstractArray, AX::AbstractArray, N)
    F = eigen(Hermitian(Matrix(X'*AX)))
    F.values[1:N], F.vectors[:,1:N]
end

function davidson(A, SS::AbstractArray; tol=1e-5, maxsubspace=8size(SS, 2), verbose=true)
    m = size(SS, 2)
    for i in 1:100
        Ass = A * SS
        rvals, rvecs = rayleigh_ritz(SS, Ass, m)
        Ax = Ass * rvecs

        R = Ax - SS * rvecs * Diagonal(rvals)
        if norm(R) < tol
            return rvals, SS * rvecs
        end

        verbose && println(i, "  ", size(SS, 2), "  ", norm(R))

        # Use QR to orthogonalise the subspace.
        if size(SS, 2) + m > maxsubspace
            SS = qrortho([SS*rvecs R])
        else
            SS = qrortho([SS       R])
        end
    end
    error("not converged.")
end
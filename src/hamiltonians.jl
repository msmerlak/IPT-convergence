using SparseArrays

function anharmonic_oscillator(g; dim=10, order=4)
    harmonic = spdiagm(dim, dim, 1:2:2dim)
    
    X = spdiagm(dim, dim,
        1 => sqrt.(2*(1:dim-1))/2,
        -1 => sqrt.(2*(1:dim-1))/2
        )
        
    return harmonic + g*X^order
end

module FunctionalMixedEffects

using LinearAlgebra
using Distributions
using Turing
using BSplines

function first_order_penalty_mat(p::Integer)
    @assert p >= 4
    pmat = zeros(Float64, p, p)
    for i in Base.OneTo(p-1)
        pmat[i,i] = 2.
        pmat[i+1,i] = -1.
        pmat[i,i+1] = -1.
    end
    pmat[p,p] = 1.
    return pmat
end

function simple_bspline_design_mat(x, low_bound, up_bound,  p::Integer)
    @assert p >= 4
    @assert low_bound < up_bound
    knots = range(low_bound, up_bound, p)
    basis = BSplineBasis(4, knots)
    return basismatrix(basis, x)
end





end # module
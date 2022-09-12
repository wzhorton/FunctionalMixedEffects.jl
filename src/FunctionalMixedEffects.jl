#module FunctionalMixedEffects

using LinearAlgebra
using Distributions
using BSplines

#--------------------------#
# General Helper Functions #
#--------------------------#
function first_order_penalty_mat(p::Integer)
    @assert p >= 4
    pmat = zeros(Float64, p, p)
    for i in Base.OneTo(p-1)
        pmat[i,i] = 2.
        pmat[i+1,i] = -1.
        pmat[i,i+1] = -1.
    end
    pmat[p,p] = 1.
    return Hermitian(pmat)
end

function simple_bspline_design_mat(x, low_bound, up_bound,  p::Integer)
    @assert p >= 4
    @assert low_bound < up_bound
    knots = range(low_bound, up_bound, p-2)
    basis = BSplineBasis(4, knots)
    return basismatrix(basis, x)
end

function identity_mat(::Type{T}, n::Integer) where {T <: Real}
    Matrix{T}(I, n, n)
end




#-------------------------#
# Main Hierarchical Model #
#-------------------------#

function mixed_functional_reg(
        Y::Matrix,
        Xfixed::Matrix,
        Xrand::Matrix,
        p::Int64,
        n_iterations::Int64,
        n_burnin::Int64
    )
    @assert size(Y,2) == size(Xfixed,2) == size(Xrand,2)

    n = size(Y,2)
    m = size(Y,1)
    qfixed = size(Xfixed,1)
    qrand = size(Xrand,1)

    P = first_order_penalty_mat(p)
    Pi = inv(P)
    H = simple_bspline_design_mat(range(0,1,m), 0, 1, p)
end

# Test code
#y = rand(101,5)
#Xfixed = vcat([1,1,1,1,1]',rand(5, 3)')
#Xrand = vcat([1,1,1,0,0]',[0,0,0,1,1]')
#out = mixed_functional_reg(y, Xfixed, Xrand, 8, 1000, 100)
#end # module
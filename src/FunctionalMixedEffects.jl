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

function conjugate_matrix_normal_regression(Y,X,U,V,C,D)
    #= Model parameters:
    Y (pxn) ~ MN(B*X, U, V)
    B (pxq) ~ MN(M, C, D)
    This draws a posterior value for B 
    Note that no validation code is used to
    either verify correct matrix formats or 
    match dimensions, other than what is
    implied by the operations themselves. 
    
    Also note that many straight inverses in this formula can't be avoided.
    For the sake of readability, only one system solve is used=#

    # I think the need for this is a bug. I've submitted a GitHub issue
    Cinv = typeof(C) <: Diagonal ? inv(C) : inv(cholesky(C))
    Dinv = typeof(D) <: Diagonal ? inv(D) : inv(cholesky(D))
    Uinv = typeof(U) <: Diagonal ? inv(U) : inv(cholesky(U))
    VinvXt = typeof(V) <: Diagonal ? V \ X' : cholesky(V) \ X'

    C_post = typeof(C) <: Diagonal && typeof(U) <: Diagonal ? inv(Uinv + Cinv) : inv(cholesky(Uinv + Cinv))
    D_post = inv(cholesky(X * VinvXt + Dinv))
    M_post = C_post * (Uinv * Y * VinvXt + Cinv * M * Dinv) * D_post

    Z = rand(Normal(), size(M_post))
    return M_post + cholesky(C_post).L * Z * cholesky(D_post).U
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
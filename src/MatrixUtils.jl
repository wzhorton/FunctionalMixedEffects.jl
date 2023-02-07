# Matrix utility tools

module MatrixUtils

using Distributions
using LinearAlgebra
using BSplines


#===================================================================================
First order penalty matrix for P-splines

Arguments:
p : Integer

Return: A Hermitian matrix.
Note: Although the structure here supports a symmetric tri-diagonal matrix, the way 
this matrix is usually used will only preserve the Hermitian type.
Exported: true =====================================================================#
export first_order_penalty_mat

function first_order_penalty_mat(p::Integer)
    @assert p >= 4
    pmat = zeros(Float64, p, p)
    for i in Base.OneTo(p-1)
        pmat[i,i] = 2.
        pmat[i+1,i] = -1.
        pmat[i,i+1] = -1.
    end
    pmat[1,1] = 1.1
    pmat[p,p] = 1.1
    return Hermitian(pmat)
end


#======================================================
Cubic B-spline design matrix for evenly spaced knots

Arguments:
x : vector of evaluation locations
low_bound/up_bound : bounds for the vector x
p : number of spline coefficients

Return: Dense matrix with row evaluations.
Exported: true =======================================#
export simple_bspline_design_mat

function simple_bspline_design_mat(x, low_bound, up_bound,  p::Integer)
    @assert p >= 4
    @assert low_bound < up_bound
    knots = range(low_bound, up_bound, p-2)
    basis = BSplineBasis(4, knots)
    return basismatrix(basis, x)
end


#=====================================================================================
Conjugate matrix-normal matrix-normal regression coefficients

Arguments:
Y, X : Observation matrices
M : Mean matrix
U, V, D : Covariance Matrices

Notes:
Model parameters follow this structure:
    Y (pxn) ~ MN(B*X, U, V)
    B (pxq) ~ MN(M, U, D)
Dimensions are checked implicitly. Diagonal matrices are handled separately due to
Cholesky compatibility. Other special types may not work, but Hermitian does work.
Against common advice, this function caches inverted matrices rather than solving
a the linear system each time. This is required in several places, so for the
sake of readability has been done in all cases.

The fact that U is common between Y and B is critical. Otherwise, the posterior is
not matrix normal, but a more general reshaped vector normal with non-kronecker
covariance structure (sum of kroneckers actually). Technically, a scalar multiple
difference is allowed, but because this is unidentifiable, users are required to
put marginal variances on V and/or D

Returns: MatrixNormal object for posterior B|Y
Exported: true ======================================================================#
export conjugate_matrix_normal_regression

function conjugate_matrix_normal_regression(Y,X,U,V,M,D)
    # Issue #46721, inv(cholesky(Diagonal(...))) fails.
    # Error fixed in Julia 1.8, type specialization pull request placed
    Dinv = typeof(D) <: Diagonal ? inv(D) : inv(cholesky(D))
    VinvXt = typeof(V) <: Diagonal ? V \ X' : cholesky(V) \ X'

    D_post = inv(cholesky(Hermitian(X * VinvXt + Dinv)))
    M_post = (Y * VinvXt + M * Dinv) * D_post

    return MatrixNormal(M_post, U, D_post)
end


#=============================================================
Conjugate matrix-normal inverse-gamma marginal variance

Arguments:
Y : Observation matrix
M : Mean matrix
R, V : Correlation/Covariance matrices
a, b: Variance hyperparameters

Notes:
Model parameters follow this structure
    Y ~ (mxn) MN(M, σR, V)
    σ ~ IG(a,b)
Note that marginal variance is unidentifiable, so we take the
convention that U = σR is factored and V is untouched.
Returns: InverseGamma object for posterior σ|Y
Exported: true ===============================================#
export conjugate_matrix_normal_variance

function conjugate_matrix_normal_variance(Y,M,R,V,a,b)
    m = size(Y,1)
    n = size(Y,2)
    exp_term = (V \ (Y-M)') * (R \ (Y-M))
    return InverseGamma(a + n*m/2, b + tr(exp_term)/2)
end

end # module
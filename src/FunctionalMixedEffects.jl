#module FunctionalMixedEffects

using LinearAlgebra
using Distributions
using BSplines

include("MatrixUtils.jl")
import .MatrixUtils as mutil


#------------------#
# Helper Functions #
#------------------#

function identity_mat(::Type{T}, n::Integer) where {T <: Real}
    Diagonal(Matrix{T}(I, n, n))
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
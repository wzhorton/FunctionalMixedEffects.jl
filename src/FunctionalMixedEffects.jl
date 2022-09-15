module FunctionalMixedEffects

using LinearAlgebra
using Distributions
using ProgressMeter

include("MatrixUtils.jl")
import .MatrixUtils as mutil

#------------------#
# Helper Functions #
#------------------#

function identity_mat(::Type{T}, n::Integer) where {T <: Real}
    Diagonal(Matrix{T}(I, n, n))
end

function imat(n::Integer)
    identity_mat(Float64,n)
end


#----------------------#
# Model config options #
#----------------------#
export OutputConfigFME
Base.@kwdef struct OutputConfigFME{N <: Integer}
    p::N = 20
    n_iterations::N = 15000
    n_burnin::N = 5000
    n_thin::N = 1
    save_random_effects::Bool = false
    save_theta::Bool = false
    function OutputConfigFME(p::Integer, n_iterations::Integer, n_burnin::Integer, 
            n_thin::Integer, save_random_effects::Bool, save_theta::Bool)
        if any(x->x <= zero(x),(n_iterations, n_burnin, n_thin))
            error("Non-positive config count found")
        end
        new(p, n_iterations, n_burnin, n_thin, save_random_effects, save_theta)
    end
end


#-----------------------#
# Model hyperparameters #
#-----------------------#
export HyperParametersFME
Base.@kwdef struct HyperParametersFME{T<:AbstractFloat}
    a_σ::T = 3.
    b_σ::T = 1.
    a_τ::T = 3.
    b_τ::T = 1.
    a_λ::T = 3.
    b_λ::T = 1.
    v_fix::T = 1000.
    v_cent::T = 1000.
end


#-------------------#
# Modal data object #
#-------------------#

Base.@kwdef struct DataFME{T<:AbstractFloat, N<:Integer}
    Y::Union{Matrix{T},Nothing}
    Xfix::Union{Matrix{T},Nothing}
    Xrand::Union{Matrix{T},Nothing}
    Xcent::Union{Matrix{T},Nothing}
    n::N
    m::N
    qfix::Union{N,Nothing}
    qrand::Union{N,Nothing}
    qcent::Union{N,Nothing}
    function DataFME(
            Y::Matrix,
            Xfix::Union{Matrix,Nothing},
            Xrand::Union{Matrix,Nothing},
            Xcent::Union{Matrix,Nothing}
        )
        @assert isnothing(Xrand) == isnothing(Xcent) # Proper hierarchical centering
        @assert !isnothing(Xrand) || !isnothing(Xfix) # At least one set of covariates
        n = size(Y,2)
        m = size(Y,1)
        qfix = isnothing(Xfix) ? nothing : size(Xfix,1)
        qrand = isnothing(Xrand) ? nothing : size(Xrand,1)
        qcent = isnothing(Xcent) ? nothing : size(Xcent,1)
        new(Y, Xfix, Xrand, Xcent, n, m, qfix, qrand, qcent)
    end
end


#--------------------------------#
# Model parameter save structure #
#--------------------------------#

Base.@kwdef mutable struct ChainsFME{T <: AbstractFloat}
    σ::Vector{T}
    τ::Vector{T}
    λ::Vector{T}
    θ::Array{T,3}
    Bfix::Array{T,3}
    Brand::Array{T,3}
    Bcent::Array{T,3}
end

function ChainsFME(data::DataFME, cfg::OutputConfigFME)
    if cfg.save_random_effects
        @assert !isnothing(data.qrand) && !isnothing(data.qcent) # save only if computation is possible
    end
    ChainsFME(
        σ = zeros(Float64, cfg.n_iterations),
        τ = zeros(Float64, cfg.n_iterations),
        λ = zeros(Float64, cfg.n_iterations),
        θ = cfg.save_theta ? zeros(Float64, cfg.p, data.n, cfg.n_iterations) : zeros(Float64, 0, 0, 0),
        Bfix = isnothing(data.qfix) ? zeros(Float64, 0, 0, 0) : zeros(Float64, cfg.p, data.qfix, cfg.n_iterations),
        Brand = cfg.save_random_effects ? zeros(Float64, cfg.p, data.qrand, cfg.n_iterations) : zeros(Float64, 0, 0, 0),
        Bcent = isnothing(data.qcent) ? zeros(Float64, 0, 0, 0) : zeros(Float64, cfg.p, data.qcent, cfg.n_iterations)
    )
end


#-------------------------#
# Hierarchical Model MCMC #
#-------------------------#
export mcmc_fme
function mcmc_fme(
        data::DataFME,
        hyps::HyperParametersFME,
        cfg::OutputConfigFME
    )

    # Constant values
    In = imat(data.n)
    Im = imat(data.m)
    Ip = imat(cfg.p)
    Iqfix = isnothing(data.qfix) ? nothing : imat(data.qfix)
    Iqrand = isnothing(data.qrand) ? nothing : imat(data.qrand)
    Iqcent = isnothing(data.qcent) ? nothing : imat(data.qcent)

    P = mutil.first_order_penalty_mat(cfg.p)
    Pi = inv(P)
    H = mutil.simple_bspline_design_mat(range(0,1,data.m), 0, 1, cfg.p)

    M_Bfix =  isnothing(data.qfix) ? nothing : zeros(Float64, cfg.p, data.qfix)
    M_Bcent = isnothing(data.qcent) ? nothing : zeros(Float64, cfg.p, data.qcent)

    # MCMC variables
    chains = ChainsFME(data, cfg)
    σ = 1.0
    τ = 1.0
    λ = 1.0
    θ = zeros(Float64, cfg.p, data.n)
    Bfix = isnothing(data.qfix) ? nothing : zeros(Float64, cfg.p, data.qfix, cfg.n_iterations)
    Brand = isnothing(data.qrand) ? nothing : zeros(Float64, cfg.p, data.qrand, cfg.n_iterations)
    Bcent = isnothing(data.qcent) ? nothing : zeros(Float64, cfg.p, data.qcent, cfg.n_iterations)

    # Helper variables
    μ = !isnothing(data.qfix) && !isnothing(data.qrand) ? Bfix * data.Xfix + Brand * data.Xrand : 
        !isnothing(data.qfix) ? Bfix * data.Xfix : Brand * data.Xrand
    E_θ = θ - μ

    # Progress bar setup
    pbar = Progress(cfg.n_iterations; dt=1, desc="MCMC Progress:", showspeed=true)
    println("Starting MCMC...")

    # MCMC Loop
    for it in -(cfg.n_burnin-1):cfg.n_iterations
        for thin in Base.OneTo(cfg.n_thin)
            # Update σ, θ, and E_θ
            σ = rand(mutil.conjugate_matrix_normal_variance(data.Y, H*θ, Im, In, hyps.a_σ, hyps.b_σ))
            τ = rand(mutil.conjugate_matrix_normal_variance(θ, μ, Pi, In, hyps.a_τ, hyps.b_τ))
            θ .= rand(mutil.conjugate_matrix_normal_regression(data.Y', H', In, σ*Im, μ', τ*Pi))'
            E_θ .= θ - μ

            # Update Bfix and μ, E_θ
            if !isnothing(data.qfix)
                μ .-= Bfix * data.Xfix
                Bfix .= rand(mutil.conjugate_matrix_normal_regression(
                    E_θ + Bfix * data.Xfix, data.Xfix, Pi, τ*In, M_Bfix, hyps.v_fix*Iqfix)
                )
                μ .+= Bfix * data.Xfix
                E_θ .= θ - μ
            end
            
            # Update λ, Brand, Bcent and μ, E_θ
            if !isnothing(data.qrand)
                λ = rand(mutil.conjugate_matrix_normal_variance(
                    Brand, Bcent * data.Xcent, Ip, Iqrand, hyps.a_λ, hyps.b_λ))

                μ .-= Brand * data.Xrand
                Brand = rand(mutil.conjugate_matrix_normal_regression(
                    E_θ + Brand * data.Xrand, data.Xrand, Pi, τ*In, Bcent * data.Xcent, λ*Iqrand))
                μ .+= Brand * data.Xrand
                E_θ .= θ - μ

                Bcent = rand(mutil.conjugate_matrix_normal_regression(
                    Brand, data.Xcent, Pi, λ*Iqrand, M_Bcent, hyps.v_cent * Iqcent))
            end
        end # thin loop

        # Save iterations
        if it > zero(it)
            next!(pbar)
            chains.σ[it] = σ
            chains.τ[it] = τ
            chains.λ[it] = λ
            if cfg.save_theta
                chains.θ[:,:,it] = θ
            end

            if !isnothing(qfix)
                chains.Bfix[:,:,it] = Bfix
            end
            if !isnothing(qrand)
                chains.Bcent[:,:,it] = Bcent
                if cfg.save_random_effects
                    chains.Brand[:,:,it] = Brand
                end
            end
        end # saves
    end # iter loop
    println("Finished!")
    return chains
end

end # module
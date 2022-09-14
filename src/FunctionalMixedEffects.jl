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


#----------------------#
# Model config options #
#----------------------#
export OutputConfigFME
Base.@kwdef struct OutputConfigFME
    n_iterations::Int64 = 5000
    n_burnin::Int64 = 1000
    n_thin::Int64 = 1
    save_random_effects::Bool = false
    save_theta::Bool = false
    function OutputConfigFME(n_iterations::Int64, n_burnin::Int64, 
            n_thin::Int64, save_random_effects::Bool, save_theta::Bool)
        if any(x->x <= zero(Int64),(n_iterations, n_burnin, n_thin))
            error("Non-positive config count found")
        end
        new(n_iterations, n_burnin, n_thin, save_random_effects, save_theta)
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
    v_fix::T = 10000.
    # B priors are fixed
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
end

function ChainsFME(p::Int64, qfix::Int64, qrand::Union{Int64,Nothing}, cfg::OutputConfigFME)
    ChainsFME(
        σ = zeros(Float64, cfg.n_iterations),
        τ = zeros(Float64, cfg.n_iterations),
        λ = zeros(Float64, cfg.n_iterations),
        θ = cfg.save_random_effects ? zeros(Float64, p, n, cfg.n_iterations) : zeros(Float64, 0, 0, 0),
        Bfix = zeros(Float64, p, qfix, cfg.n_iterations),
        Brand = cfg.save_random_effects ? zeros(Float64, p, qrand, cfg.n_iterations) : zeros(Float64, 0, 0, 0)
    )
end


#-------------------------#
# Hierarchical Model MCMC #
#-------------------------#
export mcmc_fme
function mcmc_fme(
        Y::Matrix,
        Xfix::Matrix,
        Xrand::Union{Nothing,Matrix},
        p::Int64,
        hyps::HyperParametersFME,
        cfg::OutputConfigFME
    )
    # Input checks
    @assert size(Y,2) == size(Xfix,2) # == size(Xrand,2)
    if isnothing(Xrand) && cfg.save_random_effects 
        error("Cannot save random effects; Xrand is missing.")
    end

    # Constant values
    n = size(Y,2)
    m = size(Y,1)
    qfix = size(Xfix,1)
    qrand = isnothing(Xrand) ? nothing : size(Xrand,1)

    In = identity_mat(Float64,n)
    Im = identity_mat(Float64,m)
    Ip = identity_mat(Float64,p)
    Iqfix = identity_mat(Float64,qfix)
    Iqrand = isnothing(Xrand) ? nothing : identity_mat(Float64,qrand)

    P = mutil.first_order_penalty_mat(p)
    Pi = inv(P)
    H = mutil.simple_bspline_design_mat(range(0,1,m), 0, 1, p)

    M_Bfix = zeros(Float64, p, qfix)
    M_Brand = isnothing(Xrand) ? nothing : zeros(Float64, p, qrand)

    # MCMC variables
    chains = ChainsFME(p, qfix, qrand, cfg)
    σ = 1.0
    τ = 1.0
    λ = 1.0
    θ = zeros(Float64, p, n)
    Bfix = zeros(Float64, p, qfix)
    Brand = isnothing(Xrand) ? nothing : zeros(Float64, p, qrand)

    # Helper variables
    μ = isnothing(Xrand) ? Bfix*Xfix : Bfix*Xfix + Brand*Xrand
    E_θ = θ - μ

    # Progress bar setup
    pbar = Progress(cfg.n_iterations; dt=1, desc="MCMC Progress:", showspeed=true)
    println("Starting MCMC...")

    # MCMC Loop
    for it in -(cfg.n_burnin-1):cfg.n_iterations
        for thin in Base.OneTo(cfg.n_thin)
            # Update σ, θ, and E_θ
            σ = rand(mutil.conjugate_matrix_normal_variance(Y, H*θ, Im, In, hyps.a_σ, hyps.b_σ))
            θ .= rand(mutil.conjugate_matrix_normal_regression(Y', H', In, σ*Im, μ', In, τ*Pi))'
            E_θ .= θ - μ

            # Update τ, Bfix, and μ
            τ = rand(mutil.conjugate_matrix_normal_variance(θ, μ, Pi, In, hyps.a_τ, hyps.b_τ))
            Bfix .= rand(mutil.conjugate_matrix_normal_regression(
                E_θ + Bfix*Xfix, Xfix, τ*Pi, In, M_Bfix, hyps.v_fix*Ip, Iqfix)
            )
            μ .= Bfix*Xfix

            # Update Brand, μ, and λ
            if !isnothing(Xrand)
                Brand = rand(mutil.conjugate_matrix_normal_regression(
                    E_θ + Brand*Xrand, Xrand, τ*Pi, In, M_Brand, λ*Ip, Iqrand))
                μ .+= Brand*Xrand
                λ = rand(mutil.conjugate_matrix_normal_variance(
                    Brand, M_Brand, Ip, Iqrand, hyps.a_λ, hyps.b_λ))
            end
        end # thin loop

        # Save iterations
        if it > zero(it)
            next!(pbar)
            chains.σ[it] = σ
            chains.τ[it] = τ
            chains.λ[it] = λ
            chains.Bfix[:,:,it] = Bfix
            if cfg.save_random_effects
                chains.Brand[:,:,it] = Brand
            end
            if cfg.save_theta
                chains.θ[:,:,it] = θ
            end
        end
    end # iter loop
    println("Finished!")
    return chains
end

end # module
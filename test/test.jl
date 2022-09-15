include("../src/FunctionalMixedEffects.jl")
using .FunctionalMixedEffects
using Distributions
using LinearAlgebra

x = collect(1:101)
Y = hcat([i .+ rand(Normal(0, 0.002),101) for i in 1:10]...)

Xcent = reshape(repeat([1.],5), 1, 5)
Xrand = kron(Diagonal(Matrix(1.0I, 5, 5)), [1.0 1.0])

hyps = HyperParametersFME()
cfg = OutputConfigFME(20, 15000,5000,1,false,true)
data = DataFME(Y, nothing, Xrand, Xcent)

chains = mcmc_fme(data, hyps, cfg);
println(mean(chains.σ))
println(mean(chains.τ))
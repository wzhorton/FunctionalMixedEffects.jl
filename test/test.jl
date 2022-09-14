include("../src/FunctionalMixedEffects.jl")
using .FunctionalMixedEffects
using Distributions

x = collect(1:101)
Y = hcat([i .+ rand(Normal(0, 0.002),101) for i in 1:10]...)

Xfix = reshape(repeat([1.],10), 1, 10)

hyps = HyperParametersFME()
cfg = OutputConfigFME(15000,5000,1,false,true)

chains = mcmc_fme(Y, Xfix, nothing, 20, hyps, cfg);
println(mean(chains.σ))
println(mean(chains.τ))
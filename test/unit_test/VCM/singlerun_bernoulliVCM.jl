using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem
using Ipopt
using NLopt

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
#   βtrue = ones(p)
Random.seed!(12345)
# try next
βtrue = rand(Uniform(-2, 2), p)
Σtrue = [0.5]

d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64

samplesize = 10000
ni = 5

gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = Σtrue[1] * ones(ni, ni)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η) ./ (1 .+ exp.(η))
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Bernoulli(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end

gcm = GLMCopulaVCModel(gcs);
# initialize_model!(gcm)
# @show gcm.β
# @show gcm.Σ

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-7, limited_memory_max_history = 20, accept_after_max_steps = 1, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.∇β
@show gcm.∇Σ

@test gcm.β ≈ [0.2718906573470128, 1.402611603041587, -0.5134731411739186]
@test gcm.Σ ≈ [0.49305430143517737]

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
@show mseβ
@show mseΣ
#
using Test
@test mseβ ≈ 0.00015021026740224318
@test mseΣ ≈ 4.824272855337919e-5

using BenchmarkTools
println("checking memory allocation for Bernoulli VCM")
logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)
@test logl_gradient_memory.memory == 0.0

using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
#   βtrue = ones(p)
Random.seed!(12345)
# try next
βtrue = rand(Uniform(-2, 2), p_fixed)
Σtrue = [0.5]
rtrue = 10.0

d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

samplesize = 10000
ni = 5

gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = Σtrue[1] * ones(ni, ni)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η)
    p = rtrue ./ (μ .+ rtrue)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
end

# form VarLmmModel
gcm = NBCopulaVCModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, tol = 1e-5, maxBlockIter = 50)
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.r
@show gcm.∇β
@show gcm.∇Σ
@show gcm.∇r

# ipopt version 0.9 results
# @test gcm.β ≈ [0.25203920476125224, 1.4003501217778764, -0.5073708048091656]
# @test gcm.Σ ≈ [0.4992562490328951]
# @test gcm.r ≈ [10.2357654206632]

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
mseβ, mser, mseΣ = MSE(gcm, βtrue, rtrue, Σtrue)
@show mseβ
@show mser
@show mseΣ

using Test
@test mseβ < 0.01
@test mseΣ < 0.01
@test mser < 0.1

# @test mseβ ≈ 1.3429466418437811e-5
# @test mser ≈ 0.05558533358049526
# @test mseΣ ≈ 5.531655010694497e-7

# need to optimize memory allocation 13.73 MIB
# using BenchmarkTools
# println("checking memory allocation for Nb VCM")
# logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)
# @test logl_gradient_memory.memory == 0.0

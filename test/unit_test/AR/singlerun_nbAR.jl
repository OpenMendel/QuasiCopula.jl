using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p_fixed = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p_fixed)
rtrue = 10.0
σ2true = [0.5]
ρtrue = [0.5]

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = vec[i - 1] * ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end

#simulation parameters
samplesize = 10000

st = time()
currentind = 1
d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 5 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    # X = [ones(ni) randn(ni, p - 1)]
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
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
    # push!(Ystack, y)
    V = [ones(ni, ni)]
    # V = [ones(ni, ni)]
    gcs[i] = NBCopulaARObs(y, X, d, link)
end

# form model
gcm = NBCopulaARModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter = 20, tol=1e-6)
@show fittime
@show gcm.θ
@show gcm.∇θ

# @test gcm.θ ≈ [0.25408749719321383, 1.394747799442321, -0.5060772204080676, 0.5182372977100089, 0.5088787843885787]

@show gcm.r
@show gcm.∇r

# @test gcm.r ≈ [10.135298765900169]

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

# mse and time under our model
mseβ, mseρ, mseσ2, mser = MSE(gcm, βtrue, ρtrue, σ2true, rtrue)
@show mseβ
@show mser
@show mseσ2
@show mseρ

using Test
@test mseβ < 0.01
@test mseσ2 < 0.01
@test mseρ < 0.01
@test mser < 0.1

# using Test
# @test mseβ ≈ 3.060579384225827e-5
# @test mser ≈ 0.01830575605410867
# @test mseσ2 ≈ 7.88328122188688e-5
# @test mseρ ≈ 0.00033259902776349684

# need to optimize wrt to memory 13.73 MIB
# using BenchmarkTools
# println("checking memory allocation for nb AR")
# logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)

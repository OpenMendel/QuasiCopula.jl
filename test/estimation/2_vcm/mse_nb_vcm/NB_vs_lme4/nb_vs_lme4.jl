using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

# simulation 1: generate data under QC model
# simulate samplesize = 10,000 vectors of nb base distribution to save to csv and analyze with R package lme4
p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = ones(p_fixed)
Σtrue = [0.1]
rtrue = 10.0

# #simulation parameters
samplesize = 10000
ni = 5 # number of observations per individual

Γ = Σtrue[1] * ones(ni, ni)

d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, samplesize)

# a = collect(1:samplesize)
# group = [repeat([a[i]], ni) for i in 1:samplesize]
# groupstack = vcat(group...)
# Xstack = []
# Ystack = []
# df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
for i in 1:samplesize
    Random.seed!(1234 * i)
    X = [ones(ni) randn(ni, p_fixed - 1)]
    η = X * βtrue
    μ = exp.(η)
    p = rtrue ./ (μ .+ rtrue)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    Random.seed!(1234 * i)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
    # push!(Xstack, X)
    # push!(Ystack, y)
end
gcm = NBCopulaVCModel(gcs);
fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter=100 , tol = 1e-7)
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.r
@show gcm.θ
@show gcm.∇θ
function get_CI(gcm)
    loglikelihood!(gcm, true, true)
    # @show gcm.θ
    # @show gcm.∇θ
    vcov!(gcm)
    GLMCopula.confint(gcm)
end

# Xstack = [vcat(Xstack...)][1]
# Ystack = [vcat(Ystack...)][1]
# # p = 3
# df = DataFrame(Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))

# using CSV
# CSV.write(file = "")

# simulation 2: Under GLMM
using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
using LinearAlgebra: BlasReal, copytri!

import StatsBase: sem
import GLMCopula.initialize_model!

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end

function __get_distribution(dist::Type{D}, μ, r) where D <: UnivariateDistribution
    return dist(r, μ)
end

function initialize_model!(
    gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}

    # initial guess for r = 1
    fill!(gcm.r, 1)
    fill!(gcm.τ, 1)
    fill!(gcm.β, 0)

    println("initializing variance components using MM-Algorithm")
    fill!(gcm.Σ, 1)
    update_Σ!(gcm)
    nothing
end

p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(1234)
βtrue = rand(Uniform(-0.2, 0.2), p_fixed)
Σtrue = [0.01]
rtrue = 10.0

# #simulation parameters
samplesize = 10000
ni = 5 # number of observations per individual

d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = Σtrue[1] * ones(ni, ni) + 0.00000000000001 * Matrix(I, ni, ni)
a = collect(1:samplesize)
group = [repeat([a[i]], ni) for i in 1:samplesize]
groupstack = vcat(group...)
Xstack = []
Ystack = []
for i in 1:samplesize
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    X = [ones(ni) randn(ni, p_fixed - 1)]
    η = X * βtrue
    V = [ones(ni, ni)]
    # generate mvn response
    mvn_d = MvNormal(η, Γ)
    mvn_η = rand(mvn_d)
    μ = GLM.linkinv.(link, mvn_η)
    p_nb = rtrue ./ (μ .+ rtrue)
    y = Float64.(rand.(__get_distribution.(D, p_nb, rtrue)))
    # add to data
    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
    push!(Xstack, X)
    push!(Ystack, y)
end

# form NBCopulaVCModel
gcm = NBCopulaVCModel(gcs);
fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter=100 , tol = 1e-7)
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.r
@show gcm.θ
@show gcm.∇θ
function get_CI(gcm)
    loglikelihood!(gcm, true, true)
    # @show gcm.θ
    # @show gcm.∇θ
    vcov!(gcm)
    GLMCopula.confint(gcm)
end
get_CI(gcm)

# form glmm
Xstack = [vcat(Xstack...)][1]
Ystack = [vcat(Ystack...)][1]
# p = 3
df = DataFrame(Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))

using CSV
CSV.write("N10k_ni5_NB.csv", df)

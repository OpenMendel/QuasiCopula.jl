using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

# simulate samplesize = 10,000 vectors of NB base distribution to save to csv and analyze with R package lme4
p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = ones(p_fixed)
Σtrue = [0.5]
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

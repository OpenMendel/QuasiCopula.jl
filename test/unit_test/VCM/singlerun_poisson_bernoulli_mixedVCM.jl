using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = [0.2; 0.1; -0.05; 0.2; 0.1; -0.1]
Σtrue = [0.1]

# generate data
intervals = zeros(2 * p + m, 2) #hold intervals
curcoverage = zeros(2 * p + m) #hold current coverage resutls
trueparams = [βtrue; Σtrue] #hold true parameters

#simulation parameters
samplesize = 10000
ni = 2

currentind = 1
d1 = Poisson()
d2 = Bernoulli()
vecdist = [d1, d2]

link1 = LogLink()
link2 = LogitLink()
veclink = [link1, link2]

T = Float64
VD = typeof(vecdist)
VL = typeof(veclink)

Random.seed!(12345)
Xstack = [ones(samplesize) randn(samplesize, p - 1)]
gcs = Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}(undef, samplesize)
Γ = Σtrue[1] * ones(ni, ni)
vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)

for i in 1:samplesize
    xi = Xstack[i, :]
    X = [transpose(xi) zeros(size(transpose(xi))); zeros(size(transpose(xi))) transpose(xi)]
    η = X * βtrue
    μ = zeros(ni)
    for j in 1:ni
        μ[j] = GLM.linkinv(veclink[j], η[j])
        vecd[j] = typeof(vecdist[j])(μ[j])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = Poisson_Bernoulli_VCObs(y, xi, V, vecdist, veclink)
end

gcm = Poisson_Bernoulli_VCModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", accept_after_max_steps = 5, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ

using Test
gcm.β ≈ [0.1860209299952096, 0.10317203848536331, -0.05217158982432471, 0.17357582848018224, 0.09003965923131003, -0.13325170880834192]
gcm.Σ ≈ [0.09854548500696161]

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
# coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
@show mseβ
@show mseΣ
@test mseβ ≈ 0.00035221889935475105
@test mseΣ ≈ 2.1156138649734725e-6

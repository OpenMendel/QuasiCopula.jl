using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = ones(p)
Σtrue = [0.1]

# generate data
intervals = zeros(p + m + 1, 2) # hold intervals
curcoverage = zeros(p + m + 1) # hold current coverage resutls
trueparams = [βtrue; Σtrue] # hold true parameters

# #simulation parameters
samplesize = 10000
clustersize = 5
samplesizes = [samplesize]
ns = [clustersize]

k = 2; j = 1; # for random seed
T = Float64
gcs = Vector{GaussianCopulaVCObs{T}}(undef, samplesize)

ni = ns[1] # number of observations per individual
y = Vector{Float64}(undef, ni)
res = Vector{Float64}(undef, ni)
β = ones(p)
Random.seed!(1234 * k)
X = [ones(ni) randn(ni, p - 1)]
μ = X * β
σ2 = 0.1
vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
for i in 1:length(μ)
    vecd[i] = Normal(μ[i], σ2)
end

Γ = Σtrue[1] * ones(ni, ni)
nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
Random.seed!(1234 * k + j)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, samplesize)
Ystack = vcat(Y_nsample...)
@show length(Ystack)
a = collect(1:samplesize)
group = [repeat([a[i]], ni) for i in 1:samplesize]
groupstack = vcat(group...)
Xstack = repeat(X, samplesize)
# for p = 3
df = (Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
form = @formula(Y ~ 1 + X2 + X3 + (1|group));

# # for p = 2
# df = (Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], group = string.(groupstack))
# form = @formula(Y ~ 1 + X2 + (1|group));
T = Float64
gcs = Vector{GaussianCopulaVCObs{T}}(undef, samplesize)
for i in 1:samplesize
    y = Float64.(Y_nsample[i])
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V)
end

# form VarLmmModel
gcm = GaussianCopulaVCModel(gcs);
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-4, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.τ
# @show gcm.θ
# @show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
mseβ, mseτ, mseΣ = MSE(gcm, βtrue, 0.1, Σtrue)
@show mseβ
@show mseτ
@show mseΣ

@info "Fit with MixedModels..."
fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Normal(), contrasts = Dict(:group => Grouping()))
@show fittime_GLMM
display(gm1)
@show gm1.β
# mse and time under glmm
@info "Get MSE under GLMM..."
level = 0.95
p = 3
@show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
@show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, gm1.σ .- 0.1), sum(abs2, (gm1.σs[1][1]^2) .- Σtrue[1]) / 1]

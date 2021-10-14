using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = ones(p)
Σtrue = [0.5]

#simulation parameters
samplesize = 10000

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)
ni = 2 # number of observations per individual
β = ones(p)
Γ = Σtrue[1] * ones(ni, ni)

gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)
for i in 1:samplesize
    Random.seed!(1234 * i * j * k)
    X = [ones(ni) randn(ni, p - 1)]
    η = X * β
    μ = exp.(η)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Poisson(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    Random.seed!(1234 * i * j * k)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end

# form VarLmmModel
gcm = GLMCopulaVCModel(gcs);
gcm2 = deepcopy(gcm);
gcm3 = deepcopy(gcm);
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-4, limited_memory_max_history = 18, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
sandwich!(gcm)
@show GLMCopula.confint(gcm)



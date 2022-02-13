using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-0.2, 0.2), p)
# βtrue = [log(5.0)]
σ2true = [0.5]
ρtrue = [0.5]
trueparams = [βtrue; ρtrue; σ2true]

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
samplesize = 50000

st = time()
currentind = 1
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, samplesize)

ni = 5# number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
# for i in 1:ni
#     vecd[i] = Poisson(5.0)
# end
# nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
#
# Random.seed!(1234)
# @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, samplesize)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    μ = exp.(η)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Poisson(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

# form model
gcm = GLMCopulaARModel(gcs);

# 
# N = length(gcm.data)
# di = length(gcm.data[1].y)
# Y = zeros(N, di)
# for j in 1:di
#     Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
# end
# # empirical_covariance_mat = scattermat(Y) ./ N
# StatsBase.cor(Y)
#
# initialize_model!(gcm)
# @show gcm.β
# @show gcm.ρ
# @show gcm.σ2

loglikelihood!(gcm, true, true)

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, accept_after_max_steps = 3, hessian_approximation = "limited-memory"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

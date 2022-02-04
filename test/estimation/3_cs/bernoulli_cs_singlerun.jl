using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(1234)
# βtrue = 0.1 * ones(p)
# βtrue = rand(Uniform(-0.2, 0.2), p)
# βtrue = 0.1 * ones(p)
σ2true = [0.01]
ρtrue = [0.5]

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end

#simulation parameters
samplesize = 10000

st = time()
currentind = 1
# d = Poisson()
d = Bernoulli()
# link = LogLink()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 5#  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# p = exp(βtrue[1])/(1 + exp(βtrue[1]))
#
# vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
# for i in 1:ni
#     vecd[i] = Bernoulli(p)
# end
# nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
#
# Random.seed!(1234)
# @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, samplesize)
#

for i in 1:samplesize
    X = [ones(ni) randn(ni, p - 1)]
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    η = X * βtrue
    μ = exp.(η) ./ (1 .+ exp.(η))
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Bernoulli(μ[i])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
#     # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # push!(Ystack, y)
    V = [Float64.(Matrix(I, ni, ni))]
    # V = [ones(ni, ni)]
    gcs[i] = GLMCopulaCSObs(y, X, d, link)
end

# form model
gcm = GLMCopulaCSModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2
# GLMCopula.initialize_beta!(gcm)
# copyto!(gcm.σ2, 0.1)
# copyto!(gcm.ρ, 0.2)

loglikelihood!(gcm, true, true)
#
gcm2 = deepcopy(gcm);
gcm3 = deepcopy(gcm);

fittime = @elapsed GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 30, derivative_test = "first-order", accept_after_max_steps = 2, hessian_approximation = "limited-memory"))

# fittime1 = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 200, tol = 10^-8, warm_start_init_point="yes", derivative_test = "first-order", mu_strategy = "adaptive", tau_min = 0.25, mu_oracle = "probing", hessian_approximation = "exact"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, accept_after_max_steps = 3, limited_memory_max_history = 20, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "limited-memory"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 10, tol = 10^-8, derivative_test = "first-order", hessian_approximation = "limited-memory"))
@show fittime
# @show gcm.θ
# @show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ
@show GLMCopula.confint(gcm)

# fittime = @elapsed GLMCopula.fit!(gcm3, IpoptSolver(print_level = 5, max_iter = 200, tol = 10^-7, limited_memory_max_history = 20, accept_after_max_steps = 5, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "exact"))
# # @show fittime
# @show gcm2.θ
# @show gcm2.∇θ
# vcov!(gcm2)
# @show GLMCopula.confint(gcm2)
# mseβ, mseρ, mseσ2 = MSE(gcm2, βtrue, ρtrue, σ2true)
# @show mseβ
# @show mseσ2
# @show mseρ

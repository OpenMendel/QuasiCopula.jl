using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
σ2true = [0.5]
ρtrue = [-0.05]
trueparams = [βtrue; ρtrue; σ2true]

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

ni = 25#  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[1]]
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
    # V = [Float64.(Matrix(I, ni, ni))]
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaCSObs(y, X, d, link)
end

# form model
gcm = GLMCopulaCSModel(gcs);

initialize_model!(gcm)

# GLMCopula.initialize_beta!(gcm)
# copyto!(gcm.σ2, σ2true[1])
# copyto!(gcm.ρ, ρtrue[1])
@show gcm.β
@show gcm.ρ
@show gcm.σ2

loglikelihood!(gcm, true, true)

# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 30, derivative_test = "second-order", accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, accept_after_max_steps = 5, limited_memory_max_history = 50, warm_start_init_point="yes", derivative_test = "first-order", hessian_approximation = "limited-memory"))
gcm.θ
trueparams
gcm.∇θ
# loglikelihood!(gcm, true, true)
# copyto!(gcm.σ2, σ2true[1])
# copyto!(gcm.ρ, ρtrue[1])
# loglikelihood!(gcm, true, true)
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "limited-memory"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 10, tol = 10^-8, derivative_test = "first-order", hessian_approximation = "limited-memory"))

loglikelihood!(gcm, true, true)
vcov!(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ
@show GLMCopula.confint(gcm)

####
# GLMCopula.initialize_beta!(gcm)
# N = length(gcm.data)
# di = length(gcm.data[1].y)
# Y = zeros(N, di)
# for j in 1:di
#     Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
# end
# Ycor = StatsBase.cor(Y)
# empirical_correlation_mean = mean(GLMCopula.offdiag(StatsBase.cor(Y)))
#
# update_res!(gcm)
#
# μ_ik = zeros(N)
# μ_imean = zeros(di)
#
# σ2_ik = zeros(N)
# σ2_imean = zeros(di)
#
# for k in 1:di
#     @inbounds for i in eachindex(gcm.data)
#         μ_ik[i] = mean(gcm.data[i].μ[k])
#     end
#     μ_imean[k] = mean(μ_ik)
#     @inbounds for i in eachindex(gcm.data)
#         σ2_ik[i] = mean(gcm.data[i].varμ[k])
#     end
#     σ2_imean[k] = mean(σ2_ik)
# end
#
# p_hat = zeros(di)
# for i in 1:di
#     p_hat[i] = mean(Y[:, i])
#     d = Bernoulli(p_hat[i])
#     μ, σ², sk, kt = mean(d), var(d), skewness(d), kurtosis(d, false)
#
# empirical_correlation_mean * (1 + ((di/2) * σ2true[1]) + (0.5 * (kt - 1) * σ2true[1]))

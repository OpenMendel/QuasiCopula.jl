using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
σ2true = [0.5]
ρtrue = [0.5]
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
    X = [ones(ni) X_samplesize[i]]
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

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, derivative_test = "first-order", accept_after_max_steps = 2, limited_memory_max_history = 50, warm_start_init_point="yes",  mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "limited-memory"))

# fittime2 = @elapsed GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, accept_after_max_steps = 4, limited_memory_max_history = 50, warm_start_init_point="yes", mu_strategy = "adaptive", hessian_approximation = "limited-memory"))

@show gcm.θ
@show trueparams
@show gcm.∇θ

# @test gcm.θ ≈ [0.2679016603133174, 1.3485569590033557, -0.4898578491349565, 0.5225462587214745, 0.4028766887775074]

loglikelihood!(gcm, true, true)
vcov!(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ
@show GLMCopula.confint(gcm)

using Test
@test mseβ < 0.01
@test mseσ2 < 0.1
@test mseρ < 0.01
# using Test
# @test mseβ ≈ 0.0011582605721954502
# @test mseσ2 ≈ 0.009432937582821162
# @test mseρ ≈ 0.0005083337823356664
# #
using BenchmarkTools
println("checking memory allocation for bernoulli CS")
logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)
@test logl_gradient_memory.memory == 0.0

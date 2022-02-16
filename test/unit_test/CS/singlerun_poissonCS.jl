using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
# βtrue = 0.1 * ones(p)
σ2true = [0.5]
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
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 5 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

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
    # push!(Ystack, y)
    V = [Float64.(Matrix(I, ni, ni))]
    # V = [ones(ni, ni)]
    gcs[i] = GLMCopulaCSObs(y, X, d, link)
end

# form model
gcm = GLMCopulaCSModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, accept_after_max_steps = 2, limited_memory_max_history = 20, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", derivative_test = "first-order", hessian_approximation = "limited-memory"))
loglikelihood!(gcm, true, true)
@show fittime
@show gcm.θ

# @test gcm.θ ≈ [0.2512643078372366, 1.3979789702777217, -0.5132418587872071, 0.4811172968229514, 0.504087569241636]

@show gcm.∇θ
vcov!(gcm)
@show GLMCopula.confint(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ

@test mseβ < 0.01
@test mseσ2 < 0.01
@test mseρ < 0.01

# using Test
# @test mseβ ≈ 1.1483744501546296e-6
# @test mseσ2 ≈ 1.670823046924042e-5
# @test mseρ ≈ 0.00035655646529727587

using BenchmarkTools
println("checking memory allocation for Poisson CS")
logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)
@test logl_gradient_memory.memory == 0.0

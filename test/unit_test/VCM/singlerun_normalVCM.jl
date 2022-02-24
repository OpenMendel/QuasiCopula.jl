using GLMCopula, DelimitedFiles, LinearAlgebra, Random, ToeplitzMatrices
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
Σtrue = [0.5]
τtrue = 100.0
σ2 = inv(τtrue)
σ = sqrt(σ2)

#simulation parameters
samplesize = 10000
ni = 25

# true Gamma
Γ = Σtrue[1] * ones(ni, ni)

T = Float64
gcs = Vector{GaussianCopulaVCObs{T}}(undef, samplesize)

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[i]]
    μ = X * βtrue
    vecd = Vector{ContinuousUnivariateDistribution}(undef, ni)
    for i in 1:ni
        vecd[i] = Normal(μ[i], σ)
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    gcs[i] = GaussianCopulaVCObs(y, X, V)
end

# form VarLmmModel
gcm = GaussianCopulaVCModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, limited_memory_max_history = 50, accept_after_max_steps = 4, tol = 10^-8, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.τ

loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

mseβ, mseτ, mseΣ = MSE(gcm, βtrue, τtrue[1], Σtrue)
@show mseβ
@show mseτ
@show mseΣ

using Test
@test mseβ < 0.01
@test mseΣ < 0.01
@test mseτ < 0.01

# need to optimize memory allocation 937.50 KiB
# using BenchmarkTools
# println("checking memory allocation for Normal VCM")
# logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)
# @test logl_gradient_memory.memory == 0.0

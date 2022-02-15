using GLMCopula, DelimitedFiles, LinearAlgebra, Random, ToeplitzMatrices
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-2, 2), p)
σ2true = [0.5]
ρtrue = [0.5]
τtrue = 100.0
σ2 = inv(τtrue)
σ = sqrt(σ2)

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
samplesize = 10000
ni = 5

V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

T = Float64
gcs = Vector{GaussianCopulaARObs{T}}(undef, samplesize)

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
    gcs[i] = GaussianCopulaARObs(y, X)
end

# form VarLmmModel
gcm = GaussianCopulaARModel(gcs);
initialize_model!(gcm)
@show gcm.β
@show gcm.τ
@show gcm.ρ
@show gcm.σ2

mseβ, mseρ, mseσ2, mseτ = MSE(gcm, βtrue, τtrue[1], ρtrue, σ2true)

using Test
@test mseβ < 1
@test mseτ < 1
@test mseσ2 < 1
@test mseρ < 1


# loglikelihood!(gcm, true, true)
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 300, tol = 10^-8, limited_memory_max_history = 20, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
# @show fittime
# @show gcm.θ
# @show gcm.∇θ
#
# @test gcm.θ ≈ [0.25076395120126066, 1.3994688251754344, -0.5131211706422033, 0.5245251449860135, 0.42416290525370876, 98.80166356053807]
#
# loglikelihood!(gcm, true, true)
# vcov!(gcm)
# @show GLMCopula.confint(gcm)
#
# mseβ, mseρ, mseσ2, mseτ = MSE(gcm, βtrue, τtrue[1], ρtrue, σ2true)
# @show mseβ
# @show mseτ
# @show mseσ2
# @show mseρ

# using Test
# @test mseβ ≈ 1.0039867593617186e-7
# @test mseτ ≈ 1.4710553619831918e-8
# @test mseσ2 ≈ 0.005751264939557955
# @test mseρ ≈ 0.0006014827365849855


# need to optimize wrt to memory 4.88 MIB
# using BenchmarkTools
# println("checking memory allocation for Normal AR")
# logl_gradient_memory = @benchmark loglikelihood!($gcm, true, false)

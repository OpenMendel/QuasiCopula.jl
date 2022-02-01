using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(1234)
βtrue = randn(p)
σ2true = [0.5]
ρtrue = [0.9]

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
samplesize = 1000

st = time()
currentind = 1
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 5 # number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

for i in 1:samplesize
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    X = [ones(ni) randn(ni, p - 1)]
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
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    rand(nonmixed_multivariate_dist, y, res)
    V = [Float64.(Matrix(I, ni, ni))]
    gcs[i] = GLMCopulaCSObs(y, X, d, link)
end

# form model
gcm = GLMCopulaCSModel(gcs);
fittime = NaN
initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 12, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
@show gcm.β
@show gcm.ρ
@show gcm.σ2
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

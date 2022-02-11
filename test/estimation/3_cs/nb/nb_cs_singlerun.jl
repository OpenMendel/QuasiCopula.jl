using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p_fixed = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-0.2, 0.2), p_fixed)
# βtrue = randn(p_fixed)
# βtrue = 0.1 * ones(p)
rtrue = 10.0
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
d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 5 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(1234)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[1]]
    # X = [ones(ni) randn(ni, p - 1)]
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
    η = X * βtrue
    μ = exp.(η)
    p = rtrue ./ (μ .+ rtrue)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    # push!(Ystack, y)
    V = [ones(ni, ni)]
    # V = [ones(ni, ni)]
    gcs[i] = NBCopulaCSObs(y, X, d, link)
end

# form model
gcm = NBCopulaCSModel(gcs);

# initialize_model!(gcm)
# @show gcm.β
# @show gcm.ρ
# @show gcm.σ2

# loglikelihood!(gcm, true, true)
#
# gcm2 = deepcopy(gcm);
# gcm3 = deepcopy(gcm);
# gcm4 = deepcopy(gcm);
fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter = 30, tol=1e-6)
@show fittime
@show gcm.θ
@show gcm.∇θ
@show gcm.r
@show gcm.∇r
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

# mse and time under our model
# coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseρ, mseσ2, mser = MSE(gcm, βtrue, ρtrue, σ2true, rtrue)
@show mseβ
@show mser
@show mseσ2
@show mseρ
#
# fittime = @elapsed GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 200, tol = 10^-7, accept_after_max_steps = 3, warm_start_init_point="yes", mu_strategy = "adaptive", hessian_approximation_space = "all-variables", mu_oracle = "probing", hessian_approximation = "exact"))
# @show fittime
# @show gcm2.θ
# @show gcm2.∇θ
# loglikelihood!(gcm2, true, true)
# vcov!(gcm2)
# @show GLMCopula.confint(gcm2)
# mseβ, mseρ, mseσ2 = MSE(gcm2, βtrue, ρtrue, σ2true)
# @show mseβ
# @show mseσ2
# @show mseρ

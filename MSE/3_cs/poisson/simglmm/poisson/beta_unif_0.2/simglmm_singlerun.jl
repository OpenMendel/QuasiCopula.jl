using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-0.2, 0.2), p)
# βtrue = 0.1 * ones(p)
σ2true = [1.0]
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

a = collect(1:samplesize)
group = [repeat([a[i]], ni) for i in 1:samplesize]
groupstack = vcat(group...)
Xstack = []
Ystack = []
for i in 1:samplesize
  X = [ones(ni) randn(ni, p - 1)]
  η = X * βtrue
  V = [ones(ni, ni)]
  # generate mvn response
  mvn_d = MvNormal(η, Γ)
  mvn_η = rand(mvn_d)
  μ = GLM.linkinv.(link, mvn_η)
  y = Float64.(rand.(__get_distribution.(D, μ)))
  push!(Xstack, X)
  push!(Ystack, y)
  # add to data
  gcs[i] = GLMCopulaCSObs(y, X, d, link)
end
# form model
gcm = GLMCopulaCSModel(gcs);

# initialize_model!(gcm)
GLMCopula.initialize_beta!(gcm)
copyto!(gcm.ρ, ρtrue[1])
copyto!(gcm.σ2, σ2true[1])
@show gcm.β
@show gcm.ρ
@show gcm.σ2

loglikelihood!(gcm, true, true)
#
gcm2 = deepcopy(gcm);
gcm3 = deepcopy(gcm);
gcm4 = deepcopy(gcm);
#
# # fittime = @elapsed GLMCopula.fit!(gcm3, IpoptSolver(print_level = 5, max_iter = 200, tol = 10^-7, accept_after_max_steps = 3, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "exact"))
# fittime1 = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "limited-memory"))
# # @show gcm.β
# # @show gcm.ρ
# # @show gcm.σ2
# @show fittime1
# @show gcm.θ
# @show gcm.∇θ
# loglikelihood!(gcm, true, true)
# vcov!(gcm)
# @show GLMCopula.confint(gcm)
# mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
# @show mseβ
# @show mseσ2
# @show mseρ

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, accept_after_max_steps = 2, limited_memory_max_history = 20, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", derivative_test = "first-order", hessian_approximation = "limited-memory"))
loglikelihood!(gcm, true, true)
@show fittime
@show gcm.θ
@show gcm.∇θ
vcov!(gcm)
@show GLMCopula.confint(gcm)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
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

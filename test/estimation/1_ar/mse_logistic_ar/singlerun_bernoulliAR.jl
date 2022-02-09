using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, StatsFuns, Distributions, DataFrames, ToeplitzMatrices

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
# try next
βtrue = rand(Uniform(-2, 2), p)
σ2true = [0.5]
ρtrue = [0.5]

function get_V(ρ, n)
  vec = zeros(n)
  vec[1] = 1.0
  for i in 2:n
      vec[i] = vec[i - 1] * ρ
  end
  V = ToeplitzMatrices.SymmetricToeplitz(vec)
  V
end


trueparams = [βtrue; ρtrue; σ2true] #hold true parameters

#simulation parameters
samplesize = 100
ni = 5


st = time()
currentind = 1
d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, samplesize)

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
  # simuate single vector y
  y = Vector{Float64}(undef, ni)
  res = Vector{Float64}(undef, ni)
  rand(nonmixed_multivariate_dist, y, res)
  V = [ones(ni, ni)]
  gcs[i] = GLMCopulaARObs(y, X, d, link)
end

# form model
gcm = GLMCopulaARModel(gcs);
fittime = NaN
initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, warm_start_init_point="yes", accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

# mse and time under our model
coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ

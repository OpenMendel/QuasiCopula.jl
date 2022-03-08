using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, StatsFuns, Distributions, DataFrames, ToeplitzMatrices
BLAS.set_num_threads(1)
Threads.nthreads()

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
samplesize = 10000
ni = 25

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
gcm = GLMCopulaARModel(gcs)
# precompile
println("precompiling Bernoulli AR fit")
gcm2 = deepcopy(gcm);
GLMCopula.fit!(gcm2, IpoptSolver(print_level = 0, max_iter = 20));

fittime = @elapsed GLMCopula.fit!(gcm)
@show fittime
@show gcm.β
@show gcm.σ2
@show gcm.ρ
@show gcm.∇β
@show gcm.∇σ2
@show gcm.∇ρ

@test logl(gcm) == loglikelihood!(gcm, false, false)
@show get_CI(gcm)

# mse and time under our model
mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
@show mseβ
@show mseσ2
@show mseρ

using Test
@test mseβ < 0.01
@test mseσ2 < 1
@test mseρ < 0.01

using BenchmarkTools
println("checking memory allocation for Bernoulli AR")
logl_gradient_memory = @benchmark loglikelihood!($gcm.data[1], $gcm.β, $gcm.ρ[1], $gcm.σ2[1], true, false)
@test logl_gradient_memory.memory == 0.0

# note for multi-threading we will allocate some at the model level loglikelihood!(gcm, true, false)

using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, StatsBase
import StatsBase: sem

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-0.2, 0.2), p)
Σtrue = [0.25]

# generate data
intervals = zeros(p + m, 2) #hold intervals
curcoverage = zeros(p + m) #hold current coverage resutls
trueparams = [βtrue; Σtrue] #hold true parameters

#simulation parameters
# ns = [2; 5; 10; 15; 20; 25]
samplesize = 10000
ni = 25

st = time()
currentind = 1
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = Σtrue[1] * ones(ni, ni) + 0.000000000000001 * Matrix(I, ni, ni)

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
  gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end

# form VarLmmModel
gcm = GLMCopulaVCModel(gcs)

N = length(gcm.data)
di = length(gcm.data[1].y)
Y = zeros(N, di)
for j in 1:di
    Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
end
corY = StatsBase.cor(Y)
empirical_correlation_mean = mean(GLMCopula.offdiag(corY))
initialize_model!(gcm)
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory"))
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
@show mseβ
@show mseΣ

@info "Fit with MixedModels..."
# form glmm
Xstack = [vcat(Xstack...)][1]
Ystack = [vcat(Ystack...)][1]
# p = 3
df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
form = @formula(Y ~ 1 + X2 + X3 + (1|group));

# fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(); nAGQ = 25)
fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Poisson(), contrasts = Dict(:group => Grouping()); nAGQ = 25)
@show fittime_GLMM
display(gm1)
@show gm1.β
# mse and time under glmm

@info "Get MSE under GLMM..."
level = 0.95
p = 3
@show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
@show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]

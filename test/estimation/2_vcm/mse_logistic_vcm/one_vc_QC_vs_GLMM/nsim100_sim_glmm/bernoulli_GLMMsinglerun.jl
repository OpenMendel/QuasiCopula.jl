using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end
p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(12345)
# try next
βtrue = rand(Uniform(-0.2, 0.2), p)
Σtrue = [0.25]
trueparams = [βtrue; Σtrue] #hold true parameters

d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64

samplesize = 10000
ni = 25

gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)
Γ = Σtrue[1] * ones(ni, ni) + 0.00000000001 * Matrix(I, ni, ni)

a = collect(1:samplesize)
group = [repeat([a[i]], ni) for i in 1:samplesize]
groupstack = vcat(group...)
Xstack = []
Ystack = []
for i in 1:samplesize
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    X = [ones(ni) randn(ni, p - 1)]
    η = X * βtrue
    V = [ones(ni, ni)]
    # generate mvn response
    mvn_d = MvNormal(η, Γ)
    mvn_η = rand(mvn_d)
    μ = GLM.linkinv.(link, mvn_η)
    y = Float64.(rand.(__get_distribution.(D, μ)))
    # add to data
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
    push!(Xstack, X)
    push!(Ystack, y)
end

# form VarLmmModel
gcm = GLMCopulaVCModel(gcs);


N = length(gcm.data)
di = length(gcm.data[1].y)
Y = zeros(N, di)
for j in 1:di
    Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
end
corY = StatsBase.cor(Y)
empirical_correlation_mean = mean(GLMCopula.offdiag(corY))
initialize_model!(gcm)

# form glmm
Xstack = [vcat(Xstack...)][1]
Ystack = [vcat(Ystack...)][1]
# p = 3
df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
form = @formula(Y ~ 1 + X2 + X3 + (1|group));

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, accept_after_max_steps = 1, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
@show mseβ
@show mseΣ

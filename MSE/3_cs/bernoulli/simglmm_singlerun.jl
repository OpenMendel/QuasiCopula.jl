using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end
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

d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64


gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 2 #  number of observations per individual
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
@show gcm.β
copyto!(gcm.ρ, ρtrue[1])
copyto!(gcm.σ2, σ2true[1])
@show gcm.ρ
@show gcm.σ2

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

# # for GLMM bernoulli
# function initialize_model!(
#     gcm::Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}}) where {T <: BlasReal, D<:Bernoulli, Link,  VD, VL}
#     println("initializing β using Newton's Algorithm under Independence Assumption")
#     initialize_beta!(gcm)
#     @show gcm.β
#     fill!(gcm.τ, 1.0)
#     println("initializing variance components using MM-Algorithm")
#     N = length(gcm.data)
#     di = length(gcm.data[1].y)
#     Y = zeros(N, di)
#     for j in 1:di
#         Y[:, j] = [gcm.data[i].y[j] for i in 1:N]
#     end
#     corY = StatsBase.cor(Y)
#     empirical_correlation_mean = mean(GLMCopula.offdiag(corY))
#     copyto!(gcm.Σ, empirical_correlation_mean/var(Y))
#     @show gcm.Σ
#     nothing
# end

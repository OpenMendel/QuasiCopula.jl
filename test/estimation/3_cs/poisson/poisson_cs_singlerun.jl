using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, LoopVectorization
using DataFrames, DelimitedFiles, Statistics, ToeplitzMatrices
import StatsBase: sem

p = 3    # number of fixed effects, including intercept

# true parameter values
Random.seed!(12345)
βtrue = rand(Uniform(-0.2, 0.2), p)
# βtrue = 0.1 * ones(p)
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
d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, samplesize)

ni = 25 #  number of observations per individual
V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

# for reproducibility I will simulate all the design matrices here
Random.seed!(12345)
X_samplesize = [randn(ni, p - 1) for i in 1:samplesize]

for i in 1:samplesize
    X = [ones(ni) X_samplesize[1]]
    # X = [ones(ni) randn(ni, p - 1)]
    # X = ones(ni, 1)
    # y = Float64.(Y_nsample[i])
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
    rand(nonmixed_multivariate_dist, y, res)
    # push!(Ystack, y)
    V = [Float64.(Matrix(I, ni, ni))]
    # V = [ones(ni, ni)]
    gcs[i] = GLMCopulaCSObs(y, X, d, link)
end

# form model
gcm = GLMCopulaCSModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2

gc = gcm.data[1];

loglikelihood!(gcm, true, true)


using BenchmarkTools
# @benchmark update_res!($gc, $gcm.β)
# @benchmark update_res2!($gc, $gcm.β)
#
# function std_res_differential2!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
#     @inbounds for i in 1:size(gc.X, 2)
#         @turbo for j in 1:length(gc.y)
#             gc.∇resβ[j, i] = gc.X[j, i]
#             gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
#         end
#     end
#     nothing
# end
#
# @benchmark std_res_differential!($gc)
# @benchmark std_res_differential2!($gc)

@benchmark loglikelihood!($gcm, true, false)
# revise()
# @benchmark loglikelihood!($gcm, true, true)


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

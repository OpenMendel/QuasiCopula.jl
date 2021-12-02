using GLMCopula, DelimitedFiles, LinearAlgebra, Random, ToeplitzMatrices
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

p = 3   # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = ones(p)
σ2true = [0.1]
ρtrue = [0.9]

#simulation parameters
samplesize = 10000
clustersize = 5
samplesizes = [samplesize]
ns = [clustersize]

k = 2; j = 1; # for random seed
T = Float64
gcs = Vector{GaussianCopulaARObs{T}}(undef, samplesize)

ni = ns[1] # number of observations per individual
y = Vector{Float64}(undef, ni)
res = Vector{Float64}(undef, ni)
β = ones(p)
Random.seed!(1234 * k)
X = [ones(ni) randn(ni, p - 1)]
μ = X * β
τtrue = 10
σ2 = inv(τtrue)
vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
for i in 1:length(μ)
    vecd[i] = Normal(μ[i], σ2)
end

function get_V(ρ, n)
        vec = zeros(n)
        vec[1] = 1.0
        for i in 2:n
            vec[i] = vec[i - 1] * ρ
        end
        V = ToeplitzMatrices.SymmetricToeplitz(vec)
        V
    end


V = get_V(ρtrue[1], ni)

# true Gamma
Γ = σ2true[1] * V

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
Random.seed!(1234 * k + j)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, samplesize)

T = Float64
gcs = Vector{GaussianCopulaARObs{T}}(undef, samplesize)
for i in 1:samplesize
    y = Float64.(Y_nsample[i])
    gcs[i] = GaussianCopulaARObs(y, X)
end

# form VarLmmModel
gcm = GaussianCopulaARModel(gcs);
initialize_model!(gcm)
@show gcm.β
@show gcm.τ
@show gcm.ρ
@show gcm.σ2
loglikelihood!(gcm, true, true)

gcm2 = deepcopy(gcm);
fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order", max_iter = 200, tol = 10^-4, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

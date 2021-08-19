using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM
using BenchmarkTools, Profile, Distributions

Random.seed!(1234)
n = 50
variance_component_1 = 0.1
variance_component_2 = 0.1
Γ = variance_component_1 * ones(n, n) + variance_component_2 * Matrix(I, n, n)

dist = Poisson
p = 3
β = ones(p)
X = [ones(n) randn(n, p - 1)]
η = X * β
μ = exp.(η)
vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Poisson(μ[i])
end

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
nsample = 10_000
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    V = [ones(n, n), Matrix(I, n, n)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs)
initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
loglikelihood!(gcm, true, true)
# use quasi-newton
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, hessian_approximation = "limited-memory"))
println("estimated beta = $(gcm.β); true beta value= $β")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2");

@info "get standard errors"

sandwich!(gcm)
@show GLMCopula.confint(gcm)
@show gcm.θ

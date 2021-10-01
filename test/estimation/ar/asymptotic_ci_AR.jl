using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM
using BenchmarkTools, Profile, Distributions, ToeplitzMatrices
using LinearAlgebra: BlasReal, copytri!

Random.seed!(123498523948230492456)

# sample size
N = 10000
# observations per subject
n = 50
ρ = 0.9
σ2 = 0.1

p = 3
β = ones(p)
X = [ones(n) randn(n, p - 1)]
η = X * β
μ = exp.(η) 
vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Poisson(μ[i])
end

function get_V(ρ, n)
    vec = zeros(n)
    vec[1] = 1.0
    for i in 2:n
        vec[i] = vec[i-1] * ρ
    end
    V = ToeplitzMatrices.SymmetricToeplitz(vec)
    V
end
# V = get_AR_cov(n, ρ, σ2, V)
V = get_V(ρ, n)

# true Gamma
Γ = σ2 * V

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

Y_Nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, N)
Random.seed!(1234)

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, N)

for i in 1:N
    y = Float64.(Y_Nsample[i])
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

gcm = GLMCopulaARModel(gcs);
initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2

### now sigma2 is initialized now we need rho
Y_1 = [Y_Nsample[i][1] for i in 1:N]
Y_2 = [Y_Nsample[i][2] for i in 1:N]

update_rho!(gcm, Y_1, Y_2)
@show gcm.ρ
@show gcm.σ2

loglikelihood!(gcm, true, true)

gcm2 = deepcopy(gcm);
# gcm3 = deepcopy(gcm);
# gcm4 = deepcopy(gcm);
# gcm5 = deepcopy(gcm);
# use quasi-newton

### Quasi-Newton ### 
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
println("estimated beta = $(gcm.β); true beta value= $β")
println("estimated AR rho = $(gcm.ρ[1]); true AR rho = $ρ")
println("estimated AR variance = $(gcm.σ2[1]); true AR variance = $σ2");

@time GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 25, hessian_approximation = "limited-memory"))
println("estimated beta = $(gcm2.β); true beta value= $β")
println("estimated AR rho = $(gcm2.ρ[1]); true AR rho = $ρ")
println("estimated AR variance = $(gcm2.σ2[1]); true AR variance = $σ2");

@info "get standard errors"

sandwich!(gcm)
@show GLMCopula.confint(gcm)
@show gcm.θ

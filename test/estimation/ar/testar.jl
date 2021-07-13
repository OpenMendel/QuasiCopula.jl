using DataFrames, Random, GLM, GLMCopula, Test, ToeplitzMatrices
using LinearAlgebra, BenchmarkTools

using LinearAlgebra: BlasReal, copytri!

Random.seed!(1234)

# sample size
N = 10000
# observations per subject
n = 50
ρ = 0.9
σ2 = 0.1

V = zeros(n, n) # will store the AR(1) structure without sigma2

Poisson_mean = 5

dist = Poisson

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

vecd = [dist(Poisson_mean) for i in 1:n]
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
    X = ones(n, 1)
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end

gcm = GLMCopulaARModel(gcs);
initialize_model!(gcm)
@show gcm.β
@show exp.(gcm.β)
@show gcm.ρ
@show gcm.σ2

### now sigma2 is initialized now we need rho
Y_1 = [Y_Nsample[i][1] for i in 1:N]
Y_2 = [Y_Nsample[i][2] for i in 1:N]

update_rho!(gcm, Y_1, Y_2)
@show exp.(gcm.β)
@show gcm.ρ
@show gcm.σ2

loglikelihood!(gcm, true, true)

gcm1 = deepcopy(gcm);
gcm2 = deepcopy(gcm);
gcm3 = deepcopy(gcm);
gcm4 = deepcopy(gcm);
gcm5 = deepcopy(gcm);

### Quasi-Newton ### 
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, hessian_approximation = "limited-memory"))
# @time GLMCopula.fit!(gcm1, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive", hessian_approximation = "limited-memory"))
# this one is fastest for quasi-newton
@time GLMCopula.fit!(gcm2, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive",  mu_oracle = "loqo", hessian_approximation = "limited-memory"))

### Newton ### 
# @time GLMCopula.fit!(gcm3, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, hessian_approximation = "exact"))
# this one is fastest for newton
@time GLMCopula.fit!(gcm4, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive", hessian_approximation = "exact"))
# @time GLMCopula.fit!(gcm5, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive",  mu_oracle = "loqo", hessian_approximation = "exact"))

# @show gcm.∇θ
@show gcm2.∇θ
# @show gcm3.∇θ
@show gcm4.∇θ
# @show gcm5.∇θ
# @show gcm.θ
@show gcm2.θ
# @show gcm3.θ
@show gcm4.θ
# @show gcm5.θ

using BenchmarkTools
@benchmark loglikelihood!($gcm2, true, true)
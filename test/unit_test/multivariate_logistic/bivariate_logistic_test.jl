using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM

@testset "Generate 10,000 independent bivariate poisson vectors and then fit the model to test for the correct variance component and mean. " begin
Random.seed!(12345)
n = 2
variance_component_1 = 0.2
variance_component_2 = 0.8
Γ = variance_component_1 * ones(2, 2) + variance_component_2 * [1.0 0.0; 0.0 1.0]

mean_1 = 0.3
d1 = Bernoulli(mean_1)
d2 = Bernoulli(mean_1)
vecd = [d1, d2]
nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(nonmixed_multivariate_dist, Y, res)

#### 
function simulate_nobs_independent_vectors(
    multivariate_distribution::Union{NonMixedMultivariateDistribution, MultivariateMix},
    n_obs::Integer)
    dimension = length(multivariate_distribution.vecd)
    Y = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    res = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    for i in 1:n_obs
        rand(multivariate_distribution, Y[i], res[i])
    end
    Y
end

nsample = 10_000
@info "sample $nsample independent vectors for the bivariate Poisson distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(12345)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

####
dim = 2
d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    X = ones(dim, 1)
    V = [ones(2, 2), [1.0 0.0; 0.0 1.0]]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);
initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
@test GLMCopula.loglikelihood!(gcm, true, true) ≈ -13440.070462955528

@test gcm.∇β ≈ [-86.55491199152868]
# with the extra hessian term
@test gcm.Hβ ≈ [-4382.156901109999]

# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order", mehrotra_algorithm ="yes", warm_start_init_point="yes", max_iter = 200, hessian_approximation = "exact"))

@test GLMCopula.loglikelihood!(gcm, true, true) ≈ -13439.098083194624

println("estimated mean = $(exp(gcm.β[1]) / (1 + exp(gcm.β[1]))); true mean value= $mean_1")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2")
println("gradient with respect to θ = $(gcm.∇θ)")
end
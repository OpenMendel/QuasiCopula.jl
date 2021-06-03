using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM

@testset "Generate 10,000 independent bivariate poisson vectors and then fit the model to test for the correct variance component and mean. " begin
Random.seed!(1234)
n = 50
variance_component_1 = 0.1
variance_component_2 = 0.1
Γ = variance_component_1 * ones(n, n) + variance_component_2 * Matrix(I, n, n)

mean = 0.3
dist = Bernoulli
vecd = [dist(mean) for i in 1:n]
    
nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

nsample = 10_000
@info "sample $nsample independent vectors for the bivariate Poisson distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(1234)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

####
d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    X = ones(n, 1)
    V = [ones(n, n), Matrix(I, n, n)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
@test GLMCopula.loglikelihood!(gcm, true, true) ≈ -307479.5042351658
@test gcm.∇β ≈ [-1362.849972613486]

# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, mehrotra_algorithm ="yes", warm_start_init_point="yes", max_iter = 200, hessian_approximation = "exact"))
@test GLMCopula.loglikelihood!(gcm, true, true) ≈ -307460.961986127
println("estimated mean = $(exp(gcm.β[1]) / (1 + exp(gcm.β[1]))); true mean value= $mean")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2")
println("gradient with respect to θ = $(gcm.∇θ)")
end
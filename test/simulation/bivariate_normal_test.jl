using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM

@testset "Generate 10,000 independent bivariate normal vectors and then fit the model to test for the correct random intercepts and mean. " begin
Random.seed!(12345)

variance_component_1 = 0.2
variance_component_2 = 0.8
Γ = variance_component_1 * ones(2, 2) + variance_component_2 * [1.0 0.0; 0.0 1.0]

mean_normal = 5
sd_normal = 0.5
d1 = Normal(mean_normal, sd_normal)
d2 = Normal(mean_normal, sd_normal)
vecd = [d1, d2]
nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

n = 2
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

nsample = 10000
@info "sample $nsample independent vectors for the bivariate Poisson distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(12345)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

####
dim = 2
d = Normal()
link = IdentityLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    X = ones(dim, 1)
    V = [ones(dim, dim), Matrix(I, dim, dim)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)
initial_logl = GLMCopula.loglikelihood!(gcm, true, true)
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))

post_fit_logl = GLMCopula.loglikelihood!(gcm, true, true) 
@test initial_logl < post_fit_logl

println("estimated mean = $(gcm.β[1]); true mean value= $mean_normal")
println("estimated variance (noise) = $(inv.(gcm.τ[1])); true variance value = $(sd_normal^2)")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 1 = $(gcm.Σ[2]); true variance component 1 = $variance_component_2")
println("gradient with respect to β = $(gcm.∇β)")

end

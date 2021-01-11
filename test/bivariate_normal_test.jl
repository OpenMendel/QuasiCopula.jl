using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

@testset "Generate 10,000 independent bivariate normal vectors and then fit the model to test for the correct random intercepts and mean. " begin
Random.seed!(12345)

variance_component_1 = 0.1
variance_component_2 = 0.5
Γ = variance_component_1 * [1.0 1.0; 1.0 1.0] + variance_component_2 * Matrix(I, 2, 2)
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

nsample = 10_000
@info "sample $nsample independent vectors for the bivariate Poisson distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(1234)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
# 0.083205 seconds (590.00 k allocations: 25.940 MiB)

####
dim = 2
p, m = 1, 1
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    X = ones(dim, 1)
    V = [ones(2, 2), [1.0 0.0; 0.0 1.0]]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)
GLMCopula.loglikelihood!(gcm, true, true)
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
GLMCopula.loglikelihood!(gcm, true, true)


println("estimated mean = $(gcm.β[1]); true mean value= $mean_normal")
println("estimated variance (noise) = $(inv.(gcm.τ[1])); true variance value = $(sd_normal^2)")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2")
# estimated mean = 5.001810462046591; true mean value= 5
# estimated variance (noise) = 0.24975326265954786; true variance value = 0.25
# estimated random intercept = 0.1886303755997991; true random intercept = 0.2
end
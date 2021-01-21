using CSV
using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns


@testset "Generate 10 independent multivariate normal vectors and then fit the model to test for the correct random intercepts and mean. " begin
Random.seed!(12345)
n = 400


df = CSV.read("GMMAT_example_pheno.csv")
kinship = CSV.read("GMMAT_example_pheno_GRM.csv")


variance_component_1 = 0.8
Γ = variance_component_1 * Matrix(kinship[:, 2:end])

cov1 = Float64.(df[:, :age])
cov2 = Float64.(df[:, :sex])
X = [ones(400, 1) cov1 cov2]
β = [1.2; 0.02; 0.05]
η = X * β
μ = exp.(η)
noise_sd = 0.05
vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Poisson(μ[i])
end

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(nonmixed_multivariate_dist, Y, res)

@time rand(nonmixed_multivariate_dist, Y, res)

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

nsample = 1
@info "sample $nsample independent vectors for the mvn distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(12345)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

####

dim = 400
p, m = 1, 2
d = Poisson()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    cov1 = Float64.(df[:, :age])
    cov2 = Float64.(df[:, :sex])
    X = [ones(400, 1) cov1 cov2]
    V = [Matrix(kinship[:, 2:end])]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)

GLMCopula.loglikelihood!(gcm, true, true)
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 500, hessian_approximation = "exact"))
# @time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, derivative_test = "first-order", hessian_approximation = "limited-memory"))

# check default ipopt quasi newton 
# then go back and check the hessian
GLMCopula.loglikelihood!(gcm, true, true)

println("estimated mean = $(gcm.β)); true mean value= $β")
println("estimated variance component = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
end


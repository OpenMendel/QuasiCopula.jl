module PerfTest

using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM
using BenchmarkTools, Profile

n = 50
variance_component_1 = 0.1
variance_component_2 = 0.1
Γ = variance_component_1 * ones(n, n) + variance_component_2 * Matrix(I, n, n)

mean = 5
dist = Poisson
vecd = [dist(mean) for i in 1:n]

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
nsample = 10_000
Random.seed!(1234)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

d = Poisson()
link = LogLink()
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
gcm = GLMCopulaVCModel(gcs)
initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
# @test loglikelihood!(gcm, true, true) ≈ -1.1093040785410171e6
gcm2 = deepcopy(gcm)
@time GLMCopula.fit!(gcm2, IpoptSolver(print_level = 1, max_iter = 100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", hessian_approximation = "exact"))
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, mehrotra_algorithm = "yes", warm_start_init_point = "yes", hessian_approximation = "exact"))
println("estimated mean = $(exp(gcm.β[1])); true mean value= $mean")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2");
@info "benchmarking..."
gcm.β[1] = log(5)
gcm.Σ[1] = gcm.Σ[2] = 0.1
bm = @benchmark loglikelihood!($gcm, true, true)
display(bm); println()

@info "profile..."
Profile.clear()
@profile @btime loglikelihood!($gcm, true, true)
Profile.print(format=:flat)

end
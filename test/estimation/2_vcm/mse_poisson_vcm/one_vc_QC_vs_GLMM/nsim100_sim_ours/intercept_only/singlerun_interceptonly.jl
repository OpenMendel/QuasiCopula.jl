using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

Random.seed!(1234)
n = 5
vc = 0.5
V1 = Float64.(Matrix(I, n, n))
V2 = ones(n, n)

Γ1 = vc * V1 # overdispersion
Γ2 = vc * V2 # random intercept

mean = 1
dist = Poisson
vecd = [dist(mean) for i in 1:n]

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ2)
nsample = 10_000
Random.seed!(1234)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

# Xstack = [vcat(Xstack...)][1]
Ystack = [vcat(Y_nsample...)][1]
a = collect(1:nsample)
group = [repeat([a[i]], n) for i in 1:nsample]
groupstack = vcat(group...)

df = DataFrame(Y = Ystack, group = string.(groupstack))

# using CSV
# CSV.write("N1k_ni5_Poisson_random_int.csv", df)

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    X = ones(n, 1)
    V = [V2]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs);


@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5,  max_iter = 100, tol = 10^-7, limited_memory_max_history = 10, accept_after_max_steps = 1))
println("estimated mean = $(exp.(gcm.β)[1]); true mean value= $mean")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $vc")

@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.∇β
@show gcm.∇Σ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

mseβ, mseΣ = MSE(gcm, [log(mean)], [vc])
@show mseβ
@show mseΣ

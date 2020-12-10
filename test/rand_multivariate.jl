using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns
using LinearAlgebra: BlasReal, copytri!
@testset "Trying mixtures of different discrete and cts distributions. Generate 3 element vector: Normal(0, 1), Gamma(1, 1), Exponential(10). " begin
m = 2
n = 3

# variance components
Random.seed!(12345)
# simulate desired variance components (coefficients of covariance matrices V[k] for k in 1:m)
Σ = rand(m)
Random.seed!(12345)
# simulate "specified" variance covariance matrices
V = [rand(n, n) for k in 1:m]
# desired total n x n variance matrix 
Γ = (Σ[1]*V[1]) + (Σ[2]*V[2]) 

d1 = Binomial(20, 0.5)
d2 = Poisson(5)
d3 = Normal()

vecd = [d1, d2, d3]

mixed_multivariate_gc = MultivariateMix(vecd, Γ)

Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(mixed_multivariate_gc, Y, res)

end

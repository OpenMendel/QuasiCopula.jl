using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns
using LinearAlgebra: BlasReal, copytri!
@testset "Trying mixtures of different discrete and cts distributions. Generate 3 element vector:Binomial(20, 0.5), Poisson(5), Normal(). " begin
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
mixed_multivariate_dist = MultivariateMix(vecd, Γ)

Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(mixed_multivariate_dist, Y, res)

end

### MVN only 
@testset "Trying just the normal. Generate 3 element vector: Normal(), Normal(), Normal(). " begin
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

d1 = Normal()
d2 = Normal()
d3 = Normal()

vecd = [d1, d2, d3]
mvn_dist = NonMixedMultivariateDistribution(vecd, Γ)
Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(mvn_dist, Y, res)
end

### Poisson only 
@testset "Trying just the Poisson. Generate 3 element vector: Poisson(5), Poisson(5), Poisson(5). " begin
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

d1 = Poisson(5)
d2 = Poisson(5)
d3 = Poisson(5)

vecd = [d1, d2, d3]
mvp_dist = NonMixedMultivariateDistribution(vecd, Γ)
Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(mvp_dist, Y, res)
end

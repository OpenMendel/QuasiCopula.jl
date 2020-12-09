using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns
using LinearAlgebra: BlasReal, copytri!
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
Γ_hardcoded = (Σ[1]*V[1]) + (Σ[2]*V[2])

### Normal ### 
@testset "Generate Normal(0, 1) vector, transform vector after simulation by mean and variance" begin
### trying just the standard normal base distribution
d1 = Normal(0, 1)
d2 = Normal(0, 1)
d3 = Normal(0, 1)

vecd = [d1, d2, d3]

gvc_vec1 = gvc_vec_continuous(V, Σ, vecd)
end

@testset "Trying mixtures of different continuous distributions. Generate 3 element vector: Normal(0, 1), Gamma(1, 1), Exponential(10). " begin
d1 = Normal(0, 1)
d2 = Gamma(1, 1)
d3 = Exponential(10)

vecd = [d1, d2, d3]

gvc_vec2 = gvc_vec_continuous(V, Σ, vecd)

end

### discrete ### 

@testset "Trying mixtures of different discrete distributions. Generate 3 element vector: Normal(0, 1), Gamma(1, 1), Exponential(10). " begin

d1 = Binomial(20, 0.5)
d2 = Poisson(5)
d3 = Geometric(0.5)

vecd = [d1, d2, d3]
max_value = [20.0, 50.0, 100.0]

gvc_vec3 = gvc_vec_discrete(V, Σ, vecd, max_value)

end

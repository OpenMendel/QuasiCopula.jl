using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!


m = 3
n = 5

# variance components
Random.seed!(12345)
# simulate desired variance components (coefficients of covariance matrices V[k] for k in 1:m)
Σ = rand(m)
Random.seed!(12345)
# simulate "specified" variance covariance matrices
V = [rand(n, n) for k in 1:m]
# desired total n x n variance matrix 
Γ_hardcoded = (Σ[1]*V[1]) + (Σ[2]*V[2]) + (Σ[3]*V[3])

# specify distributions of the residuals
d1 = Normal(0, 1)
d2 = Gamma(1, 1)
d3 = Normal(0, 1)
d4 = Gamma(1, 1)
d5 = Normal(0, 1)

vector_distributions = [d1, d2, d3, d4, d5]

gvc_vec1 = GVCVec(V, Σ, vector_distributions)

@test gvc_vec1.Γ == Γ_hardcoded

####
# create the vector object given just the total variance Γ
gvc_vec2 = GVCVec(Γ_hardcoded, vector_distributions)

# test our construction
@test gvc_vec2.Γ == sum(gvc_vec2.Σ[k] * gvc_vec2.V[k] for k in 1:gvc_vec2.m)

# GenR1(gvc_vec2.Γ, gvc_vec2.vecd[1], gvc_vec2.res[1])

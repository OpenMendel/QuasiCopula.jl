using DataFrames, MixedModels, Random, GLMCopula, GLM, StatsFuns
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


# simulating the first residual using marginal distribution as a mixture model.
res1_obj = genR1(gvc_vec2)
@test gvc_vec2.res[1] != 0.0

# create the vector object given just the total variance Γ, now let r_1 ~ gamma(1, 1)
vector_distributions_gamma1 = [d2, d3, d4, d5, d1]
gvc_vec3 = GVCVec(Γ_hardcoded, vector_distributions_gamma1)

# simulating the first residual using marginal distribution as a mixture model.
genR1(gvc_vec3)
@test gvc_vec3.res[1] != 0.0

##### inverse cdf method using Roots.jl ###
# using Roots

# conditional_cdf(x) =  x^5 - x + 1/2
# uniform_cdf(x)
# F(x) = conditional_cdf(x) - uniform_cdf(x)
# find_zero(F, (0, 1))

##### above we have checked that we write to res[1] the first residual value simulated from the mixture density.
# we have checked for normal and gamma.
##### next we will start to simulate from the conditional density of R_2 | R_1
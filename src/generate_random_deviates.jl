using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!

"""
GVCVecObs
GVCVecObs()
GLM copula variance component model vector of observations, which contains a vector of
`GVCObs as data and appropriate vectorized fields for easy access when simulating from conditional densities.
"""
struct GVCVec{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
    # data
    n::Int     # total number of singleton observations
    m::Int          # number of variance components
    res::Vector{T}  # residual vector res_i
    Y::Vector{T}
    V::Vector{Matrix{T}}
    Σ::Vector{T}
    Γ::Matrix{T}
    # normalizing constant
    trΓ::T
    term1::Vector{T}
    term2::Vector{T}
    term3::Vector{T}
    # working arrays
    storage_n::Vector{T}
    storage_m::Vector{T}
    vecd::Vector{D}
end

function GVCVec(
    V::Vector{Matrix{T}},
    Σ::Vector{T},    # m-vector: [σ12, ..., σm2],
    vecd::Vector{D}  # vector of univariate densities
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}    
    n, m = length(vecd), length(V)
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    Γ = sum(Σ[k] * V[k] for k in 1:m)
    trΓ = tr(Γ)
    term1 = zeros(T, n)    # term1[i] is term1 in the conditional density of r_i given r_1, ..., r_{i-1}
    term2 = zeros(T, n)    # term2[i] is term2 in the conditional density of r_i given r_1, ..., r_{i-1}
    term3 = zeros(T, n)    # term3[i] is term2 in the conditional density of r_i given r_1, ..., r_{i-1}
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    GVCVec(n, m, res, Y, V, Σ, Γ, trΓ, term1, term2, term3,
        storage_n, storage_m, vecd)
end

function GVCVec(
    Γ::Matrix{T},
    vecd::Vector{D}
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    n = length(vecd)
    m = 1
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    V = [Γ]
    Σ = ones(T, m)
    trΓ = tr(Γ)
    term1 = zeros(T, n)    # term1[i] is term1 in the conditional density of r_i given r_1, ..., r_{i-1}
    term2 = zeros(T, n)    # term2[i] is term2 in the conditional density of r_i given r_1, ..., r_{i-1}
    term3 = zeros(T, n)    # term3[i] is term2 in the conditional density of r_i given r_1, ..., r_{i-1}
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    GVCVec(n, m, res, Y, V, Σ, Γ, trΓ, term1, term2, term3,
    storage_n, storage_m, vecd)
end   

# ### sampling R_1 from the marginal density of R_1 ~ f(y_1 = R_1) where f in Normal(0, 1), Gamma(1.0, 1.0)

# """
# GenR1
# GenR1()
# create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities, depending on the distribution.
# """
# struct GenR1{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
#     # data
#     R1::T
#     Γ::Matrix{T}
#     # normalizing constant
#     trΓ::T
#     # working arrays
#     term1::T
#     term23::T
#     d::D
# end

# function GenR1(
#     Γ::Matrix{T},
#     d::Normal{T},  # univariate density of R_1,
#     R1::T) where {T <: BlasReal}    
#     trΓ = tr(Γ)
#     term1 = 1 + 0.5 * tr(Γ)
#     term23 = 1 + 0.5 * (Γ[1, 1] *  R_1^2 + tr(Γ[2:end, 2:end]))
#     GenR1{T, D}(R1, Γ, trΓ, term1, term23, d)
# end

# # create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities:

# # create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities:
# mixing_probabilities = [(1 + 0.5 * tr(Γ[2:end, 2:end])) / (1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5*tr(Γ))]

# D_1 = MixtureModel(
#    [Normal(0.0, 1.0),
#    Chi(3),
#    Chi(3)], mixing_probabilities
#    )

# function generate_R1_mixture(d::Distributions.Distribution)
#     csamplers = map(sampler, d.components)
#     psampler = sampler(d.prior)
#     random_deviate = csamplers[rand(psampler)]
    
#     if typeof(random_deviate) == Normal{Float64}
#         println("using standard normal")
#         return rand(random_deviate)
#     else
#         println("if chi (3), one is positive and one is negative with equal probabilty")
#         return rand([-1, 1]) * rand(random_deviate)
#     end
# end

# R_1 = generate_R1_mixture(D_1)

# # D_1 = MixtureModel(
# #    [Normal(0.0, 1.0),
# #    Chi(3),
# #    Chi(3)], mixing_probabilities
# #    )

# # function generate_R1_mixture(d)
# #     csamplers = map(sampler, d.components)
# #     psampler = sampler(d.prior)
# #     random_deviate = csamplers[rand(psampler)]
    
# #     if typeof(random_deviate) == Normal{Float64}
# #         println("using standard normal")
# #         return rand(random_deviate)
# #     else
# #         println("if chi (3), one is positive and one is negative with equal probabilty")
# #         return rand([-1, 1]) * rand(random_deviate)
# #     end
# # end

# # R_1 = generate_R1_mixture(D_1)
using DataFrames, MixedModels, Random, GLMCopula, GLM, StatsFuns
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

#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# function generate_resvec(
#     gvc_vector::GVCVec{T, D}
#     ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    
# end

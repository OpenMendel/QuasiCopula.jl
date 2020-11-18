using DataFrames, MixedModels, Random, GLMCopula, GLM, StatsFuns
using ForwardDiff, Test, LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!

## make field for conditional terms in the 

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
    ## conditionalterms::ConditionalTerms{T}
    term1::Vector{T}
    term2::Vector{T}
    term3::Vector{Any}
    conditional_pdf::Vector{Any}
    conditional_cdf::Vector{Any}
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
    term3 = Vector{Any}(undef, n)
    conditional_pdf = Vector{Any}(undef, n)
    conditional_cdf = Vector{Any}(undef, n)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    GVCVec(n, m, res, Y, V, Σ, Γ, trΓ, term1, term2, term3, conditional_pdf, conditional_cdf,
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
    term3 = Vector{Any}(undef, n)
    conditional_pdf = Vector{Any}(undef, n)
    conditional_cdf = Vector{Any}(undef, n)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    GVCVec(n, m, res, Y, V, Σ, Γ, trΓ, term1, term2, term3, conditional_pdf, conditional_cdf,
    storage_n, storage_m, vecd)
end   


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
## currently for normal density it is using the residuals to simulate the vectors.

function conditional_terms!(gvc_vec::GVCVec{T, D}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    for i in 1:gvc_vec.n
        gvc_vec.term1[i] = 1 + 0.5 * transpose( gvc_vec.res[1:i-1]) *  gvc_vec.Γ[1:i-1, 1:i-1] *  gvc_vec.res[1:i-1] +  0.5 * tr( gvc_vec.Γ[i:end, i:end])
        gvc_vec.term2[i] = sum(crossterm_res( gvc_vec.res, i, gvc_vec.Γ))
        fun(x) = (0.5 * gvc_vec.Γ[i, i] * (x^2 - 1))
        gvc_vec.term3[i] = fun
        fun2(x) = inv(gvc_vec.term1[i]) * pdf(gvc_vec.vecd[i], x) * (gvc_vec.term1[i] +  gvc_vec.term2[i] +  gvc_vec.term3[i](x))
        gvc_vec.conditional_pdf[i] = fun2
        # fun3(x) = inv(gvc_vec.term1[i]) * (gvc_vec.term1[i] - 0.5 * Γ[i, i]) * cdf(gvc_vec.vecd[i], x) - sum(gvc_vec.res[j] * gvc_vec.Γ[i, j] for j in 1:i-1) * gvc_vec.conditional_pdf[i](res[i]) + 0.5 * Γ[i, i] * (0.5 + 0.5 * sign(x) * cdf(Chisq(3), x^2))
        # gvc_vec.conditional_cdf[i] = fun3
    end
end
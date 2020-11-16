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

### sampling R_1 from the marginal density of R_1 ~ f(y_1 = R_1) where f in Normal(0, 1), Gamma(1.0, 1.0)
# ####
# """
# GenR1
# GenR1()
# create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities, depending on the distribution.
# """
struct GenR1{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
    # data
    gvc_vector::GVCVec{T, D} # we will update gvc_vector.res[1]
    # working arrays
    term1::T
    term2::T
end


### GAUSSIAN BASE ### 
"""
GenR1
GenR1()
Let R1~N(0, 1) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities. Given d = distribution of R1.
"""
function GenR1(
    gvc_vector::GVCVec{T, D},
    d::Distributions.Normal{T}
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    term1 = 1 + 0.5 * gvc_vector.trΓ
    term2 = 1 + 0.5 * tr(gvc_vector.Γ[2:end, 2:end])
    mixture_probabilities = [inv(term1) * term2, inv(term1) * (0.25 * gvc_vector.Γ[1, 1]), inv(term1) * (0.25 * gvc_vector.Γ[1, 1])]
    mixture_model = MixtureModel(
    [Normal(0.0, 1.0),
    Chi(3),
    Chi(3)], mixture_probabilities
    )
    gvc_vector.res[1] = generate_R1_mixture_Normal(mixture_model)
    GenR1{T, D}(gvc_vector, term1, term2)
end

"""
GenR1
GenR1()
Let R1~N(0, 1) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities. Just given without distribution of R1
"""
function GenR1(
    gvc_vector::GVCVec{T, D}
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    GenR1(gvc_vector, gvc_vector.vecd[1])
end


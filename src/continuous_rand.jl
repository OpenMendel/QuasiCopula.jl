@reexport using Distributions
import Distributions: mean, var, logpdf, pdf, cdf, maximum, minimum, insupport, quantile
export ContinuousUnivariateCopula, marginal_pdf_constants, mvsk_to_absm, conditional_pdf_constants, crossterm_res, gvc_vec_continuous, update_res! 
using LinearAlgebra: BlasReal, copytri!

struct ContinuousUnivariateCopula{
    DistT <: ContinuousUnivariateDistribution, 
    T     <: Real
    } <: ContinuousUnivariateDistribution
    d  :: DistT
    μ :: T
    σ2 :: T
    c0 :: T
    c1 :: T
    c2 :: T
    c  :: T # normalizing constant
end

function ContinuousUnivariateCopula(
    d  :: DistT, 
    c0 :: T, 
    c1 :: T, 
    c2 :: T) where {DistT <: ContinuousUnivariateDistribution, T <: Real}
    μ = mean(d)
    σ2 = var(d)
    c  = inv(c0 + c1 * μ + c2 * (σ2 + abs2(μ)))
    Tc = typeof(c)
    ContinuousUnivariateCopula(d, Tc(μ), Tc(σ2), Tc(c0), Tc(c1), Tc(c2), c)
end

# this function will fill out the appropriate constants to form the ContinuousUnivariateCopula structure. 
function marginal_pdf_constants(Γ::Matrix{T}, dist::Union{Gamma{T}, Beta{T}, Exponential{T}}) where T <: Real
    μ = mean(dist)
    σ2 = var(dist)
    c_0 = μ^2 * inv(σ2)
    c_1 = -2μ * inv(σ2)
    c_2 = inv(σ2)
    c0 = 1 + 0.5 * tr(Γ[2:end, 2:end]) + 0.5 * Γ[1, 1] * c_0
    c1 = 0.5 * Γ[1, 1] * c_1
    c2 = 0.5 * Γ[1, 1] * c_2
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end

# this function will fill out the appropriate constants to form the ContinuousUnivariateCopula
# structure when y ~ Normal. We will use the base distribution of the standardized residuals
# (standard normal) here instead.
function marginal_pdf_constants(Γ::Matrix{T}, dist::Normal{T}) where T <: Real
    c0 = 1 + 0.5 * tr(Γ[2:end, 2:end])
    c2 = 0.5 * Γ[1, 1]
    ContinuousUnivariateCopula(dist, c0, 0.0, c2)
end


# # this function will fill out the appropriate constants for conditional distribution to form the ContinuousUnivariateCopula structure. 
function conditional_pdf_constants(Γ::Matrix{T}, res::Vector{T}, i::Int64, dist::Normal{T}) where T <: Real
    μ = mean(dist)
    σ2 = var(dist)
    c0 = 1 + 0.5 * transpose(res[1:i-1]) *  Γ[1:i-1, 1:i-1] *  res[1:i-1] +  0.5 * tr(Γ[i+1:end, i+1:end])
    c1 = crossterm_res(res, i, Γ)[1]
    c2 = 0.5 * Γ[i, i]
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end

# this function will fill out the appropriate constants for conditional distribution to form the ContinuousUnivariateCopula structure. 
function conditional_pdf_constants(Γ::Matrix{T}, res::Vector{T}, i::Int64, dist::Union{Gamma{T}, Beta{T}, Exponential{T}}) where T <: Real
    μ = mean(dist)
    σ2 = var(dist)
    c_0 = μ^2 * inv(σ2)
    c__0 = μ * inv(sqrt(σ2)) * crossterm_res(res, i, Γ)
    c_1 = -2μ * inv(σ2)
    c__1 = inv(sqrt(σ2)) * crossterm_res(res, i, Γ) 
    c_2 = inv(σ2)
    c0 = 1 + 0.5 * transpose(res[1:i-1]) *  Γ[1:i-1, 1:i-1] *  res[1:i-1] +  0.5 * tr(Γ[i+1:end, i+1:end]) + 0.5 * Γ[i, i] * c_0  - c__0[1]
    c1 = 0.5 * Γ[i, i] * c_1  + c__1[1]
    c2 = 0.5 * Γ[i, i] * c_2
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end

function crossterm_res(res::Vector{T}, s::Integer, Γ::Matrix{T}; all = false) where {T<: BlasReal}
    results = []
    if s == 1
        return 0.0
    elseif s > 1
        if all == true
            for i in 2:s
                for j in 1:i - 1
                    push!(results, sum(res[j] * Γ[i, j]))
                end
            end
        else
            for j in 1:s - 1
                push!(results, sum(res[j] * Γ[s, j]))
            end
        end
    end
    return results
 end

"""
gvc_vec_continuous
gvc_vec_continuous()
GLM copula variance component model vector of observations, which contains a vector of
`ContinuousUnivariateCopula as data and appropriate vectorized fields for easy access when simulating from conditional densities.
"""
struct gvc_vec_continuous{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
    # data
    n::Int     # total number of singleton observations
    m::Int          # number of variance components
    gc_obs::Vector{ContinuousUnivariateCopula}
    res::Vector{T}  # residual vector res_i
    Y::Vector{T}
    V::Vector{Matrix{T}}
    Σ::Vector{T}
    Γ::Matrix{T}
    # normalizing constant
    trΓ::T
    ## conditionalterms::ConditionalTerms{T}
    vecd::Vector{D}
end

function gvc_vec_continuous(
    V::Vector{Matrix{T}},
    Σ::Vector{T},    # m-vector: [σ12, ..., σm2],
    vecd::Vector{D}  # vector of univariate densities
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    n, m = length(vecd), length(V)
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    Γ = sum(Σ[k] * V[k] for k in 1:m)
    gc_obs = Vector{ContinuousUnivariateCopula}(undef, n)

    # form constants for the marginal density
    gc_obs[1] = marginal_pdf_constants(Γ, vecd[1])
    # generate y_1 
    Y[1] = rand(gc_obs[1])

    for i in 2:length(vecd)
        # update residuals 1,..., i-1
        res[i-1] = update_res!(Y[i-1], res[i-1], gc_obs[i-1])
        # form constants for conditional density of i given 1, ..., i-1
        gc_obs[i] = conditional_pdf_constants(Γ, res, i, vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = rand(gc_obs[i])
     end
    res[end] = update_res!(Y[end], res[end], gc_obs[end])
    trΓ = tr(Γ)
    gvc_vec_continuous(n, m, gc_obs, res, Y, V, Σ, Γ, trΓ, vecd)
end

function gvc_vec_continuous(
    Γ::Matrix{T},
    vecd::Vector{D}
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    n = length(vecd)
    m = 1
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    V = [Γ]
    Σ = ones(T, m)
    gc_obs = Vector{ContinuousUnivariateCopula}(undef, n)

    # form constants for the marginal density
    gc_obs[1] = marginal_pdf_constants(Γ, vecd[1])
    # generate y_1 
    Y[1] = rand!(gc_obs[1], [Y[1]])[1]

    for i in 2:length(vecd)
        # update residuals 1,..., i-1
        res[i-1] = update_res!(Y[i-1], res[i-1], gc_obs[i-1])
        # form constants for conditional density of i given 1, ..., i-1
        gc_obs[i] = conditional_pdf_constants(Γ, res, i, vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = rand!(gc_obs[i], [Y[i]])[1]
     end
    res[end] = update_res!(Y[end], res[end], gc_obs[end])
    trΓ = tr(Γ)
    gvc_vec_continuous(n, m, gc_obs, res, Y, V, Σ, Γ, trΓ, vecd)
end   

minimum(d::ContinuousUnivariateCopula) = minimum(d.d)
maximum(d::ContinuousUnivariateCopula) = maximum(d.d)
insupport(d::ContinuousUnivariateCopula) = insupport(d.d)

"""
    mvsk_to_absm(μ, σ², sk, kt)
Convert mean `μ`, variance `σ²`, skewness `sk`, and kurtosis `kt` to first four 
moments about zero. See formula at <https://en.wikipedia.org/wiki/Central_moment#Relation_to_moments_about_the_origin>.
"""
function mvsk_to_absm(μ, σ², sk, kt)
    σ = sqrt(σ²)
    m1 = μ # E[X]
    m2 = σ² + abs2(μ) # E[X^2] - E[X]^2 + E[X]^2 = E[X^2]
    m3 = sk * σ^3 + 3μ * m2 - 2μ^3 # E[X^3] = E[((X-μ)/σ)^3] * (1 / σ^3) + 3 μ E[X^2] - 3μ^2 * E[X] + μ^3 
    m4 = kt * abs2(σ²) + 4μ * m3 - 6 * abs2(μ) * m2 + 3μ^4
    m1, m2, m3, m4
end

"""
    mean(d::ContinuousUnivariateCopula)
Theoretical mean under the copula model. 
"""
function mean(d::ContinuousUnivariateCopula)
    μ, σ², sk, kt = mean(d.d), var(d.d), skewness(d.d), kurtosis(d.d, false) # proper kurtosis (un-corrected) when false 
    m1, m2, m3, _ = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m1 + d.c1 * m2 + d.c2 * m3)
end

"""
    var(d::ContinuousUnivariateCopula)
Theoretical variance under the copula model. 
"""
function var(d::ContinuousUnivariateCopula)
    μ, σ², sk, kt = mean(d.d), var(d.d), skewness(d.d), kurtosis(d.d, false)
    _, m2, m3, m4 = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m2 + d.c1 * m3 + d.c2 * m4) - abs2(mean(d))
end

function logpdf(d::ContinuousUnivariateCopula, x::Real)
    log(d.c * (d.c0 + d.c1 * x + d.c2 * abs2(x))) + logpdf(d.d, x)
end

function pdf(d::ContinuousUnivariateCopula, x::Real)
    d.c * pdf(d.d, x) * (d.c0 + d.c1 * x + d.c2 * abs2(x))
end

# this function specialized to Normal base distribution
function cdf(d::ContinuousUnivariateCopula{Normal{T},T}, x::Real) where T <: Real
    μ, σ    = d.d.μ, d.d.σ
    z       = (x - μ) / σ
    result  = (d.c0 + d.c1 * μ + d.c2 * abs2(μ)) * cdf(Normal(), z)
    result += (d.c1 + 2d.c2 * μ) * σ / sqrt(2π) *
        ifelse(z < 0, - ccdf(Chi(2), -z), cdf(Chi(2), z) - 1)
    result += d.c2 * abs2(σ) / 2 * ifelse(z < 0, ccdf(Chi(3), -z), cdf(Chi(3), z) + 1)
    result *= d.c
end

# this function specialized to Gamma base distribution
function cdf(d::ContinuousUnivariateCopula{Gamma{T},T}, x::Real) where T <: Real
    α, θ = params(d.d)
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) * cdf(Gamma(α + 1, θ), x)
    result += d.c2 * (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α)) * cdf(Gamma(α + 2, θ), x)
    result *= d.c
end

#this function specialized to exponential base distribution
function cdf(d::ContinuousUnivariateCopula{Exponential{T},T}, x::Real) where T <: Real
    θ = params(d.d)[1]
    normalizing_c1 = θ * (StatsFuns.gamma(2)/ StatsFuns.gamma(1))
    normalizing_c2 = θ^2 * (StatsFuns.gamma(3)/ StatsFuns.gamma(1))
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * normalizing_c1 * cdf(Gamma(2, θ), x)
    result += d.c2 * normalizing_c2 * cdf(Gamma(3, θ), x)
    result *= d.c
end

# this function specialized to beta base distribution
function cdf(d::ContinuousUnivariateCopula{Beta{T},T}, x::Real) where T <: Real
    α, β = params(d.d)
    normalizing_c1 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 1)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 1))
    normalizing_c2 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 2)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 2))
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * normalizing_c1 * cdf(Beta(α + 1, β), x)
    result += d.c2 * normalizing_c2 * cdf(Beta(α + 2, β), x)
    result *= d.c
end

# Gamma has no mode when shape < 1.
function quantile(d::ContinuousUnivariateCopula{Gamma{T}, T}, p::T) where T <: Real
    α, θ = params(d.d)
    if α ≥ 1
        Distributions.quantile_newton(d, p, (d.d.α - 1) * d.d.θ)
    else 
        error("Gamma has no mode when shape < 1")
    end
end

function quantile(d::ContinuousUnivariateCopula{Normal{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, mean(d))
end

function quantile(d::ContinuousUnivariateCopula{Exponential{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, 0.0)
end

# since Beta density has finite extrema, we can use the bisection method to implement the inverse cdf sampling 
function quantile(d::ContinuousUnivariateCopula{Beta{T}, T}, p::T) where T <: Real
    min, max = extrema(d.d) 
    Distributions.quantile_bisect(d, p, min, max, 1.0e-12)
end

# function quantile(d::ContinuousUnivariateCopula{Beta{T}, T}, p::T) where T <: Real
#     (α, β) = params(d.d)
#     if (α > 1 && β > 1)
#         Distributions.quantile_newton(d, p, (α - 1) * inv(α + β - 2))
#     elseif (α ≤ 1 && β > 1)
#         Distributions.quantile_newton(d, p, 0.0)
#     elseif (α > 1 && β ≤ 1)
#         Distributions.quantile_newton(d, p, 1.0)
#     elseif (α < 1 && β < 1) # only when bimodal, use the bisection method.
#         min, max = extrema(d.d) 
#         Distributions.quantile_bisect(d, p, min, max, 1.0e-12)
#     end
# end
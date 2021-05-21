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

"""
    ContinuousUnivariateCopula(d, c0, c1, c2)
The distribution with density `c * P(x = x) * (c0 + c1 * x + c2 * x^2)`, where `f` 
is the density of the base distribution `d` and `c` is the normalizing constant.
"""
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

"""
    pdf_constants(Γ::Matrix{<:Real}, res::Vector{<:Real}, i::Int64, dist::ContinuousUnivariateDistribution)
This function will fill out the appropriate constants, c0, c1, c2 for each conditional distribution to form the `ContinuousUnivariateCopula` structure. 
"""
function pdf_constants(Γ::Matrix{T}, res::Vector{T}, i::Int64, dist::ContinuousUnivariateDistribution) where T <: Real
    μ = Distributions.mean(dist)
    σ2 = Distributions.var(dist)
    c_0 = μ^2 * inv(σ2)
    c__0 = μ * inv(sqrt(σ2)) * sum(crossterm_res(res, i, Γ))
    c_1 = -2μ * inv(σ2)
    c__1 = inv(sqrt(σ2)) * sum(crossterm_res(res, i, Γ)) 
    c_2 = inv(σ2)
    storage = zeros(i-1)
    # first multiply Γ[1:i-1, 1:i-1] *  res[1:i-1]
    mul!(storage, Γ[1:i-1, 1:i-1], res[1:i-1])
    # then multiply to get transpose(res[1:i-1]) *  Γ[1:i-1, 1:i-1] *  res[1:i-1]
    # dot(storage, res[1:i-1])
    c0 = 1 + 0.5 * dot(storage, res[1:i-1]) +  0.5 * tr(Γ[i+1:end, i+1:end]) + 0.5 * Γ[i, i] * c_0  - c__0
    # c0 = 1 + 0.5 * transpose(res[1:i-1]) *  Γ[1:i-1, 1:i-1] *  res[1:i-1] +  0.5 * tr(Γ[i+1:end, i+1:end]) + 0.5 * Γ[i, i] * c_0  - c__0
    c1 = 0.5 * Γ[i, i] * c_1  + c__1
    c2 = 0.5 * Γ[i, i] * c_2
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end


"""
    pdf_constants(Γ::Matrix{<:Real}, dist::DiscreteUnivariateDistribution)
This function will fill out the appropriate constants, c0, c1, c2 for the univariate marginal distribution to form the `DiscreteUnivariateCopula` structure. 
"""
function pdf_constants(Γ::Matrix{T}, i::Int64, dist::ContinuousUnivariateDistribution) where T <: Real
    μ = Distributions.mean(dist)
    σ2 = Distributions.var(dist)
    c_0 = μ^2 * inv(σ2)
    c_1 = -2μ * inv(σ2)
    c_2 = inv(σ2)
    c0 = 1  +  0.5 * (tr(Γ) - Γ[i, i]) + 0.5 * Γ[i, i] * c_0
    c1 = 0.5 * Γ[i, i] * c_1 
    c2 = 0.5 * Γ[i, i] * c_2
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end


"""
    pdf_constants(γ::T, dist::ContinuousUnivariateDistribution)
This function will fill out the appropriate constants, c0, c1, c2 for the univariate marginal distribution to form the `ContinuousUnivariateCopula` structure. 
When the number of observations in the cluster is 1
"""
function pdf_constants(γ::T, dist::ContinuousUnivariateDistribution) where T <: Real
    μ = Distributions.mean(dist)
    σ2 = Distributions.var(dist)
    c_0 = μ^2 * inv(σ2)
    c_1 = -2μ * inv(σ2)
    c_2 = inv(σ2)
    c0 = 1  + 0.5 * γ * c_0
    c1 = 0.5 * γ * c_1 
    c2 = 0.5 * γ * c_2
    ContinuousUnivariateCopula(dist, c0, c1, c2)
end


"""
    crossterm_res(res::Vector{<:Real}, i::Int64, Γ::Matrix{<:Real}; all = false)
This function will compute the crossterm involving the residual values for constants c0 and c1 in the conditional densities. 
Default all = false (conditional density):sum_{j = 1}^{i-1} γ_ij * r_j, but if all = true then will output (marginal density): sum_{j = 1}^{i} γ_ij * r_j
"""
function crossterm_res(res::Vector{T}, i::Integer, Γ::Matrix{T}; all = false) where {T<: BlasReal}
    results = zeros(i-1)
    if i == 1
        return results
    else
        for j in 1:i - 1
            results[j] = res[j] * Γ[i, j]
        end
    end
    results
 end


"""
    update_res!(Y::Real, res:Real, gc_obs::Union{ContinuousUnivariateCopula{<:ContinuousUnivariateDistribution, <:Real}, DiscreteUnivariateCopula{<:DiscreteUnivariateDistribution, <:Real}})
This function will update the residual value, given the base distributions mean and variance. This step is necessary when constructing the constants, c0, c1, c2, in the conditional density. 
"""
 function update_res!(
    Y::T,
    res::T,
    gc_obs::Union{ContinuousUnivariateCopula{D, T}, DiscreteUnivariateCopula{D, T}}) where {T <: BlasReal, D}
    res = (Y - gc_obs.μ) * inv(sqrt(gc_obs.σ2))
 end

minimum(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}) = minimum(d.d)
maximum(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}) = maximum(d.d)
insupport(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}) = insupport(d.d)

"""
    mvsk_to_absm(μ, σ², sk, kt)
Convert mean `μ`, variance `σ²`, skewness `sk`, and kurtosis `kt` to first four 
moments about zero. See formula at <https://en.wikipedia.org/wiki/Central_moment#Relation_to_moments_about_the_origin>.
"""
function mvsk_to_absm(μ, σ², sk, kt)
    σ = sqrt(σ²)
    m1 = μ # E[X]
    m2 = σ² + abs2(μ) # Var(X) + E[X]^2 = E[X^2]
    m3 = sk * σ^3 + 3μ * m2 - 2μ^3 # E[X^3] ; where sk = m3 - 3μ(m2) + 2μ^3
    m4 = kt * abs2(σ²) + 4μ * m3 - 6 * abs2(μ) * m2 + 3μ^4 # E[X^4]; where kt = m4 - 4μ (m3) + 6μ^2 (m2) -3μ^4
    m1, m2, m3, m4
end

"""
    mean(d::ContinuousUnivariateCopula)
Theoretical mean under the copula model. 
"""
function mean(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula})
    μ, σ², sk, kt = Distributions.mean(d.d), Distributions.var(d.d), Distributions.skewness(d.d), Distributions.kurtosis(d.d, false) # proper kurtosis (un-corrected) when false 
    m1, m2, m3, _ = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m1 + d.c1 * m2 + d.c2 * m3)
end

"""
    var(d::ContinuousUnivariateCopula)
Theoretical variance under the copula model. 
"""
function var(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula})
    μ, σ², sk, kt = mean(d.d), var(d.d), skewness(d.d), kurtosis(d.d, false)
    _, m2, m3, m4 = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m2 + d.c1 * m3 + d.c2 * m4) - abs2(mean(d)) # E[Y_k^2] - E[Y_k]^2
end

"""
    logpdf(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}, x::Real)
Theoretical log pdf under the copula model. 
"""
function logpdf(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}, x::Real)
    log(d.c * (d.c0 + d.c1 * x + d.c2 * abs2(x))) + logpdf(d.d, x)
end

"""
    pdf(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}, x::Real)
Theoretical pdf under the copula model. 
"""
function pdf(d::Union{ContinuousUnivariateCopula, DiscreteUnivariateCopula}, x::Real)
    d.c * pdf(d.d, x) * (d.c0 + d.c1 * x + d.c2 * abs2(x))
end

"""
    cdf(d::Union{ContinuousUnivariateCopula{Normal{<:Real}}, x::Real)
Theoretical cdf for the Normal base distribution derived under the copula model. 
"""
function cdf(d::ContinuousUnivariateCopula{Normal{T},T}, x::Real) where T <: Real
    μ, σ    = d.d.μ, d.d.σ
    z       = (x - μ) / σ
    result  = (d.c0 + d.c1 * μ + d.c2 * abs2(μ)) * cdf(Normal(), z)
    result += (d.c1 + 2d.c2 * μ) * σ / sqrt(2π) *
        ifelse(z < 0, - ccdf(Chi(2), -z), cdf(Chi(2), z) - 1)
    result += d.c2 * abs2(σ) / 2 * ifelse(z < 0, ccdf(Chi(3), -z), cdf(Chi(3), z) + 1)
    result *= d.c
end

"""
    cdf(d::Union{ContinuousUnivariateCopula{Gamma{<:Real}}, x::Real)
Theoretical cdf for the Gamma base distribution derived under the copula model. 
"""
function cdf(d::ContinuousUnivariateCopula{Gamma{T},T}, x::Real) where T <: Real
    α, θ = params(d.d)
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) * cdf(Gamma(α + 1, θ), x)
    result += d.c2 * (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α)) * cdf(Gamma(α + 2, θ), x)
    result *= d.c
end

"""
    cdf(d::Union{ContinuousUnivariateCopula{Exponential{<:Real}}, x::Real)
Theoretical cdf for the Exponential base distribution derived under the copula model. 
"""
function cdf(d::ContinuousUnivariateCopula{Exponential{T},T}, x::Real) where T <: Real
    θ = params(d.d)[1]
    normalizing_c1 = θ * (StatsFuns.gamma(2)/ StatsFuns.gamma(1))
    normalizing_c2 = θ^2 * (StatsFuns.gamma(3)/ StatsFuns.gamma(1))
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * normalizing_c1 * cdf(Gamma(2, θ), x)
    result += d.c2 * normalizing_c2 * cdf(Gamma(3, θ), x)
    result *= d.c
end

"""
    cdf(d::Union{ContinuousUnivariateCopula{Beta{<:Real}}, x::Real)
Theoretical cdf for the Beta base distribution derived under the copula model. 
"""
function cdf(d::ContinuousUnivariateCopula{Beta{T},T}, x::Real) where T <: Real
    α, β = params(d.d)
    normalizing_c1 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 1)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 1))
    normalizing_c2 = inv(StatsFuns.gamma(α) * StatsFuns.gamma(α + β + 2)) * (StatsFuns.gamma(α + β) * StatsFuns.gamma(α + 2))
    result  = d.c0 * cdf(d.d, x)
    result += d.c1 * normalizing_c1 * cdf(Beta(α + 1, β), x)
    result += d.c2 * normalizing_c2 * cdf(Beta(α + 2, β), x)
    result *= d.c
end

"""
    quantile(d::Union{ContinuousUnivariateCopula{Normal{<:Real}}, p::Real)
Finds the quantile value for the specified cumulative probability `p`, under the Normal Base distribution of the copula model, using Newtons Method. 
"""
function quantile(d::ContinuousUnivariateCopula{Normal{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, mean(d))
end

"""
    quantile(d::Union{ContinuousUnivariateCopula{Gamma{<:Real}}, p::Real)
Finds the quantile value for the specified cumulative probability `p`, under the Gamma Base distribution of the copula model, using Newtons method. 
"""
function quantile(d::ContinuousUnivariateCopula{Gamma{T}, T}, p::T) where T <: Real
    α, θ = params(d.d)
    if α ≥ 1
        Distributions.quantile_newton(d, p, ((d.d.α - 1) * d.d.θ))
    else 
        error("Gamma has no mode when shape < 1")
    end
end

"""
    quantile(d::Union{ContinuousUnivariateCopula{Exponential{<:Real}}, p::Real)
Finds the quantile value for the specified cumulative probability `p`, under the Normal Base distribution of the copula model, using Newtons method. 
"""
function quantile(d::ContinuousUnivariateCopula{Exponential{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, 0.0)
end

"""
    quantile(d::Union{ContinuousUnivariateCopula{Exponential{<:Real}}, p::Real)
Finds the quantile value for the specified cumulative probability `p`, under the Beta Base distribution of the copula model, using Bisection Method. 
Since the Beta density has finite extrema, [0, 1], we can use the bisection method to implement the inverse cdf sampling rather than Newtons Method (greater stability). 
"""
function quantile(d::ContinuousUnivariateCopula{Beta{T}, T}, p::T) where T <: Real
    min, max = extrema(d.d) 
    Distributions.quantile_bisect(d, p, min, max, 1.0e-12)
end

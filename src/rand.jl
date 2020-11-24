@reexport using Distributions
import Distributions: mean, var, logpdf, pdf, cdf, maximum, minimum, insupport, quantile
export ContinuousUnivariateCopula

"""
    ContinuousUnivariateCopula(d, c0, c1, c2)
The distribution with density `c * f(x) * (c0 + c1 * x + c2 * x^2)`, where `f` 
is the density of the base distribution `d` and `c` is the normalizing constant.
"""
struct ContinuousUnivariateCopula{
    DistT <: ContinuousUnivariateDistribution, 
    T     <: Real
    } <: ContinuousUnivariateDistribution
    d  :: DistT
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
    c  = inv(c0 + c1 * mean(d) + c2 * (var(d) + abs2(mean(d))))
    Tc = typeof(c)
    ContinuousUnivariateCopula(d, Tc(c0), Tc(c1), Tc(c2), c)
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

# this function specialized to exponential base distribution
# function cdf(d::ContinuousUnivariateCopula{Exponential{T},T}, x::Real) where T <: Real
#     λ = params(d.d)
#     # normalizing_c1 = 
#     # normalizing_c2 = 
#     result  = d.c0 * cdf(d.d, x)
#     result += d.c1 * normalizing_c1 * cdf(Exponential(), x)
#     result += d.c2 * normalizing_c2 * cdf(Exponential(), x)
#     result *= d.c
# end

# this function specialized to beta base distribution
# function cdf(d::ContinuousUnivariateCopula{Beta{T},T}, x::Real) where T <: Real
#     α, β = params(d.d)
#     # normalizing_c1 = 
#     # normalizing_c2 = 
#     result  = d.c0 * cdf(d.d, x)
#     result += d.c1 * normalizing_c1 * cdf(Beta(), x)
#     result += d.c2 * normalizing_c2 * cdf(Beta(), x)
#     result *= d.c
# end

function quantile(d::ContinuousUnivariateCopula{Gamma{T}, T}, p::T) where T <: Real
    α, θ = params(d.d)
    if α > 1
        Distributions.quantile_newton(d, p, (d.d.α - 1) * d.d.θ)
    else
        Distributions.quantile_newton(d, p, mean(d.d))
    end
end

function quantile(d::ContinuousUnivariateCopula{Normal{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, mean(d))
end


function quantile(d::ContinuousUnivariateCopula{Exponential{T}, T}, p::T) where T <: Real
    Distributions.quantile_newton(d, p, mode(d.d))
end

function quantile(d::ContinuousUnivariateCopula{Beta{T}, T}, p::T) where T <: Real
    α, β = params(d.d)
    if α > 1 && β > 1
        Distributions.quantile_newton(d, p, mode(d.d))
    elseif α <= 1 && β > 1
        Distributions.quantile_newton(d, p, 0.0)
    elseif α > 1 && β <= 1 
        Distributions.quantile_newton(d, p, 1.0)
    end
end
@reexport using Distributions
import Distributions: mean, var, logpdf, pdf, cdf, maximum, minimum, insupport, quantile
export DiscreteUnivariateCopula, marginal_pdf_constants
export pmf_copula, reorder_pmf, discrete_rand, discrete_rand!

"""
    DiscreteUnivariateCopula(d, c0, c1, c2)
The distribution with density `c * P(x = x) * (c0 + c1 * x + c2 * x^2)`, where `f` 
is the density of the base distribution `d` and `c` is the normalizing constant.
"""
struct DiscreteUnivariateCopula{
    DistT <: DiscreteUnivariateDistribution, 
    T     <: Real
    } <: DiscreteUnivariateDistribution
    d  :: DistT
    c0 :: T
    c1 :: T
    c2 :: T
    c  :: T # normalizing constant
end

function DiscreteUnivariateCopula(
    d  :: DistT, 
    c0 :: T, 
    c1 :: T, 
    c2 :: T) where {DistT <: DiscreteUnivariateDistribution, T <: Real}
    c  = inv(c0 + c1 * mean(d) + c2 * (var(d) + abs2(mean(d))))
    Tc = typeof(c)
    DiscreteUnivariateCopula(d, Tc(c0), Tc(c1), Tc(c2), c)
end

# this function will fill out the appropriate constants to form the ContinuousUnivariateCopula structure. 
function marginal_pdf_constants(Γ::Matrix{T}, dist::DiscreteUnivariateDistribution) where T <: Real
    μ = mean(dist)
    σ2 = var(dist)
    c_0 = μ^2 * inv(σ2)
    c_1 = -2μ * inv(σ2)
    c_2 = inv(σ2)
    c0 = 1 + 0.5 * tr(Γ[2:end, 2:end]) + 0.5 * Γ[1, 1] * c_0
    c1 = 0.5 * Γ[1, 1] * c_1
    c2 = 0.5 * Γ[1, 1] * c_2
    DiscreteUnivariateCopula(dist, c0, c1, c2)
end

"""
    mean(d::DiscreteUnivariateCopula)
Theoretical mean under the copula model. 
"""
function mean(d::DiscreteUnivariateCopula)
    μ, σ², sk, kt = mean(d.d), var(d.d), skewness(d.d), kurtosis(d.d, false) # proper kurtosis (un-corrected) when false 
    m1, m2, m3, _ = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m1 + d.c1 * m2 + d.c2 * m3)
end

"""
    var(d::DiscreteUnivariateCopula)
Theoretical variance under the copula model. 
"""
function var(d::DiscreteUnivariateCopula)
    μ, σ², sk, kt = mean(d.d), var(d.d), skewness(d.d), kurtosis(d.d, false)
    _, m2, m3, m4 = mvsk_to_absm(μ, σ², sk, kt)
    d.c * (d.c0 * m2 + d.c1 * m3 + d.c2 * m4) - abs2(mean(d))
end


function pdf(d::DiscreteUnivariateCopula, x::Real)
    d.c * pdf(d.d, x) * (d.c0 + d.c1 * x + d.c2 * abs2(x))
end

#### discrete specific #### 

# using our pdf function
function pmf_copula(maximum::T, dist::DiscreteUnivariateCopula) where T<: Real
    y_sample = collect(0:maximum)
    pmf_vec = zeros(length(y_sample)) # marginal pmf
    for k in 1:maximum
        pmf_vec[k] = pdf(dist, y_sample[k]) # get the pmf given maximum and our pdf implementation
    end
    pmf_vec
end

function reorder_pmf(pmf::Vector{T}, μ) where T <: Real
    listofj = zeros(Int64, length(pmf))
    k = Integer(floor.(μ))
    reordered_pmf = zeros(length(pmf))
    i = 1
    j = k[1]
    while(i < length(pmf) && j > 0)
        listofj[i] = j
        reordered_pmf[i] = pmf[j + 1]
        if i%2 == 1
            j = j + i
            elseif i%2 == 0
            j = j - i
        end
        i = i + 1
    end
    if j == 0
        listofj[i] = 0
        reordered_pmf[i] = pmf[1]
        for s in i+1:length(pmf)
            listofj[s] = s - 1
            reordered_pmf[s] = pmf[s]
            end
        end
    return(listofj, reordered_pmf)
end

# for a single mu, generate a single poisson.
function discrete_rand(maximum::Int64, dist::DiscreteUnivariateCopula, μ::T) where T <: Real
    pmf_vec = pmf_copula(maximum, dist) # get pmf under our copula density 
    listofj, reordered_pmf = reorder_pmf(pmf_vec, μ) # re-order the pmf 
    # listofj[1] == μ  #### make sure the first entry of the pmf is the mean.

    sample = rand() # generate x from uniform(0, 1)
    (random_deviate, s) = listofj[1], reordered_pmf[1] # if the cumulative probability mass is less than the P(X = listofj[1]) then leave it as the mean
    # precompute cumsum #
    for i in 2:length(reordered_pmf)  
        if sample < s
            random_deviate = listofj[i - 1]
            break
        else
            s += reordered_pmf[i]
        end
    end
    random_deviate
end

function discrete_rand(maximum::Int64, dist::DiscreteUnivariateCopula, μ, n_reps::Int64) where T <: Real
    random_deviate = zeros(Int64, n_reps)
    for l in 1:n_reps
        random_deviate[l] = discrete_rand(maximum, dist, μ)
    end
    random_deviate
end

function discrete_rand!(maximum::Int64, dist::DiscreteUnivariateCopula, μ, sample::Vector) where T <: Real
    for i in 1:length(sample)
        sample[i] = discrete_rand(maximum, dist, μ)
    end
    sample
end
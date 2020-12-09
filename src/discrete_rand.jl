@reexport using Distributions
import Distributions: mean, var, logpdf, pdf, cdf, maximum, minimum, insupport, quantile
export DiscreteUnivariateCopula, marginal_pdf_constants
export pmf_copula, reorder_pmf, discrete_rand, discrete_rand!, gvc_vec_discrete

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
    μ :: T
    σ2 :: T
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
    μ = mean(d)
    σ2 = var(d)
    c  = inv(c0 + c1 * μ + c2 * (σ2 + abs2(μ)))
    Tc = typeof(c)
    DiscreteUnivariateCopula(d, Tc(μ), Tc(σ2), Tc(c0), Tc(c1), Tc(c2), c)
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

# this function will fill out the appropriate constants for conditional distribution to form the ContinuousUnivariateCopula structure. 
function conditional_pdf_constants(Γ::Matrix{T}, res::Vector{T}, i::Int64, dist::DiscreteUnivariateDistribution) where T <: Real
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
    DiscreteUnivariateCopula(dist, c0, c1, c2)
end

"""
gvc_vec
gvc_vec()
GLM copula variance component model vector of observations, which contains a vector of
`ContinuousUnivariateCopula as data and appropriate vectorized fields for easy access when simulating from conditional densities.
"""
struct gvc_vec_discrete{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
    # data
    n::Int     # total number of singleton observations
    m::Int          # number of variance components
    gc_obs::Vector{DiscreteUnivariateCopula}
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

function gvc_vec_discrete(
    V::Vector{Matrix{T}},
    Σ::Vector{T},    # m-vector: [σ12, ..., σm2],
    vecd::Vector{D},  # vector of univariate densities
    max_value::Vector{T}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    n, m = length(vecd), length(V)
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    Γ = sum(Σ[k] * V[k] for k in 1:m)
    
    gc_obs = Vector{DiscreteUnivariateCopula}(undef, n)

    # form constants for the marginal density
    gc_obs[1] = marginal_pdf_constants(Γ, vecd[1])
    # generate y_1 
    Y[1] = discrete_rand(Integer(max_value[1]), gc_obs[1], gc_obs[1].μ)
    
    for i in 2:length(vecd)
        # update residuals 1,..., i-1
        res[i-1] = update_res!(Y[i-1], res[i-1], gc_obs[i-1])
        # form constants for conditional density of i given 1, ..., i-1
        gc_obs[i] = conditional_pdf_constants(Γ, res, i, vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = discrete_rand(Integer(max_value[i]), gc_obs[i], gc_obs[i].μ)
     end
    res[end] = update_res!(Y[end], res[end], gc_obs[end])
    trΓ = tr(Γ)
    gvc_vec_discrete(n, m, gc_obs, res, Y, V, Σ, Γ, trΓ, vecd)
end

function gvc_vec_discrete(
    Γ::Matrix{T},
    vecd::Vector{D},
    max_value::Vector{T}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}
    n = length(vecd)
    m = 1
    res = Vector{T}(undef, n)  # simulated residual vector
    Y = Vector{T}(undef, n)    # vector of simulated outcome values transformed from residuals using hypothesized densities
    V = [Γ]
    Σ = ones(T, m)
    gc_obs = Vector{DiscreteUnivariateCopula}(undef, n)

    # form constants for the marginal density
    gc_obs[1] = marginal_pdf_constants(Γ, vecd[1])
    # generate y_1 
    Y[1] = discrete_rand(max_value[1], gc_obs[1], gc_obs[1].μ)
    
    for i in 2:length(vecd)
        # update residuals 1,..., i-1
        res[i-1] = update_res!(Y[i-1], res[i-1], gc_obs[i-1])
        # form constants for conditional density of i given 1, ..., i-1
        gc_obs[i] = conditional_pdf_constants(Γ, res, i, vecd[i])
        # generate y_i given y_1, ..., y_i-1
        Y[i] = discrete_rand(max_value[i], gc_obs[i], gc_obs[i].μ)
     end
    res[end] = update_res!(Y[end], res[end], gc_obs[end])
    trΓ = tr(Γ)
    gvc_vec_discrete(n, m, gc_obs, res, Y, V, Σ, Γ, trΓ, vecd)
end   

function update_res!(
    Y::T,
    res::T,
    gc_obs::Union{ContinuousUnivariateCopula{D, T}, DiscreteUnivariateCopula{D, T}}) where {T <: BlasReal, D}
    res = (Y - gc_obs.μ) * inv(sqrt(gc_obs.σ2))
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


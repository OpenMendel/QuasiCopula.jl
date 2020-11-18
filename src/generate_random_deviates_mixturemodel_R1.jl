using GLMCopula
########################### MARGINAL R_1 MIXTURE #####################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
### sampling R_1 from the marginal density of R_1 ~ f(y_1 = R_1) where f in Normal(0, 1), Gamma(1.0, 1.0)
# ####
# """
# GenR1
# GenR1()
# create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities, depending on the distribution.
# """
# struct genR1{T <: BlasReal, D <: Distributions.UnivariateDistribution} #<: MathProgBase.AbstractNLPEvaluator
#     # data
#     gvc_vector::GVCVec{T, D} # we will update gvc_vector.res[1]
#     # working arrays
#     term1::T
#     term2::T
# end

"""
genR1
genR1()
Let R1~f and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities. Just given without distribution of R1
"""
function genR1(
    gvc_vector::GLMCopula.GVCVec{T, D}
    ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    genR1!(gvc_vector, gvc_vector.vecd[1])
end


### GAUSSIAN BASE ### 
"""
genR1
genR1()
Let R1~N(0, 1) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities. Given d = distribution of R1.
"""
function genR1!(
    gvc_vector::GLMCopula.GVCVec{T, D},
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
end

#### Gamma Base ######
# """
# genR1
# genR1()
# Let R1~ Gamma(α = 1.0, θ = 1.0) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities. Given d = distribution of R1.
# """
# function genR1!(
#     gvc_vector::GLMCopula.GVCVec{T, D},
#     d::Distributions.Gamma{T}
#     ) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
#     term1 = 1 + 0.5 * gvc_vector.trΓ
#     term2 = 1 + 0.5 * tr(gvc_vector.Γ[2:end, 2:end])
#     α, θ = params(d) # shape and scale of gamma
#     # normalizing constant
#     c1 =  (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) * θ
#     c2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α)) * θ^2
#     # mixture probabilities
#     b0 = (inv(term1) * term2) + (inv(term1) * (0.5 * gvc_vector.Γ[1, 1] * (mean(d)^2/ var(d))))
#     b1 = -inv(term1) * gvc_vector.Γ[1, 1] * mean(d) * c1
#     b2 = inv(term1) * gvc_vector.Γ[1, 1] * inv(var(d)) * c2
#     mixture_probabilities = [b0, b1, b2]
#     mixture_model = MixtureModel(
#     [Gamma(α, θ), # Gamma(α, θ)
#     Gamma(α + 2, θ)], mixture_probabilities
#     )
#     gvc_vector.res[1] = rand(mixture_model)
# end

### we may not want to use mixture distribution just use the inverse cdf method.
"""
generate_R1_mixture_Normal
generate_R1_mixture_Normal()
Let R1~N(0, 1) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities.
"""
function generate_R1_mixture_Normal(d::Distributions.Distribution)
    csamplers = map(sampler, d.components)
    psampler = sampler(d.prior)
    random_deviate = csamplers[rand(psampler)]
    
    if typeof(random_deviate) == Normal{Float64}
        println("using standard normal")
        return rand(random_deviate)
    else
        println("if chi (3), one is positive and one is negative with equal probabilty")
        return rand([-1, 1]) * rand(random_deviate)
    end
end
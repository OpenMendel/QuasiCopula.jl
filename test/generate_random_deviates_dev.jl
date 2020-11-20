using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!

# If we are able to generate a residual vectorRfrom the (standardized) Gaussian copula model[1 +12tr(Γ)]−1(1√2π)ne−‖r‖222(1 +12rTΓr),2
# then Y = σ0R + μ is a desired sample from density (1).

# To generate a sample from the standardized Gaussian copula model,
# we first sample R1 from its marginal distribution

# we recognize it as a mixture of three distributions Normal(0,1), √χ2_(3) and − √χ2_(3)
# with mixing probabilities 1 + 0.5 ∑{i = 2 ^n} γii / 1+0.5∑{i=1^n} γii,
# 0.25γ_11 /1+0.5∑{i=1^n} γii and 0.25 γ_11/ 1+0.5∑{i=1^n} γii respectively.

#### using MixedModels to contstrut mixture of distributions


######## FIRST SIMULATE Y_1 ##############
#### USER SPECIFIES:
n_1 = 5 # 5 observations in the fist vector
Γ = ones(n_1, n_1)
σ_0 = 1.0 # per subject level noise
# covariates
p = 3
Random.seed!(12345)
y = rand(n_1)

Random.seed!(12345)
X = rand(n_1, 3)

# initialize beta and mu as eta for the normal density
Random.seed!(12345)
β = rand(3)
# link function is identity for the normal density
μ = X*β

res = y - μ 

#### mixture distribution exploration ##
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

# # Y_1 = σ_0 * R_1 + μ[1]

# # generalized conditional density


# # using our joint_density function

# function joint_density_value(density, res)
#    pdf_vector = pdf(density, res)
#    joint_pdf = 1.0
#    for i in 1:length(pdf_vector)
#        joint_pdf = pdf_vector[i] * joint_pdf
#    end
#    return joint_pdf
# end

# # this function will get the cross terms for s, and all the cross terms up to s if all = true; used in marginal density of i in S
# function crossterm_res(res, s, Γ; all = false)
#    results = []
#    if s == 1
#        return 0.0
#    elseif s > 1
#        if all == true
#            for i in 2:s
#                for j in 1:i - 1
#                    push!(results, res[i] * sum(res[j] * Γ[i, j]))
#                end
#            end
#        else
#            for j in 1:s - 1
#                push!(results, res[s] * sum(res[j] * Γ[s, j]))
#            end
#        end
#    end
#    return results
# end


# STEP 2-n: then generate remaining components sequentially
# from the conditional distributions R_k|R_1,...,R_k−1 for k = 2,...,n.
# however since the resulting conditional density is not a mixture of known pdf's we will get the cdf:

############################################################################################################################################################################
#################################################################### MARGINAL OF R_1 ########################################################################
############################################################################################################################################################################
s = 1
################## hardcoded numerical check
marginal_r1_hardcode = inv(1 + 0.5 * tr(Γ)) * (1/sqrt(2*pi)) * exp(-0.5 * res[s]^2) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s, Γ)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.3332318759238957

#####################################################################################################################################################################
########################################################## using marginal formula of R_S where S = {1, 2, ..., s} ######################################################
############################################################################################################################################################################
d = Normal()


s = 1 # first we check for the marginal of r_1 and then we can use this general formula for the joint density of R_s
################## break up into 3 terms according to recursive conditional density formula ####################################
term1 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])
@test term1 == 1 + 0.5 * tr(Γ)

marginal_r_s1 = inv(term1)[1] * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s, Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.33323187592389564

################################################################################################################################################################################################
################################################################################################################################################################################################
############################################################################## NOW USING Recursive Conditional Formula # check if recursive conditional formula will work as expected ##########################################################################################################################################################
############################################################################################## in the normalizing constant of the conditional density ##############################################################################################################

term2 = sum(crossterm_res(res, s, Γ)) # only for s = 1 do we have no cross terms
@test term2 == 0.0

term3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4613819816401454

@test [1 + 0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s) + sum(crossterm_res(res, s, Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1] == (term1 + term2 + term3)[1] # using the marginal of R_S and conditional of R1| nothing, respectively
@test marginal_r_s1 ≈ marginal_r1_hardcode
# 0.33323187592389564

d = Normal()
conditionalr1 = inv(term1) * pdf(d, res[s]) * (term1 + term2 + term3)[1]
# 0.33323187592389564

@test conditionalr1 ≈ marginal_r1_hardcode
# 0.33323187592389564

################################################################################################################################################
############################################################### s = 2 ########################################################################################################
########################################### Joint Density of S = {1, 2} for f(R_1, R_2) ######################################################################################
############################################################################################################################################################################

s = 2
################## hardcoded numerical check for f(R_1, R_2)
marginal_densities_s = [(1/sqrt(2*pi)) * exp(-0.5 * res[i]^2) for i in 1:s]
marginal_r1_r2_hardcode = inv(1 + 0.5 * tr(Γ)) * marginal_densities_s[1] * marginal_densities_s[2] * [1 + 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + res[1] * res[2] * Γ[1, 2] + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.10187907575145581

@test inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) == inv(1 + 0.5 * tr(Γ)) * marginal_densities_s[1] * marginal_densities_s[2] 
@test 1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s,  Γ; all = true)) == 1 + (0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2)) + res[1] * res[2] * Γ[1, 2]

### using marginal R_s where s = 2 formula:
marginal_r_s2 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s,  Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.10187907575145581

@test marginal_r_s2 ≈ marginal_r1_r2_hardcode # check marginal of R_1, R_2 fits in the marginal R_S formula

#########################################################################################################################################################################################################################################################
########################################### Conditional Density f(R_2 | R_1) ###################################################################################################################
######################################## using Bayes rule we have P(R_2 | R_1) = P(R_1, R_2) / P(R_1) ######################################################################
conditional_r2_r1_Bayes = marginal_r_s2 / marginal_r_s1
# 0.30573028306188577

########################################################################################################################################################################################################################
## for s = 2 check the conditional recursive formula ############################################################################################################################################

term1_s2 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])
# @test 1 + 0.5 * (diagonal_terms2) + cross_terms2 + 0.5 * remainder_trace2 == term1
# 3.0386180183598546

term2_s2 = sum(crossterm_res(res, s,  Γ))
# 0.19529686717517308

term3_s2 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.25308892099907326

conditional_r2_r1 = inv(term1_s2) * pdf(d, res[s]) * (term1_s2 + term2_s2 + term3_s2)
# 0.30573028306188565

@test conditional_r2_r1 ≈ conditional_r2_r1_Bayes

################################################################################ S = 3 ################################################################################################################################################################
##################################################################################### JOINT R_1, R_2, R_3 ########################################################################################################################

#############################################################################################################################################
# bayes rule for conditional distribution of R_3 given R_1 and R_2
s = 3
marginal_r_s3 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s,  Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.03661772700845761

conditional_r3_r12_Bayes = marginal_r_s3 / marginal_r_s2
# 0.3594234315375045

@test sum(Γ[i, i] * res[i]^2 for i in 1:s) ≈ Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2 + Γ[3, 3] * res[3]^2
@test sum(crossterm_res(res, s,  Γ; all = true)) == res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[3] * res[2] * Γ[3, 2]

marginal_densities_s = [(1/sqrt(2*pi)) * exp(-0.5 * res[i]^2) for i in 1:s]
marginal_r1_r2_r3_hardcode = inv(1 + 0.5 * tr(Γ)) * marginal_densities_s[1] * marginal_densities_s[2] *  marginal_densities_s[3] * [1 + 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + 0.5 * Γ[3, 3] * res[3]^2 + res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[3] * res[2] * Γ[3, 2] + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.03661772700845762

@test marginal_r_s3 ≈ marginal_r1_r2_r3_hardcode

########################################################################################################################################################################
## for s = 3 check the conditional recursive formula ############################################################################################################################################
term1_s3 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

t1 = 1 + 0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2) + res[1] * res[2] * Γ[1, 2] + 0.5 * tr(Γ[s:end, s:end])
@test term1_s3 == t1
# 2.9808259645359545

term2_s3 = sum(crossterm_res(res, s,  Γ))
# 0.27113634223316985

term3_s3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4617767542199328

conditional_r3_r12 = inv(term1_s3) * pdf(d, res[s]) * (term1_s3 + term2_s3 + term3_s3)
# 0.35942343153750456

@test conditional_r3_r12 ≈ conditional_r3_r12_Bayes
############  Now we have checked that both the bayes rule and the conditional formula are numerically the same. 

########################################################################### S = 4 ######################################################################################################################################################
#################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################################################################
s = 4
@test (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) == 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + 0.5 * Γ[3, 3] * res[3]^2 + 0.5 * Γ[4, 4] * res[4]^2
@test sum(crossterm_res(res, s,  Γ; all = true)) == res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[3] * res[2] * Γ[3, 2] + res[4] * res[1] * Γ[4, 1]  + res[4] * res[2] * Γ[4, 2] + res[4] * res[3] * Γ[4, 3] 

marginal_densities_s = [(1/sqrt(2*pi)) * exp(-0.5 * res[i]^2) for i in 1:s]
marginal_r1_r2_r3_r4_hardcode = inv(1 + 0.5 * tr(Γ)) * marginal_densities_s[1] * marginal_densities_s[2] *  marginal_densities_s[3] * marginal_densities_s[4] * [1 + 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + 0.5 * Γ[3, 3] * res[3]^2 + 0.5 * Γ[4, 4] * res[4]^2 +
 res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[3] * res[2] * Γ[3, 2] + res[4] * res[1] * Γ[4, 1]  + res[4] * res[2] * Γ[4, 2] + res[4] * res[3] * Γ[4, 3] + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.014029880582104255

marginal_r_s4 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s,  Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.014029880582104253


@test marginal_r_s4 ≈ marginal_r1_r2_r3_r4_hardcode


conditional_r4_r123_Bayes = marginal_r_s4 / marginal_r_s3
# 0.38314449662218975

########################################################################################################################################################################
## for s = 4 check the conditional recursive formula ############################################################################################################################################
term1_s4 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

t1 = 1 + 0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2 + Γ[3, 3] * res[3]^2) + res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[2] * res[3] * Γ[2, 3] + 0.5 * tr(Γ[s:end, s:end])
@test term1_s4 == t1
# 2.7901855525491914

term2_s4 = sum(crossterm_res(res, s,  Γ))
# 0.8531432405546888

term3_s4 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4617767542199328

conditional_r4_r123 = inv(term1_s4) * pdf(d, res[s]) * (term1_s4 + term2_s4 + term3_s4)
# -0.2697207363017294

@test conditional_r4_r123 ≈ conditional_r4_r123_Bayes
# 0.3831444966221897

######
s = 5
@test (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) == 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + 0.5 * Γ[3, 3] * res[3]^2 + 0.5 * Γ[4, 4] * res[4]^2 + 0.5 * Γ[5, 5] * res[5]^2
@test sum(crossterm_res(res, s, Γ; all = true)) ≈  res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[1] * res[5] * Γ[1, 5] + res[3] * res[2] * Γ[3, 2] + res[3] * res[5] * Γ[3, 5]  + res[4] * res[1] * Γ[4, 1]  + res[4] * res[2] * Γ[4, 2] + res[5] * res[2] * Γ[5, 2] + res[4] * res[3] * Γ[4, 3] + res[4] * res[5] * Γ[4, 5] 

marginal_densities_s = [(1/sqrt(2*pi)) * exp(-0.5 * res[i]^2) for i in 1:s]
marginal_r1_r2_r3_r4_r5_hardcode = inv(1 + 0.5 * tr(Γ)) * marginal_densities_s[1] * marginal_densities_s[2] *  marginal_densities_s[3] * marginal_densities_s[4] * marginal_densities_s[5] * [1 + 0.5 * Γ[1, 1] * res[1]^2 + 0.5 * Γ[2, 2] * res[2]^2 + 0.5 * Γ[3, 3] * res[3]^2 + 0.5 * Γ[4, 4] * res[4]^2 + 0.5 * Γ[5, 5] * res[5]^2 + res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[1] * res[5] * Γ[1, 5] + res[3] * res[2] * Γ[3, 2] + res[3] * res[5] * Γ[3, 5]  + res[4] * res[1] * Γ[4, 1]  + res[4] * res[2] * Γ[4, 2] + res[5] * res[2] * Γ[5, 2] + res[4] * res[3] * Γ[4, 3] + res[4] * res[5] * Γ[4, 5] + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.005057446050339986

marginal_r_s5 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s,  Γ; all = true)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.005057446050339985


@test marginal_r_s5 ≈ marginal_r1_r2_r3_r4_r5_hardcode


conditional_r5_r1234_Bayes = marginal_r_s5 / marginal_r_s4
# 0.3604767710418709

########################################################################################################################################################################
## for s = 5 check the conditional recursive formula ############################################################################################################################################
term1_s5 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

# check if all the ones till right before s-1
t1_5 = 1 + 0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2 + Γ[3, 3] * res[3]^2 + Γ[4, 4] * res[4]^2) + res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[1] * res[4] * Γ[1, 4] + res[2] * res[3] * Γ[2, 3] + res[2] * res[4] * Γ[2, 4] + res[3] * res[4] * Γ[3, 4] + 0.5 * tr(Γ[s:end, s:end])
@test term1_s5 == t1_5
# 2.7901855525491914

term2_s5 = sum(crossterm_res(res, s, Γ))
# 0.8531432405546888

term3_s5 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4617767542199328

conditional_r5_r1234 = inv(term1_s5) * pdf(d, res[s]) * (term1_s5 + term2_s5 + term3_s5)
# -0.2697207363017294

@test conditional_r5_r1234 ≈ conditional_r5_r1234_Bayes
# 0.36047677104187104

######################################################## everything works above here. ######################################################################################################################
## summary: 
# (1) I just got to marginal density of R_1 using both the recursive conditional density formula and the marginal density formula in the paper. 
# (2) Then I checked the recursion formula using bayes rule.

function recursive_conditional_densities(res, d; s = length(res))
    term1, term2, term3 = (0.0, 0.0, 0.0)
    R_s = res[s]
    if s == 1 # return marginal density of R_1
        ################## break up into 3 terms according to recursive conditional density formula ####################################
        term1 = 1 + 0.5 * tr(Γ)
        # @test term1 == 1 + 0.5 * tr(Γ)
        
        term2 = sum(crossterm_res(res, s, Γ)) # only for s = 1 do we have no cross terms
        # @test term2 == 0.0

        term3 = (0.5 * Γ[s, s] * (R_s^2 - 1))

        conditional_pdf = inv(term1) * pdf(d, R_s) * (term1 + term2 + term3)[1]
    elseif s > 1
        term1 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

        term2 = sum(crossterm_res(res, s,  Γ))

        term3 = (0.5 * Γ[s, s] * (R_s^2 - 1))

        conditional_pdf = inv(term1) * pdf(d, R_s) * (term1 + term2 + term3)
    end

    # conditional_cdf = inv(term1) * (term1 - 0.5 * Γ[s, s]) * cdf(d, R_s) - sum(res[j] * Γ[s, j] for j in 1:s-1) * conditional_pdf + 0.5 * Γ[s, s] * (0.5 + 0.5 * sign * cdf(Chisq(3), R_s^2))

    return term1, term2, term3, conditional_pdf #, conditional_cdf
end

@test recursive_conditional_densities(res, d; s = 1)[4] == 0.33323187592389564
@test recursive_conditional_densities(res, d; s = 2)[4] == 0.30573028306188565
@test recursive_conditional_densities(res, d; s = 3)[4] == 0.35942343153750456
@test recursive_conditional_densities(res, d; s = 4)[4] == 0.3831444966221897
@test recursive_conditional_densities(res, d; s = 5)[4] == 0.36047677104187104



### for the hard coded values of res we will test if the function and constructors work.
# specify distributions of the residuals
d = Normal()
vector_distributions = [d, d, d, d, d]

######## FIRST SIMULATE Y_1 ##############
#### USER SPECIFIES:
n_1 = 5 # 5 observations in the fist vector
Γ = ones(n_1, n_1)
σ_0 = 1.0 # per subject level noise
# covariates
p = 3
Random.seed!(12345)
y = rand(n_1)

Random.seed!(12345)
X = rand(n_1, 3)

# initialize beta and mu as eta for the normal density
Random.seed!(12345)
β = rand(3)
# link function is identity for the normal density
μ = X*β

res = y - μ 

gvc_vec = GVCVec(Γ, vector_distributions)
gvc_vec.res .= res

# conditional_terms_ken!(gvc_vec)
# # conditional pdf using the original version in kens notes
# ## check to see the implementation is the same at the evaluated res
# @test gvc_vec.conditional_pdf[1](res[1]) == 0.33323187592389564
# @test gvc_vec.conditional_pdf[2](res[2]) == 0.30573028306188565
# @test gvc_vec.conditional_pdf[3](res[3]) == 0.35942343153750456
# @test gvc_vec.conditional_pdf[4](res[4]) == 0.3831444966221897
# @test gvc_vec.conditional_pdf[5](res[5]) == 0.36047677104187104

## using the simplified version in huas notes
gvc_vec = GVCVec(Γ, vector_distributions)
gvc_vec.res .= res

conditional_terms!(gvc_vec)
conditional_pdf_cdf!(gvc_vec)
# pdf
## check to see the implementation is the same at the evaluated res
@test gvc_vec.conditional_pdf[1](res[1]) == 0.33323187592389564

# check these 
# @test gvc_vec.conditional_pdf[2](res[2]) == 0.30573028306188565
# @test gvc_vec.conditional_pdf[3](res[3]) == 0.35942343153750456
# @test gvc_vec.conditional_pdf[4](res[4]) == 0.3831444966221897
# @test gvc_vec.conditional_pdf[5](res[5]) == 0.36047677104187104


# cdf
# @test gvc_vec.conditional_cdf[1](res[1])

# function recursive_conditional_densities(gvc; s = length(res))
#     term1, term2, term3 = (0.0, 0.0, 0.0)
#     for i in 1:length(res)
#         term1 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

#         term2 = sum(crossterm_res(res, s, Γ))

#         term3 = (0.5 * Γ[s, s] * (R_s^2 - 1))

#         conditional_pdf = inv(term1) * pdf(d, R_s) * (term1 + term2 + term3)
#     end


# function f(x)
#     term1 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

#     term2 = sum(crossterm_res(res, s, Γ))

#     term3 = (0.5 * Γ[s, s] * (x^2 - 1))

#     conditional_density = inv(term1) * pdf(d, x) * (term1 + term2 + term3)
#     SVector(inv(term1) * (term1 - 0.5 * Γ[s, s]) * cdf(d, x) - sum(res[j] * Γ[s, j] for j in 1:s-1) * conditional_pdf + 0.5 * Γ[s, s] * (0.5 + 0.5 * sign * cdf(Chisq(3), x^2)))
# end

# # this function will get the cross terms for s, and all the cross terms up to s if all = true; used in marginal density of i in S
# function crossterm_res(res, s, Γ; all = false)
#     results = []
#     if s == 1
#         return 0.0
#     elseif s > 1
#         if all == true
#             for i in 2:s
#                 for j in 1:i - 1
#                     push!(results, res[i] * sum(res[j] * Γ[i, j]))
#                 end
#             end
#         else
#             for j in 1:s - 1
#                 push!(results, res[s] * sum(res[j] * Γ[s, j]))
#             end
#         end
#     end
#     return results
# end


# # now using the inverse CDF approach:
# function inverse_cdf_sample(res, d; s = length(res))
#     term1, term2, term3, conditional_pdf = recursive_conditional_pdf(res, d; s = s)

#     # (1) Draw U1 ~ uniform(0, 1)
#     U = rand(Uniform(0, 1))
#     # (2) and  use  nonlinear  root  finding  to locate R_2 such that the cdf of R_2 vector : F(R_2) = U
    

# end


# (2) and  use  nonlinear  root  finding  to locate R_2 such that the cdf of R_2 vector : F(R_2) = U

# #### old random stuff trying out inverse CDF approach as in kens original paper.
# function reorder_pmf(pmf, μ)
#     listofj = zeros(Int64, length(pmf))
#     k = floor.(μ[1])
#     reordered_pmf = zeros(length(pmf))
#     i = 1
#     j = k[1]
#     while(i < length(pmf) && j > 0)
#         listofj[i] = j
#         reordered_pmf[i] = pmf[j + 1]
#         if i%2 == 1
#             j = j + i
#             elseif i%2 == 0
#             j = j - i
#         end
#         i = i + 1
#     end
#     if j == 0
#         listofj[i] = 0
#         reordered_pmf[i] = pmf[1]
#         for s in i+1:length(pmf)
#             listofj[s] = s - 1
#             reordered_pmf[s] = pmf[s]
#             end
#         end
#     return(listofj, reordered_pmf)
# end

# pmf = [0.3; 0.2; 0.1; 0.4]
# μ = 2
# reordered_k, reordered_pmf = reorder_pmf(pmf, μ)

# # for a single mu, generate a single poisson.
# function generate_random_deviate(pmf, μ)
#     listofj, reordered_pmf = reorder_pmf(pmf, μ)
#     sample = rand() # generate x from uniform(0, 1)
#     random_deviate = listofj[1] # if the cumulative probability mass is less than the P(X = x_1) then leave it as 1
#     s = reordered_pmf[1]
#     for i in 2:length(pmf)
#         if sample < s
#             random_deviate = listofj[i - 1]
#             break
#         else
#             s = s + reordered_pmf[i]
#         end
#     end
#     return(random_deviate)
# end

# function generate_random_deviate(pmf, μ, n_reps)
#     random_deviate = zeros(eltype(μ), n_reps)
#     for l in 1:n_reps
#         random_deviate[l] = generate_random_deviate(pmf, μ)
#     end
#     return(random_deviate)
# end

# n = 10
# a = generate_random_deviate(pmf, μ, n)

# using Distributions
# const UnivariateDistribution{S<:ValueSupport} = Distribution{Univariate,S}

# const DiscreteUnivariateDistribution = Distribution{Univariate, Discrete}

# function pdf_tonda(ys, dist::UnivariateDistribution, μ, Γ, s)
# pdf_g = ones(1)
# for i in 1:s
#         pdf_g[1] *= pdf(dist(μ), ys)
# end
#     #build W
#     D = Diagonal(μ)
#     W = (ys .- μ) * transpose(ys .- μ) .- D
#     pdf_g = pdf_g * (1 + 0.5*tr(W*Γ))
#     return(pdf_g)
# end

# # returns a vector that sums to about 1 on the poisson density for s = # of random deviates in marginal distribution
# function pmf_vector(max, Γ, dist::UnivariateDistribution, μ, s, ken)
#     y = collect.(Iterators.product(ntuple(_ -> 0:max, s)...))[:]
#     y_sample = zeros(Int64, length(y), s)
#     for i in 1:s
#         y_sample[:, i] = collect(y[j][i] for j in 1:length(y))
#     end

#     pmf_value = zeros(size(y_sample)[1])

#     for k in 1:size(y_sample)[1]
#         if ken == true
#             pmf_value[k] = pdf_ken(y_sample[k, :], dist, μ, Γ, s)[1]
#         else
#             pmf_value[k] = pdf_tonda(y_sample[k, :], dist, μ, Γ, s)[1]
#             end
#     end
#     return(y_sample, pmf_value)
# end

# #when gamma is a constant theres no indexing solution is to feed gamma as a matrix.
# function pdf_ken(ys, dist::UnivariateDistribution, μ, Γ, s)
#     pdf_g = ones(1)
#     hardbracketken = zero(Float64)

#     for i in 1:s
#         pdf_g[1] *= pdf(dist(μ), ys)
#     end
#     #creates hard bracket
#     D = diagm(μ[1:s])
#     res = inv(D)*(ys .- μ[1:s])
#     Γs = Γ[1:s, 1:s]
#     if s == 1
#         Γt = 0.0
#     else
#         Γt = Γ[s+1:end, s+1:end]
#     end

#     hardbracketken = (1 + 0.5(transpose(res)*Γs*res) + 0.5*tr(Γt))

#     if s == 1
#         pdf_g = inv(1 + 0.5(tr(Γ[1]))) * pdf_g * hardbracketken
#     else
#         pdf_g = inv(1 + 0.5(tr(Γ))) * pdf_g * hardbracketken
#     end
#     return(pdf_g)
# end

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

# create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities:
mixing_probabilities = [(1 + 0.5 * tr(Γ[2:end, 2:end])) / (1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5*tr(Γ))]

D_1 = MixtureModel(
   [Normal(0.0, 1.0),
   Chi(3.0),
   Chi(3.0)], [(1 + 0.5 * tr(Γ[2:end, 2:end])) / (1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5*tr(Γ))]
   )

R_1 = rand(D_1)

Y_1 = σ_0 * R_1 + μ[1]

# STEP 2-n: then generate remaining components sequentially
# from the conditional distributions R_k|R_1,...,R_k−1 for k = 2,...,n.
# however since the resulting conditional density is not a mixture of known pdf's we will get the cdf:

############################################################################################################################################################################
#################################################################### MARGINAL OF R_1 ########################################################################
############################################################################################################################################################################
s = 1
################## hardcoded numerical check
marginal_r1_hardcode = inv(1 + 0.5 * tr(Γ)) * (1/sqrt(2*pi)) * exp(-0.5 * res[s]^2) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.3332318759238957

#####################################################################################################################################################################
########################################################## using marginal formula of R_S where S = {1, 2, ..., s} ######################################################
############################################################################################################################################################################
d = Normal()

# using our joint_density function

function joint_density_value(density, res)
    pdf_vector = pdf(density, res)
    joint_pdf = 1.0
    for i in 1:length(pdf_vector)
        joint_pdf = pdf_vector[i] * joint_pdf
    end
    return joint_pdf
end

# this function will get ALL the cross terms including i = s; used in marginal density of i in S
function crossterm_res(res, s)
    results = []
    if s == 1
        return 0.0
    elseif s > 1
        for i in 2:s
            for j in 1:i - 1
                push!(results, res[i] * sum(res[j] * Γ[i, j]))
            end
        end
        return results
    end
end

s = 1 # first we check for the marginal of r_1 and then we can use this general formula for the joint density of R_s
################## break up into 3 terms according to recursive conditional density formula ####################################
term1 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])
@test term1 == 1 + 0.5 * tr(Γ)

marginal_r_s1 = inv(term1)[1] * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.33323187592389564

@test [1 + 0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s) + 0.5 * sum(crossterm_res(res, s)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1] == (term1 + term2 + term3)[1] # using the marginal of R_S and conditional of R1| nothing, respectively
@test marginal_r_s1 ≈ marginal_r1_hardcode
# 0.33323187592389564

################################################################################################################################################################################################
################################################################################################################################################################################################
############################################################################## NOW USING Recursive Conditional Formula # check if recursive conditional formula will work as expected ##########################################################################################################################################################
############################################################################################## in the normalizing constant of the conditional density ##############################################################################################################

term2 = sum(crossterm_res(res, s)) # only for s = 1 do we have no cross terms
@test term2 == 0.0

term3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4613819816401454

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
@test 1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s)) == 1 + (0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2)) + res[1] * res[2] * Γ[1, 2]

### using marginal R_s where s = 2 formula:
marginal_r_s2 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
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

term2_s2 = sum(crossterm_res(res, s)[s-1:end])
# 0.19529686717517308

term3_s2 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.25308892099907326

conditional_r2_r1 = inv(term1_s2) * pdf(d, res[s]) * (term1_s2 + term2_s2 + term3_s2)
# 0.30573028306188565

@test conditional_r2_r1 ≈ conditional_r2_r1_Bayes
######################################################## everything works above here. ######################################################################################################################
## summary: 
# (1) I just got to marginal density of R_1 using both the recursive conditional density formula and the marginal density formula in the paper. 
# (2) Then I checked the recursion formula using bayes rule.

################################################################################ S = 3 ################################################################################################################################################################
##################################################################################### JOINT R_1, R_2, R_3 ########################################################################################################################


# term1_s3 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

# term2_s3 = sum(crossterm_res(res, s)) # keep all terms 
# # 0.466433209408343

# term3_s3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# # -0.4617767542199328

# conditional_r3_r21 = inv(term1) * pdf(d, res[s]) * (term1 + term2 + term3)
# # 0.38458099502086984

#############################################################################################################################################
# bayes rule for conditional distribution of R_3 given R_1 and R_2
s = 3
marginal_r_s3 = inv(1 + 0.5 * tr(Γ)) * joint_density_value(d, res[1:s]) * [1 + (0.5 * sum(Γ[i, i] * res[i]^2 for i in 1:s)) + sum(crossterm_res(res, s)) + 0.5 * tr(Γ[s+1:end, s+1:end])][1]
# 0.03661772700845761

conditional_r3_r12_Bayes = marginal_r_s3 / marginal_r_s2
# 0.3594234315375045

diagonal_terms = sum(Γ[i, i] * res[i]^2 for i in 1:s)
cross_terms = sum(crossterm_res(res, s))
remainder_trace = tr(Γ[s+1:end, s+1:end])

@test diagonal_terms ≈ Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2 + Γ[3, 3] * res[3]^2
@test cross_terms == res[1] * res[2] * Γ[1, 2] + res[1] * res[3] * Γ[1, 3] + res[3] * res[2] * Γ[3, 2]

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

term2_s3 = sum(crossterm_res(res, s)[s - 1:end])
# 0.27113634223316985

term3_s3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# -0.4617767542199328

conditional_r3_r12 = inv(term1_s3) * pdf(d, res[s]) * (term1_s3 + term2_s3 + term3_s3)
# 0.35942343153750456

@test conditional_r3_r12 ≈ conditional_r3_r12_Bayes
############  Now we have checked that both the bayes rule and the conditional formula are numerically the same. 

# now using the inverse CDF approach:
# (1) Draw U1 ~ uniform(0, 1)
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

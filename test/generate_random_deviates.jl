using DataFrames, MixedModels, Random, GLMCopula, GLM
using ForwardDiff, Test, LinearAlgebra
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
Γ =  Matrix(I, n_1, n_1)
σ_0 = 0.1 # lets use a small per subject level noise
# covariates
p = 3
X = rand(n_1, 3)

# initialize beta and mu as eta for the normal density
β = rand(3)
# link function is identity for the normal density
μ = X*β

# create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities:
mixing_probabilities = [(1 + 0.5 * tr(Γ[2:end, 2:end])) / (1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5*tr(Γ))]

D_1 = MixtureModel(
   [Normal(0.0, 1.0),
   Chi(3.0),
   Chi(3.0)], [(1 + 0.5 * tr(Γ[2:end, 2:end])) / (1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5 * tr(Γ)), (0.25 * Γ[1, 1])/(1 + 0.5*tr(Γ))]
   )

R_1 = zeros(n_1)

rand!(D_1, R_1)

Y_1 = σ_0 * R_1 + μ

# STEP 2-n: then generate remaining components sequentially
# from the conditional distributions R_k|R_1,...,R_k−1 for k = 2,...,n.

# using Bayes rule we have P(R_2 | R_1) = P(R_1, R_2) / P(R_1)
# however since the resulting conditional density is not a mixture of known pdf's we will get the cdf:


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

using DataFrames, MixedModels, Random, GLMCopula, GLM, Roots
using ForwardDiff, Test, LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!

######## FIRST SIMULATE Y_1 ##############
#### USER SPECIFIES:
n_1 = 5 # 5 observations in the fist vector
Γ = ones(n_1, n_1)
d = Normal()
vector_distributions = [d, d, d, d, d]

gvc_vec = GVCVec(Γ, vector_distributions)

conditional_terms!(gvc_vec)

conditional_pdf_cdf!(gvc_vec)

generate_res_vec!(gvc_vec)

@test gvc_vec.res[1] == 0.26778824840862536

####
##### inverse cdf method using Roots.jl ###
# using Roots
## taking out whats inside the function ### 
i = 1
conditional_terms!(gvc_vec) 
conditional_pdf_cdf!(gvc_vec)
Random.seed!(1234)
gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
gvc_vec.res[i] = find_zero(F_r, (-50, 50), Bisection())

i = 2 
conditional_terms!(gvc_vec) 
conditional_pdf_cdf!(gvc_vec)
Random.seed!(1234)
gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
gvc_vec.res[i] = find_zero(F_r, (-50, 50), Bisection())

i = 3
conditional_terms!(gvc_vec) 
conditional_pdf_cdf!(gvc_vec)
Random.seed!(1234)
gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
gvc_vec.res[i] = find_zero(F_r, (-50, 50), Bisection())

i = 4
conditional_terms!(gvc_vec) 
conditional_pdf_cdf!(gvc_vec)
Random.seed!(1234)
gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
gvc_vec.res[i] = find_zero(F_r, (-50, 50), Bisection())

i = 5
conditional_terms!(gvc_vec) 
conditional_pdf_cdf!(gvc_vec)
Random.seed!(1234)
gvc_vec.storage_n[i] = rand(Uniform(0, 1)) # simulate uniform random variable U1~uni(0, 1)
F_r(x) = gvc_vec.conditional_cdf[i](x)[1] - gvc_vec.storage_n[i] # make new function that subtracts the uniform value we simulated from conditonal cdf
gvc_vec.res[i] = find_zero(F_r, (-50, 50), Bisection())



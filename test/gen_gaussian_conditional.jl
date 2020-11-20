using DataFrames, Random, GLMCopula, GLM, Roots
using LinearAlgebra, Distributions
using LinearAlgebra: BlasReal, copytri!

Random.seed!(1234)
######## FIRST SIMULATE Y_1 ##############
#### USER SPECIFIES:
n_1 = 5 # 5 observations in the fist vector
Γ = ones(n_1, n_1)
d = Normal()
vector_distributions = [d, d, d, d, d]

gvc_vec = GVCVec(Γ, vector_distributions)

####
##### using inverse cdf method to fill up residual vector using Roots.jl ###
generate_res_vec!(gvc_vec)

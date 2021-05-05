using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, StatsBase
using LinearAlgebra: BlasReal, copytri!
# set up the component densities
mean_rate = 5
n = 3
d1 = Poisson(mean_rate)
d2 = Poisson(mean_rate + 2)
d3 = Poisson(mean_rate)

# vector of distributions
vecd = [d1, d2, d3]

# specify the covariance matrix  
vc1 = 0.1
Γ = vc1 * ones(n, n)

# create the tri-variate distribution object 
gc_vec = NonMixedMultivariateDistribution(vecd, Γ)

# pre allocate for sampling nsample times from conditional density
Random.seed!(1234)
nsample = 100_000
Y_nsample = simulate_nobs_independent_vectors(gc_vec, nsample)
Y_1 = [Y_nsample[i, 1][1] for i in 1:nsample]
Y_2 = [Y_nsample[i, 1][2] for i in 1:nsample]
Y_3 = [Y_nsample[i, 1][3] for i in 1:nsample]

# using ours
function covariance_matrix(gc_vec)
    n = length(gc_vec.gc_obs)
    Covariance = zeros(n, n)
    for i in 1:n 
        Covariance[i, i] = GLMCopula.var(gc_vec.gc_obs[i])
    end
    Covariance[1, 2] = GLMCopula.cov(gc_vec, 1, 2)
    Covariance[1, 3] = GLMCopula.cov(gc_vec, 1, 3)
    Covariance[2, 3] = GLMCopula.cov(gc_vec, 2, 3)
    Covariance[2, 1] = Covariance[1, 2]
    Covariance[3, 2] = Covariance[2, 3]
    Covariance[3, 1] = Covariance[1, 3]
    Covariance
end

#### check the 3 by 3 covariance 
Matrix_Y = hcat(Y_1, Y_2, Y_3)
@show empirical_covariance = scattermat(Matrix_Y) ./ nsample

# using ours 
@show theoretical_covariance = covariance_matrix(gc_vec)

@test theoretical_covariance[1, 2] == theoretical_covariance[2, 3]
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
Y = zeros(nsample, n)
for j in 1:n
    Y[:, j] = [Y_nsample[i][j] for i in 1:nsample]
end
Y

#### check the 3 by 3 covariance (Sample covariance)
empirical_covariance = scattermat(Y) ./ nsample

# using ours covariance
theoretical_covariance = covariance_matrix(gc_vec)

@test norm(empirical_covariance - theoretical_covariance) < 1

@test theoretical_covariance[1, 2] == theoretical_covariance[2, 3]

# using ours correlation
theoretical_corr = correlation_matrix(gc_vec)

@test theoretical_corr[1, 2] == theoretical_corr[2, 3]

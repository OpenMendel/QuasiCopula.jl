module PkgTest
# example datasets
include("dyestuff.jl")
include("sleepstudy.jl")

# simulating data
include("rand_continuous.jl")
include("rand_discrete.jl")
include("rand_multivariate.jl")

# # # # analyze simulated data
include("bivariate_poisson_test.jl")
include("bivariate_normal_test.jl")

# # using mixed models to simulate data
include("simulate_logistic_mixedmodels.jl")
include("simulate_poisson_mixedmodels.jl")
end

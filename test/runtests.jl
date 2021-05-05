module PkgTest
# # # # simulating data
include("rand_continuous.jl")
include("rand_discrete.jl")

# # # # simulating random vector using conditional densities
include("rand_multivariate.jl")

# # # # analyze simulated data
include("bivariate_poisson_test.jl")
include("bivariate_normal_test.jl")

# # # trivariate poisson test
include("trivariate_poisson.jl")

# # # using mixed models to simulate data + checking gradient and hessian pieces using forward diff
include("simulate_logistic_mixedmodels.jl")
include("simulate_poisson_mixedmodels.jl")

# # example datasets
include("dyestuff.jl")
# include("sleepstudy.jl")
end

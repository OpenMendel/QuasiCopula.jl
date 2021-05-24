module PkgTest
# # # # simulating data
include("simulation/rand_continuous.jl")
include("simulation/rand_discrete.jl")

# # # # # simulating random vector using conditional densities
include("simulation/rand_multivariate.jl")

# # # # analyze simulated data
include("simulation/bivariate_poisson_test.jl")
include("simulation/bivariate_normal_test.jl")

# # # # trivariate poisson test
include("simulation/trivariate_poisson.jl")

# # # # using mixed models to simulate data + checking gradient and hessian pieces using forward diff
include("estimation/simulate_logistic_mixedmodels.jl")
include("estimation/simulate_poisson_mixedmodels.jl")

# # example datasets
include("examples/dyestuff.jl")
end

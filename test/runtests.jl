module PkgTest
# include("generate_random_deviates_dev.jl")
# include("gen_random_dev_constructors.jl")
# include("gen_gaussian_conditional.jl")
include("rand_continuous.jl")
include("rand_discrete.jl")
include("rand_multivariate.jl")
include("bivariate_poisson_test.jl")
include("bivariate_normal_test.jl")
# include("dyestuff.jl")
# include("simulate_logistic_mixedmodels.jl")
# include("simulate_poisson_mixedmodels.jl")
end

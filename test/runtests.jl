module PkgTest
# # # # # # # simulating data
# include("simulation/generate_random_deviates/rand_continuous.jl")
# include("simulation/generate_random_deviates/rand_discrete.jl")

# # # # # # # # simulating random vector using conditional densities + check covariances/correlation on multivariate poisson
# include("simulation/generate_random_deviates/rand_multivariate.jl")
# include("simulation/multivariate_poisson/trivariate_poisson.jl")

# # # # # # # # analyze simulated data for Poisson base distribution
# include("simulation/multivariate_poisson/bivariate_poisson_test.jl")
# include("simulation/multivariate_poisson/multivariate_n50_poisson_test.jl")

# # # # # # # # analyze simulated data for Bernoulli base distribution
# include("simulation/multivariate_logistic/bivariate_logistic_test.jl")
# include("simulation/multivariate_logistic/multivariate_n50_logistic_test.jl")

# # inference/ confidence intervals
# include("estimation/ar/asymptotic_ci_AR.jl")
# include("estimation/vcm/asymptotic_ci_VCM.jl")

# estimate simulated vcm covariance structure
# # include("estimation/vcm/simulate_logistic_mixedmodels.jl")
# # include("estimation/vcm/simulate_poisson_mixedmodels.jl")
# # include("estimation/vcm/mse_logistic_vcm/two_vc_QC/mse.jl")

# mean squared error 
# AR 
include("estimation/ar/mse_logistic_ar/mse.jl")
include("estimation/ar/mse_poisson_ar/mse.jl")
# one vc vs glmm
include("estimation/vcm/mse_poisson_vcm/one_vc_QC_vs_GLMM/mse_poisson_vs_glmm.jl")
include("estimation/vcm/mse_logistic_vcm/one_vc_QC_vs_GLMM/mse_logistic_vs_glmm.jl")
# two vcm
include("estimation/vcm/mse_poisson_vcm/two_vc_QC/mse.jl")
include("estimation/vcm/mse_logistic_vcm/two_vc_QC/mse.jl")

# profiling and benchmarking
# include("perf.jl")

end
module PkgTest
using Test
@testset "Generating Random Deviates" begin
    include("unit_test/generate_random_deviates/rand_continuous.jl")
    include("unit_test/generate_random_deviates/rand_discrete.jl")
    include("unit_test/generate_random_deviates/rand_multivariate.jl")
end

# VCM
@testset "VCM Covariance" begin
    include("unit_test/VCM/singlerun_bernoulliVCM.jl")
    include("unit_test/VCM/singlerun_nbVCM.jl")
    include("unit_test/VCM/singlerun_normalVCM.jl")
    include("unit_test/VCM/singlerun_poissonVCM.jl")
    include("unit_test/VCM/singlerun_poisson_bernoulli_mixedVCM.jl")
end

### AR
@testset "AR Covariance" begin
    include("unit_test/AR/singlerun_bernoulliAR.jl")
    include("unit_test/AR/singlerun_nbAR.jl")
    include("unit_test/AR/singlerun_normalAR.jl")
    include("unit_test/AR/singlerun_poissonAR.jl")
    include("unit_test/AR/Poisson_AR_turbo_macro_LoopVectorization.jl")
    include("unit_test/AR/NB_AR_turbo_macro_LoopVectorization.jl")
end

### CS
@testset "CS Covariance" begin
    include("unit_test/CS/singlerun_bernoulliCS.jl")
    include("unit_test/CS/singlerun_nbCS.jl")
    include("unit_test/CS/singlerun_normalCS.jl")
    include("unit_test/CS/singlerun_poissonCS.jl")
end

## two vc
## poisson
# include("unit_test/multivariate_poisson/trivariate_poisson.jl")
# include("unit_test/multivariate_poisson/bivariate_poisson_test.jl")
# include("unit_test/multivariate_poisson/multivariate_n50_poisson_test.jl")
# # inference/ confidence intervals
# include("unit_test/multivariate_poisson/asymptotic_ci_AR.jl")
# include("unit_test/multivariate_poisson/asymptotic_ci_VCM.jl")
# # bernoulli
# include("unit_test/multivariate_logistic/bivariate_logistic_test.jl")
# include("unit_test/multivariate_logistic/multivariate_n50_logistic_test.jl")

# profiling and benchmarking
# include("perf.jl")

end

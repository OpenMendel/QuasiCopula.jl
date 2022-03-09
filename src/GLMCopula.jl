module GLMCopula
using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Distributions, StatsFuns, ToeplitzMatrices
using LoopVectorization, DataFrames
using LinearAlgebra: BlasReal, copytri!
using SpecialFunctions
using FFTW
@reexport using Ipopt
import Base: show
export fit!, update_θ_jensen!, init_β!, initialize_model!, loglikelihood!, standardize_res!, std_res_differential!
export update_res!, update_θ!
export update_∇θ!, update_Hθ! # update gradient and hessian of variance components
export glm_regress_jl, glm_regress_model, glm_score_statistic!  # these are to initialize our model
export component_loglikelihood, glm_gradient, hessian_glm
export GLMCopulaVCObs, GLMCopulaVCModel
export Poisson_Bernoulli_VCObs, Poisson_Bernoulli_VCModel

include("parameter_estimation/gaussian_CS.jl")
include("parameter_estimation/NBCopulaCS.jl")
include("parameter_estimation/GLM_CS.jl")
include("parameter_estimation/bivariate_mixed.jl")
include("parameter_estimation/GLM_VC.jl")
include("parameter_estimation/gaussian_VC.jl")
include("parameter_estimation/gaussian_AR.jl")
include("parameter_estimation/NBCopulaAR.jl")
include("parameter_estimation/NBCopulaVC.jl")
include("parameter_estimation/GLM_AR.jl")
include("generate_random_deviates/discrete_rand.jl")
include("generate_random_deviates/continuous_rand.jl")
include("generate_random_deviates/multivariate_rand.jl")
include("parameter_estimation/update_sigma_and_residuals.jl")
include("parameter_estimation/initialize_model.jl")
include("parameter_estimation/component_loglikelihood.jl")
include("parameter_estimation/gradient_hessian.jl")
include("parameter_estimation/fit_glm_ar_cs.jl")
include("parameter_estimation/fit_gaussian_ar_cs.jl")
include("parameter_estimation/fit_glm_vc.jl")
include("parameter_estimation/fit_nb.jl")
include("parameter_estimation/inference_ci.jl")
include("parameter_estimation/fit_gaussian_vc.jl")
include("model_interface/AR_interface.jl")
include("model_interface/CS_interface.jl")
include("model_interface/VC_interface.jl")
include("model_interface/show_io.jl")
end # module

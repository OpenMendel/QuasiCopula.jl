module GLMCopula
using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Distributions, StatsFuns, Statistics, StatsBase, ToeplitzMatrices
using LoopVectorization, DataFrames
using LinearAlgebra: BlasReal, copytri!
using SpecialFunctions, Random
using FFTW
@reexport using Ipopt
@reexport using NLopt

export fit!, update_θ_jensen!, init_β!, initialize_model!, loglikelihood!, standardize_res!, std_res_differential!
export update_res!, update_θ!
export update_∇θ!, update_Hθ! # update gradient and hessian of variance components
export glm_regress_jl, glm_regress_model, glm_score_statistic!  # these are to initialize our model
export component_loglikelihood, glm_gradient, hessian_glm
export GLMCopulaVCObs, GLMCopulaVCModel
export Poisson_Bernoulli_VCObs, Poisson_Bernoulli_VCModel

mutable struct GLMCopulaVCObs{T <: BlasReal, D, Link}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    n::Int
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇θ::Vector{T}   # gradient wrt θ
    Hβ::Matrix{T}   # Hessian wrt β
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    storage_p1::Vector{T}
    storage_p2::Vector{T}
    storage_np::Matrix{T}
    storage_pp::Matrix{T}
    added_term_numerator::Matrix{T}
    added_term2::Matrix{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    link::Link      # link function ()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function GLMCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    m1        = Vector{T}(undef, m)
    m2        = Vector{T}(undef, m)
    storage_p1 = Vector{T}(undef, p)
    storage_p2 = Vector{T}(undef, p)
    storage_np = Matrix{T}(undef, n, p)
    storage_pp = Matrix{T}(undef, p, p)
    added_term_numerator = Matrix{T}(undef, n, p)
    added_term2 = Matrix{T}(undef, p, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    wt = Vector{T}(undef, n)
    fill!(wt, one(T))
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    GLMCopulaVCObs{T, D, Link}(y, X, V, n, p, m, ∇β, ∇resβ, ∇τ, ∇θ, Hβ, Hθ,
        Hτ, res, t, q, xtx, storage_n, m1, m2, storage_p1, storage_p2, storage_np,
        storage_pp, added_term_numerator, added_term2, η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
GLMCopulaVCModel
GLMCopulaVCModel(gcs)
Gaussian copula variance component model, which contains a vector of
`GLMCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GLMCopulaVCModel{T <: BlasReal, D, Link} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GLMCopulaVCObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    θ::Vector{T}    # m-vector: [θ12, ..., θm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇θ::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hθ::Matrix{T}
    Hτ::Matrix{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    TR::Matrix{T}         # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}         # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_θ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
end

function GLMCopulaVCModel(gcs::Vector{GLMCopulaVCObs{T, D, Link}}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β       = Vector{T}(undef, p)
    τ       = [1.0]
    θ       = Vector{T}(undef, m)
    ∇β      = Vector{T}(undef, p)
    ∇τ      = Vector{T}(undef, 1)
    ∇θ      = Vector{T}(undef, m)
    XtX     = zeros(T, p, p) # sum_i xi'xi
    Hβ      = Matrix{T}(undef, p, p)
    Hθ      = Matrix{T}(undef, m, m)
    Hτ      = Matrix{T}(undef, 1, 1)
    Ainv    = zeros(T, p + m, p + m)
    Aevec   = zeros(T, p + m, p + m)
    M       = zeros(T, p + m, p + m)
    vcov    = zeros(T, p + m, p + m)
    ψ       = Vector{T}(undef, p + m)
    TR      = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal  = 0.0
    ntotal  = 0.0
    d       = Vector{D}(undef, n)
    link    = Vector{Link}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
        d[i] = gcs[i].d
        link[i] = gcs[i].link
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_θ = Vector{T}(undef, m)
    GLMCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, p, m, β, τ, θ,
        ∇β, ∇τ, ∇θ, XtX, Hβ, Hθ, Hτ, Ainv, Aevec, M, vcov, ψ, TR, QF,
        storage_n, storage_m, storage_θ, d, link)
end

#
# function GLMCopulaVCModel(gcs::Vector{GLMCopulaVCObs{T, D, Link}}) where {T <: BlasReal, D<:Normal, Link}
#     n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
#     β       = Vector{T}(undef, p)
#     τ       = [1.0]
#     θ       = Vector{T}(undef, m)
#     θ       = Vector{T}(undef, m + p + 1)
#     ∇β      = Vector{T}(undef, p)
#     ∇τ      = Vector{T}(undef, 1)
#     ∇θ      = Vector{T}(undef, m)
#     XtX     = zeros(T, p, p) # sum_i xi'xi
#     Hβ      = Matrix{T}(undef, p, p)
#     Hθ      = Matrix{T}(undef, m, m)
#     Hτ      = Matrix{T}(undef, 1, 1)
#     Ainv    = zeros(T, p + m + 1, p + m + 1)
#     Aevec   = zeros(T, p + m + 1, p + m + 1)
#     M       = zeros(T, p + m + 1, p + m + 1)
#     vcov    = zeros(T, p + m + 1, p + m + 1)
#     ψ       = Vector{T}(undef, p + m + 1)
#     TR      = Matrix{T}(undef, n, m) # collect trace terms
#     Ytotal  = 0.0
#     ntotal  = 0.0
#     d       = Vector{D}(undef, n)
#     link    = Vector{Link}(undef, n)
#     for i in eachindex(gcs)
#         ntotal  += length(gcs[i].y)
#         Ytotal  += sum(gcs[i].y)
#         BLAS.axpy!(one(T), gcs[i].xtx, XtX)
#         TR[i, :] = gcs[i].t
#         d[i] = gcs[i].d
#         link[i] = gcs[i].link
#     end
#     QF        = Matrix{T}(undef, n, m)
#     storage_n = Vector{T}(undef, n)
#     storage_m = Vector{T}(undef, m)
#     storage_θ = Vector{T}(undef, m)
#     GLMCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, p, m, β, τ, θ, θ,
#         ∇β, ∇τ, ∇θ, ∇θ, XtX, Hβ, Hθ, Hτ, Ainv, Aevec, M, vcov, ψ, TR, QF,
#         storage_n, storage_m, storage_θ, d, link)
# end

include("parameter_estimation/gaussian_CS.jl")
include("parameter_estimation/NBCopulaCS.jl")
include("parameter_estimation/compound_symmetric.jl")
include("parameter_estimation/bivariate_mixed.jl")
include("parameter_estimation/gaussian_VC.jl")
include("parameter_estimation/gaussian_AR.jl")
include("parameter_estimation/NBCopulaAR.jl")
include("parameter_estimation/NBCopulaVC.jl")
include("parameter_estimation/autoregressive.jl")
include("generate_random_deviates/discrete_rand.jl")
include("generate_random_deviates/continuous_rand.jl")
include("generate_random_deviates/multivariate_rand.jl")
include("parameter_estimation/initialize_model.jl")
include("parameter_estimation/splitting_loglikelihood.jl")
include("parameter_estimation/gradient_hessian.jl")
include("parameter_estimation/update_sigma_and_residuals.jl")
include("parameter_estimation/fit_ar.jl")
include("parameter_estimation/fit_gaussian_ar.jl")
include("parameter_estimation/fit_new.jl") # only initializes using MM-algorithm does joint estimation using newton after
include("parameter_estimation/fit_nb.jl")
include("parameter_estimation/inference_ci.jl")
include("parameter_estimation/fit_newton_normal.jl")
include("model_interface/AR_interface.jl")
include("model_interface/CS_interface.jl")
include("model_interface/VC_interface.jl")
end # module

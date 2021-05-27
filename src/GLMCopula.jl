module GLMCopula


using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Distributions, StatsFuns
using LinearAlgebra: BlasReal, copytri!
@reexport using Ipopt
@reexport using NLopt

export fit2!, update_Σ_jensen!, init_β!, initialize_model!, loglikelihood!, standardize_res!, std_res_differential!
export update_res!, update_Σ!

export update_∇Σ! # update gradient of variance components

export glm_regress_jl, glm_regress_model, glm_score_statistic  # these are to initialize our model

export copula_loglikelihood, copula_loglikelihood_addendum, component_loglikelihood
export loglikelihood3!
export copula_gradient, glm_gradient, copula_gradient_addendum
export hessian_glm, hessian_copula_addendum, copula_hessian

export GLMCopulaVCObs, GLMCopulaVCModel
export GaussianCopulaVCObs, GaussianCopulaVCModel

export GaussianCopulaLMMObs, GaussianCopulaLMMModel

"""
GaussianCopulaVCObs
GaussianCopulaVCObs(ys, X, V)
A realization of Gaussian copula variance component data.
"""
struct GaussianCopulaVCObs{T <: BlasReal, D}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    storage_n::Vector{T}
    storage_p::Vector{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function GaussianCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D
    ) where {T <: BlasReal, D}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    wt = ones(T, n)
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    GaussianCopulaVCObs{T, D}(y, X, V, ∇β, ∇resβ, ∇τ, ∇Σ, Hβ,
        Hτ, res, xtx, t, q, storage_n, storage_p, η, μ, varμ, dμ, d, wt, w1, w2)
end

"""
GaussianCopulaVCModel
GaussianCopulaVCModel(gcs)
Gaussian copula variance component model, which contains a vector of
`GaussianCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaVCModel{T <: BlasReal, D} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaVCObs{T, D}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    Hβ::Matrix{T}    # Hessian from all observations
    Hτ::Matrix{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::D
end

function GaussianCopulaVCModel(gcs::Vector{GaussianCopulaVCObs{T, D}}) where {T <: BlasReal, D}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = ones(T)
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    GaussianCopulaVCModel{T, D}(gcs, Ytotal, ntotal, p, m, β, τ, Σ,
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, XtX, TR, QF,
        storage_n, storage_m, storage_Σ, gcs[1].d)
end

struct GLMCopulaVCObs{T <: BlasReal, D, Link}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β
    HΣ::Matrix{T}   # Hessian wrt variance components Σ
    HΣ1::Matrix{T}   # Hessian wrt variance components Σ term 1
    HΣ2::Matrix{T}   # Hessian wrt variance components Σ term 2
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
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
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    HΣ  = Matrix{T}(undef, m, m)
    HΣ1  = Matrix{T}(undef, m, m)
    HΣ2  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
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
    GLMCopulaVCObs{T, D, Link}(y, X, V, ∇β, ∇resβ, ∇τ, ∇Σ, Hβ, HΣ, HΣ1, HΣ2,
        Hτ, res, t, q, xtx, storage_n, storage_p1, storage_p2, storage_np, storage_pp, added_term_numerator, added_term2, η, μ, varμ, dμ, d, link, wt, w1, w2)
end

# function fill_in_weights(
#     wt::Vector{T},
#     d::D
#     ) where {T <: BlasReal, D<:Binomial}
#     fill!(wt, d.n)
#     nothing
# end

# function fill_in_weights(
#     wt::Vector{T},
#     d::D
#     ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli, Normal, NegativeBinomial}}
#     fill!(wt, one(T))
#     nothing
# end

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
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    ∇Σ1::Vector{T}  # gradient term 1
    ∇Σ2::Vector{T}  # gradient term 2
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    HΣ::Matrix{T}
    HΣ1::Matrix{T}
    HΣ2::Matrix{T}
    Hτ::Matrix{T}
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_n2::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
end

function GLMCopulaVCModel(gcs::Vector{GLMCopulaVCObs{T, D, Link}}) where {T <: BlasReal, D, Link}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    ∇Σ1  = Vector{T}(undef, m)
    ∇Σ2  = Vector{T}(undef, m)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    HΣ  = Matrix{T}(undef, m, m)
    HΣ1  = Matrix{T}(undef, m, m)
    HΣ2  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0.0
    ntotal = 0.0
    d = Vector{D}(undef, n)
    link = Vector{Link}(undef, n)
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
    storage_n2 = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    GLMCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, p, m, β, τ, Σ,
        ∇β, ∇τ, ∇Σ, ∇Σ1, ∇Σ2, XtX, Hβ, HΣ, HΣ1, HΣ2, Hτ, TR, QF,
        storage_n, storage_n2, storage_m, storage_Σ, d, link)
end

#######

"""
GaussianCopulaLMMObs
GaussianCopulaLMMObs(y, X, Z)
A realization of Gaussian copula linear mixed model data instance.
"""
struct GaussianCopulaLMMObs{T <: LinearAlgebra.BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Matrix{T}   # gradient wrt Σ 
    Hβ::Matrix{T}   # Hessian wrt β
    Hτ::Matrix{T}   # Hessian wrt τ
    HΣ::Matrix{T}   # Hessian wrt Σ
    res::Vector{T}  # residual vector
    xtx::Matrix{T}  # Xi'Xi (p-by-p)
    ztz::Matrix{T}  # Zi'Zi (q-by-q)
    xtz::Matrix{T}  # Xi'Zi (p-by-q)
    storage_q1::Vector{T}
    storage_q2::Vector{T}
end

function GaussianCopulaLMMObs(
    y::Vector{T},
    X::Matrix{T},
    Z::Matrix{T}
    ) where T <: BlasReal
    n, p, q = size(X, 1), size(X, 2), size(Z, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    ztz = transpose(Z) * Z
    xtz = transpose(X) * Z
    storage_q1 = Vector{T}(undef, q)
    storage_q2 = Vector{T}(undef, q)
    # constructor
    GaussianCopulaLMMObs{T}(y, X, Z, 
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ,
        res, xtx, ztz, xtz,
        storage_q1, storage_q2)
end

"""
GaussianCopulaLMMModel
GaussianCopulaLMMModel(gcs)
Gaussian copula linear mixed model, which contains a vector of 
`GaussianCopulaLMMObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaLMMModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaLMMObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    q::Int          # number of random effects
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Matrix{T}    # q-by-q (psd) matrix
    # working arrays
    ΣL::Matrix{T}
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Matrix{T}
    Hβ::Matrix{T}   # Hessian from all observations
    Hτ::Matrix{T}
    HΣ::Matrix{T}
    XtX::Matrix{T}      # X'X = sum_i Xi'Xi
    storage_qq::Matrix{T}
    storage_nq::Matrix{T}
end

function GaussianCopulaLMMModel(gcs::Vector{GaussianCopulaLMMObs{T}}) where T <: BlasReal
    n, p, q = length(gcs), size(gcs[1].X, 2), size(gcs[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Matrix{T}(undef, q, q)
    ΣL  = similar(Σ)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Matrix{T}(undef, q, q)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    HΣ  = Matrix{T}(undef, abs2(q), abs2(q))
    XtX = zeros(T, p, p) # sum_i xi'xi
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
    end
    storage_qq = Matrix{T}(undef, q, q)
    storage_nq = Matrix{T}(undef, n, q)
    GaussianCopulaLMMModel{T}(gcs, ntotal, p, q, 
        β, τ, Σ, ΣL,
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, HΣ, 
        XtX, storage_qq, storage_nq)
end

include("initialize_model.jl")
include("splitting_loglikelihood.jl")
include("logl.jl")
include("splitting_gradient.jl")
include("splitting_hessian.jl")
# include("fit_new.jl")
include("fit_old.jl")
include("update_sigma_and_residuals.jl")
include("discrete_rand.jl")
include("continuous_rand.jl")
include("multivariate_rand.jl")
include("gaussian_lmm.jl")

end # module
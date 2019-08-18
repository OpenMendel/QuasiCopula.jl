__precompile__()

module GLMCopula

using Convex, LinearAlgebra, MathProgBase, Reexport
@reexport using Ipopt
@reexport using NLopt

export GaussianCopulaVCObs, GaussianCopulaVCModel
export fit!, fitted, init_β!, loglikelihood!, standardize_res!
export update_res!, update_Σ!, update_quadform!

export GaussianCopulaLMMObs, GaussianCopulaLMMModel

"""
GaussianCopulaVCObs
GaussianCopulaVCObs(y, X, V)

A realization of Gaussian copula variance component data.
"""
struct GaussianCopulaVCObs{T <: LinearAlgebra.BlasFloat}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt σ2
    H::Matrix{T}    # Hessian H
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    storage_n::Vector{T}
end

function GaussianCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}}
    ) where T <: LinearAlgebra.BlasFloat
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    H   = Matrix{T}(undef, p + 1 + m, p + 1 + m)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    t   = [tr(V[k])/2 for k in 1:m] 
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    # constructor
    GaussianCopulaVCObs{T}(y, X, V, ∇β, ∇τ, ∇Σ, H, res, xtx, t, q, storage_n)
end

"""
GaussianCopulaVCModel
GaussianCopulaVCModel(gcs)

Gaussian copula variance component model, which contains a vector of 
`GaussianCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaVCModel{T <: LinearAlgebra.BlasFloat} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaVCObs{T}}
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
    H::Matrix{T}    # Hessian from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
end

function GaussianCopulaVCModel(gcs::Vector{GaussianCopulaVCObs{T}}) where T <: LinearAlgebra.BlasFloat
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    npar = p + m + 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    H   = Matrix{T}(undef, npar, npar)
    XtX = zeros(T, p, p) # sum_i xi'xi
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    GaussianCopulaVCModel{T}(gcs, ntotal, p, m, β, τ, Σ, 
        ∇β, ∇τ, ∇Σ, H, XtX, TR, QF, 
        storage_n, storage_m, storage_Σ)
end

"""
GaussianCopulaLMMObs
GaussianCopulaLMMObs(y, X, Z)

A realization of Gaussian copula linear mixed model data.
"""
struct GaussianCopulaLMMObs{T <: LinearAlgebra.BlasFloat}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::AbstractMatrix{T}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt Σ 
    H::Matrix{T}    # Hessian H
    res::Vector{T}  # residual vector
    xtx::Matrix{T}  # Xi'Xi
    storage_q::Vector{T}
    storage_nq::Matrix{T}
end

function GaussianCopulaLMMObs(
    y::Vector{T},
    X::Matrix{T},
    Z::AbstractMatrix{T}
    ) where T <: LinearAlgebra.BlasFloat
    n, p, q = size(X, 1), size(X, 2), size(Z, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, abs2(q))
    H   = Matrix{T}(undef, p + 1 + abs2(q), p + 1 + abs2(q))
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    storage_q1 = Vector{T}(undef, q)
    storage_q2 = Vector{T}(undef, q)
    storage_nq = Matrix{T}(undef, n, q)
    # constructor
    GaussianCopulaLMMObs{T}(y, X, Z, ∇β, ∇τ, ∇Σ, H, res, xtx, 
        storage_q1, storage_q2, storage_nq)
end

"""
GaussianCopulaLMMModel
GaussianCopulaLMMModel(gcs)

Gaussian copula linear mixed model, which contains a vector of 
`GaussianCopulaLMMObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaLMMModel{T <: LinearAlgebra.BlasFloat} <: MathProgBase.AbstractNLPEvaluator
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
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    H::Matrix{T}    # Hessian from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    storage_n::Vector{T}
    storage_p::Vector{T}
    storage_Σ::Vector{T}
end

function GaussianCopulaLMMModel(gcs::Vector{GaussianCopulaLMMObs{T}}) where T <: LinearAlgebra.BlasFloat
    n, p, q = length(gcs), size(gcs[1].X, 2), size(gcs[1].Z, 2)
    npar = p + abs2(q) + 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Matrix{T}(undef, q, q)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, abs2(q))
    H   = Matrix{T}(undef, npar, npar)
    XtX = zeros(T, p, p) # sum_i xi'xi
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
    end
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    storage_Σ = Matrix{T}(undef, q, q)
    GaussianCopulaLMMModel{T}(gcs, ntotal, p, q, β, τ, Σ, 
        ∇β, ∇τ, ∇Σ, H, XtX,
        storage_n, storage_p, storage_Σ)
end

include("gaussian_vc.jl")
include("gaussian_lmm.jl")

end#module
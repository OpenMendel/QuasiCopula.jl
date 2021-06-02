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
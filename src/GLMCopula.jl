__precompile__()

module GLMCopula

using Convex, LinearAlgebra, MathProgBase, Reexport, SCS
@reexport using Ipopt
@reexport using NLopt

export GaussianCopulaVCObs, GaussianCopulaVCModel
export fit!, init_β!, loglikelihood!, standardize_res!, update_σ2!, update_quadform!

"""
GaussianCopulaVCObs
GaussianCopulaVCObs(y, X, V)

A single instance of Gaussian copula variance component data.
"""
struct GaussianCopulaVCObs{T <: LinearAlgebra.BlasFloat}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇σ2::Vector{T}  # gradient wrt σ2
    H::Matrix{T}    # Hessian H
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i
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
    ∇σ2 = Vector{T}(undef, m)
    H   = Matrix{T}(undef, p + 1 + m, p + 1 + m)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    t   = [tr(V[k])/2 for k in 1:m] 
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    # constructor
    GaussianCopulaVCObs{T}(y, X, V, ∇β, ∇τ, ∇σ2, H, res, xtx, t, q, storage_n)
end

struct GaussianCopulaVCModel{T <: LinearAlgebra.BlasFloat} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaVCObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    σ2::Vector{T}   # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}    # gradient from all observations
    ∇τ::Vector{T}
    ∇σ2::Vector{T}
    H::Matrix{T}    # Hessian from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_σ2::Vector{T}
end

function GaussianCopulaVCModel(gcs::Vector{GaussianCopulaVCObs{T}}) where T <: LinearAlgebra.BlasFloat
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    npar = p + m + 1
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    σ2  = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇σ2 = Vector{T}(undef, m)
    H   = Matrix{T}(undef, npar, npar)
    XtX = zeros(T, p, p) # sum_i xi'xi
    TR  = Matrix{T}(undef, n, m)
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
        TR[i, :] = gcs[i].t
    end
    QF         = Matrix{T}(undef, n, m)
    storage_n  = Vector{T}(undef, n)
    storage_m  = Vector{T}(undef, m)
    storage_σ2 = Vector{T}(undef, m)
    GaussianCopulaVCModel{T}(gcs, ntotal, p, m, β, τ, σ2, 
        ∇β, ∇τ, ∇σ2, H, XtX, TR, QF, 
        storage_n, storage_m, storage_σ2)
end

"""
init_β(gc)

Initialize the linear regression parameters `β` and `σ20` by the least squares 
solution.
"""
function init_β!(gcm::GaussianCopulaVCModel{T}) where T <: LinearAlgebra.BlasFloat   
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p) 
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    gcm.β .= cholesky(Symmetric(gcm.XtX)) \ xty
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    gcm.τ[1] = gcm.ntotal / rss
    gcm.β
end

"""
update_res!(gc, β)

Update the residual vector according to `β`.
"""
function update_res!(
    gc::GaussianCopulaVCObs{T}, 
    β::Vector{T}
    ) where T <: LinearAlgebra.BlasFloat
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(gcm::GaussianCopulaVCModel{T}) where T <: LinearAlgebra.BlasFloat
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(gc::GaussianCopulaVCObs{T}, σinv::T) where T <: LinearAlgebra.BlasFloat
    gc.res .*= σinv
end

function standardize_res!(gcm::GaussianCopulaVCModel{T}) where T <: LinearAlgebra.BlasFloat
    σinv = sqrt(gcm.τ[1])
    # standardize residual
    for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i], σinv)
    end
    nothing
end

"""
update_quadform!(gc)

Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::GaussianCopulaVCObs)
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

function update_quadform!(gcm::GaussianCopulaVCModel, updateqi::Bool=false)
    for i in eachindex(gcm.data)
        updateqi && update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q
    end
    nothing
end

"""
update_σ2!(gc)

Update the variance components `σ2` according to the current value of 
`β` and `σ02`.
"""
function update_σ2!(gcm::GaussianCopulaVCModel, maxiter::Integer=1000, reltol::Number=1e-6)
    # MM iteration
    for iter in 1:maxiter
        copyto!(gcm.storage_σ2, gcm.σ2)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.σ2)
        gcm.storage_n .= inv.(gcm.storage_n .+ 1)
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n)
        gcm.σ2 .*= gcm.storage_m
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.σ2)
        gcm.storage_n .= inv.(gcm.storage_n .+ 1)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.σ2 ./= gcm.storage_m
        # monotonicity diagnosis
        # println(sum(log, (gcm.QF * gcm.σ2) .+ 1) - sum(log, gcm.TR * gcm.σ2 .+ 1))
        # convergence check
        gcm.storage_m .= gcm.σ2 .- gcm.storage_σ2
        norm(gcm.storage_m) < reltol * (norm(gcm.storage_σ2) + 1) && break
        iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.σ2
end

include("fit_nlp.jl")

end#module
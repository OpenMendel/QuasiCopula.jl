export GaussianCopulaVCObs, GaussianCopulaVCModel
export fit!, fitted, init_β!, loglikelihood!, standardize_res!, update_σ2!, update_quadform!

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

"""
GaussianCopulaVCModel
GaussianCopulaVCModel(gcs)

Gaussian copula variance component model, which contains a vector of 
`GaussianCopulaVCObs` as data `gcs`, parameters, and working arrays.
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
    σ2::Vector{T}   # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
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
init_β(gcm)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least squares 
solution.
"""
function init_β!(gcm::GaussianCopulaVCModel{T}) where T <: LinearAlgebra.BlasFloat   
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p) 
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
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
function update_σ2!(gcm::GaussianCopulaVCModel, maxiter::Integer=5000, reltol::Number=1e-6)
    # MM iteration
    # @show gcm.σ2
    # @show gcm.τ
    # @show gcm.QF
    # @show gcm.TR
    for iter in 1:maxiter
        # store previous iterate
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

function fitted(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T,
    σ2::Vector{T}) where T <: LinearAlgebra.BlasFloat
    n, m = length(gc.y), length(gc.V)
    μ̂ = gc.X * β
    Ω = Matrix{T}(undef, n, n)
    for k in 1:m
        Ω .+= σ2[k] .* gc.V[k]
    end
    σ02 = inv(τ)
    c = inv(1 + dot(σ2, gc.t)) # normalizing constant
    V̂ = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:j-1
            V̂[i, j] = c * σ02 * Ω[i, j]
        end
        V̂[j, j] = c * σ02 * (1 + Ω[j, j] + tr(Ω) / 2) 
    end
    LinearAlgebra.copytri!(V̂, 'U')
    μ̂, V̂
end

function loglikelihood!(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    σ2::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇σ2, 0) 
    end
    needhess && fill!(gc.H, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss  = abs2(norm(gc.res))
    tsum = dot(σ2, gc.t)
    logl = - log(1 + tsum) - (n * log(2π) -  n * log(τ) + rss) / 2
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad
            BLAS.gemv!('T', σ2[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(σ2, gc.q)
    logl += log(1 + qsum)
    # gradient
    if needgrad
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv(1 + qsum), gc.∇β)
        gc.∇β  .*= sqrtτ
        gc.∇τ[1] = (n - rss + 2qsum / (1 + qsum)) / 2τ
        gc.∇σ2  .= inv(1 + qsum) .* gc.q .- inv(1 + tsum) .* gc.t 
    end
    # Hessian: TODO
    if needhess; end;
    # output
    logl
end

function loglikelihood!(
    gcm::GaussianCopulaVCModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇σ2, 0) 
    end
    needhess && fill!(gcm.H, 0)
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.σ2, needgrad, needhess)
        if needgrad
            gcm.∇β  .+= gcm.data[i].∇β
            gcm.∇τ  .+= gcm.data[i].∇τ
            gcm.∇σ2 .+= gcm.data[i].∇σ2
        end
        needhess && (gcm.H .+= gcm.data[i].H)
    end
    logl
end

function fit!(
    gcm::GaussianCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    npar = gcm.p + 1
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, [gcm.β; log(gcm.τ[1])])
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm)
    gcm
end

function MathProgBase.initialize(gcm::GaussianCopulaVCModel, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(gcm::GaussianCopulaVCModel, par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 at current β and τ using MM
    update_res!(gcm)
    standardize_res!(gcm)
    update_quadform!(gcm, true)
    update_σ2!(gcm)
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(gcm::GaussianCopulaVCModel, grad::Vector, par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 at current β and τ using MM
    update_res!(gcm)
    standardize_res!(gcm)
    update_quadform!(gcm, true)
    update_σ2!(gcm)
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, 1, gcm.∇β, 1, gcm.p)
    grad[gcm.p+1] = gcm.∇τ[1] * gcm.τ[1]
    logl
end

function copy_par!(gcm::GaussianCopulaVCModel, par::Vector)
    copyto!(gcm.β, 1, par, 1, gcm.p)
    gcm.τ[1] = exp(par[end])
    par
end

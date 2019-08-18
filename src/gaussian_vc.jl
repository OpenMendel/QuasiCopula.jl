"""
init_β(gcm)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    gcm::Union{GaussianCopulaVCModel{T},GaussianCopulaLMMModel{T}}
    ) where T <: LinearAlgebra.BlasFloat
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
    gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}}, 
    β::Vector{T}
    ) where T <: LinearAlgebra.BlasFloat
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: LinearAlgebra.BlasFloat
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}}, 
    σinv::T
    ) where T <: LinearAlgebra.BlasFloat
    gc.res .*= σinv
end

function standardize_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: LinearAlgebra.BlasFloat
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
update_Σ!(gc)

Update the variance components `Σ` according to the current value of 
`β` and `τ`.
"""
function update_Σ!(
    gcm::GaussianCopulaVCModel, 
    maxiter::Integer=5000, 
    reltol::Number=1e-6)
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.Σ)
        gcm.storage_n .= inv.(gcm.storage_n .+ 1)
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n)
        gcm.Σ .*= gcm.storage_m
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.Σ)
        gcm.storage_n .= inv.(gcm.storage_n .+ 1)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.Σ ./= gcm.storage_m
        # monotonicity diagnosis
        # println(sum(log, (gcm.QF * gcm.σ2) .+ 1) - sum(log, gcm.TR * gcm.σ2 .+ 1))
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
        iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

function fitted(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}) where T <: LinearAlgebra.BlasFloat
    n, m = length(gc.y), length(gc.V)
    μ̂ = gc.X * β
    Ω = Matrix{T}(undef, n, n)
    for k in 1:m
        Ω .+= Σ[k] .* gc.V[k]
    end
    σ02 = inv(τ)
    c = inv(1 + dot(Σ, gc.t)) # normalizing constant
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
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇Σ, 0) 
    end
    needhess && fill!(gc.H, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    tsum = dot(Σ, gc.t)
    logl = - log(1 + tsum) - (n * log(2π) -  n * log(τ) + rss) / 2
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad
            BLAS.gemv!('T', Σ[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)
    # gradient
    if needgrad
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv(1 + qsum), gc.∇β)
        gc.∇β  .*= sqrtτ
        gc.∇τ[1] = (n - rss + 2qsum / (1 + qsum)) / 2τ
        gc.∇Σ   .= inv(1 + qsum) .* gc.q .- inv(1 + tsum) .* gc.t 
    end
    # Hessian: TODO
    if needhess; end;
    # output
    logl
end

function loglikelihood!(
    gcm::Union{GaussianCopulaVCModel{T},GaussianCopulaLMMModel{T}},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇Σ, 0)
    end
    needhess && fill!(gcm.H, 0)
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇τ .+= gcm.data[i].∇τ
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        needhess && (gcm.H .+= gcm.data[i].H)
    end
    logl
end

function fit!(
    gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel},
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

function MathProgBase.initialize(
    gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel}, 
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel}) = [:Grad]

function MathProgBase.eval_f(
    gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel}, 
    par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 at current β and τ using MM
    update_res!(gcm)
    standardize_res!(gcm)
    update_quadform!(gcm, true)
    update_Σ!(gcm)
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel}, 
    grad::Vector, 
    par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 at current β and τ using MM
    update_res!(gcm)
    standardize_res!(gcm)
    update_quadform!(gcm, true)
    update_Σ!(gcm)
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, 1, gcm.∇β, 1, gcm.p)
    grad[gcm.p+1] = gcm.∇τ[1] * gcm.τ[1]
    logl
end

function copy_par!(
    gcm::Union{GaussianCopulaVCModel,GaussianCopulaLMMModel}, 
    par::Vector)
    copyto!(gcm.β, 1, par, 1, gcm.p)
    gcm.τ[1] = exp(par[end])
    par
end

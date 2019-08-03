function loglikelihood!(
    gc::GaussianCopulaVC{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    σ2::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    npar = p + 1 + m
    if needgrad; fill!(gc.∇, 0); fill!(gc.storage_p, 0); end
    needhess && fill!(gc.H, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss = abs2(norm(gc.res))
    tsum = dot(σ2, gc.t)
    logl  = - log(1 + tsum) - (n * log(2π) -  n * log(τ) + rss) / 2
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad
            BLAS.gemv!('T', σ2[k], gc.X, gc.storage_n, one(T), gc.storage_p)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum = dot(σ2, gc.q)
    logl += log(1 + qsum)
    # gradient
    if needgrad
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv(1 + qsum), gc.storage_p)
        gc.∇[1:p] .= sqrtτ .* gc.storage_p
        gc.∇[p+1] = (n - rss + 2qsum / (1 + qsum)) / 2τ
        gc.∇[p+2:end] .= inv(1 + qsum) .* gc.q .- inv(1 + tsum) .* gc.t 
    end
    # Hessian
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
    needgrad && fill!(gcm.∇, 0)
    needhess && fill!(gcm.H, 0)
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.σ2, needgrad, needhess)
        needgrad && (gcm.∇ .+= gcm.data[i].∇)
        needhess && (gcm.H .+= gcm.data[i].H)
    end
    logl
end

function fit!(
    gcm::GaussianCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    npar = gcm.p + gcm.m + 1
    optm = MathProgBase.NonlinearModel(solver)
    lb = [fill(-Inf, gcm.p); fill(0, gcm.m + 1)]
    ub = fill(Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, [gcm.β; gcm.τ; gcm.σ2])
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    update_res!(gcm)
    gcm
end

function MathProgBase.initialize(gc::GaussianCopulaVCModel, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(gcm::GaussianCopulaVCModel, par::Vector)
    copy_par!(gcm, par)
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(gcm::GaussianCopulaVCModel, grad::Vector, par::Vector)
    copy_par!(gcm, par)
    loglikelihood!(gcm, true, false)
    copyto!(grad, gcm.∇)
end

function copy_par!(gcm::GaussianCopulaVCModel, par::Vector)
    copyto!(gcm.β, 1, par, 1, gcm.p)
    gcm.τ[1] = max(par[gcm.p+1], 0)
    copyto!(gcm.σ2, 1, par, gcm.p+2, gcm.m)
    par
end

function loglikelihood!(
    gc::GaussianCopulaLMMObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    n, p, q = size(gc.X, 1), size(gc.X, 2), size(gc.Z, 2)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇Σ, 0)
    end
    if needhess
        fill!(gc.Hβ, 0)
        fill!(gc.Hτ, 0)
        fill!(gc.HΣ, 0)
    end
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss = abs2(norm(gc.res)) # RSS of standardized residual
    tr = (1//2)dot(gc.ztz, Σ)
    mul!(gc.storage_q1, transpose(gc.Z), gc.res) # storage_q1 = Z' * std residual
    mul!(gc.storage_q2, Σ, gc.storage_q1)        # storage_q2 = Σ * Z' * std residual
    qf = (1//2)dot(gc.storage_q1, gc.storage_q2)
    logl = - (n * log(2π) -  n * log(τ) + rss) / 2 - log(1 + tr) + log(1 + qf)
    # gradient
    if needgrad
        # wrt β
        mul!(gc.∇β, transpose(gc.X), gc.res)
        BLAS.gemv!('N', -inv(1 + qf), gc.xtz, gc.storage_q2, one(T), gc.∇β)
        gc.∇β .*= sqrtτ
        # wrt τ
        gc.∇τ[1] = (n - rss + 2qf / (1 + qf)) / 2τ
        # wrt Σ
        copyto!(gc.∇Σ, gc.ztz)
        BLAS.syrk!('U', 'N', (1//2)inv(1 + qf), gc.storage_q1, (-1//2)inv(1 + tr), gc.∇Σ)
        copytri!(gc.∇Σ, 'U')
    end
    # Hessian: TODO
    if needhess; end;
    # output
    logl
end

function fit!(
    gcm::GaussianCopulaLMMModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    npar = p + 1 + (q * (q + 1)) >> 1
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    # starting point
    par0 = Vector{Float64}(undef, npar)
    modelpar_to_optimpar!(par0, gcm)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # refresh gradient and Hessian
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, true)
    gcm
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
    gcm::GaussianCopulaLMMModel, 
    par::Vector)
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    copyto!(gcm.β, 1, par, 1, p)
    gcm.τ[1] = exp(par[p+1])
    fill!(gcm.ΣL, 0)
    offset = p + 2
    for j in 1:q
        gcm.ΣL[j, j] = exp(par[offset])
        offset += 1
        for i in j+1:q
            gcm.ΣL[i, j] = par[offset]
            offset += 1
        end
    end
    mul!(gcm.Σ, gcm.ΣL, transpose(gcm.ΣL))
    nothing
end

"""
    modelpar_to_optimpar!(gcm, par)

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    gcm::GaussianCopulaLMMModel
    )
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    copyto!(par, gcm.β)
    par[p+1] = log(gcm.τ[1])
    Σchol = cholesky(Symmetric(gcm.Σ))
    gcm.ΣL .= Σchol.L
    offset = p + 2
    for j in 1:q
        par[offset] = log(gcm.ΣL[j, j])
        offset += 1
        for i in j+1:q
            par[offset] = gcm.ΣL[i, j]
            offset += 1
        end
    end
    par
end

function MathProgBase.initialize(
    gcm::GaussianCopulaLMMModel, 
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaLMMModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::GaussianCopulaLMMModel, 
    par::Vector)
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::GaussianCopulaLMMModel, 
    grad::Vector, 
    par::Vector)
    p, q = size(gcm.data[1].X, 2), size(gcm.data[1].Z, 2)
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt log(τ)
    grad[p+1] = gcm.∇τ[1] * gcm.τ[1]
    # gradient wrt L
    mul!(gcm.storage_qq, gcm.∇Σ, gcm.ΣL)
    offset = p + 2
    for j in 1:q
        grad[offset] = 2gcm.storage_qq[j, j] * gcm.ΣL[j, j]
        offset += 1
        for i in j+1:q
            grad[offset] = 2gcm.storage_qq[i, j]
            offset += 1
        end
    end
    nothing
end

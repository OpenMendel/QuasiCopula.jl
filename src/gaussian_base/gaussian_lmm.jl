export init_β!, loglikelihood!

"""
init_β(gcm)
Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    gcm::Union{GaussianCopulaVCModel{T},GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
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
    ) where T <: BlasReal
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}}, 
    σinv::T
    ) where T <: BlasReal
    gc.res .*= σinv
end

function standardize_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
    σinv = sqrt(gcm.τ[1])
    # standardize residual
    for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i], σinv)
    end
    nothing
end

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

function loglikelihood!(
    gcm::GaussianCopulaLMMModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇Σ, 0)
    end
    if needhess
        gcm.Hβ .= - gcm.XtX
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇τ .+= gcm.data[i].∇τ
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hτ .+= gcm.data[i].Hτ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
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
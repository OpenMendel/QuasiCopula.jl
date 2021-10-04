function fit2!(
    gcm::GLMCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill( Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm)
    # gcm
    nothing
end

function MathProgBase.initialize(
    gcm::GLMCopulaVCModel, 
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GLMCopulaVCModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
    gcm::GLMCopulaVCModel, 
    par::Vector)
    copy_par!(gcm, par)
    # update nuisance parameter
    if typeof(gcm.data[1].d) <: NegativeBinomial
        new_d = update_r!(gcm)
        @show new_d
    end
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    @show gcm.Σ
    # evaluate loglikelihood
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::GLMCopulaVCModel, 
    grad::Vector, 
    par::Vector)
    # println("reached eval_grad_f")
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    @show gcm.β
    # update nuisance parameter
    if typeof(gcm.data[1].d) <: NegativeBinomial
        new_d = update_r!(gcm)
        @show new_d
    end
    update_Σ!(gcm)
    @show gcm.Σ
    # evaluate gradient
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    nothing
end

function copy_par!(
    gcm::GLMCopulaVCModel, 
    par::Vector)
    copyto!(gcm.β, par)
    par
end

function MathProgBase.hesslag_structure(gcm::GLMCopulaVCModel)
    Iidx = Vector{Int}(undef, (gcm.p * (gcm.p + 1)) >> 1)
    Jidx = similar(Iidx)
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            Iidx[ct] = i
            Jidx[ct] = j
            ct += 1
        end
    end
    Iidx, Jidx
end

function MathProgBase.eval_hesslag(
    gcm::GLMCopulaVCModel,
    H::Vector{T},
    par::Vector{T},
    σ::T,
    μ::Vector{T}) where {T}
    # println("reached eval_hesslag")
    copy_par!(gcm, par)
    if typeof(gcm.data[1].d) <: NegativeBinomial
        new_d = update_r!(gcm)
        @show new_d
    end
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate Hessian
    loglikelihood!(gcm, true, true)
    # copy Hessian elements into H
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            H[ct] = gcm.Hβ[i, j]
            ct += 1
        end
    end
    H .*= σ
end

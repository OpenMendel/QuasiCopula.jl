export ◺
"""
    fit!(gcm::GaussianCopulaARModel)

Fit an `GaussianCopulaARModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `gcm.β`, `gcm.τ`, `gcm.ρ`, `gcm.σ2`.
"""
function fit!(
        gcm::GaussianCopulaARModel,
        solver=Ipopt.IpoptSolver(print_level = 5)
    )
    npar = gcm.p + 3 # tau, rho and sigma squared
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill(Inf, npar)
    offset = gcm.p + 1
    ub[offset] = 1
    for k in 1:3
        lb[offset] = 0
        offset += 1
    end
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, gcm)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, false)
    # gcm
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: GaussianCopulaARModel
    )
    # β
    copyto!(par, gcm.β)
    # ρ, σ2
    par[end - 2] = gcm.ρ[1]
    par[end - 1] = gcm.σ2[1]
    # τ
    par[end] = gcm.τ[1]
    par
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: GaussianCopulaARModel,
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # ρ, σ2
    gcm.ρ[1] = par[gcm.p + 1]
    gcm.σ2[1] = par[gcm.p + 2]
    # τ
    gcm.τ[1] = par[gcm.p + 3]
    copyto!(gcm.θ, par)
    gcm
end

function MathProgBase.initialize(
    gcm::GaussianCopulaARModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaARModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
        gcm :: GaussianCopulaARModel,
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
        gcm    :: GaussianCopulaARModel,
        grad :: Vector,
        par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt ρ
    grad[gcm.p + 1] = gcm.∇ρ[1]
    # gradient wrt σ2
    grad[gcm.p + 2] = gcm.∇σ2[1]
    # gradient wrt τ
    grad[gcm.p + 3] = gcm.∇τ[1]
    # @show gcm.θ
    # @show gcm.θ
    copyto!(gcm.∇θ, grad)
    # @show gcm.∇θ
    # return objective
    obj
end

MathProgBase.eval_g(gcm::GaussianCopulaARModel, g, par) = nothing
MathProgBase.jac_structure(gcm::GaussianCopulaARModel) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::GaussianCopulaARModel, J, par) = nothing


function MathProgBase.hesslag_structure(gcm::GaussianCopulaARModel)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p + 1)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p + 1)
    # Hββ block
    idx  = 1    
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # rho and sigma2
    for j in 1:2
        arr1[idx] = gcm.p + j
        arr2[idx] = gcm.p + j
        idx += 1
    end
    arr1[idx] = gcm.p + 1
    arr2[idx] = gcm.p + 2
    idx += 1
    for k in 1:gcm.p
        arr1[idx] = gcm.p + 2
        arr2[idx] = k
        idx += 1
    end
    # for tau
    arr1[idx] = gcm.p + 3
    arr2[idx] = gcm.p + 3
    # for k in 1:gcm.p
    #     arr1[idx] = gcm.p + 1
    #     arr2[idx] = k
    #     idx += 1
    # end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
        gcm   :: GaussianCopulaARModel,
        H   :: Vector{T},
        par :: Vector{T},
        σ   :: T,
        μ   :: Vector{T}
    ) where {T}    
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, true)
    # Hβ block
    idx = 1    
    @inbounds for j in 1:gcm.p, i in 1:j
        H[idx] = gcm.Hβ[i, j]
        idx   += 1
    end
    # Hrho block
    H[idx] = gcm.Hρ[1, 1]
    idx += 1
    H[idx] = gcm.Hσ2[1, 1]
    idx += 1
    H[idx] = gcm.Hρσ2[1, 1]
    idx += 1
    for k in 1:gcm.p
        H[idx] = gcm.Hβσ2[k]
        idx += 1
    end
    # Hτ block
    H[idx] = gcm.Hτ[1, 1]
    idx += 1
    # for k in 1:gcm.p
    #     H[idx] = gcm.Hβρ[k]
    #     idx += 1
    # end
    # lmul!(σ, H)
    H .*= σ
end
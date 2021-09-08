"""
    fit!(gcm::GLMCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `GLMCopulaVCModel` object by MLE using a nonlinear programming solver. Start point 
should be provided in `gcm.β`, `gcm.Σ`.
"""
function fit!(
        gcm::GLMCopulaVCModel,
        solver=Ipopt.IpoptSolver(print_level=5)
    )
    npar = gcm.p + gcm.m
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    # diagonal entries of Cholesky factor L should be >= 0
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    offset = gcm.p + 1
    for k in 1:gcm.m
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
        gcm :: GLMCopulaVCModel
    )
    # β
    copyto!(par, gcm.β)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.Σ[k]
        offset += 1
    end
    par
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: GLMCopulaVCModel, 
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.Σ[k] = par[offset]
        offset   += 1
    end
    copyto!(gcm.θ, par)
    gcm
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
        gcm :: GLMCopulaVCModel, 
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
        gcm    :: GLMCopulaVCModel, 
        grad :: Vector, 
        par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par) 
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        grad[offset] = gcm.∇Σ[k]
        offset += 1
    end
    # update nuisance parameter
    # @show gcm.θ
    copyto!(gcm.∇θ, grad)
    # @show gcm.∇θ
    # return objective
    obj
end

MathProgBase.eval_g(gcm::GLMCopulaVCModel, g, par) = nothing
MathProgBase.jac_structure(gcm::GLMCopulaVCModel) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::GLMCopulaVCModel, J, par) = nothing

function MathProgBase.hesslag_structure(gcm::GLMCopulaVCModel)
    m◺ = ◺(gcm.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + m◺)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + m◺)
    # Hββ block
    idx  = 1    
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # Haa block
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    return (arr1, arr2)
end
    
function MathProgBase.eval_hesslag(
        gcm   :: GLMCopulaVCModel, 
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
    # Haa block
    @inbounds for j in 1:gcm.m, i in 1:j
        H[idx] = gcm.HΣ[i, j]
        idx   += 1
    end
    # lmul!(σ, H)
    H .*= σ
end
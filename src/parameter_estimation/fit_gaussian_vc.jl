"""
    fit_quasi!(gcm::GaussianCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `GaussianCopulaVCModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `gcm.β`, `gcm.θ`, `gcm.τ` this is for Normal base.
"""
function fit!(
        gcm::GaussianCopulaVCModel,
        solver=Ipopt.IpoptSolver(print_level = 3, tol = 10^-6, max_iter = 100,
        limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")
    )
    initialize_model!(gcm)
    npar = gcm.p + gcm.m + 1
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    offset = gcm.p + 1
    for k in 1:gcm.m + 1
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

Translate model parameters in `gcm` to optimization variables in `par` for Normal base.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: GaussianCopulaVCModel
    )
    # β
    copyto!(par, gcm.β)
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.θ[k]
        offset += 1
    end
    par[offset] = gcm.τ[1]
    par
end

"""
    optimpar_to_modelpar_quasi!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: GaussianCopulaVCModel,
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset   += 1
    end
    gcm.τ[1] = par[offset]
    gcm
end

function MathProgBase.initialize(
    gcm::GaussianCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
        gcm :: GaussianCopulaVCModel,
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
    gcm  :: GaussianCopulaVCModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt variance comps
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        grad[offset] = gcm.∇θ[k]
        offset += 1
    end
    grad[offset] = gcm.∇τ[1]
obj
end

MathProgBase.eval_g(gcm::GaussianCopulaVCModel, g, par) = nothing
MathProgBase.jac_structure(gcm::GaussianCopulaVCModel) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::GaussianCopulaVCModel, J, par) = nothing

function MathProgBase.hesslag_structure(gcm::GaussianCopulaVCModel)
    m◺ = ◺(gcm.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    # Hββ block
    idx  = 1    
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # variance components
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    arr1[idx] = gcm.p + gcm.m + 1
    arr2[idx] = gcm.p + gcm.m + 1
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
        gcm   :: GaussianCopulaVCModel,
        H   :: Vector{T},
        par :: Vector{T},
        σ   :: T,
        μ   :: Vector{T}
    )where {T <: BlasReal}
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
        H[idx] = gcm.Hθ[i, j]
        idx   += 1
    end
    H[idx] = gcm.Hτ[1, 1]
    # lmul!(σ, H)
    H .*= σ
end

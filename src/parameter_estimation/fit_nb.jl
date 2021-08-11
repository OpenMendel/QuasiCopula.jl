"""
    fit!(gcm::NBCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `NBCopulaVCModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.Σ`, `gcm.r`. In our block updates,
we fit 10 iterations of `gcm.β`, `gcm.Σ` using IPOPT, followed by 10 iterations of 
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.
"""
function fit!(
        gcm::NBCopulaVCModel,
        solver=Ipopt.IpoptSolver(print_level=5,max_iter=10,
                                hessian_approximation = "limited-memory"),
        tol::Float64 = 1e-4,
        maxIter::Int=100
    )
    npar = gcm.p + gcm.m
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
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
    logl0 = MathProgBase.getobjval(optm)
    # optimize
    r_diff = Inf
    curr_r = gcm.r[1]
    for i in 1:maxIter
        MathProgBase.optimize!(optm)
        logl = MathProgBase.getobjval(optm)
        update_r!(gcm)
        if abs(logl - logl0) ≤ tol
            break
        else
            println("iter $i r = $(gcm.r[1]), logl = $logl, tol = $(abs(logl - logl0))")
            logl0 = logl
        end
    end
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
        gcm :: NBCopulaVCModel
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
        gcm :: NBCopulaVCModel, 
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
    gcm::NBCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::NBCopulaVCModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
        gcm :: NBCopulaVCModel, 
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
        gcm    :: NBCopulaVCModel, 
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
    copyto!(gcm.∇θ, grad)
    # @show gcm.∇θ
    # return objective
    obj
end

MathProgBase.eval_g(gcm::NBCopulaVCModel, g, par) = nothing
MathProgBase.jac_structure(gcm::NBCopulaVCModel) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::NBCopulaVCModel, J, par) = nothing

# """
#     ◺(n::Integer)
# Triangular number n * (n+1) / 2
# """
# @inline ◺(n::Integer) = (n * (n + 1)) >> 1

function MathProgBase.hesslag_structure(gcm::NBCopulaVCModel)
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
        gcm   :: NBCopulaVCModel, 
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
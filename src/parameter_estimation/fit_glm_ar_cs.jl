export ◺
"""
    fit!(gcm::GLMCopulaARModel, solver=Ipopt.IpoptSolver)

Fit an `GLMCopulaARModel` object by MLE using a nonlinear programming solver.
This is for Poisson and Bernoulli base distributions with no additional base distribution parameters than the mean.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`.

# Arguments
- `gcm`: A `GLMCopulaARModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, hessian_approximation = "limited-memory")`)
"""
function fit!(
        gcm::GLMCopulaARModel,
        solver=Ipopt.IpoptSolver(print_level = 3, max_iter = 100, tol = 10^-6,
        limited_memory_max_history = 20, hessian_approximation = "limited-memory")
    )
    initialize_model!(gcm)
    npar = gcm.p + 2 # rho and sigma squared
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill(Inf, npar)
    offset = gcm.p + 1
    ub[offset] = 1
    for k in 1:2
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
end

"""
    fit!(gcm::GLMCopulaCSModel, solver=Ipopt.IpoptSolver)

Fit an `GLMCopulaCSModel` object by MLE using a nonlinear programming solver.
This is for Poisson and Bernoulli base distributions with no additional base distribution parameters than the mean.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`.

# Arguments
- `gcm`: A `GLMCopulaCSModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, hessian_approximation = "limited-memory")`)
"""
function fit!(
        gcm::GLMCopulaCSModel,
        solver=Ipopt.IpoptSolver(print_level = 3, max_iter = 100, tol = 10^-6,
        limited_memory_max_history = 20, hessian_approximation = "limited-memory")
    )
    initialize_model!(gcm)
    npar = gcm.p + 2 # rho and sigma squared
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill(Inf, npar)
    offset = gcm.p + 1
    # rho
    ub[offset] = 1
    # lb[offset] = 0
    lb[offset] = -inv(gcm.data[1].n - 1)
    offset += 1
    # sigma2
    lb[offset] = 0
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, gcm)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    # @show par0
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, false)
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}
    )
    # β
    copyto!(par, gcm.β)
    # ρ, σ2
    par[end - 1] = gcm.ρ[1]
    par[end] = gcm.σ2[1]
    par
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
        par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # ρ, σ2
    gcm.ρ[1] = par[gcm.p + 1]
    gcm.σ2[1] = par[gcm.p + 2]
    gcm
end

function MathProgBase.initialize(
    gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}) = [:Grad, :Hess]

function MathProgBase.eval_f(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
        gcm  :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel},
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
    obj
end

MathProgBase.eval_g(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}, g, par) = nothing
MathProgBase.jac_structure(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel, GLMCopulaCSModel}, J, par) = nothing

"""
    ◺(n::Integer)
Triangular number n * (n+1) / 2
"""
@inline ◺(n::Integer) = (n * (n + 1)) >> 1

function MathProgBase.hesslag_structure(gcm::GLMCopulaCSModel)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
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
    # for k in 1:gcm.p
    #     arr1[idx] = gcm.p + 1
    #     arr2[idx] = k
    #     idx += 1
    # end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
        gcm :: GLMCopulaCSModel,
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
    # for k in 1:gcm.p
    #     H[idx] = gcm.Hβρ[k]
    #     idx += 1
    # end
    # lmul!(σ, H)
    H .*= σ
end

function MathProgBase.hesslag_structure(gcm::Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel})
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + ◺(2) + gcm.p)
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
    # for k in 1:gcm.p
    #     arr1[idx] = gcm.p + 1
    #     arr2[idx] = k
    #     idx += 1
    # end
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
        gcm :: Union{GLMCopulaARModel, NBCopulaARModel, NBCopulaCSModel},
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
    # for k in 1:gcm.p
    #     H[idx] = gcm.Hβρ[k]
    #     idx += 1
    # end
    # lmul!(σ, H)
    H .*= σ
end

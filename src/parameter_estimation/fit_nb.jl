"""
    fit!(gcm::NBCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaVCModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.θ`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.θ` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
        gcm::NBCopulaVCModel,
        solver=Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                                warm_start_init_point = "yes",  mu_strategy = "adaptive",
                                hessian_approximation = "limited-memory");
        tol::Float64 = 1e-6,
        maxBlockIter::Int=10
    )
    # initialize model
    initialize_model!(gcm)
    npar = gcm.p + gcm.m
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
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
    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    # optimize
    for i in 1:maxBlockIter
        MathProgBase.optimize!(optm)
        logl = MathProgBase.getobjval(optm)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
        # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            logl0 = logl
        end
    end
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, false)
    # gcm
end

"""
    fit!(gcm::NBCopulaARModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaARModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.ρ`, `gcm.σ2` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaARModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
    gcm::NBCopulaARModel,
    solver=Ipopt.Ipopt.IpoptSolver(print_level = 0, max_iter = 15,
                            limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory");
    tol::Float64 = 1e-6,
    maxBlockIter::Int=10
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
    logl0 = MathProgBase.getobjval(optm)
    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    # optimize
    for i in 1:maxBlockIter
        MathProgBase.optimize!(optm)
        logl = MathProgBase.getobjval(optm)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
            # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            logl0 = logl
        end
    end
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, false)
    # gcm
end

"""
    fit!(gcm::NBCopulaCSModel, solver=Ipopt.IpoptSolver)

Fit an `NBCopulaCSModel` object by block MLE using a nonlinear programming solver.
Start point should be provided in `gcm.β`, `gcm.ρ`, `gcm.σ2`, `gcm.r`.
In our block updates, we fit 15 iterations of `gcm.β`, `gcm.ρ`, `gcm.σ2` using IPOPT, followed by 10 iterations of
Newton on nuisance parameter `gcm.r`. Convergence is declared when difference of
successive loglikelihood is less than `tol`.

# Arguments
- `gcm`: A `NBCopulaCSModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes",  mu_strategy = "adaptive",
                            hessian_approximation = "limited-memory")`)

# Optional Arguments
- `tol`: Convergence tolerance for the max block iter updates (default `tol = 1e-6`).
- `maxBlockIter`: Number of maximum block iterations to update `gcm.β`, `gcm.θ` and  `gcm.r` (default `maxBlockIter = 10`).
"""
function fit!(
    gcm::NBCopulaCSModel,
    solver=Ipopt.IpoptSolver(print_level = 0, max_iter = 15, limited_memory_max_history = 20,
                            warm_start_init_point = "yes", mu_strategy = "adaptive",
                             hessian_approximation = "limited-memory");
    tol::Float64 = 1e-6,
    maxBlockIter::Int=10
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
    logl0 = MathProgBase.getobjval(optm)
    println("Converging when tol ≤ $tol (max block iter = $maxBlockIter)")
    # optimize
    for i in 1:maxBlockIter
        MathProgBase.optimize!(optm)
        logl = MathProgBase.getobjval(optm)
        update_r!(gcm)
        if abs(logl - logl0) / (1 + abs(logl0)) ≤ tol # this is faster but has wider confidence intervals
            # if abs(logl - logl0) ≤ tol # this is slower but has very tight confidence intervals
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
            break
        else
            println("Block iter $i r = $(round(gcm.r[1], digits=2))," *
            " logl = $(round(logl, digits=2)), tol = $(abs(logl - logl0) / (1 + abs(logl0)))")
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
        par[offset] = gcm.θ[k]
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
        gcm.θ[k] = par[offset]
        offset   += 1
    end
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
        grad[offset] = gcm.∇θ[k]
        offset += 1
    end
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
        H[idx] = gcm.Hθ[i, j]
        idx   += 1
    end
    # lmul!(σ, H)
    H .*= σ
end

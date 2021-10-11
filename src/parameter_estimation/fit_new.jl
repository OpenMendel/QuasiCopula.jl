"""
    fit!(gcm::GLMCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `GLMCopulaVCModel` object by MLE using a nonlinear programming solver. Start point 
should be provided in `gcm.β`, `gcm.Σ`, this is for Poisson and Bernoulli base with no additional parameters than the mean.
"""
function fit!(
        gcm::GLMCopulaVCModel{T, D, Link},
        solver=Ipopt.IpoptSolver(print_level=5)
    )  where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    initialize_model!(gcm)
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
    fit!(gcm::GLMCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `GLMCopulaVCModel` object by MLE using a nonlinear programming solver. Start point 
should be provided in `gcm.β`, `gcm.Σ`. This is for the normal base with the additional precision parameter.
"""
function fit!(
        gcm::GLMCopulaVCModel{T, D, Link},
        solver=Ipopt.IpoptSolver(print_level=5)
    )  where {T <: BlasReal, D<:Normal, Link}
    initialize_model!(gcm)
    npar = gcm.p + gcm.m + 1
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

Translate model parameters in `gcm` to optimization variables in `par` for Poisson and Bernoulli base with only mean parameters.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: GLMCopulaVCModel{T, D, Link}
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
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
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par` for the Normal base, we have the precision parameter.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: GLMCopulaVCModel{T, D, Link}
    ) where {T <: BlasReal, D<:Normal, Link}
    # β
    copyto!(par, gcm.β)
    # L
    offset = gcm.p + 1
    par[offset] = gcm.τ[1]
    offset += 1
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
        gcm :: GLMCopulaVCModel{T, D, Link},
        par :: Vector
    )  where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.Σ[k] = par[offset]
        offset   += 1
    end
    copyto!(gcm.θ, par)
    gcm
end

"""
    optimpar_to_modelpar!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm` for Normal base with additional precision parameter.
"""
function optimpar_to_modelpar!(
        gcm :: GLMCopulaVCModel{T, D, Link}, 
        par :: Vector
    ) where {T <: BlasReal, D<:Normal, Link}
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # L
    offset = gcm.p + 1
    gcm.τ[1] = par[offset]
    offset   += 1
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
    gcm  :: GLMCopulaVCModel{T, D, Link}, 
    grad :: Vector, 
    par  :: Vector
) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
optimpar_to_modelpar!(gcm, par) 
obj = loglikelihood!(gcm, true, false)
# gradient wrt β
copyto!(grad, gcm.∇β)
# gradient wrt variance comps
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

function MathProgBase.eval_grad_f(
        gcm  :: GLMCopulaVCModel{T, D, Link}, 
        grad :: Vector, 
        par  :: Vector
    ) where {T <: BlasReal, D<:Normal, Link}
    optimpar_to_modelpar!(gcm, par) 
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt dispersion
    offset = gcm.p + 1
    grad[offset] = gcm.∇τ[1]
    offset += 1
    # gradient wrt variance comps
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

function MathProgBase.hesslag_structure(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
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
    # variance components
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    return (arr1, arr2)
end

function MathProgBase.hesslag_structure(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D<:Normal, Link}
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
    # precision
    arr1[idx] = gcm.p + 1
    arr2[idx] = gcm.p + 1
    idx += 1
    # variance components
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = (gcm.p + 1) + i
            arr2[idx] = (gcm.p + 1) + j
            idx += 1
        end
    end
    return (arr1, arr2)
end
    
function MathProgBase.eval_hesslag(
        gcm   :: GLMCopulaVCModel{T, D, Link}, 
        H   :: Vector{T},
        par :: Vector{T}, 
        σ   :: T, 
        μ   :: Vector{T}
    )where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link}
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

function MathProgBase.eval_hesslag(
            gcm   :: GLMCopulaVCModel{T, D, Link}, 
            H   :: Vector{T},
            par :: Vector{T}, 
            σ   :: T, 
            μ   :: Vector{T}
        ) where {T <: BlasReal, D<:Normal, Link}
        optimpar_to_modelpar!(gcm, par)
        loglikelihood!(gcm, true, true)
        # Hβ block
        idx = 1    
        @inbounds for j in 1:gcm.p, i in 1:j
            H[idx] = gcm.Hβ[i, j]
            idx   += 1
        end
        H[idx] = gcm.Hτ[1, 1]
        idx   += 1
        # Haa block
        @inbounds for j in 1:gcm.m, i in 1:j
            H[idx] = gcm.HΣ[i, j]
            idx   += 1
        end
        # lmul!(σ, H)
        H .*= σ
    end
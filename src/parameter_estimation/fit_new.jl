"""
    fit!(gcm::GLMCopulaVCModel, solver=Ipopt.IpoptSolver(print_level=5))

Fit an `GLMCopulaVCModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `gcm.β`, `gcm.θ`, this is for Poisson and Bernoulli base with no additional parameters than the mean.
"""
function fit!(
        gcm::Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}},
        solver=Ipopt.IpoptSolver(print_level=3, tol = 10^-6, max_iter = 100,
                                    limited_memory_max_history = 50,  hessian_approximation = "limited-memory")
    )  where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
    initialize_model!(gcm)
    npar = gcm.p + gcm.m
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    # diagonal entries of Cholesky factor L should be >= 0
    lb   = fill(-Inf, npar)
    ub   = fill(Inf, npar)
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
    return optm
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par` for Poisson and Bernoulli base with only mean parameters.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        gcm :: Union{GLMCopulaVCModel{T, D, Link},  Poisson_Bernoulli_VCModel{T, VD, VL}}
    ) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
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
        gcm :: Union{GLMCopulaVCModel{T, D, Link},  Poisson_Bernoulli_VCModel{T, VD, VL}},
        par :: Vector
    )  where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset   += 1
    end
    gcm
end

function MathProgBase.initialize(
    gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel},
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}) = [:Grad, :Hess]

function MathProgBase.eval_f(
        gcm :: Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel},
        par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
    gcm  :: Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}},
    grad :: Vector,
    par  :: Vector
) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
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
obj
end

MathProgBase.eval_g(gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}, g, par) = nothing
MathProgBase.jac_structure(gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel}, J, par) = nothing

function MathProgBase.hesslag_structure(gcm::Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}}) where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
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

function MathProgBase.eval_hesslag(
        gcm   :: Union{GLMCopulaVCModel{T, D, Link}, Poisson_Bernoulli_VCModel{T, VD, VL}},
        H   :: Vector{T},
        par :: Vector{T},
        σ   :: T,
        μ   :: Vector{T}
    )where {T <: BlasReal, D<:Union{Poisson, Bernoulli}, Link, VD, VL}
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

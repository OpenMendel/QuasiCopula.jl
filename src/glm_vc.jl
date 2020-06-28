
"""
update_res!(gcm, β)

Update the residual vector according to `β` for the model object.
"""
function update_res!(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}},
    β::Vector{T}
    ) where {T <: BlasReal, D<:Normal{T}}
    mul!(gc.η, gc.X, β)
    copyto!(gc.μ, gc.η)
    fill!(gc.dμ, 1.0)
    fill!(gc.w1, 1.0)
    fill!(gc.w2, 1.0)
    fill!(gc.varμ, 1.0)
    copyto!(gc.res, gc.y)
    BLAS.axpy!(-1, gc.μ, gc.res)
    # BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

"""
update_res!(gc, β)
Update the residual vector according to `β` and the canonical inverse link to the given distribution.
"""
function update_res!(
   gc::GLMCopulaVCObs{T, D},
   β::Vector{T}
   ) where {T <: BlasReal, D}
   mul!(gc.η, gc.X, β)
   for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(canonicallink(gc.d), gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i])
       gc.dμ[i] = GLM.mueta(canonicallink(gc.d), gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.dμ[i]^2 / gc.varμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end


function update_res!(
    gcm::Union{GLMCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}},
    σinv::T
    ) where {T <: BlasReal, D}
    for j in eachindex(gc.y)
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}}
    ) where {T <: BlasReal, D}
    for j in eachindex(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}}
    ) where {T <: BlasReal, D}
    # standardize residual
    if gcm.d == Normal()
        σinv = sqrt(gcm.τ[1])# general variance
        for i in eachindex(gcm.data)
            standardize_res!(gcm.data[i], σinv)
        end
    else
        for i in eachindex(gcm.data)
            standardize_res!(gcm.data[i])
        end
    end
    nothing
end

"""
update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::GLMCopulaVCObs{T, D}) where {T<:Real, D}
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

"""

MM update to minimize ``n \\log (\\tau) - rss / 2 \\ln (\\tau) +
\\sum_i \\log (1 + \\tau * q_i)``.
"""
function update_τ(
    τ0::T,
    q::Vector{T},
    n::Integer,
    rss::T,
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    ) where T <: BlasReal
    @assert τ0 ≥ 0 "τ0 has to be nonnegative"
    τ = τ0
    for τiter in 1:maxiter
        τold = τ
        tmp = zero(T)
        for i in eachindex(q)
            tmp += q[i] / (1 + τ * q[i])
        end
        τ = (n + 2τ * tmp) / rss
        abs(τ - τold) < reltol * (abs(τold) + 1) && break
    end
    τ
end

"""
update_Σ!(gc)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
function update_Σ!(gcm::GLMCopulaVCModel{T, D}) where {T <: BlasReal, D}
    #distT = Base.typename(typeof(gcm.d)).wrapper
    update_Σ_jensen!(gcm)
end

function update_Σ_jensen!(
    gcm::GLMCopulaVCModel{T, D},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        if gcm.d  ==  Normal()
            rsstotal += abs2(norm(gcm.data[i].res))
        else
            standardize_res!(gcm.data[i])
            fill!(gcm.τ, 1.0)
        end
        GLMCopula.update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q
        end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = sum_k^m qi[k] sigmai_[k] # denom of numerator
        if gcm.d  ==  Normal()
            gcm.τ[1] = GLMCopula.update_τ(gcm.τ[1], gcm.storage_n, gcm.ntotal, rsstotal, 1)
        end
        # gcm.storage_n .= inv.(1 .+ gcm.storage_n) # 1/ (1 + sum_k^m qi[k] sigmai_[k]) # denom of numerator
        gcm.storage_n .= inv.(inv(gcm.τ[1]) .+ gcm.storage_n) # use newest τ to update Σ
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n) # store numerator = b_i / (1 + a b_i)
        gcm.Σ .*= gcm.storage_m # multiply
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.Σ ./= gcm.storage_m
        # monotonicity diagnosis
        verbose && println(sum(log, 1 .+ (gcm.QF * gcm.Σ)) -
            sum(log, 1 .+ gcm.TR * gcm.Σ))
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
            verbose && println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

"""
std_res_differential!(gc)
compute the gradient of residual vector (standardized residual) with respect to beta. For Normal it will be X
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T <: BlasReal, D<:Normal{T}}
        copyto!(gc.∇resβ, gc.X)
    gc
end

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Poisson{T}}
    ∇μβ = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇μβ[j, :] = gc.dμ[j] .* transpose(gc.X[j, :])
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * ∇μβ[j, :] - (1/2gc.varμ[j])*gc.res[j] * ∇μβ[j, :]
    end
    gc
end

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Bernoulli{T}}
∇σ2β = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇σ2β[j, :] = (1 - 2*gc.μ[j]) * gc.dμ[j] .* transpose(gc.X[j, :])  # 1.3298137228856906
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j]))*gc.dμ[j].*transpose(gc.X[j, :]) - (1/2gc.varμ[j])*gc.res[j] .* transpose(∇σ2β[j, :])
    end
    gc
end

function fit2!(
    gcm::GLMCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0),
    )
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill(Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    GLMCopula.copy_par!(gcm, MathProgBase.getsolution(optm))
    copula_loglikelihood(gcm)
end

MathProgBase.features_available(gcm::GLMCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::GLMCopulaVCModel,
    par::Vector)
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate loglikelihood
    copula_loglikelihood(gcm)[1]
end

function MathProgBase.eval_grad_f(
    gcm::GLMCopulaVCModel{T, D},
    grad::Vector,
    par::Vector) where {T<:BlasReal, D<:Normal{T}}
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate gradient
    gcm.∇β .= beta_gradient_hessian(gcm)[1]
    copyto!(grad, gcm.∇β)
    nothing
end

function MathProgBase.eval_hesslag(
    gcm::GLMCopulaVCModel{T, D},
    H::Vector{T},
    par::Vector{T},
    σ::T) where {T <: BlasReal, D<:Normal{T}}
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate Hessian
    gcm.Hβ .= beta_gradient_hessian(gcm)[2]
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

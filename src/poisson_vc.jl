
"""
update_res!(gc, β)

Update the residual vector according to `β` and the canonical inverse link to the given distribution.
"""
function update_res!(
    gc::glm_VCobs{T, D},
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


function standardize_res!(
    gc::glm_VCobs{T, D}
    ) where {T <: BlasReal, D}
    for j in 1:length(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end

function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(ϕ)), y)
#loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)

# to ensure no- infinity from loglikelihood!!
function loglik_obs(::Poisson, y, μ, wt, ϕ)
    y * log(μ) - μ
end

"""
update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::glm_VCobs{T, D}) where {T <:BlasReal, D}
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

function update_Σ_jensen!(
    gcm::glm_VCModel{T, D},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        standardize_res!(gcm.data[i])
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q # now QF is formed from the standardized residuals
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = sum_k^m qi[k] sigmai_[k] # denom of numerator
        gcm.storage_n .= inv.(1 .+ gcm.storage_n) # 1/ (1 + sum_k^m qi[k] sigmai_[k]) # denom of numerator
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
update_Σ!(gc)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
update_Σ! = update_Σ_jensen!


"""
    loglikelihood!(gc::glm_VCobs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
"""

function loglikelihood!(
    gc::glm_VCobs{T, D},
    β::Vector{T},
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    component_score = zeros(p)
    needgrad = needgrad || needhess
    update_res!(gc, β)
    standardize_res!(gc)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇Σ, 0)
        fill!(gc.∇resβ, 0.0)
        std_res_differential!(gc)
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t)
    logl = - log(1 + tsum)
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # component_score stores ∇resβ*Γ*res (standardized residual)
            BLAS.gemv!('N', Σ[k], gc.∇resβ, gc.storage_n, 1.0, component_score)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)

    # sum up the component loglikelihood
    for j = 1:length(gc.y)
        logl += loglik_obs(gc.d, gc.y[j], gc.μ[j], 1.0, 1.0)
    end
    # gradient
    if needgrad
        x = zeros(p)
        c = 0.0
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), component_score, 1.0, gc.Hβ) # only lower triangular
        end
        for j in 1:length(gc.y)
              c = gc.res[j] * gc.w1[j]
              copyto!(x, gc.X[j, :])
              BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
              BLAS.axpy!(-inv1pq, component_score, gc.∇β) # first term for each glm score
              BLAS.ger!(gc.w2[j], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
        end
        # BLAS.gemv!('N', 1.0, Diagonal(ones), component_score, -inv1pq, gc.∇β)
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
    end
    # output
    logl
end
#


function std_res_differential!(gc)
    for j in 1:length(gc.y)
        gc.∇resβ[:, j] = -inv(sqrt(gc.varμ[j]))*gc.dμ[j].*transpose(gc.X[j, :]) - (1/2gc.dμ[j])*gc.res[j] * gc.dμ[j].*transpose(gc.X[j, :])
    end
    gc
end

#-102.95885585666409 + 8.646784118270125 = -94.31207173839397 for obs 14

function loglikelihood!(
    gcm::glm_VCModel{T, D},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇Σ, 0)
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.Σ, needgrad, needhess)
        #println(logl)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    needhess && (gcm.Hβ)
    logl
end

function fit!(
    gcm::glm_VCModel{T, D},
    solver=NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000)
    ) where {T<:BlasReal, D}
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill(Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm)
    gcm
end

function MathProgBase.initialize(
    gcm::glm_VCModel{T, D},
    requested_features::Vector{Symbol}) where {T<:BlasReal, D}
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

function copy_par!(
    gcm::glm_VCModel{T, D},
    par::Vector) where {T<:BlasReal, D}
    copyto!(gcm.β, par)
    par
end

MathProgBase.features_available(gcm::glm_VCModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::glm_VCModel{T, D},
    par::Vector)  where {T<:BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate loglikelihood
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::glm_VCModel{T, D},
    grad::Vector,
    par::Vector)  where {T<:BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    nothing
end

function MathProgBase.hesslag_structure(gcm::glm_VCModel{T, D})  where {T<:BlasReal, D}
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
    gcm::glm_VCModel{T, D},
    H::Vector{T},
    par::Vector{T},
    σ::T) where {T <: BlasReal, D}
    copy_par!(gcm, par)
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

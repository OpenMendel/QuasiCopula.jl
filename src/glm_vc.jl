
"""
    loglikelihood!(gc::GLMCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
"""

function loglikelihood!(
    gc::GLMCopulaVCObs{T, D},
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
        if needgrad # component_score stores transpose(∇resβ)*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, component_score)
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
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
    end
    # output
    logl
end
#

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Poisson{T}}
    ∇μβ = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇μβ[j, :] = gc.dμ[j] .* transpose(gc.X[j, :])
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])).* ∇μβ[j, :] - (1/2gc.varμ[j])*gc.res[j] .* ∇μβ[j, :]
    end
    gc
end

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Bernoulli{T}}
∇σ2β = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇σ2β[j, :] = (1 - 2*gc.μ[j]) * gc.dμ[j] .* transpose(gc.X[j, :])
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j]))*gc.dμ[j].*transpose(gc.X[j, :]) - (1/2gc.varμ[j])*gc.res[j] .* transpose(∇σ2β[j, :])
    end
    gc
end

#20 :  -0.0612945  -0.33712

function loglikelihood!(
    gcm::GLMCopulaVCModel{T, D},
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

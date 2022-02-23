"""
    loglik_obs!(d, y, μ, wt, ϕ)
Get the loglikelihood from the GLM.jl package for each observation
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(abs(ϕ))), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = logpdf(Poisson(μ), y)

# this gets the loglikelihood from the glm.jl package for the component density
"""
    component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T <: BlasReal, D, Link}
  logl = zero(T)
    @inbounds for j in 1:gc.n
      logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], 1.0)
  end
  logl
end

### loglikelihood functions
"""
    component_loglikelihood!(gc::Union{NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}, r::T)
Calculates the loglikelihood of observing `y` given parameters for `μ` and `r` for Negative Binomial distribution using the GLM.jl package.
"""
function component_loglikelihood(gc::Union{NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}, NBCopulaCSObs{T, D, Link}}, r::T) where {T <: BlasReal, D<:NegativeBinomial{T}, Link}
    logl = zero(T)
    @inbounds for j in 1:gc.n
        logl += logpdf(D(r, r / (gc.μ[j] + r)), gc.y[j])
    end
    logl
end

"""
    component_loglikelihood!(gc::Poisson_Bernoulli_VCObs{T, VD, VL})
Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
`d` with mixed types of poisson and bernoulli distributions using the GLM.jl package.
"""
function component_loglikelihood(gc::Poisson_Bernoulli_VCObs{T, VD, VL}) where {T <: BlasReal, VD, VL}
  logl = zero(T)
    @inbounds for j in 1:gc.n
      logl += GLMCopula.loglik_obs(gc.vecd[j], gc.y[j], gc.μ[j], gc.wt[j], 1.0)
  end
  logl
end

"""
    loglikelihood!(gc, β, τ, Σ)
Calculates the loglikelihood of observing `y` given mean `μ`, for the Poisson and Bernoulli base distribution using the GLM.jl package.
"""
function loglikelihood!(
    gc::Union{GLMCopulaVCObs, Poisson_Bernoulli_VCObs},
    β::Vector{T},
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal}
    # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇Σ, 0)
    end
    needhess && fill!(gc.Hβ, 0)
    update_res!(gc, β)
    # @show gc.res
    standardize_res!(gc)
    # @show gc.res
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # evaluate copula loglikelihood
    @inbounds for k in 1:gc.m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    # loglikelihood
    logl = GLMCopula.component_loglikelihood(gc)
    tsum = dot(Σ, gc.t)
    logl += -log(1 + tsum)
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)

    if needgrad
        inv1pq = inv(1 + qsum)
        inv1pt = inv(1 + tsum) #
        # gc.∇Σ .= inv1pq * gc.q .- inv1pt * gc.t
        gc.m1 .= gc.q
        gc.m1 .*= inv1pq
        gc.m2 .= gc.t
        gc.m2 .*= inv1pt
        gc.∇Σ .= gc.m1 .- gc.m2
        # BLAS.gemv!('N', inv1pq, Float64.(Matrix(I, m, m)), gc.q, -one(T), gc.∇Σ)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
            # does adding this term to the approximation of the hessian violate negative semidefinite properties?
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            @inbounds for k in 1:gc.m
                mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
                BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
            end
            gc.added_term2 .*= inv1pq
            gc.Hβ .+= gc.added_term2
            gc.Hβ .+= GLMCopula.glm_hessian(gc)

            # hessian for vc
            fill!(gc.HΣ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.HΣ)
            BLAS.syr!('U', -one(T), gc.m1, gc.HΣ)
            copytri!(gc.HΣ, 'U')
            # gc.HΣ .= gc.m2 * transpose(gc.m2) - gc.m1 * transpose(gc.m1)
        end
        gc.storage_p2 .= gc.∇β .* inv1pq
        # @show gc.res
        gc.res .= gc.y .- gc.μ
        gc.∇β .= GLMCopula.glm_gradient(gc)
        # @show gc.res
        gc.∇β .+= gc.storage_p2
    end
    logl
end

function loglikelihood!(
    gcm::Union{GLMCopulaVCModel, Poisson_Bernoulli_VCModel},
    needgrad::Bool = false,
    needhess::Bool = false
    )
    logl = 0.0
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇Σ, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0)
        fill!(gcm.HΣ, 0)
    end
    @inbounds for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.HΣ .+= gcm.data[i].HΣ
        end
    end
    logl
end

# """
#     loglikelihood!(gc::GLMCopulaVCObs{T, D, Link}, β, τ, Σ)
# Calculates the loglikelihood of observing `y` given mean `μ`, for the Normal base distribution.
# """
# function loglikelihood!(
#     gc::GLMCopulaVCObs{T, D, Link},
#     β::Vector{T},
#     τ::T, # inverse of linear regression variance
#     Σ::Vector{T},
#     needgrad::Bool = false,
#     needhess::Bool = false
#     ) where {T <: BlasReal, D<:Normal, Link}
#     n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
#     needgrad = needgrad || needhess
#     if needgrad
#         fill!(gc.∇β, 0)
#         fill!(gc.∇τ, 0)
#         fill!(gc.∇Σ, 0)
#     end
#     needhess && fill!(gc.Hβ, 0)
#     # evaluate copula loglikelihood
#     sqrtτ = sqrt(abs(τ))
#     update_res!(gc, β)
#     standardize_res!(gc, sqrtτ)
#     rss  = abs2(norm(gc.res)) # RSS of standardized residual
#     tsum = dot(Σ, gc.t)
#
#     fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
#     std_res_differential!(gc) # this will compute ∇resβ
#
#     for k in 1:m
#         mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
#         if needgrad # ∇β stores X'*Γ*res (standardized residual)
#             BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, one(T), gc.∇β)
#         end
#         gc.q[k] = dot(gc.res, gc.storage_n) / 2
#     end
#     # test passed gc.∇β ≈ [8.846661533432094]
#     # loglikelihood
#     qsum = abs(dot(Σ, gc.q)) #  test passed: 1.8124610637883112
#     logl = GLMCopula.component_loglikelihood(gc, τ)
#     tsum = dot(Σ, gc.t)
#     logl += -log(1 + tsum)
#     logl += log(1 + qsum) # test passed: -27.795829678091444
#     # gradient
#     if needgrad
#         inv1pq = inv(1 + qsum)
#         inv1pt = inv(1 + tsum)
#         gc.m1 .= gc.q
#         gc.m1 .*= inv1pq
#         gc.m2 .= gc.t
#         gc.m2 .*= inv1pt
#         gc.∇Σ .= gc.m1 .- gc.m2
#         if needhess
#             # gc.HΣ .= gc.m2 * transpose(gc.m2) - τ * gc.m1 * transpose(gc.m1)
#             # BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
#             # gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
#             BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
#             # does adding this term to the approximation of the hessian violate negative semidefinite properties?
#             fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
#             fill!(gc.added_term2, 0.0) # fill hessian with 0
#             @inbounds for k in 1:m
#                 mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
#                 BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
#             end
#             gc.added_term2 .*= inv1pq
#             gc.Hβ .+= gc.added_term2
#             gc.Hβ .+= GLMCopula.glm_hessian(gc, β)
#             gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
#             # hessian for vc
#             fill!(gc.HΣ, 0.0)
#             BLAS.syr!('U', one(T), gc.m2, gc.HΣ)
#             BLAS.syr!('U', -one(T), gc.m1, gc.HΣ)
#             copytri!(gc.HΣ, 'U')
#         end
#         # @show gc.∇β
#         BLAS.gemv!('T', one(T), gc.X, gc.res, inv1pq, gc.∇β)
#         # @show gc.∇β
#         gc.∇β .*= sqrtτ # test passed: 0.01997344639809115
#         gc.∇τ .= (n - rss + 2qsum * inv1pq) / 2τ
#     end
#     # output
#     logl
# end
#
# function loglikelihood!(
#     gcm::GLMCopulaVCModel{T, D, Link},
#     needgrad::Bool = false,
#     needhess::Bool = false
#     ) where {T <: BlasReal, D<:Normal, Link}
#     logl = zero(T)
#     if needgrad
#         fill!(gcm.∇β, 0)
#         fill!(gcm.∇τ, 0)
#         fill!(gcm.∇Σ, 0)
#     end
#     if needhess
#         gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
#     end
#     for i in eachindex(gcm.data)
#         logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
#         if needgrad
#             gcm.∇β .+= gcm.data[i].∇β
#             gcm.∇τ .+= gcm.data[i].∇τ
#             gcm.∇Σ .+= gcm.data[i].∇Σ
#         end
#         if needhess
#             gcm.Hβ .+= gcm.data[i].Hβ
#             gcm.Hτ .+= gcm.data[i].Hτ
#             gcm.HΣ .+= gcm.data[i].HΣ
#         end
#     end
#     needhess && (gcm.Hβ .*= gcm.τ[1])
#     logl
# end

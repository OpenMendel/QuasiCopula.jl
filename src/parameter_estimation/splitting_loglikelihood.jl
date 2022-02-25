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
    loglikelihood!(gc, β, τ, θ)
Calculates the loglikelihood of observing `y` given mean `μ`, for the Poisson and Bernoulli base distribution using the GLM.jl package.
"""
function loglikelihood!(
    gc::Union{GLMCopulaVCObs, Poisson_Bernoulli_VCObs},
    β::Vector{T},
    θ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal}
    # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇θ, 0)
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
            BLAS.gemv!('T', θ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    # loglikelihood
    logl = GLMCopula.component_loglikelihood(gc)
    tsum = dot(θ, gc.t)
    logl += -log(1 + tsum)
    qsum  = dot(θ, gc.q)
    logl += log(1 + qsum)

    if needgrad
        inv1pq = inv(1 + qsum)
        inv1pt = inv(1 + tsum) #
        # gc.∇θ .= inv1pq * gc.q .- inv1pt * gc.t
        gc.m1 .= gc.q
        gc.m1 .*= inv1pq
        gc.m2 .= gc.t
        gc.m2 .*= inv1pt
        gc.∇θ .= gc.m1 .- gc.m2
        # BLAS.gemv!('N', inv1pq, Float64.(Matrix(I, m, m)), gc.q, -one(T), gc.∇θ)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
            # does adding this term to the approximation of the hessian violate negative semidefinite properties?
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            @inbounds for k in 1:gc.m
                mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
                BLAS.gemm!('T', 'N', θ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
            end
            gc.added_term2 .*= inv1pq
            gc.Hβ .+= gc.added_term2
            gc.Hβ .+= GLMCopula.glm_hessian(gc)

            # hessian for vc
            fill!(gc.Hθ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            copytri!(gc.Hθ, 'U')
            # gc.Hθ .= gc.m2 * transpose(gc.m2) - gc.m1 * transpose(gc.m1)
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
        fill!(gcm.∇θ, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0)
        fill!(gcm.Hθ, 0)
    end
    @inbounds for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.θ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇θ .+= gcm.data[i].∇θ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hθ .+= gcm.data[i].Hθ
        end
    end
    logl
end

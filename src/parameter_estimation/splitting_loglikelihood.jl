"""
    loglik_obs!(d, y, μ, wt, ϕ)
Get the loglikelihood from the GLM.jl package for each observation
"""
function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = logpdf(Poisson(μ), y)

# this gets the loglikelihood from the glm.jl package for the component density
# """
#     component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link}, τ, logl)
# Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
# `d` using the GLM.jl package.
# """
# function component_loglikelihood(gc::GLMCopulaVCObs{T, D, Link}, τ::T) where {T <: BlasReal, D, Link}
#   logl = zero(T)
#     @inbounds for j in eachindex(gc.y)
#       logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], one(T))
#   end
#   logl
# end

"""
    component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the loglikelihood of observing `y` given mean `μ`, Bernoulli or Poisson distribution using the GLM.jl package.
"""
function component_loglikelihood(gc::GLMCopulaVCObs{T, D, Link}) where {T <: BlasReal, D<:Union{Bernoulli{T}, Poisson{T}}, Link}
    logl = zero(T)
    @inbounds for j in 1:length(gc.y)
        logl += logpdf(D(gc.μ[j]), gc.y[j])
    end
    logl
end

"""
    component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the loglikelihood of observing `y` given mean `μ`, Negative Binomial distribution using the GLM.jl package.
"""
function component_loglikelihood(gc::GLMCopulaVCObs{T, D, Link}) where {T <: BlasReal, D<:NegativeBinomial{T}, Link}
    logl = zero(T)
    r = gc.d.r
    @inbounds for j in 1:length(gc.y)
        logl += logpdf(D(r, r/(gc.μ[j] + r)), gc.y[j])
    end
    logl
end

function loglikelihood!(
  gc::GLMCopulaVCObs{T, D, Link},
  β::Vector{T},
  τ::T, # inverse of linear regression variance
  Σ::Vector{T},
  needgrad::Bool = false,
  needhess::Bool = false
  ) where {T <: BlasReal, D, Link}
  n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
  needgrad = needgrad || needhess
  if needgrad
      fill!(gc.∇β, 0)
      fill!(gc.∇τ, 0)
      fill!(gc.∇Σ, 0) 
  end
  needhess && fill!(gc.Hβ, 0)
  fill!(gc.∇β, 0.0)
  update_res!(gc, β)
  standardize_res!(gc)
  fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
  std_res_differential!(gc) # this will compute ∇resβ

  # evaluate copula loglikelihood
  @inbounds for k in 1:m
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
      if needhess
          BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
          # does adding this term to the approximation of the hessian violate negative semidefinite properties?
          fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
          fill!(gc.added_term2, 0.0) # fill hessian with 0
          @inbounds for k in 1:m
              mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
              BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
          end
          gc.added_term2 .*= inv1pq
          gc.Hβ .+= gc.added_term2
          gc.Hβ .+= GLMCopula.glm_hessian(gc, β)
      end
      gc.storage_p2 .= gc.∇β .* inv1pq
      gc.∇β .= GLMCopula.glm_gradient(gc, β, τ)
      gc.∇β .+= gc.storage_p2
  end
  logl
end

function loglikelihood!(
  gcm::GLMCopulaVCModel{T, D, Link},
  needgrad::Bool = false,
  needhess::Bool = false
  ) where {T <: BlasReal, D, Link}
  logl = zero(T)
  if needgrad
      fill!(gcm.∇β, 0)
      fill!(gcm.∇Σ, 0)
  end
  if needhess
      fill!(gcm.Hβ, 0)
      fill!(gcm.HΣ, 0)
  end
  @inbounds for i in eachindex(gcm.data)
      logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
      if needgrad
          gcm.∇β .+= gcm.data[i].∇β
      end
      if needhess
          gcm.Hβ .+= gcm.data[i].Hβ
      end
  end
    if needgrad
        gcm.∇Σ .= update_∇Σ!(gcm)
    end
    if needhess
        gcm.HΣ .= update_HΣ!(gcm)
    end
  logl
end


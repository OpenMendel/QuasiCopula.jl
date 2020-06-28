"""
    copula_loglikelihood_addendum!(gc::GLMCopulaVCObs{T, D})
Calculates the parts of the loglikelihood that is particular to our density for a single observation. These parts are an addendum to the component loglikelihood.
"""

function copula_loglikelihood_addendum(gc::GLMCopulaVCObs{T, D}, Σ::Vector{T}) where {T<: BlasReal, D}
  m = length(gc.V)
  for k in 1:m
    mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
    # stores ∇resβ*Γ*res (standardized residual
    gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q = 5.858419861984103
  end
  tsum = dot(Σ, gc.t)
  logl = -log(1 + tsum)
  qsum  = dot(Σ, gc.q) # 1.8124610637883112
  inv1pq = inv(1 + qsum)
  logl += log(1 + qsum) # -0.11620740109213168
  logl
end

"""
    copula_loglikelihood_addendum!(gcm::GLMCopulaVCModel{T, D})
Calculates the parts of the loglikelihood that is particular to our density for the collection of all observations. These parts are an addendum to the component loglikelihood.
"""
function copula_loglikelihood_addendum(gcm::GLMCopulaVCModel{T, D}) where {T<: BlasReal, D}
  logl = 0.0
  Σ = gcm.Σ
  update_res!(gcm)
  standardize_res!(gcm)
  for i in 1:length(gcm.data)
    logl += copula_loglikelihood_addendum(gcm.data[i], Σ)
  end
  logl
end

# this gets the loglikelihood from the glm.jl package for the component density
"""
    component_loglikelihood!(gc::GLMCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::GLMCopulaVCObs{T, D}, τ::T, logl::T) where {T <: BlasReal, D<:Normal{T}}
    ϕ = inv(τ)
    @inbounds for j in eachindex(gc.y)
        logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], one(T), ϕ)
    end
    logl
end

function component_loglikelihood(gc::GLMCopulaVCObs{T, D}, τ::T, logl::T) where {T <: BlasReal, D<:Union{Bernoulli{T}, Poisson{T}}}
    @inbounds for j in eachindex(gc.y)
        logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], τ, one(T))
    end
    logl
end

function component_loglikelihood(gcm::GLMCopulaVCModel{T, D}) where {T <: BlasReal, D}
  logl = 0.0
  if GLM.dispersion_parameter(gcm.d)
    τ = gcm.τ[1]
  else
    τ = one(T)
  end
    for i in 1:length(gcm.data)
        logl += component_loglikelihood(gcm.data[i], τ, zero(T))
    end
    logl
end

"""
    copula_loglikelihood(gcm::GLMCopulaVCModel{T, D})
Calculates the full loglikelihood for our copula model
"""
function copula_loglikelihood(gcm::GLMCopulaVCModel{T, D}) where {T<: BlasReal, D}
  logl = 0.0
  # first get the loglikelihood from the component density with glm.jl
  logl += component_loglikelihood(gcm)
  # second we add the parts of the loglikelihood from the copula density
  logl += copula_loglikelihood_addendum(gcm)
  logl
end

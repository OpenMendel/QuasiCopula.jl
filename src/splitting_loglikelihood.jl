
"""
copula_loglikelihood_addendum!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the parts of the loglikelihood that is particular to our density for a single observation. These parts are an addendum to the component loglikelihood from the GLM density.
"""
function copula_loglikelihood_addendum(gc::GLMCopulaVCObs{T, D, Link}, Σ::Vector{T}) where {T<: BlasReal, D, Link}
m = length(gc.V)
for k in 1:m
mul!(gc.storage_n, gc.V[k], gc.res)
  gc.q[k] = dot(gc.res, gc.storage_n) / 2
end
tsum = dot(Σ, gc.t)
logl = -log(1 + tsum)
qsum  = dot(Σ, gc.q)
inv1pq = inv(1 + qsum)
logl += log(1 + qsum)
logl
end

"""
copula_loglikelihood_addendum!(gcm::GLMCopulaVCModel{T, D, Link})
Calculates the parts of the loglikelihood that is particular to our density for the collection of all observations.
These parts are an addendum to the component loglikelihood coming from the GLM density.
"""
function copula_loglikelihood_addendum(gcm::GLMCopulaVCModel{T, D, Link}) where {T<: BlasReal, D, Link}
logl = 0.0
update_res!(gcm)
standardize_res!(gcm)
for i in 1:length(gcm.data)
 logl += copula_loglikelihood_addendum(gcm.data[i], gcm.Σ)
end
logl
end

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
#loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)

# this gets the loglikelihood from the glm.jl package for the component density
"""
component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::GLMCopulaVCObs{T, D, Link},
τ::T, logl::T) where {T <: BlasReal, D, Link}
if GLM.dispersion_parameter(gc.d) == false
 τ = one(T)
end
ϕ = inv(τ)
@inbounds for j in eachindex(gc.y)
    logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], ϕ)
end
logl
end

"""
component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
Calculates the loglikelihood of observing the our density
"""
function component_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
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
copula_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link})
Calculates the full loglikelihood for our copula model for a single observation
"""
function copula_loglikelihood(gc::Union{GLMCopulaVCObs{T, D, Link}, GaussianCopulaVCObs{T, D}}, β::Vector{T}, τ::T,
Σ::Vector{T}) where {T<: BlasReal, D, Link}
# first get the loglikelihood from the component density with glm.jl
logl = 0.0
update_res!(gc, β)
standardize_res!(gc)
logl += GLMCopula.copula_loglikelihood_addendum(gc, Σ)
logl += GLMCopula.component_loglikelihood(gc, τ, zero(T))
logl
end


"""
copula_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link})
Calculates the full loglikelihood for our copula model
"""
function copula_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link}) where {T<: BlasReal, D, Link}
logl = 0.0
# first get the loglikelihood from the component density with glm.jl
logl += component_loglikelihood(gcm)
# second we add the parts of the loglikelihood from the copula density
logl += copula_loglikelihood_addendum(gcm)
logl
end

"""
loglikelihood!(gcm::GLMCopulaVCModel{T, D, Link})
Calculates the loglikelihood along with the gradient and hessian with respect to β
using our split up functions.
"""
function loglikelihood!(
gcm::GLMCopulaVCModel{T, D, Link},
needgrad::Bool = false,
needhess::Bool = false
) where {T <: BlasReal, D, Link}
logl = zero(T)
if needgrad
    fill!(gcm.∇β, 0.0)
    fill!(gcm.∇Σ, 0.0)
end
if needgrad
    gcm.∇β .= copula_gradient(gcm)
end
if needhess
    gcm.Hβ .= copula_hessian(gcm)
end
logl += copula_loglikelihood(gcm)
logl
end
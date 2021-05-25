function loglikelihood3!(
    gc::GLMCopulaVCObs{T, D, Link},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Vector{T},
    needgrad::Bool = true,
    needhess::Bool = true
    ) where {T<: BlasReal, D, Link}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    
    if needgrad
        fill!(gc.∇β, 0.0) # fill gradient with 0
        fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
        mul!(gc.storage_n, Diagonal(gc.w1), gc.res) 
        mul!(gc.storage_p1, transpose(gc.X), gc.storage_n)
        gc.storage_p1 .*= τ[1]
    end
    if gc.d == Normal()
        standardize_res!(gc, sqrtτ)
    else
        standardize_res!(gc)
    end
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    tsum = dot(Σ, gc.t)
    logl = GLMCopula.component_loglikelihood(gc, τ, zero(T))
    logl -= log(1 + tsum)
    
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        gc.storage_p2 .= gc.∇β .* inv1pq
        gc.storage_p2 .*= sqrtτ
        if needhess
            fill!(gc.Hβ, 0.0) # fill hessian with 0
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
            mul!(gc.storage_np, Diagonal(gc.w1), gc.X) 
            mul!(gc.storage_pp, transpose(gc.X), gc.storage_np)
            gc.storage_pp .*= -one(T)
            gc.Hβ .+= gc.storage_pp
        end
        # BLAS.gemv!('T', one(T), gc.X, gc.res, -inv1pq, gc.∇β)
        gc.∇β .= gc.storage_p1 .+ gc.storage_p2
    end
    # output
    logl
end

function loglikelihood3!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, GaussianCopulaLMMModel{T}},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T<: BlasReal, D, Link}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇Σ, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0.0)
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood3!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    logl
end

# """
# loglik_obs!(d, y, μ, wt, ϕ)
# Get the loglikelihood from the GLM.jl package for each observation
# """
# function loglik_obs end

# loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
# loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
# loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
# loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
# loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(ϕ)), y)
# #loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)
# loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*logpdf(Poisson(μ), y)

# # this gets the loglikelihood from the glm.jl package for the component density
# """
# component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
# Calculates the loglikelihood of observing `y` given mean `μ`, a distribution
# `d` using the GLM.jl package.
# """
# function component_loglikelihood(gc::GLMCopulaVCObs{T, D, Link},
# τ::T, logl::T) where {T <: BlasReal, D, Link}
#     if GLM.dispersion_parameter(gc.d) == false
#         τ = one(T)
#     end
#     ϕ = inv(τ)
#     @inbounds for j in eachindex(gc.y)
#         logl += GLMCopula.loglik_obs(gc.d, gc.y[j], gc.μ[j], gc.wt[j], ϕ)
#     end
#     logl
# end

# """
# component_loglikelihood!(gc::GLMCopulaVCObs{T, D, Link})
# Calculates the loglikelihood of observing the our density
# """
# function component_loglikelihood(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
#     logl = 0.0
#     if GLM.dispersion_parameter(gcm.d)
#         τ = gcm.τ[1]
#     else
#         τ = one(T)
#     end
#     for i in 1:length(gcm.data)
#         logl += component_loglikelihood(gcm.data[i], τ, zero(T))
#     end
#     logl
# end
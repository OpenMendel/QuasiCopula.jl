
"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta. For Normal it will be X.
"""
function std_res_differential!(gc::Union{GaussianCopulaVCObs{T, D}, GLMCopulaVCObs{T, D}}) where {T <: BlasReal, D<:Normal{T}}
    gc.∇resβ .= gc.X
    gc.∇resβ .*= -one(T)
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Poisson.
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Poisson{T}}
    for j in 1:length(gc.y)
        gc.∇μβ[j, :] = gc.dμ[j] .* gc.X[j, :]
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * gc.∇μβ[j, :] - (1/2gc.varμ[j])*gc.res[j] * gc.∇μβ[j, :]
    end
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Negative Binomial.
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:NegativeBinomial{T}}
    for j in 1:length(gc.y)
        gc.∇μβ[j, :] = gc.dμ[j] .* gc.X[j, :]
        gc.∇σ2β[j, :] = (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) * gc.∇μβ[j, :]
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * gc.∇μβ[j, :] - (0.5 * inv(gc.varμ[j])) * gc.res[j] * gc.∇σ2β[j, :]
    end
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Bernoulli.
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Bernoulli{T}}
    for j in 1:length(gc.y)
        gc.∇σ2β[j, :] = (1 - 2 * gc.μ[j]) * gc.dμ[j] .* gc.X[j, :]
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * gc.dμ[j] .* gc.X[j, :] - (1 / 2gc.varμ[j]) * gc.res[j] .* gc.∇σ2β[j, :]
    end
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Binomial.
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Binomial{T}}
    for j in 1:length(gc.y)
        gc.∇μβ[j, :] = gc.varμ[j] .* gc.X[j, :]
        gc.∇σ2β[j, :] = (1 - 2*gc.μ[j]) * gc.dμ[j] .* gc.X[j, :]
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * gc.∇μβ[j, :] - (1 / 2gc.varμ[j]) * gc.res[j] .* gc.∇σ2β[j, :]
    end
    nothing
end

"""
    glm_gradient(gc::GLMCopulaVCObs{T, D, Link})
Calculates the gradient with respect to beta for our the glm portion for one obs. Keeps the residuals standardized.
"""
function glm_gradient(gc::Union{GaussianCopulaVCObs{T, D}, GLMCopulaVCObs{T, D, Link}}, β::Vector, τ) where {T<:Real, D, Link}
    (n, p) = size(gc.X)
    update_res!(gc, β)
    mul!(gc.storage_n, Diagonal(gc.w1), gc.res) 
    mul!(gc.storage_p1, transpose(gc.X), gc.storage_n)
    gc.storage_p1 .*= τ[1]
    gc.storage_p1
end

"""
    copula_gradient_addendum(gc)
Compute the part of gradient specific to copula density with respect to beta for a single observation
"""
function copula_gradient_addendum(
    gc::GLMCopulaVCObs{T, D, Link},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D, Link}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    if gc.d  ==  Normal()
            sqrtτ = sqrt.(τ[1])
            standardize_res!(gc, sqrtτ)
        else
            sqrtτ = 1.0
            standardize_res!(gc)
        end
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # evaluate copula loglikelihood
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end

    qsum  = dot(Σ, gc.q)
    # gradient
        denom = 1 .+ qsum
        inv1pq = inv(denom) #0.9625492359318475
        # component_score = W1i(Yi -μi)
        gc.storage_p2 .= gc.∇β .* inv1pq
        gc.storage_p2 .*= sqrtτ # since we already standardized it above
        gc.storage_p2
end

"""
    copula_gradient(gc::GLMCopulaVCObs{T, D})
Calculates the full gradient with respect to beta for one observation
"""
function copula_gradient(gc::GLMCopulaVCObs{T, D, Link}, β, τ, Σ)  where {T<:BlasReal, D, Link}
    fill!(gc.∇β, 0.0)
    gc.∇β .= glm_gradient(gc, β, τ) .+ GLMCopula.copula_gradient_addendum(gc, β, τ[1], Σ)
    gc.∇β
end

"""
    copula_gradient(gcm::GLMCopulaVCModel{T, D})
Calculates the full gradient with respect to beta for our copula model
"""
function copula_gradient(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}
    ) where {T <: BlasReal, D}
        fill!(gcm.∇β, 0.0)
        if GLM.dispersion_parameter(gcm.d) == false
            fill!(gcm.τ, 1.0)
        end
        update_res!(gcm)
    for i in 1:length(gcm.data)
        gcm.data[i].∇β .= copula_gradient(gcm.data[i], gcm.β, gcm.τ, gcm.Σ)
        gcm.∇β .+= gcm.data[i].∇β
    end
    gcm.∇β
end


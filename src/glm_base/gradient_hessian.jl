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
    @inbounds for j in 1:length(gc.y)
        gc.∇μβ[j, :] .= gc.dμ[j] .* gc.X[j, :]
        gc.∇resβ[j, :] .= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) .* gc.∇μβ[j, :]
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
    glm_hessian(gc)
Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
"""
function glm_hessian(gc::Union{GLMCopulaVCObs{T, D, Link}, GaussianCopulaVCObs{T, D}}, β) where {T <: BlasReal, D, Link}
    update_res!(gc, β)
    mul!(gc.storage_np, Diagonal(gc.w2), gc.X) 
    mul!(gc.storage_pp, transpose(gc.X), gc.storage_np)
    gc.storage_pp .*= -one(T)
end

"""
    update_∇Σ!(gcm)

Update Σ gradient for Newton's Algorithm, given β.
"""
function update_∇Σ!(
    gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    rsstotal = zero(T)
    @inbounds for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))  # needed for updating τ in normal case
        standardize_res!(gcm.data[i])            # standardize the residuals GLM variance(μ)
        GLMCopula.update_quadform!(gcm.data[i]) # with standardized residuals
        gcm.QF[i, :] = gcm.data[i].q
    end
    mul!(gcm.storage_n, gcm.QF, gcm.Σ)
    gcm.storage_n .= inv.(1 .+ gcm.storage_n)
    
    mul!(gcm.storage_n2, gcm.TR, gcm.Σ)
    gcm.storage_n2 .= inv.(1 .+ gcm.storage_n2)
    gcm.storage_n2 .*= -one(T)
    
    mul!(gcm.∇Σ1, transpose(gcm.QF), gcm.storage_n)
    mul!(gcm.∇Σ2, transpose(gcm.TR), gcm.storage_n2)
    gcm.∇Σ .+= gcm.∇Σ1
    gcm.∇Σ .+= gcm.∇Σ2
end

"""
    update_HΣ!(gcm)

Update Σ Hessian for Newton's Algorithm, given β.
"""
function update_HΣ!(
    gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    fill!(gcm.HΣ, 0.0)
    gcm.diagonal_n .= Diagonal(gcm.storage_n)
    mul!(gcm.hess1, transpose(gcm.QF), gcm.diagonal_n)
    
    mul!(gcm.HΣ1, gcm.hess1, transpose(gcm.hess1))
    gcm.HΣ1 .*= -one(T)
    
    gcm.diagonal_n .= Diagonal(gcm.storage_n2)
    mul!(gcm.hess2, transpose(gcm.TR), gcm.diagonal_n)
    mul!(gcm.HΣ2, gcm.hess2, transpose(gcm.hess2))
    gcm.HΣ .+= gcm.HΣ1
    gcm.HΣ .+= gcm.HΣ2
end
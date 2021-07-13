"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta. For Normal it will be X.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link},GLMCopulaARObs{T, D, Link}}) where {T <: BlasReal, D<:Normal{T}, Link}
    gc.∇resβ .= gc.X
    gc.∇resβ .*= -one(T)
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Poisson.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
    @inbounds for i in 1:size(gc.X, 2)
        @simd for j in 1:length(gc.y)
            gc.∇resβ[j, i] = gc.X[j, i]
            gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
        end
    end
    nothing
end
"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Negative Binomial.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @inbounds for j in 1:length(gc.y)
        gc.∇μβ[j, :] .= gc.dμ[j] .* @view(gc.X[j, :])
        gc.∇σ2β[j, :] .= (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) * @view(gc.∇μβ[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) * @view(gc.∇μβ[j, :]) - (0.5 * inv(gc.varμ[j])) * gc.res[j] * @view(gc.∇σ2β[j, :])
    end
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Bernoulli.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}) where {T<: BlasReal, D<:Bernoulli{T}, Link}
    @inbounds for j in 1:length(gc.y)
        gc.∇σ2β[j, :].= (1 - 2 * gc.μ[j]) * gc.dμ[j] .* @view(gc.X[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) * gc.dμ[j] .* @view(gc.X[j, :]) - (0.5 * inv(gc.varμ[j])) * gc.res[j] .* @view(gc.∇σ2β[j, :])
    end
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Binomial.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}) where {T<: BlasReal, D<:Binomial{T}, Link}
    @inbounds for j in 1:length(gc.y)
        gc.∇μβ[j, :] .= gc.varμ[j] .* @view(gc.X[j, :])
        gc.∇σ2β[j, :] .= (1 - 2*gc.μ[j]) * gc.dμ[j] .* @view(gc.X[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) * gc.∇μβ[j, :] - (1 / 2gc.varμ[j]) * gc.res[j] .* @view(gc.∇σ2β[j, :])
    end
    nothing
end

"""
    glm_gradient(gc::GLMCopulaVCObs{T, D, Link})
Calculates the gradient with respect to beta for our the glm portion for one obs. Keeps the residuals standardized.
"""
function glm_gradient(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}, β::Vector, τ) where {T<:Real, D, Link}
    (n, p) = size(gc.X)
    gc.storage_n .= gc.w1 .* gc.res
    mul!(gc.storage_p1, transpose(gc.X), gc.storage_n)
    gc.storage_p1 .*= τ[1]
    gc.storage_p1
end

"""
    glm_hessian(gc, β)
Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
"""
function glm_hessian(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}, β) where {T <: BlasReal, D, Link}
    mul!(gc.storage_np, Diagonal(gc.w2), gc.X) 
    BLAS.gemm!('T', 'N', -T(1), gc.X, gc.storage_np, T(0), gc.storage_pp)
end

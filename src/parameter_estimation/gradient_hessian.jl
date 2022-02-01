"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta. For Normal it will be X.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T <: BlasReal, D<:Normal{T}, Link}
    gc.∇resβ .= gc.X
    gc.∇resβ .*= -one(T)
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Negative Binomial.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, NBCopulaARObs{T, D, Link}}
    ) where {T<: BlasReal, D<:NegativeBinomial{T}, Link}
    @inbounds for j in 1:length(gc.y)
        gc.∇μβ[j, :] .= gc.dμ[j] .* @view(gc.X[j, :])
        gc.∇σ2β[j, :] .= (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) .* @view(gc.∇μβ[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) .* @view(gc.∇μβ[j, :]) .- (0.5 * inv(gc.varμ[j])) .* gc.res[j] .* @view(gc.∇σ2β[j, :])
    end
    nothing
end


"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Poisson.
"""
function std_res_differential!(gc::Poisson_Bernoulli_VCObs{T, VD, VL}) where {T<: BlasReal, VD, VL}
    fill!(gc.∇resβ, 0.0)
    fill!(gc.∇μβ, 0.0)
    @inbounds for i in 1:Integer(size(gc.X, 2) / 2)
            # first is poisson
            gc.∇resβ[1, i] = gc.X[1, i]
            gc.∇resβ[1, i] *= -(inv(sqrt(gc.varμ[1])) + (0.5 * inv(gc.varμ[1])) * gc.res[1]) * gc.dμ[1]
        end
            # second is bernoulli
            gc.∇μβ[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))] .= gc.varμ[2] .* @view(gc.X[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))])
            gc.∇σ2β[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))].= gc.varμ[2] * (1 - 2 * gc.μ[2]) .* @view(gc.X[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))])
            gc.∇resβ[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))] .= -inv(sqrt(gc.varμ[2])) .* @view(gc.∇μβ[2, Integer((size(gc.X, 2) / 2) + 1):Integer(size(gc.X, 2))]).- (0.5 * inv(sqrt(gc.varμ[2]))) .* gc.res[2] .* @view(gc.∇σ2β[2, Integer((size(gc.X, 2) / 2) + 1):Integer((size(gc.X, 2)))])
    nothing
end

"""
    std_res_differential!(gc)
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Poisson.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Poisson{T}, Link}
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
compute the gradient of residual vector ∇resβ (standardized residual) with respect to beta, for Bernoulli.
"""
function std_res_differential!(gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, GLMCopulaCSObs{T, D, Link}}) where {T<: BlasReal, D<:Bernoulli{T}, Link}
    @inbounds for j in 1:length(gc.y)
        gc.∇μβ[j, :] .= gc.varμ[j] .* @view(gc.X[j, :])
        gc.∇σ2β[j, :].= gc.varμ[j] * (1 - 2 * gc.μ[j]) .* @view(gc.X[j, :])
        gc.∇resβ[j, :] .= -inv(sqrt(gc.varμ[j])) .* @view(gc.∇μβ[j, :]) .- (0.5 * inv(sqrt(gc.varμ[j]))) .* gc.res[j] .* @view(gc.∇σ2β[j, :])
    end
    nothing
end

"""
    glm_gradient(gc::GLMCopulaVCObs{T, D, Link}, β)
Calculates the gradient with respect to beta for our the glm portion for one obs. Keeps the residuals standardized.
"""
function glm_gradient(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, NBCopulaARObs{T, D, Link}, Poisson_Bernoulli_VCObs{T, VD, VL}, GLMCopulaCSObs{T, D, Link}}) where {T<:Real, D, Link, VD, VL}
    gc.storage_n .= gc.w1 .* gc.res
    mul!(gc.storage_p1, transpose(gc.X), gc.storage_n)
    gc.storage_p1
end

"""
    glm_hessian(gc, β)

Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
"""
function glm_hessian(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}, NBCopulaARObs{T, D, Link}, Poisson_Bernoulli_VCObs{T, VD, VL}, GLMCopulaCSObs{T, D, Link}}) where {T <: BlasReal, D, Link, VD, VL}
    mul!(gc.storage_np, Diagonal(gc.w2), gc.X)
    BLAS.gemm!('T', 'N', -1.0, gc.X, gc.storage_np, 0.0, gc.storage_pp)
end

"""
    update_∇Σ!(gcm)

Update Σ gradient for Newton's Algorithm, given β.
"""
function update_∇Σ!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    @inbounds for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i])            # standardize the residuals GLM variance(μ)
        GLMCopula.update_quadform!(gcm.data[i]) # with standardized residuals
        gcm.QF[i, :] .= gcm.data[i].q
    end
    BLAS.gemv!('N', T(1), gcm.QF,  gcm.Σ, T(0), gcm.storage_n)
    gcm.storage_n .= inv.(1 .+ gcm.storage_n)
    BLAS.gemv!('N', T(1), gcm.TR,  gcm.Σ, T(0), gcm.storage_n2)
    gcm.storage_n2 .= inv.(1 .+ gcm.storage_n2)
    BLAS.gemv!('T', T(1), gcm.QF, gcm.storage_n, T(0), gcm.∇Σ)
    BLAS.gemv!('T', -T(1), gcm.TR, gcm.storage_n2, T(1), gcm.∇Σ)
end

"""
    update_HΣ!(gcm)

Update Σ Hessian for Newton's Algorithm, given β.
"""
function update_HΣ!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    fill!(gcm.HΣ, 0.0)
    @inbounds for j in 1:gcm.m
        @simd for i in 1:length(gcm.storage_n)
            gcm.hess1[j, i] = gcm.QF[i, j] * gcm.storage_n[i]
            gcm.hess2[j, i] = gcm.TR[i, j] * gcm.storage_n2[i]
        end
    end
    BLAS.gemm!('N', 'T', -T(1), gcm.hess1, gcm.hess1, T(0), gcm.HΣ)
    BLAS.gemm!('N', 'T', T(1), gcm.hess2, gcm.hess2, T(1), gcm.HΣ)
end

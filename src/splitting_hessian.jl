# # """
# # glm_hessian(gc)
# # Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
# # """
function glm_hessian(gc::Union{GLMCopulaVCObs{T, D, Link}, GaussianCopulaVCObs{T, D}}, β) where {T <: BlasReal, D, Link}
    update_res!(gc, β)
    mul!(gc.storage_np, Diagonal(gc.w2), gc.X) 
    mul!(gc.storage_pp, transpose(gc.X), gc.storage_np)
    gc.storage_pp .*= -one(T)
end

"""
hessian_copula_addendum(gc)
compute the part of the hessian relevant to just the copula density with respect to beta.
"""
function hessian_copula_addendum(
        gc::Union{GLMCopulaVCObs{T, D, Link}, GaussianCopulaVCObs{T, D}},
        β::Vector{T},
        τ::T,
        Σ::Vector{T}
        ) where {T <: BlasReal, D, Link}
        n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
        fill!(gc.∇β, 0.0) # fill gradient with 0
        fill!(gc.Hβ, 0.0) # fill hessian with 0
        fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
        update_res!(gc, β)
        if gc.d  ==  Normal()
            sqrtτ = sqrt.(τ[1])
            standardize_res!(gc, sqrtτ)
        else
            sqrtτ = 1.0
            standardize_res!(gc)
        end
        std_res_differential!(gc) # this will compute ∇resβ
        tsum = dot(Σ, gc.t)
        for k in 1:m
            mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
            gc.q[k] = dot(gc.res, gc.storage_n) / 2
        end
        # @show gc.∇β this is numerator vector 
        #@show gc.q
        qsum  = dot(Σ, gc.q)
        #@show qsum
        inv1pq = inv(1 + qsum)
        #@show inv1pq
        BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular

        # does adding this term to the approximation of the hessian violate negative semidefinite properties?
        fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
        fill!(gc.added_term2, 0.0) # fill hessian with 0
        for k in 1:m
            mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
            BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
        end
        #@show added_term2
        gc.added_term2 .*= inv1pq
        #@show added_term2
        gc.Hβ .+= gc.added_term2
        gc.Hβ
    end


"""
copula_hessian(gcm)
Compute the full hessian for our copula model
"""
function copula_hessian(gcm::Union{GLMCopulaVCModel{T, D, Link}, GaussianCopulaVCModel{T, D}}) where {T <: BlasReal, D, Link}
    fill!(gcm.Hβ, 0.0)
    if GLM.dispersion_parameter(gcm.d) == false
        fill!(gcm.τ, 1.0)
    end
    for i in 1:length(gcm.data)
        hessian_copula_addendum(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
        gcm.data[i].Hβ .+= glm_hessian(gcm.data[i], gcm.β)
        gcm.Hβ .+= gcm.data[i].Hβ
    end
    gcm.Hβ .*= gcm.τ[1]
    gcm.Hβ
end

#### with respect to variance component vector

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
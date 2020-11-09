"""
glm_hessian(gc)
Compute the part of the hessian relevant to the glm density with respect to beta for a single obs
"""
function hessian_glm(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}},
    β) where {T <: BlasReal, D}
    score = zeros(size(gc.∇β))
    inform = zeros(size(gc.∇β, 1), size(gc.∇β, 1))
    update_res!(gc, β)
        for j in 1:length(gc.y)
              c = gc.dμ[j]^2 / gc.varμ[j]
              BLAS.ger!(-c, gc.X[j, :], gc.X[j, :], inform) # inform = inform + c * x * x'
        end
   inform
end

"""
glm_hessian(gcm)
Compute the part of the hessian relevant to the glm density with respect to beta for the model object
"""
function hessian_glm(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}
    ) where {T <: BlasReal, D}
        fill!(gcm.Hβ, 0.0)
        if GLM.dispersion_parameter(gcm.d) == false
                fill!(gcm.τ, 1.0)
        end
    for i in 1:length(gcm.data)
        gcm.data[i].Hβ .= glm_hessian(gcm.data[i], gcm.β)
        gcm.Hβ .+= gcm.data[i].Hβ
    end
    gcm.Hβ .*= gcm.τ[1]
    gcm.Hβ
end

"""
hessian_copula(gc)
compute the part of the hessian relevant to the copula density with respect to beta.
"""
function hessian_copula_addendum(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    update_res!(gc, β)
    if gc.d  ==  Normal()
        sqrtτ = sqrt.(τ[1])
        standardize_res!(gc, sqrtτ)
    else
        sqrtτ = 1.0
        standardize_res!(gc)
    end
    fill!(gc.∇β, 0.0) # fill gradient with 0
    fill!(gc.Hβ, 0) # fill hessian with 0
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ
    tsum = dot(Σ, gc.t)
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    #@show gc.∇β
    #@show gc.q
    qsum  = dot(Σ, gc.q)
    #@show qsum
        inv1pq = inv(1 + qsum)
    #@show inv1pq
        BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular

# does adding this term to the approximation of the hessian violate negative semidefinite properties?
#     added_term_numerator = zeros(n, p)
#     added_term2 = zeros(p, p)
#     for k in 1:m
#         mul!(added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
#         BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, added_term_numerator, 1.0, added_term2)
#     end
#     #@show added_term2
#     added_term2 .*= inv1pq
#     #@show added_term2
#     gc.Hβ .+= added_term2
    gc.Hβ
end


"""
copula_hessian(gcm)
Compute the full hessian for our copula model
"""
function copula_hessian(
    gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}
    ) where {T <: BlasReal, D}
        fill!(gcm.Hβ, 0.0)
        if GLM.dispersion_parameter(gcm.d) == false
                fill!(gcm.τ, 1.0)
        end
    for i in 1:length(gcm.data)
        gcm.data[i].Hβ .= hessian_copula_addendum(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
        gcm.data[i].Hβ .+= hessian_glm(gcm.data[i], gcm.β)
        gcm.Hβ .+= gcm.data[i].Hβ
    end
    gcm.Hβ .*= gcm.τ[1]
    gcm.Hβ
end

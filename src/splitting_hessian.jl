"""
beta_gradient_hessian(gc)
compute the gradient and hessian with respect to beta
"""
function beta_hessian_term1(
    gc::GLMCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    update_res!(gc, β)
    if gc.d  ==  Normal()
        sqrtτ = sqrt.(τ[1]) #sqrtτ = 0.018211123993574548
        standardize_res!(gc, sqrtτ)
    else
        sqrtτ = 1.0
        standardize_res!(gc)
    end
    rss = abs2(norm(gc.res)) # RSS of standardized residual
    fill!(gc.∇β, 0.0) # fill gradient with 0
    fill!(gc.Hβ, 0) # fill hessian with 0
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    GLMCopula.std_res_differential!(gc) # this will compute ∇resβ
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t)
    #logl = - log(1 + tsum)#  -1.150267324540463
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q = 5.858419861984103
    end
    #gc.∇β = -8.84666153343209
    qsum  = dot(Σ, gc.q) # 1.8124610637883112
    # gradient
#         x = zeros(p)
#         (score, inform) = (zeros(p), zeros(p, p))
        inv1pq = inv(1 + qsum)
        BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular =  -9.894316220056414
        gc.Hβ
end

function beta_hessian_term2(
    gc::GLMCopulaVCObs{T, D},
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

function beta_hessian_term2(
    gcm::GLMCopulaVCModel{T, D}
    ) where {T <: BlasReal, D}
        fill!(gcm.Hβ, 0.0)
        if GLM.dispersion_parameter(gcm.d) == false
                fill!(gcm.τ, 1.0)
        end
    for i in 1:length(gcm.data)
        gcm.data[i].Hβ .= beta_hessian_term2(gcm.data[i], gcm.β)
        gcm.Hβ .+= gcm.data[i].Hβ
    end
    gcm.Hβ .*= gcm.τ[1]
    gcm.Hβ
end

function beta_hessians(
    gcm::GLMCopulaVCModel{T, D}
    ) where {T <: BlasReal, D}
        fill!(gcm.Hβ, 0.0)
        if GLM.dispersion_parameter(gcm.d) == false
                fill!(gcm.τ, 1.0)
        end
    for i in 1:length(gcm.data)
        gcm.data[i].Hβ .= beta_hessian_term1(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
        gcm.data[i].Hβ .+= beta_hessian_term2(gcm.data[i], gcm.β)
        gcm.Hβ .+= gcm.data[i].Hβ
    end
    gcm.Hβ .*= gcm.τ[1]
    gcm.Hβ
end

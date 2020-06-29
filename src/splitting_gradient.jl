
"""
std_res_differential!(gc)
compute the gradient of residual vector (standardized residual) with respect to beta. For Normal it will be X
"""
function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T <: BlasReal, D<:Normal{T}}
        copyto!(gc.∇resβ, gc.X)
    gc
end

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Poisson{T}}
    ∇μβ = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇μβ[j, :] = gc.dμ[j] .* transpose(gc.X[j, :])
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j])) * ∇μβ[j, :] - (1/2gc.varμ[j])*gc.res[j] * ∇μβ[j, :]
    end
    gc
end

function std_res_differential!(gc::GLMCopulaVCObs{T, D}) where {T<: BlasReal, D<:Bernoulli{T}}
∇σ2β = zeros(size(gc.X))
    for j in 1:length(gc.y)
        ∇σ2β[j, :] = (1 - 2*gc.μ[j]) * gc.dμ[j] .* transpose(gc.X[j, :])  # 1.3298137228856906
        gc.∇resβ[j, :] = -inv(sqrt(gc.varμ[j]))*gc.dμ[j] .* transpose(gc.X[j, :]) - (1/2gc.varμ[j])*gc.res[j] .* transpose(∇σ2β[j, :])
    end
    gc
end

"""
beta_gradient_hessian(gc)
compute the gradient and hessian with respect to beta
"""
function beta_gradient_hessians(
    gc::GLMCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D<:Normal{T}}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    component_score = zeros(n)
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
        x = zeros(p)
        inv1pq = inv(1 + qsum)
        BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular =  -9.894316220056414
        for j in 1:length(gc.y)
            # the first term in the score, where res is standardized!
             component_score[j] = gc.res[j] * sqrt(gc.varμ[j]) # note for normal the last 2 terms go away so its just the residual
             BLAS.ger!(-1.0, gc.X[j, :], gc.X[j, :], gc.Hβ) # gc.Hβ = gc.Hβ - xtx # part from glm_regress
         end
        # component_score = W1i(Yi -μi)
        BLAS.gemv!('T', 1.0, gc.X, component_score, -inv1pq, gc.∇β) # gc.∇β = 1.0967717536346686 subtract the second term from the component score
        # BLAS.gemv!('N', 1.0, Diagonal(ones), component_score, -inv1pq, gc.∇β)
        gc.∇β .*= sqrtτ # 0.019973446398091146 note for normal distribution we need this but other dist without dispersion we made it 1
        gc.∇β, gc.Hβ
end

function beta_gradient_hessians(
    gcm::GLMCopulaVCModel{T, D}
    ) where {T <: BlasReal, D}
    logl = zero(T)
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇Σ, 0.0)
        fill!(gcm.∇τ, 0.0)
        fill!(gcm.Hβ, 0.0)
    for i in 1:length(gcm.data)
        score, hess = beta_gradient_hessians(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
        @show score
        @show hess
        gcm.∇β .+= score
        gcm.Hβ .+= hess
    end
    @show gcm.Hβ
    gcm.Hβ .*= gcm.τ[1]
    gcm.∇β, gcm.Hβ
end

"""
beta_gradient_term2(gc)
compute the gradient and hessian with respect to beta
"""
function beta_gradient_term2(
    gc::GLMCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    #component_score = zeros(p)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    if gc.d  ==  Normal()
        sqrtτ = sqrt.(τ[1]) #sqrtτ = 0.018211123993574548
        standardize_res!(gc, sqrtτ)
    else
        sqrtτ = 1.0
        standardize_res!(gc)
    end
    rss = abs2(norm(gc.res)) # RSS of standardized residual
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    GLMCopula.std_res_differential!(gc) # this will compute ∇resβ
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t)
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
        gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q = 5.858419861984103
    end
    @show gc.∇β
    #gc.∇β = -8.84666153343209
    qsum  = dot(Σ, gc.q) # 1.8124610637883112
    # gradient
        x = zeros(p)
        denom = 1 .+ qsum
        @show denom
        inv1pq = inv(denom)
        # component_score = W1i(Yi -μi)
        secondterm = gc.∇β .* inv1pq
        secondterm .*= sqrtτ
        secondterm
end

function beta_gradient_term2(
    gcm::GLMCopulaVCModel{T, D}
    ) where {T <: BlasReal, D}
    fill!(gcm.∇β, 0.0)
    for i in 1:length(gcm.data)
        beta_gradient_term2(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
        gcm.∇β .+= gcm.data[i].∇β
    end
    gcm.∇β
end

function hardcode_term1(gc::GLMCopulaVCObs{T, D}, τ::T) where {T,  D<:Normal{T}}
    #standardize_res!(gc)
    fill!(gc.∇β, 0.0)
    fill!(gc.varμ, inv(τ))
    gc.∇β .+= transpose(gc.X)*(gc.y - gc.μ)
    gc.∇β .*= τ
    gc.∇β
end

function hardcode_term1(gc::GLMCopulaVCObs{T, D}, τ::T) where {T,  D<:Union{Poisson{T}, Bernoulli{T}}}
    fill!(gc.∇β, 0.0)
    gc.∇β .+= transpose(gc.X)*(gc.y - gc.μ)
    gc.∇β
    #gc.∇β .+= transpose(gc.X)*Diagonal(gc.dμ ./ sqrt.(gc.varμ))*gc.res
end
## poisson test data
# julia> score # poisson test data
# 1-element Array{Float64,1}:
#  -2.1032064978498966e-12
## logistic test data
# julia> hardcode_term1(gc, gcm.τ[1])
# 1-element Array{Float64,1}:
#  2.087476023952942e-27
# function hardcode_term1(gcm::GLMCopulaVCModel{T, D}) where {T,  D}
#     fill!(gcm.∇β, 0.0)
#     for i in 1:length(gcm.data)
#         gcm.data[i].∇β .= hardcode_term1(gcm.data[i], gcm.τ[1])
#         gcm.∇β .+= gcm.data[i].∇β
#     end
#     gcm.∇β
# end


# we checked if the gradient of each part of the normal is good and it is
# then we tried to match the parts of the gradient for the poisson and logistic example data
# now i think i just need to check if the model object for each part of the gradient works and then put it in the big gradient function
"""
    copula_gradient(gcm::GLMCopulaVCModel{T, D})
Calculates the full loglikelihood for our copula model
"""
function copula_gradient(gcm::GLMCopulaVCModel{T, D}) where {T<: BlasReal, D}
  fill!(gcm.∇β, 0.0)
  # first get the loglikelihood from the component density with glm.jl
  for i in 1:length(gcm.data)
      gcm.data[i].∇β = hardcode_term1(gcm.data[i], gcm.τ[1])

      gcm.data[i].∇β .+= beta_gradient_term2(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ)
  end
end

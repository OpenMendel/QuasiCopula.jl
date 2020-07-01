
"""
    loglikelihood!(gc::GLMCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
For each logpdf from Normal, Gamma, and InverseGaussian, we scale by dispersion.
"""

function loglikelihood!(
    gc::GLMCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D <:Normal{T}}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    component_score = zeros(n)
    needgrad = needgrad || needhess
    update_res!(gc, β)
    if gc.d  ==  Normal()
        sqrtτ = sqrt.(τ[1]) #sqrtτ = 0.018211123993574548
        standardize_res!(gc, sqrtτ)
    else
        sqrtτ = 1.0
        standardize_res!(gc)
    end
    if needgrad
        fill!(gc.∇β, 0.0)
        fill!(gc.∇Σ, 0.0)
        fill!(gc.∇τ, 0.0)
        rss = abs2(norm(gc.res)) # RSS of standardized residual
        fill!(gc.∇resβ, 0.0)
        GLMCopula.std_res_differential!(gc)
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t) # 4.332948841735151 gcm_all
    logl = - log(1 + tsum)#  -1.150267324540463
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # stores ∇resβ*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q = 5.858419861984103
    end
    #gc.∇β = -8.84666153343209
    qsum  = dot(Σ, gc.q) # 1.8124610637883112 < normal , 4.25375319607831 < gcm_all logistic
    logl += log(1 + qsum) # -0.11620740109213168
    logl += GLMCopula.component_loglikelihood(gc, τ, 0.0)
    # gradient
    if needgrad
        x = zeros(p)
        inv1pq = inv(1 + qsum) # 0.1903401173748426
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular =  -9.894316220056414
        end
        for j in 1:length(gc.y)
            # the first term in the score, where res is standardized!
             component_score[j] = gc.res[j] * gc.w1[j] * inv(sqrt(gc.varμ[j]))
             BLAS.ger!(gc.w2[j], gc.X[j, :], gc.X[j, :], gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
         end
        # component_score = W1i(Yi -μi)
        BLAS.gemv!('T', 1.0, gc.X, component_score, -inv1pq, gc.∇β) # gc.∇β = 1.0967717536346686
        # BLAS.gemv!('N', 1.0, Diagonal(ones), component_score, -inv1pq, gc.∇β)
        gc.∇β .*= sqrtτ # 0.019973446398091146
        gc.∇τ  .= (n - rss + 2qsum * inv1pq) / 2τ # gc.∇τ = 265.7155003051121
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t # gc.∇Σ = -0.045168675742349174
    end
    # output
    logl # -27.795829678091444
end

function loglikelihood!(
    gcm::GLMCopulaVCModel{T, D},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D<:Normal{T}}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇Σ, 0.0)
        fill!(gcm.∇τ, 0.0)
    end
    τ = 1.0
    if GLM.dispersion_parameter(gcm.d)
        τ = gcm.τ[1]
    end
    if needhess
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        #println(logl)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇τ .+= gcm.data[i].∇τ
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    logl
end

"""
    loglikelihood!(gc::GLMCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
"""

function loglikelihood!(
    gc::GLMCopulaVCObs{T, D},
    β::Vector{T},
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    component_score = zeros(p)
    needgrad = needgrad || needhess
    update_res!(gc, β)
    standardize_res!(gc)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇Σ, 0)
        fill!(gc.∇resβ, 0.0)
        std_res_differential!(gc)
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    tsum = dot(Σ, gc.t)
    logl = - log(1 + tsum)
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # component_score stores ∇resβ*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, component_score)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)

    # sum up the component loglikelihood
    for j = 1:length(gc.y)
        logl += loglik_obs(gc.d, gc.y[j], gc.μ[j], 1.0, 1.0)
    end
    # gradient
    if needgrad
        x = zeros(p)
        c = 0.0
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), component_score, 1.0, gc.Hβ) # only lower triangular
        end
        for j in 1:length(gc.y)
              c = gc.res[j] * gc.w1[j]
              copyto!(x, gc.X[j, :])
              BLAS.axpy!(c, x, gc.∇β) # gc.∇β = gc.∇β + r_ij(β) * mueta* x
              BLAS.axpy!(-inv1pq, component_score, gc.∇β) # first term for each glm score
              BLAS.ger!(gc.w2[j], x, x, gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
        end
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
    end
    # output
    logl
end

# function loglikelihood!(
#     gcm::GLMCopulaVCModel{T, D},
#     needgrad::Bool = false,
#     needhess::Bool = false
#     ) where {T <: BlasReal, D<:Union{Poisson{T}, Bernoulli{T}}}
#     logl = zero(T)
#     if needgrad
#         fill!(gcm.∇β, 0)
#         fill!(gcm.∇Σ, 0)
#     end
#     for i in eachindex(gcm.data)
#         logl += loglikelihood!(gcm.data[i], gcm.β, gcm.Σ, needgrad, needhess)
#         #println(logl)
#         if needgrad
#             gcm.∇β .+= gcm.data[i].∇β
#             gcm.∇Σ .+= gcm.data[i].∇Σ
#         end
#         if needhess
#             gcm.Hβ .+= gcm.data[i].Hβ
#         end
#     end
#     needhess && (gcm.Hβ)
#     logl
# end

function loglikelihood2!(
    gcm::GLMCopulaVCModel{T, D},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D<:Union{Poisson{T}, Bernoulli{T}}}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇Σ, 0.0)
    end
    logl += copula_loglikelihood(gcm)
    if needgrad
        gcm.∇β .= copula_gradient(gcm)
        # gcm.∇Σ .+= gcm.data[i].∇Σ
    end
    if needhess
        gcm.Hβ .= beta_hessians(gcm)
    end
    logl
end

function fit2!(
    gcm::GLMCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0),
    )
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill(Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    GLMCopula.copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood2!(gcm)
    gcm
end

function MathProgBase.initialize(
    gcm::GLMCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GLMCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::GLMCopulaVCModel,
    par::Vector)
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    # GLMCopula.update_Σ!(gcm)
    # evaluate loglikelihood
    loglikelihood2!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::GLMCopulaVCModel,
    grad::Vector,
    par::Vector)
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    # GLMCopula.update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood2!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    #gcm = glm_score_statistic(gcm, gcm.β)
    #copyto!(grad, gcm.∇β)
    nothing
end

function copy_par!(
    gcm::GLMCopulaVCModel,
    par::Vector)
    copyto!(gcm.β, par)
    par
end

function MathProgBase.hesslag_structure(gcm::GLMCopulaVCModel)
    Iidx = Vector{Int}(undef, (gcm.p * (gcm.p + 1)) >> 1)
    Jidx = similar(Iidx)
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            Iidx[ct] = i
            Jidx[ct] = j
            ct += 1
        end
    end
    Iidx, Jidx
end

function MathProgBase.eval_hesslag(
    gcm::GLMCopulaVCModel{T, D},
    H::Vector{T},
    par::Vector{T},
    σ::T) where {T <: BlasReal, D}
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    #GLMCopula.update_Σ!(gcm)
    # evaluate Hessian
    loglikelihood2!(gcm, true, true)
    # copy Hessian elements into H
    ct = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            H[ct] = gcm.Hβ[i, j]
            ct += 1
        end
    end
    H .*= σ
end

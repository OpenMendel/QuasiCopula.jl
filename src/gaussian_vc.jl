
"""
update_res!(gcm, β)

Update the residual vector according to `β` for the model object.
"""
function update_res!(
    gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}},
    β::Vector{T}
    ) where T <: BlasReal
    mul!(gc.η, gc.X, β)
    copyto!(gc.μ, gc.η)
    fill!(gc.dμ, 0.0)
    fill!(gc.w1, 1.0)
    fill!(gc.w2, 1.0)
    fill!(gc.varμ, 1.0)
    copyto!(gc.res, gc.y)
    BLAS.axpy!(-1, gc.μ, gc.res)
    # BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

"""
update_res!(gc, β)
Update the residual vector according to `β` and the canonical inverse link to the given distribution.
"""
function update_res!(
   gc::GLMCopulaVCObs{T, D},
   β::Vector{T}
   ) where {T <: BlasReal, D}
   mul!(gc.η, gc.X, β)
   for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(canonicallink(gc.d), gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i])
       gc.dμ[i] = GLM.mueta(canonicallink(gc.d), gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.dμ[i]^2 / gc.varμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end


function update_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}},
    σinv::T
    ) where {T <: BlasReal, D}
    gc.res .*= σinv
end

function standardize_res!(
    gc::Union{GLMCopulaVCObs{T, D}, GaussianCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}}
    ) where {T <: BlasReal, D}
    for j in eachindex(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}, GLMCopulaVCModel{T, D}}
    ) where {T <: BlasReal, D}
    # standardize residual
    if gcm.d == Normal()
        σinv = sqrt(gcm.τ[1])# general variance
        for i in eachindex(gcm.data)
            standardize_res!(gcm.data[i], σinv)
        end
    else
        for i in eachindex(gcm.data)
            standardize_res!(gcm.data[i])
        end
    end
    nothing
end

"""
update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::Union{GaussianCopulaVCObs{T, D}, GLMCopulaVCObs{T, D}}) where {T<:Real, D}
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

"""

MM update to minimize ``n \\log (\\tau) - rss / 2 \\ln (\\tau) +
\\sum_i \\log (1 + \\tau * q_i)``.
"""
function update_τ(
    τ0::T,
    q::Vector{T},
    n::Integer,
    rss::T,
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    ) where T <: BlasReal
    @assert τ0 ≥ 0 "τ0 has to be nonnegative"
    τ = τ0
    for τiter in 1:maxiter
        τold = τ
        tmp = zero(T)
        for i in eachindex(q)
            tmp += q[i] / (1 + τ * q[i])
        end
        τ = (n + 2τ * tmp) / rss
        abs(τ - τold) < reltol * (abs(τold) + 1) && break
    end
    τ
end

"""
update_Σ!(gc)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
function update_Σ!(gcm::Union{GLMCopulaVCModel{T, D}, GaussianCopulaVCModel{T, D}}) where {T <: BlasReal, D}
    distT = Base.typename(typeof(gcm.d)).wrapper
    update_Σ_jensen!(gcm)
end


function update_Σ_jensen!(
    gcm::GaussianCopulaVCModel{T, D},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # update τ
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = q[i]
        gcm.τ[1] = update_τ(gcm.τ[1], gcm.storage_n, gcm.ntotal, rsstotal, 1)
        # numerator in the multiplicative update
        gcm.storage_n .= inv.(inv(gcm.τ[1]) .+ gcm.storage_n) # use newest τ to update Σ
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n)
        gcm.Σ .*= gcm.storage_m
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.Σ ./= gcm.storage_m
        # monotonicity diagnosis
        verbose && println(sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.Σ)) -
            sum(log, 1 .+ gcm.TR * gcm.Σ) +
            gcm.ntotal / 2 * (log(gcm.τ[1]) - log(2π)) -
             rsstotal / 2 * gcm.τ[1])
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
            verbose && println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end


function update_Σ_jensen!(
    gcm::GLMCopulaVCModel{T, D},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        standardize_res!(gcm.data[i])
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q # now QF is formed from the standardized residuals
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # numerator in the multiplicative update
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = sum_k^m qi[k] sigmai_[k] # denom of numerator
        gcm.storage_n .= inv.(1 .+ gcm.storage_n) # 1/ (1 + sum_k^m qi[k] sigmai_[k]) # denom of numerator
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n) # store numerator = b_i / (1 + a b_i)
        gcm.Σ .*= gcm.storage_m # multiply
        # denominator in the multiplicative update
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        gcm.Σ ./= gcm.storage_m
        # monotonicity diagnosis
        verbose && println(sum(log, 1 .+ (gcm.QF * gcm.Σ)) -
            sum(log, 1 .+ gcm.TR * gcm.Σ))
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
            verbose && println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

# function update_Σ_quadratic!(
#     gcm::GaussianCopulaVCModel{T, D},
#     maxiter::Integer=50000,
#     reltol::Number=1e-6,
#     qpsolver=Ipopt.IpoptSolver(print_level=0),
#     verbose::Bool=false) where {T <: BlasReal, D}
#     n, m = length(gcm.data), length(gcm.data[1].V)
#     # pre-compute quadratic forms and RSS
#     rsstotal = zero(T)
#     for i in eachindex(gcm.data)
#         update_res!(gcm.data[i], gcm.β)
#         rsstotal += abs2(norm(gcm.data[i].res))
#         update_quadform!(gcm.data[i])
#         gcm.QF[i, :] = gcm.data[i].q
#     end
#     qcolsum = sum(gcm.QF, dims=1)[:]
#     # define NNLS optimization problem
#     H = Matrix{T}(undef, m, m)  # quadratic coefficient in QP
#     c = Vector{T}(undef, m)     # linear coefficient in QP
#     w = Vector{T}(undef, n)
#     # MM iteration
#     for iter in 1:maxiter
#         # store previous iterate
#         copyto!(gcm.storage_Σ, gcm.Σ)
#         # update τ
#         mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = q[i]
#         tmp = zero(T)
#         for i in eachindex(gcm.data)
#             tmp += gcm.storage_n[i] / (1 + gcm.τ[1] * gcm.storage_n[i])
#         end
#         gcm.τ[1] = (gcm.ntotal + 2gcm.τ[1] * tmp) / rsstotal  # update τ
#         # update variance components
#         for i in eachindex(gcm.data)
#             w[i] = abs2(gcm.τ[1]) / (1 + gcm.τ[1] * gcm.storage_n[i])
#         end
#         mul!(H, transpose(gcm.QF) * Diagonal(w), gcm.QF)
#         mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
#         gcm.storage_n .= inv.(1 .+ gcm.storage_n)
#         mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
#         c .= gcm.τ[1] .* qcolsum .- gcm.storage_m
#         # try unconstrained solution first
#         ldiv!(gcm.Σ, cholesky(Symmetric(H)), c)
#         # if violate nonnegativity constraint, resort to quadratic programming
#         if any(x -> x < 0, gcm.Σ)
#             @show "use QP"
#             qpsol = quadprog(-c, H, Matrix{T}(undef, 0, m),
#                 Vector{Char}(undef, 0), Vector{T}(undef, 0),
#                 fill(T(0), m), fill(T(Inf), m), qpsolver)
#             gcm.Σ .= qpsol.sol
#         end
#         # monotonicity diagnosis
#         verbose && println(sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.Σ)) -
#             sum(log, 1 .+ gcm.TR * gcm.Σ) +
#             gcm.ntotal / 2 * (log(gcm.τ[1]) - log(2π)) -
#              rsstotal / 2 * gcm.τ[1])
#         # convergence check
#         gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
#         if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
#             println("iters=$iter")
#             break
#         end
#         verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
#     end
#     gcm.Σ
# end

"""
std_res_differential!(gc)
compute the gradient of residual vector (standardized)
"""
function std_res_differential!(gc::GaussianCopulaVCObs{T, D}) where {T <: BlasReal, D}
        copyto!(gc.∇resβ, gc.X)
    gc
end

"""
    component_loglikelihood!(gc::GaussianCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d` using the GLM.jl package.
"""
function component_loglikelihood(gc::GaussianCopulaVCObs{T, D}, τ, logl::T) where {T <: BlasReal, D}
    ϕ = inv(τ)
    @inbounds for j in eachindex(gc.y)
        logl += GLM.loglik_obs(gc.d, gc.y[j], gc.μ[j], 1.0, ϕ)
    end
    logl
end

function component_loglikelihood(gc::GLMCopulaVCObs{T, D}, logl::T) where {T <: BlasReal, D}
    @inbounds for j in eachindex(gc.y)
        logl += GLM.loglik_obs(gc.d, gc.y[j], gc.μ[j], 1.0, 1.0)
    end
    logl
end

"""
    loglikelihood!(gc::GaussianCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution
`d`.
Note that loglikelihood is the sum of the logpdfs for each observation.
For each logpdf from Normal, Gamma, and InverseGaussian, we scale by dispersion.
"""

function loglikelihood!(
    gc::GaussianCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    component_score = zeros(n)
    needgrad = needgrad || needhess
    update_res!(gc, β)
    if gc.d  ==  Normal()
        sqrtτ = sqrt.(τ[1])
        standardize_res!(gc, sqrtτ)
    else
        sqrtτ = 1.0
        standardize_res!(gc)
    end
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
            BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)
    logl += GLMCopula.component_loglikelihood(gc, τ, 0.0)
    # gradient
    if needgrad
        x = zeros(p)
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
        end
        for j in 1:length(gc.y)
                # the first term in the score, where res is standardized!
              component_score[j] = gc.res[j] * gc.w1[j] * inv(sqrt(gc.varμ[j]))
             BLAS.ger!(gc.w2[j], gc.X[j, :], gc.X[j, :], gc.Hβ) # gc.Hβ = gc.Hβ + r_ij(β) * x * x'
         end
        # component_score = W1i(Yi -μi)
        BLAS.gemv!('T', 1.0, gc.X, component_score, -inv1pq, gc.∇β)
        # BLAS.gemv!('N', 1.0, Diagonal(ones), component_score, -inv1pq, gc.∇β)
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
    end
    # output
    logl
end
#


#-102.95885585666409 + 8.646784118270125 = -94.31207173839397 for obs 14

function loglikelihood!(
    gcm::GaussianCopulaVCModel{T, D},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇Σ, 0)
    end
    τ = 1.0
    if GLM.dispersion_parameter(gcm.d)
        τ = gcm.τ[1]
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        #println(logl)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    needhess && (gcm.Hβ)
    logl
end


function fit!(
    gcm::Union{GaussianCopulaVCModel, GLMCopulaVCModel},
    solver=NLopt.NLoptSolver(algorithm = :LN_BOBYQA, maxeval = 4000)
    )
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill( Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    GLMCopula.copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm)
    gcm
end

function MathProgBase.initialize(
    gcm::Union{GaussianCopulaVCModel, GLMCopulaVCModel},
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::Union{GaussianCopulaVCModel, GLMCopulaVCModel}) = [:Grad]

function MathProgBase.eval_f(
    gcm::Union{GaussianCopulaVCModel, GLMCopulaVCModel},
    par::Vector)
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate loglikelihood
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::Union{GaussianCopulaVCModel,GLMCopulaVCModel},
    grad::Vector,
    par::Vector)
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    nothing
end

function copy_par!(
    gcm::Union{GaussianCopulaVCModel,GLMCopulaVCModel},
    par::Vector)
    copyto!(gcm.β, par)
    par
end

function MathProgBase.hesslag_structure(gcm::Union{GaussianCopulaVCModel,GLMCopulaVCModel})
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
    gcm::GaussianCopulaVCModel{T, D},
    H::Vector{T},
    par::Vector{T},
    σ::T) where {T <: BlasReal, D}
    GLMCopula.copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    GLMCopula.update_Σ!(gcm)
    # evaluate Hessian
    loglikelihood!(gcm, true, true)
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

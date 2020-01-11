"""
init_β(gcm)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}}
    ) where {T <: BlasReal, D}
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p) 
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β s.t gcm.β = inv(cholesky(Symmetric(gcm.XtX)))*xty
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res2!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    gcm.τ[1] = gcm.ntotal / rss
    gcm.β
end

# """
# update_res!(gc, β)
# Update the residual vector according to `β`.
# """
# function update_res!(
#     gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}}, 
#     β::Vector{T}
#     ) where T <: BlasReal
#     copyto!(gc.res, gc.y)
#     BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
#     gc.res
# end

# function update_res!(
#     gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
#     ) where T <: BlasReal
#     #println("hi")
#     for i in eachindex(gcm.data)
#         update_res!(gcm.data[i], gcm.β)
#     end
#     nothing
# end

"""
update_res2!(gc, β)

Update the residual vector according to `β` and the canonical inverse link to the given distribution.
"""
function update_res2!(
    gc::Union{GaussianCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}}, 
    β::Vector{T}
    ) where {T <: BlasReal, D}
    gc.μ .= GLM.linkinv.(canonicallink(gc.d), gc.X * β)
    gc.res .= gc.y .- gc.μ
    return (gc.res)
end

function update_res2!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}}
    ) where {T <: BlasReal, D}
    for i in eachindex(gcm.data)
        update_res2!(gcm.data[i], gcm.β)
    end
    nothing
end


function standardize_res!(
    gc::Union{GaussianCopulaVCObs{T, D}, GaussianCopulaLMMObs{T}}, 
    σinv::T
    ) where {T <: BlasReal, D}
    gc.res .*= σinv
end

function standardize_res!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}}
    ) where {T <: BlasReal, D}
    σinv = sqrt(gcm.τ[1])
    # standardize residual
    for i in eachindex(gcm.data)
        standardize_res!(gcm.data[i], σinv)
    end
    nothing
end

"""
update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::GaussianCopulaVCObs)
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

# from GLM package

function loglik_obs end

loglik_obs(::Bernoulli, y, μ, wt, ϕ) = wt*GLM.logpdf(Bernoulli(μ), y)
loglik_obs(::Binomial, y, μ, wt, ϕ) = GLM.logpdf(Binomial(Int(wt), μ), Int(y*wt))
loglik_obs(::Gamma, y, μ, wt, ϕ) = wt*GLM.logpdf(Gamma(inv(ϕ), μ*ϕ), y)
loglik_obs(::InverseGaussian, y, μ, wt, ϕ) = wt*GLM.logpdf(InverseGaussian(μ, inv(ϕ)), y)
loglik_obs(::Normal, y, μ, wt, ϕ) = wt*GLM.logpdf(Normal(μ, sqrt(ϕ)), y)
loglik_obs(::Poisson, y, μ, wt, ϕ) = wt*GLM.logpdf(Poisson(μ), y)
# We use the following parameterization for the Negative Binomial distribution:
#    (Γ(r+y) / (Γ(r) * y!)) * μ^y * r^r / (μ+r)^{r+y}
# The parameterization of NegativeBinomial(r=r, p) in Distributions.jl is
#    Γ(r+y) / (y! * Γ(r)) * p^r(1-p)^y
# Hence, p = r/(μ+r)
loglik_obs(d::NegativeBinomial, y, μ, wt, ϕ) = wt*GLM.logpdf(NegativeBinomial(d.r, d.r/(μ+d.r)), y)

# function update_Σ_jensenHUA!(
#     gcm::GaussianCopulaVCModel{T, D}, 
#     maxiter::Integer=50000,
#     reltol::Number=1e-6,
#     verbose::Bool=false) where {T <: BlasReal, D}
#     rsstotal = zero(T)
#     println(gcm.data[1].res)
#     println(gcm.data[1].q)
#     for i in eachindex(gcm.data)
#         update_res!(gcm.data[i], gcm.β)
#         rsstotal += abs2(norm(gcm.data[i].res))
#         GLMCopula.update_quadform!(gcm.data[i])
#         gcm.QF[i, :] = gcm.data[i].q        
#     end
#     # MM iteration
#     for iter in 1:maxiter
#         # store previous iterate
#         copyto!(gcm.storage_Σ, gcm.Σ)
#         # update τ
#         mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = q[i]
#         gcm.τ[1] = GLMCopula.update_τ(gcm.τ[1], gcm.storage_n, gcm.ntotal, rsstotal, 1)
#         # numerator in the multiplicative update
#         gcm.storage_n .= inv.(inv(gcm.τ[1]) .+ gcm.storage_n) # use newest τ to update Σ
#         mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n)
#         gcm.Σ .*= gcm.storage_m
#         # denominator in the multiplicative update
#         mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
#         gcm.storage_n .= inv.(1 .+ gcm.storage_n)
#         mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
#         gcm.Σ ./= gcm.storage_m
#         # monotonicity diagnosis
#         verbose && println(sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.Σ)) - 
#             sum(log, 1 .+ gcm.TR * gcm.Σ) + 
#             gcm.ntotal / 2 * (log(gcm.τ[1]) - log(2π)) - 
#             rsstotal / 2 * gcm.τ[1])
#         # convergence check
#         gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
#         # norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1) && break
#         if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
#             verbose && println("iters=$iter")
#             break
#         end
#         verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
#     end
#     gcm.Σ
# end

function update_Σ_jensen!(
    gcm::GaussianCopulaVCModel{T, D}, 
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res2!(gcm.data[i], gcm.β)
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
        verbose && println(get_loglikelihood(gcm))
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

function update_Σ_quadratic!(
    gcm::GaussianCopulaVCModel{T, D}, 
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    qpsolver=Ipopt.IpoptSolver(print_level=0),
    verbose::Bool=false) where {T <: BlasReal, D}
    n, m = length(gcm.data), length(gcm.data[1].V)
    # pre-compute quadratic forms and RSS
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res2!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))
        update_quadform!(gcm.data[i])
        gcm.QF[i, :] = gcm.data[i].q
    end
    qcolsum = sum(gcm.QF, dims=1)[:]
    # define NNLS optimization problem
    H = Matrix{T}(undef, m, m)  # quadratic coefficient in QP
    c = Vector{T}(undef, m)     # linear coefficient in QP
    w = Vector{T}(undef, n)
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)
        # update τ
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = q[i]
        # a, b = zero(T), - rsstotal / 2
        # for i in eachindex(gcm.data)
        #     a += abs2(gcm.storage_n[i]) / (1 + gcm.τ[1] * gcm.storage_n[i])
        #     b += gcm.storage_n[i]
        # end
        # gcm.τ[1] = (b + sqrt(abs2(b) + 2a * gcm.ntotal)) / 2a
        tmp = zero(T)
        for i in eachindex(gcm.data)
            tmp += gcm.storage_n[i] / (1 + gcm.τ[1] * gcm.storage_n[i])
        end
        gcm.τ[1] = (gcm.ntotal + 2gcm.τ[1] * tmp) / rsstotal  # update τ
        # update variance components
        for i in eachindex(gcm.data)
            w[i] = abs2(gcm.τ[1]) / (1 + gcm.τ[1] * gcm.storage_n[i])
        end
        mul!(H, transpose(gcm.QF) * Diagonal(w), gcm.QF)
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n)
        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n)
        c .= gcm.τ[1] .* qcolsum .- gcm.storage_m
        # try unconstrained solution first
        ldiv!(gcm.Σ, cholesky(Symmetric(H)), c)
        # if violate nonnegativity constraint, resort to quadratic programming
        if any(x -> x < 0, gcm.Σ)
            @show "use QP"
            qpsol = quadprog(-c, H, Matrix{T}(undef, 0, m), 
                Vector{Char}(undef, 0), Vector{T}(undef, 0), 
                fill(T(0), m), fill(T(Inf), m), qpsolver)
            gcm.Σ .= qpsol.sol
        end
        # monotonicity diagnosis
        verbose && println(get_loglikelihood(gcm))
        # convergence check
        gcm.storage_m .= gcm.Σ .- gcm.storage_Σ
        if norm(gcm.storage_m) < reltol * (norm(gcm.storage_Σ) + 1)
            println("iters=$iter")
            break
        end
        verbose && iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

"""
update_Σ!(gc)

Update `τ` and variance components `Σ` according to the current value of 
`β` by an MM algorithm. `gcm.QF` needs to hold qudratic forms calculated from 
un-standardized residuals.
"""
update_Σ! = update_Σ_jensen!

function fitted(
    gc::GaussianCopulaVCObs{T, D},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}) where {T <: BlasReal, D}
    n, m = length(gc.y), length(gc.V)
    μ̂ = glm_residual(gc, β)[1]
    Ω = Matrix{T}(undef, n, n)
    for k in 1:m
        Ω .+= Σ[k] .* gc.V[k]
    end
    σ02 = inv(τ)
    c = inv(1 + dot(Σ, gc.t)) # normalizing constant
    V̂ = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:j-1
            V̂[i, j] = c * σ02 * Ω[i, j]
        end
        V̂[j, j] = c * σ02 * (1 + Ω[j, j] + tr(Ω) / 2)
    end
    LinearAlgebra.copytri!(V̂, 'U')
    μ̂, V̂
end


# """
#     score = X^T * W * (y - g(x^T b))
# Calculates the score (gradient) for different glm models. 
# W is a diagonal matrix where w[i, i] = g'(x^T b) / var(μ). 
# """
# function score!(d::UnivariateDistribution, gcm) where {T <: Float}
#     @inbounds for i in eachindex(y)
#         # η = clamp(v.xb[i] + v.zc[i], -20, 20)
#         η = #gcm xb 
#         w = mueta(l, η) / glmvar(d, v.μ[i])
#         v.r[i] = w * (y[i] - v.μ[i])
#     end
#     At_mul_B!(v.df, v.df2, x, z, v.r, v.r)
# end

"""
The deviance of a GLM can be evaluated as the sum of the squared deviance residuals. Calculation
of sqared deviance residuals is accomplished by `devresid` which is implemented in GLM.jl
"""
function deviance(gc::GaussianCopulaVCObs{T, D}) where {T <: BlasReal, D}
    dev = 0.0
    @inbounds for j in eachindex(gc.y)
        dev += GLM.devresid(gc.d, gc.y[j], gc.μ[j])
    end
    return dev
end

"""
    loglikelihood2!(gc::GaussianCopulaVCObs{T, D})
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution 
`d`. 
Note that loglikelihood is the sum of the logpdfs for each observation. 
For each logpdf from Normal, Gamma, and InverseGaussian, we scale by dispersion. 
"""

function loglikelihood2!(
    gc::GaussianCopulaVCObs{T, D},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇Σ, 0) 
    end
    needhess && fill!(gc.Hβ, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res2!(gc, β)
    standardize_res!(gc, sqrtτ)
    tsum = dot(Σ, gc.t)
    logl = - log(1 + tsum) 
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # ∇β stores X'*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)
    rss = abs2(norm(gc.res))
    ϕ = GLMCopula.deviance(gc) / n
    @inbounds for j in eachindex(gc.y)
        logl += GLM.loglik_obs(gc.d, gc.y[j], gc.μ[j], 1, ϕ)
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
            gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
        end
        BLAS.gemv!('T', 1.0, gc.X, gc.res, -inv1pq, gc.∇β)
        gc.∇β .*= sqrtτ
        gc.∇τ  .= (n - rss + 2qsum * inv1pq) / 2τ
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t 
    end
    # output
    logl
end

function loglikelihood2!(
    gcm::Union{GaussianCopulaVCModel{T, D}, GaussianCopulaLMMModel{T}},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇τ, 0)
        fill!(gcm.∇Σ, 0)
    end
    if needhess
        gcm.Hβ .= - gcm.XtX
        gcm.Hτ .= - gcm.ntotal / 2abs2(gcm.τ[1])
    end
    for i in eachindex(gcm.data)
        logl += loglikelihood2!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        #println(logl)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇τ .+= gcm.data[i].∇τ
            gcm.∇Σ .+= gcm.data[i].∇Σ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hτ .+= gcm.data[i].Hτ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    logl
end

function fit!(
    gcm::GaussianCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level=0)
    )
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, gcm.p)
    ub = fill( Inf, gcm.p)
    MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    MathProgBase.setwarmstart!(optm, gcm.β)
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    copy_par!(gcm, MathProgBase.getsolution(optm))
    loglikelihood2!(gcm)
    gcm
end

function MathProgBase.initialize(
    gcm::GaussianCopulaVCModel, 
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(
    gcm::GaussianCopulaVCModel, 
    par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate loglikelihood
    loglikelihood2!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::GaussianCopulaVCModel, 
    grad::Vector, 
    par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood2!(gcm, true, false)
    copyto!(grad, gcm.∇β)
    nothing
end

function copy_par!(
    gcm::GaussianCopulaVCModel, 
    par::Vector)
    copyto!(gcm.β, par)
    par
end

function MathProgBase.hesslag_structure(gcm::GaussianCopulaVCModel)
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
    σ::T,
    μ::Vector{T}) where {T <: BlasReal, D}
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
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

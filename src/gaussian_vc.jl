"""
init_β(gcm)

Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution.
"""
function init_β!(
    gcm::Union{GaussianCopulaVCModel{T},GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
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
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    gcm.τ[1] = gcm.ntotal / rss
    gcm.β
end

"""
update_res!(gc, β)

Update the residual vector according to `β`.
"""
function glm_residual(d, gcs, β)
    # get mu = canonicallinkinv(XB)
    μ = map(x -> GLM.inverselink(canonicallink(d), x)[1], gcs.X*β)
    # get residual gcs.res = (yi - mu)
    return (μ, sqrt.(GLM.devresid.(d, gcs.y, μ)))
end

"""
update_res!(gc, β)

Update the residual vector according to `β`.
"""
function update_res!(
    gc::Union{GaussianCopulaVCObs{T, D}, GaussianCopulaLMMModel{T}},
    β::Vector{T}
    ) where {T <: BlasReal, D}
    copyto!(gc.res, gc.y)
    μ, residual = glm_residual(gc.d, gc, β)
    copyto!(gc.res, residual)
    copyto!(gc.μ, μ)
    gc.res
end

function update_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where {T <: BlasReal}
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::Union{GaussianCopulaVCObs{T}, GaussianCopulaLMMObs{T}}, 
    σinv::T
    ) where T <: BlasReal
    gc.res .*= σinv
end

function standardize_res!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}}
    ) where T <: BlasReal
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




"""
    glmloglikelihood(d::UnivariateDistribution, y::AbstractVector, μ::AbstractVector)
Calculates the loglikelihood of observing `y` given mean `μ` and some distribution 
`d`. 
Note that loglikelihood is the sum of the logpdfs for each observation. 
For each logpdf from Normal, Gamma, and InverseGaussian, we scale by dispersion. 
"""
function glmloglikelihood(d, gcm)
    μ = [zeros(size(gcm.data[i].X)) for i in eachindex(gcm.data)]
    logl = 0.0
    rsstotal = 0.0
    β = gcm.β
    for i in eachindex(gcm.data)
        map!(x -> GLM.inverselink.(canonicallink(d), x)[1], μ[i], gcm.data[i].X*β)
        rsstotal += abs2(norm(gcm.data[i].res))
    end
    
    ϕ = rsstotal / (gcm.ntotal - 1)
    for i in eachindex(gcm.data), j in eachindex(gcm.data[i].y)
        logl += loglik_obs(d, gcm.data[i].y[j], μ[i][j], 1, ϕ) #wt = 1 because only have 1 sample to estimate a given mean 
    end
    return logl
end

# rss  = abs2(norm(gc.res)) # RSS of standardized residual
#     tsum = dot(Σ, gc.t)
#     #logl = - log(1 + tsum) - (n * log(2π) -  n * log(τ) + rss) / 2

# function get_loglikelihood(gcs, rss, tsum)
#     logpdf_term1 = -log(1 + tsum)
#     logpdf_term2 = (n * log(τ) + rss) / 2


# end


function get_loglikelihood(gcm)
    logpdf_term1 = -sum(log, 1 .+ gcm.TR * gcm.Σ)
    logpdf_term2 = sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.Σ))
    logpdf_term3 = glmloglikelihood(gcm.d, gcm)
    loglikelihood = logpdf_term1 + logpdf_term2 + logpdf_term3
    return loglikelihood
end

            
function update_Σ_jensen!(
    gcm::GaussianCopulaVCModel{T}, 
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where T <: BlasReal
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
    gcm::GaussianCopulaVCModel{T}, 
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    qpsolver=Ipopt.IpoptSolver(print_level=0),
    verbose::Bool=false) where T <: BlasReal
    n, m = length(gcm.data), length(gcm.data[1].V)
    # pre-compute quadratic forms and RSS
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
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
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T,
    Σ::Vector{T}) where T <: BlasReal
    n, m = length(gc.y), length(gc.V)
    μ̂ = gc.X * β
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

function loglikelihood!(
    gc::GaussianCopulaVCObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Vector{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
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
    update_res!(gc, β)
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
    ϕ = rss/ n
    for i in eachindex(gc.y)
        logl += loglik_obs(gc.d, gc.y[i], gc.μ[i], 1, ϕ) #wt = 1 because only have 1 sample to estimate a given mean 
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
            gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
        end
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv1pq, gc.∇β)
        gc.∇β .*= sqrtτ
        gc.∇τ  .= (n - rss + 2qsum * inv1pq) / 2τ
        gc.∇Σ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t 
    end
    # output
    logl
end

function loglikelihood!(
    gcm::Union{GaussianCopulaVCModel{T}, GaussianCopulaLMMModel{T}},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
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
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
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
    loglikelihood!(gcm)
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
    loglikelihood!(gcm, false, false)
end

function MathProgBase.eval_grad_f(
    gcm::GaussianCopulaVCModel, 
    grad::Vector, 
    par::Vector)
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
    # evaluate gradient
    logl = loglikelihood!(gcm, true, false)
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
    gcm::GaussianCopulaVCModel{T},
    H::Vector{T},
    par::Vector{T},
    σ::T,
    μ::Vector{T}) where T <: BlasReal
    copy_par!(gcm, par)
    # maximize σ2 and τ at current β using MM
    update_Σ!(gcm)
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

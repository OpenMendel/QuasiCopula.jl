export GaussianCopulaVCObs, GaussianCopulaVCModel
export fitted

"""
GaussianCopulaVCObs
GaussianCopulaVCObs(y, X, V)
A realization of Gaussian copula variance component data.
"""
struct GaussianCopulaVCObs{T <: BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β 
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    storage_n::Vector{T}
    storage_p::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    HΣ::Matrix{T}   # Hessian wrt variance components Σ
end

function GaussianCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}}
    ) where T <: BlasReal
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    t   = [tr(V[k])/2 for k in 1:m] 
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    m1        = Vector{T}(undef, m)
    m2        = Vector{T}(undef, m)
    HΣ  = Matrix{T}(undef, m, m)
    # constructor
    GaussianCopulaVCObs{T}(y, X, V, ∇β, ∇τ, ∇Σ, Hβ, 
        Hτ, res, xtx, t, q, storage_n, storage_p, m1, m2, HΣ)
end

"""
GaussianCopulaVCModel
GaussianCopulaVCModel(gcs)
Gaussian copula variance component model, which contains a vector of 
`GaussianCopulaVCObs` as data, model parameters, and working arrays.
"""
struct GaussianCopulaVCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GaussianCopulaVCObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    Hβ::Matrix{T}    # Hessian from all observations
    Hτ::Matrix{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    # hessain with resp to vc
    # asymptotic covariance for inference
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    HΣ::Matrix{T}   # Hessian wrt variance components Σ
end

function GaussianCopulaVCModel(gcs::Vector{GaussianCopulaVCObs{T}}) where T <: BlasReal
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = Vector{T}(undef, 1)
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hτ  = Matrix{T}(undef, 1, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        XtX    .+= gcs[i].xtx
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    Ainv    = zeros(T, p + m + 1, p + m + 1)
    Aevec   = zeros(T, p + m + 1, p + m + 1)
    M       = zeros(T, p + m + 1, p + m + 1)
    vcov    = zeros(T, p + m + 1, p + m + 1)
    ψ       = Vector{T}(undef, p + m + 1)
    HΣ  = Matrix{T}(undef, m, m)
    GaussianCopulaVCModel{T}(gcs, ntotal, p, m, β, τ, Σ, 
        ∇β, ∇τ, ∇Σ, Hβ, Hτ, XtX, TR, QF,
        storage_n, storage_m, storage_Σ, Ainv, Aevec, M, vcov, ψ, HΣ)
end

"""
initialize_model!(gcm)
Initialize the linear regression parameters `β` and `τ=σ0^{-2}` by the least 
squares solution. 
"""
function initialize_model!(
    gcm::GaussianCopulaVCModel{T}
    ) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, gcm.p) 
    for i in eachindex(gcm.data)
        BLAS.gemv!('T', one(T), gcm.data[i].X, gcm.data[i].y, one(T), xty)
    end
    # least square solution for β
    ldiv!(gcm.β, cholesky(Symmetric(gcm.XtX)), xty)
    @show gcm.β
    # accumulate residual sum of squares
    rss = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rss += abs2(norm(gcm.data[i].res))
    end
    println("initializing dispersion using residual sum of squares")
    gcm.τ[1] = gcm.ntotal / rss
    @show gcm.τ
    println("initializing variance components using MM-Algorithm")
    fill!(gcm.Σ, 1.0)
    update_Σ!(gcm)
    @show gcm.Σ
    nothing
end

"""
update_res!(gc, β)
Update the residual vector according to `β`.
"""
function update_res!(
    gc::GaussianCopulaVCObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(gc.res, gc.y)
    BLAS.gemv!('N', -one(T), gc.X, β, one(T), gc.res)
    gc.res
end

function update_res!(
    gcm::GaussianCopulaVCModel{T}
    ) where T <: BlasReal
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::GaussianCopulaVCObs{T}, 
    σinv::T
    ) where T <: BlasReal
    gc.res .*= σinv
end

function standardize_res!(
    gcm::GaussianCopulaVCModel{T}
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
        verbose && println(sum(log, 1 .+ gcm.τ[1] .* (gcm.QF * gcm.Σ)) - 
            sum(log, 1 .+ gcm.TR * gcm.Σ) + 
            gcm.ntotal / 2 * (log(gcm.τ[1]) - log(2π)) - 
            rsstotal / 2 * gcm.τ[1])
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
    sqrtτ = sqrt(abs(τ))
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    tsum = dot(abs.(Σ), gc.t)
    logl = - log(1 + tsum) - (n * log(2π) -  n * log(abs(τ)) + rss) / 2
    for k in 1:m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # ∇β stores X'*Γ*res (standardized residual)
            BLAS.gemv!('T', Σ[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(Σ, gc.q)
    logl += log(1 + qsum)
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
            gc.Hτ[1, 1] = - abs2(qsum * inv1pq / τ)
            # # hessian of vc vector use with fit_newton_normal.jl
            inv1pt = inv(1 + tsum)
            gc.m1 .= gc.q
            gc.m1 .*= inv1pq
            gc.m2 .= gc.t
            gc.m2 .*= inv1pt
            # hessian for vc
            fill!(gc.HΣ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.HΣ)
            BLAS.syr!('U', -one(T), gc.m1, gc.HΣ)
            copytri!(gc.HΣ, 'U')
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
    gcm::GaussianCopulaVCModel{T},
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
            gcm.HΣ .+= gcm.data[i].HΣ
        end
    end
    needhess && (gcm.Hβ .*= gcm.τ[1])
    logl
end

# uncomment this and exclude fit_newton_normal.jl from GLMCopula.jl to fit variance components separately using MM-algorithm instead of Joint Newton.
# function fit!(
#     gcm::GaussianCopulaVCModel,
#     solver=Ipopt.IpoptSolver(print_level=0)
#     )
#     initialize_model!(gcm)
#     optm = MathProgBase.NonlinearModel(solver)
#     lb = fill(-Inf, gcm.p)
#     ub = fill( Inf, gcm.p)
#     MathProgBase.loadproblem!(optm, gcm.p, 0, lb, ub, Float64[], Float64[], :Max, gcm)
#     MathProgBase.setwarmstart!(optm, gcm.β)
#     MathProgBase.optimize!(optm)
#     optstat = MathProgBase.status(optm)
#     optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
#     copy_par!(gcm, MathProgBase.getsolution(optm))
#     loglikelihood!(gcm)
#     gcm
# end

# function MathProgBase.initialize(
#     gcm::GaussianCopulaVCModel, 
#     requested_features::Vector{Symbol})
#     for feat in requested_features
#         if !(feat in [:Grad, :Hess])
#             error("Unsupported feature $feat")
#         end
#     end
# end

# MathProgBase.features_available(gcm::GaussianCopulaVCModel) = [:Grad, :Hess]

# function MathProgBase.eval_f(
#     gcm::GaussianCopulaVCModel, 
#     par::Vector)
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_Σ!(gcm)
#     # evaluate loglikelihood
#     loglikelihood!(gcm, false, false)
# end

# function MathProgBase.eval_grad_f(
#     gcm::GaussianCopulaVCModel, 
#     grad::Vector, 
#     par::Vector)
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_Σ!(gcm)
#     # evaluate gradient
#     logl = loglikelihood!(gcm, true, false)
#     copyto!(grad, gcm.∇β)
#     nothing
# end

# function copy_par!(
#     gcm::GaussianCopulaVCModel, 
#     par::Vector)
#     copyto!(gcm.β, par)
#     par
# end

# function MathProgBase.hesslag_structure(gcm::GaussianCopulaVCModel)
#     Iidx = Vector{Int}(undef, (gcm.p * (gcm.p + 1)) >> 1)
#     Jidx = similar(Iidx)
#     ct = 1
#     for j in 1:gcm.p
#         for i in j:gcm.p
#             Iidx[ct] = i
#             Jidx[ct] = j
#             ct += 1
#         end
#     end
#     Iidx, Jidx
# end

# function MathProgBase.eval_hesslag(
#     gcm::GaussianCopulaVCModel{T},
#     H::Vector{T},
#     par::Vector{T},
#     σ::T,
#     μ::Vector{T}) where T <: BlasReal
#     copy_par!(gcm, par)
#     # maximize σ2 and τ at current β using MM
#     update_Σ!(gcm)
#     # evaluate Hessian
#     loglikelihood!(gcm, true, true)
#     # copy Hessian elements into H
#     ct = 1
#     for j in 1:gcm.p
#         for i in j:gcm.p
#             H[ct] = gcm.Hβ[i, j]
#             ct += 1
#         end
#     end
#     H .*= σ
# end
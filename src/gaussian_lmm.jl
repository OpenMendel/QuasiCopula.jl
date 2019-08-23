"""
update_Σ!(gcm)

Update the variance components `σ2` according to the current value of 
`β` and `τ`.
"""
function update_Σ!(
    gcm::GaussianCopulaLMMModel{T}, 
    maxiter::Integer=5000, 
    reltol::Number=1e-6) where T<:BlasReal
    # pre-compute
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))
        # storage_q1 = Zi' * ri
        mul!(gcm.data[i].storage_q1, transpose(gcm.data[i].Z), gcm.data[i].res)
    end
    H = Matrix{T}(undef, abs2(gcm.q). abs2(gcm.q))
    # set up SDP optimization
    Σ = Convex.Semidefinite(gcm.q)
    C = Matrix{T}(undef, gcm.q, gcm.q)
    problem = minimize(dot(Σ, H * Convex.vec(Σ)) + dot(C, Σ))
    # MM iteration
    for iter in 1:maxiter
        # update τ        
        for i in eachindex(gcm.data)
            # storage_q2 = Σ * Zi' * ri
            mul!(gcm.data[i].storage_q2, gcm.Σ, gcm.data[i].storage_q1)
            # storage_n[i] = (ri' * Zi * Σ * Zi' * ri) / 2
            gcm.storage_n[i] = dot(gcm.data[i].storage_q1, gcm.data[i].storage_q2) / 2
        end
        for τiter in 1:10
            tmp = zero(T)
            for i in eachindex(gcm.storage_n)
                tmp += gcm.storage_n[i] / (1 + gcm.τ[1] * gcm.storage_n[i])
            end
            gcm.τ[1] = (gcm.ntotal + 2gcm.τ[1] * tmp) / rsstotal
        end
        # update Σ by SDP
        
        # # monotonicity diagnosis
        # # println(sum(log, (gcm.QF * gcm.σ2) .+ 1) - sum(log, gcm.TR * gcm.σ2 .+ 1))
        # # convergence check
        # gcm.storage_m .= gcm.σ2 .- gcm.storage_σ2
        # norm(gcm.storage_m) < reltol * (norm(gcm.storage_σ2) + 1) && break
        # iter == maxiter && @warn "maximum iterations $maxiter reached"
    end
    gcm.Σ
end

function loglikelihood!(
    gc::GaussianCopulaLMMObs{T},
    β::Vector{T},
    τ::T, # inverse of linear regression variance
    Σ::Matrix{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: LinearAlgebra.BlasFloat
    n, p, q = size(gc.X, 1), size(gc.X, 2), size(gc.Z, 2)
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇τ, 0)
        fill!(gc.∇Σ, 0)
    end
    needhess && fill!(gc.H, 0)
    # evaluate copula loglikelihood
    sqrtτ = sqrt(τ)
    update_res!(gc, β)
    standardize_res!(gc, sqrtτ)
    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    mul!(gc.storage_q1, transpose(gc.Z), gc.res)
    mul!(gc.storage_q2, Σ, gc.storage_q1)
    tr = dot(gc.storage_nq, gc.Z) / 2
    qf = dot(gc.storage_q1, gc.storage_q2) / 2
    mul!(gc.storage_nq, gc.Z, Σ)
    logl = - log(1 + tr) - (n * log(2π) -  n * log(τ) + rss) / 2 + log(1 + qf)
    # gradient
    if needgrad
        mul!(gc.∇β, transpose(gc.X), gc.res)
        BLAS.gemv!('T', -one(T), gc.Z, gc.storage_q2, one(T), gc.∇β)
        gc.∇β  .*= sqrtτ
        gc.∇τ[1] = (n - rss + 2qf / (1 + qf)) / 2τ
        gc.∇σ2  .= 0 # TODO
    end
    # Hessian: TODO
    if needhess; end;
    # output
    logl
end


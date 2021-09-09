
"""
update_Σ!(gcm)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
function update_Σ!(gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}}) where {T <: BlasReal, D, Link}
    update_Σ_jensen!(gcm)
end

"""
update_Σ_jensen!(gcm)

Update Σ using the MM algorithm and Jensens inequality, given β.
"""
function update_Σ_jensen!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}, GLMCopulaARModel{T, D, Link}},
    maxiter::Integer=50000,
    reltol::Number=1e-6,
    verbose::Bool=false) where {T <: BlasReal, D, Link}
    rsstotal = zero(T)
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
        rsstotal += abs2(norm(gcm.data[i].res))  # needed for updating τ in normal case
        standardize_res!(gcm.data[i])            # standardize the residuals GLM variance(μ)
        GLMCopula.update_quadform!(gcm.data[i]) # with standardized residuals
        gcm.QF[i, :] = gcm.data[i].q
    end
    # MM iteration
    for iter in 1:maxiter
        # store previous iterate
        copyto!(gcm.storage_Σ, gcm.Σ)

        # update τ if necessary
        mul!(gcm.storage_n, gcm.QF, gcm.Σ) # gcm.storage_n[i] = sum_k^m qi[k] sigmai_[k] # denom of numerator
        if gcm.d[1] == Normal()
            gcm.τ[1] = GLMCopula.update_τ(gcm.τ[1], gcm.storage_n, gcm.ntotal, rsstotal, 1)
        else
            fill!(gcm.τ, 1.0)
        end

        ##### Numerator in the multiplicative update ##########
        # when its logistic gcm.τ = 1.0
        gcm.storage_n .= inv.(inv.(gcm.τ) .+ gcm.storage_n) # 1/ (1 + sum_k^m qi[k] sigmai_[k]) # denom of numerator
        mul!(gcm.storage_m, transpose(gcm.QF), gcm.storage_n) # store numerator of numerator
        gcm.Σ .*= gcm.storage_m # multiply by the numerator

        ######## Denominator in the multiplicative update  ###########
        mul!(gcm.storage_n, gcm.TR, gcm.storage_Σ)
        gcm.storage_n .= inv.(1 .+ gcm.storage_n) # denominator of the denominator

        mul!(gcm.storage_m, transpose(gcm.TR), gcm.storage_n) # numerator of the denominator
        gcm.Σ ./= gcm.storage_m # divide by the denominator

        # monotonicity diagnosis (for the normal distribution there is an extra term)
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

"""
update_res!(gc, β)
Update the residual vector according to `β` given link function and distribution.
"""
function update_res!(
   gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}},
   β::Vector{T}) where {T <: BlasReal, D, Link}
   mul!(gc.η, gc.X, β)
   for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(gc.link, gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i]) # Note: for negative binomial, d.r is used
       gc.dμ[i] = GLM.mueta(gc.link, gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.dμ[i]^2 / gc.varμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end

function update_res!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}
    ) where {T <: BlasReal, D, Link}
    for i in eachindex(gcm.data)
        update_res!(gcm.data[i], gcm.β)
    end
    nothing
end

function standardize_res!(
    gc::GLMCopulaVCObs{T, D, Link},
    σinv::T
    ) where {T <: BlasReal, D, Link}
    for j in eachindex(gc.y)
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}
    ) where {T <: BlasReal, D, Link}
    for j in eachindex(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gcm::Union{GLMCopulaVCModel{T, D, Link}, NBCopulaVCModel{T, D, Link}}
    ) where {T <: BlasReal, D, Link}
    # standardize residual
    if gcm.d[1] == Normal()
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
function update_quadform!(gc::Union{GLMCopulaVCObs{T, D, Link}, NBCopulaVCObs{T, D, Link}}) where {T<:Real, D, Link}
    for k in 1:length(gc.V)
        gc.q[k] = dot(gc.res, mul!(gc.storage_n, gc.V[k], gc.res)) / 2
    end
    gc.q
end

"""
    update_quadform!(gc)
Update the quadratic forms `(r^T V[k] r) / 2` according to the current residual `r`.
"""
function update_quadform!(gc::GLMCopulaARObs{T, D, Link}) where {T<:Real, D, Link}
    gc.q .= dot(gc.res, mul!(gc.storage_n, gc.V, gc.res)) / 2
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
    update_r!(gc::GLMCopulaVCObs{T, D, Link})

Performs maximum loglikelihood estimation of the nuisance paramter for negative 
binomial model using Newton's algorithm. Will run a maximum of `maxIter` and
convergence is defaulted to `convTol`.
"""
function update_r_newton!(gcm::NBCopulaVCModel; maxIter=100, convTol=1e-6)

    T = eltype(gcm.β)
    r = gcm.r[1] # estimated r in previous iteration

    function first_derivative(gcm, r)
        s = zero(T)
        tmp(yi, μi) = -(yi+r)/(μi+r) - log(μi+r) + 1 + log(r) + digamma(r+yi) - digamma(r)
        @inbounds for i in eachindex(gcm.data)
            # 2nd term of logl
            for j in eachindex(gcm.data[i].y)
                s += tmp(gcm.data[i].y[j], gcm.data[i].μ[j])
            end
            # 3rd term of logl
            resid = gcm.data[i].res
            Γ = gcm.Σ' * gcm.data[i].V # Γ = a1*V1 + ... + am*Vm
            η = gcm.data[i].η
            # D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
            # dD = Diagonal([-exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r))) for j in 1:length(η)])
            # dresid = -inv(D)*dD*resid
            # s += resid'*Γ*dresid / (1 + 0.5resid'*Γ*resid)
            for j in 1:length(η)
                gcm.data[i].storage_n[j] = inv(sqrt(exp(η[j])*(exp(η[j])+r) / r)) # storage_n[i] = 1 / Di = 1 / sqrt(var(yi))
            end
            for j in 1:length(η)
                gcm.data[i].storage_n[j] *= -exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r))) # storage_n = inv(D) * dD
            end
            gcm.data[i].storage_n .*= -resid # storage_n = dr(β) (derivative of residuals)
            mul!(gcm.data[i].storage_n2, Γ, gcm.data[i].storage_n) # storage_n2 = Γ * dresid
            numer = dot(resid, gcm.data[i].storage_n2) # numer = r' * Γ * dr
            mul!(gcm.data[i].storage_n2, Γ, resid) # storage_n = Γ * resid
            denom = 1 + 0.5 * dot(resid, gcm.data[i].storage_n2) # denom = 1 + 0.5(r * Γ * r)
            s += numer / denom
        end
        return s
    end

    function second_derivative(gcm, r)
        tmp(yi, μi) = (yi+r)/(μi+r)^2 - 2/(μi+r) + 1/r + trigamma(r+yi) - trigamma(r)
        s = zero(T)
        @inbounds for i in eachindex(gcm.data)
            # 2nd term of logl
            for j in eachindex(gcm.data[i].y)
                s += tmp(gcm.data[i].y[j], gcm.data[i].μ[j])
            end
            # 3rd term of logl
            Γ = gcm.Σ' * gcm.data[i].V # Γ = a1*V1 + ... + am*Vm
            η = gcm.data[i].η
            resid = gcm.data[i].res
            # D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
            # dD = Diagonal([-exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r))) for j in 1:length(η)])
            # d2D = Diagonal([(exp(3η[j]) / (4r^1.5 * (exp(η[j])*(exp(η[j])+r))^(1.5))) + 
            #     (3exp(2η[j]) / (4r^(2.5)*sqrt(exp(η[j])*(exp(η[j])+r)))) for j in 1:length(η)])
            # resid = gcm.data[i].res
            # dresid = -inv(D)*dD*resid
            # d2resid = (2inv(D)*dD*inv(D)*dD - inv(D)*d2D)*resid
            # denom = 1 + 0.5resid'*Γ*resid
            # term1 = (resid'*Γ*dresid / denom)^2
            # term2 = dresid'*Γ*dresid / denom
            # term3 = resid'*Γ*d2resid / denom
            # s += -term1 + term2 + term3
            for j in 1:length(η)
                gcm.data[i].storage_n[j] = inv(sqrt(exp(η[j])*(exp(η[j])+r) / r)) # storage_n = inv(Di) = 1 / sqrt(var(yi))
            end
            for j in 1:length(η)
                # storage_n2 = -inv(D) * d2D
                gcm.data[i].storage_n2[j] = -gcm.data[i].storage_n[j] * 
                    ((exp(3η[j]) / (4r^1.5 * (exp(η[j])*(exp(η[j])+r))^(1.5))) + 
                    (3exp(2η[j]) / (4r^(2.5)*sqrt(exp(η[j])*(exp(η[j])+r)))))
            end
            for j in 1:length(η)
                # storage_n = inv(D) * dD
                gcm.data[i].storage_n[j] *= -exp(2η[j]) / (2r^1.5 * sqrt(exp(η[j])*(exp(η[j])+r)))
            end
            for j in 1:length(η)
                # storage_n2 = 2inv(D)*dD*inv(D)*dD -inv(D)*d2D
                gcm.data[i].storage_n2[j] += 2 * abs2(gcm.data[i].storage_n[j])
            end
            gcm.data[i].storage_n .*= -resid # storage_n = dr(β) = derivative of residuals
            gcm.data[i].storage_n2 .*= resid # storage_n2 = dr²(β) = 2nd derivative of residuals
            mul!(gcm.data[i].storage_n3, Γ, resid) # storage_n3 = Γ * resid
            denom = 1 + 0.5 * dot(resid, gcm.data[i].storage_n3)
            mul!(gcm.data[i].storage_n3, Γ, gcm.data[i].storage_n) # storage_n3 = Γ * dresid
            term1 = (dot(resid, gcm.data[i].storage_n3) / denom)^2 # (resid'*Γ*dresid / denom)^2
            term2 = dot(gcm.data[i].storage_n, gcm.data[i].storage_n3) / denom # term2 = dresid'*Γ*dresid / denom
            mul!(gcm.data[i].storage_n3, Γ, gcm.data[i].storage_n2) # storage_n3 = Γ * d2resid
            term3 = dot(resid, gcm.data[i].storage_n3) / denom # term3 = resid'*Γ*d2resid / denom
            s += -term1 + term2 + term3
        end
        return s
    end

    function negbin_component_loglikelihood(gcm, r)
        logl = zero(T)
        for gc in gcm.data
            n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
            # fill!(gc.∇β, 0.0)
            update_res!(gc, gcm.β)
            standardize_res!(gc)
            # fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
            std_res_differential!(gc) # this will compute ∇resβ
            @inbounds for k in 1:m
                mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
                # BLAS.gemv!('T', gcm.Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # gc.∇β += ∇resβ*Γ*res (standardized residual) 
                gc.q[k] = dot(gc.res, gc.storage_n) / 2 # gc.q[k] = 0.5res' * V[k] * res
            end
            # 2nd term of logl
            logl += component_loglikelihood(gc, r)
            # 3rd term of logl
            qsum  = dot(gcm.Σ, gc.q)
            # println("fdsafdsafdsafdsa 1 + qsum = $(1 + qsum)")
            logl += log(1 + qsum)
        end
        return logl
    end

    function newton_increment(gcm, r)
        dx = first_derivative(gcm, r)
        dx2 = second_derivative(gcm, r)
        increment = dx / dx2
        # use gradient ascent if hessian not negative definite
        # if dx2 < 0
        #     increment = dx / dx2
        # else 
        #     increment = dx
        # end
        return increment
    end

    new_r = one(T)
    stepsize = one(T)
    for i in 1:maxIter
        old_logl = negbin_component_loglikelihood(gcm, r)

        # run 1 iteration of Newton's algorithm
        increment = newton_increment(gcm, r)
        new_r = r - stepsize * increment

        # linesearch
        # for j in 1:20
        #     if new_r <= 0
        #         stepsize = stepsize / 2
        #         new_r = r - stepsize * increment
        #     else
        #         new_logl = negbin_component_loglikelihood(gcm, new_r)
        #         if old_logl >= new_logl
        #             stepsize = stepsize / 2
        #             new_r = r - stepsize * increment
        #         else
        #             break
        #         end
        #     end
        # end

        #check convergence
        if abs(r - new_r) ≤ convTol
            break
        else
            r = new_r
        end
    end

    return new_r
end

function update_r!(gcm::NBCopulaVCModel)
    new_r = update_r_newton!(gcm, maxIter=10)
    gcm.r[1] = new_r
    for gc in gcm.data
        gc.d = NegativeBinomial(new_r)
    end
    return nothing
end

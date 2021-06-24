
"""
update_Σ!(gcm)

Update variance components `Σ` according to the current value of
`β` by an MM algorithm. `gcm.QF` now needs to hold qudratic forms calculated from standardized residuals.
"""
function update_Σ!(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    update_Σ_jensen!(gcm)
end

"""
update_Σ_jensen!(gcm)

Update Σ using the MM algorithm and Jensens inequality, given β.
"""
function update_Σ_jensen!(
    gcm::GLMCopulaVCModel{T, D, Link},
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
   gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}},
   β::Vector{T}) where {T <: BlasReal, D, Link}
   mul!(gc.η, gc.X, β)
   for i in 1:length(gc.y)
       gc.μ[i] = GLM.linkinv(gc.link, gc.η[i])
       gc.varμ[i] = GLM.glmvar(gc.d, gc.μ[i])
       gc.dμ[i] = GLM.mueta(gc.link, gc.η[i])
       gc.w1[i] = gc.dμ[i] / gc.varμ[i]
       gc.w2[i] = gc.dμ[i]^2 / gc.varμ[i]
       gc.res[i] = gc.y[i] - gc.μ[i]
   end
   return gc.res
end

function update_res!(
    gcm::GLMCopulaVCModel{T, D, Link}
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
    gc::Union{GLMCopulaVCObs{T, D, Link}, GLMCopulaARObs{T, D, Link}}
    ) where {T <: BlasReal, D, Link}
    for j in eachindex(gc.y)
        σinv = inv(sqrt(gc.varμ[j]))
        gc.res[j] *= σinv
    end
end

function standardize_res!(
    gcm::GLMCopulaVCModel{T, D, Link}
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
function update_quadform!(gc::GLMCopulaVCObs{T, D, Link}) where {T<:Real, D, Link}
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
    update_r!(gc::GLMCopulaVCObs{T, D, Link})

Performs maximum loglikelihood estimation of the nuisance paramter for negative 
binomial model using Newton's algorithm. Will run a maximum of `maxIter` and
convergence is defaulted to `convTol`.
"""
function update_r_newton!(gcm::GLMCopulaVCModel; maxIter=100, convTol=1e-6)
    
    T = eltype(gcm.β)
    r = gcm.data[1].d.r # estimated r in previous iteration

    function first_derivative(gcm, r)
        s = zero(T)
        tmp(yi, μi) = -(yi+r)/(μi+r) - log(μi+r) + 1 + log(r) + digamma(r+yi) - digamma(r)
        for i in eachindex(gcm.data)
            # 2nd term of logl
            for j in eachindex(gcm.data[i].y)
                s += tmp(gcm.data[i].y[j], gcm.data[i].μ[j])
            end
            # 3rd term of logl
            resid = gcm.data[i].res
            Γ = gcm.Σ' * gcm.data[i].V # Γ = a1*V1 + ... + am*Vm
            η = gcm.data[i].η
            D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
            dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
            dresid = -inv(D)*dD*resid
            s += resid'*Γ*dresid / (1 + 0.5resid'*Γ*resid)
        end
        return s
    end

    function second_derivative(gcm, r)
        tmp(yi, μi) = (yi+r)/(μi+r)^2 - 2/(μi+r) + 1/r + trigamma(r+yi) - trigamma(r)
        s = zero(T)
        for i in eachindex(gcm.data)
            # 2nd term of logl
            for j in eachindex(gcm.data[i].y)
                s += tmp(gcm.data[i].y[j], gcm.data[i].μ[j])
            end
            # 3rd term of logl
            Γ = gcm.Σ' * gcm.data[i].V # Γ = a1*V1 + ... + am*Vm
            η = gcm.data[i].η
            D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
            dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
            d2D = Diagonal([(exp(3η[i]) / (4r^1.5 * (exp(η[i])*(exp(η[i])+r))^(1.5))) + 
                (3exp(2η[i]) / (4r^(2.5)*sqrt(exp(η[i])*(exp(η[i])+r)))) for i in 1:length(η)])
            resid = gcm.data[i].res
            dresid = -inv(D)*dD*resid
            d2resid = (2inv(D)*dD*inv(D)*dD - inv(D)*d2D)*resid
            denom = 1 + 0.5resid'*Γ*resid
            term1 = (resid'*Γ*dresid / denom)^2
            term2 = dresid'*Γ*dresid / denom
            term3 = resid'*Γ*d2resid / denom
            s += -term1 + term2 + term3
        end
        return s
    end

    function negbin_component_loglikelihood(gcm, r)
        logl = zero(T)
        for (i, gc) in enumerate(gcm.data)
            gc.d = NegativeBinomial(r, T(0.5))
            # 2nd term of logl
            logl += component_loglikelihood(gc)
            # 3rd term of logl
            resid = gcm.data[i].res
            Γ = gcm.Σ' * gc.V # Γ = a1*V1 + ... + am*Vm
            logl += log(1 + 0.5resid'*Γ*resid)
        end
        return logl
    end

    function newton_increment(gcm, r)
        dx = first_derivative(gcm, r)
        dx2 = second_derivative(gcm, r)
        if dx2 < 0
            increment = dx / dx2
        else 
            increment = dx # use gradient ascent if hessian not negative definite
        end
        return increment
    end

    new_r = one(T)
    stepsize = one(T)
    for i in 1:maxIter
        # run 1 iteration of Newton's algorithm
        increment = newton_increment(gcm, r)
        new_r = r - stepsize * increment

        # linesearch
        old_logl = negbin_component_loglikelihood(gcm, r)
        for j in 1:20
            if new_r <= 0
                stepsize = stepsize / 2
                new_r = r - stepsize * increment
            else 
                new_logl = negbin_component_loglikelihood(gcm, new_r)
                if old_logl >= new_logl
                    stepsize = stepsize / 2
                    new_r = r - stepsize * increment
                else
                    break
                end
            end
        end

        #check convergence
        if abs(r - new_r) <= convTol
            return NegativeBinomial(new_r, T(0.5))
        else
            r = new_r
        end
    end

    return NegativeBinomial(r, T(0.5))
end

function update_r!(gcm::GLMCopulaVCModel)
    new_d = update_r_newton!(gcm)
    for gc in gcm.data
        gc.d = new_d
    end
    return new_d
end

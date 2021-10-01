export sandwich!, coef, stderror, confint, MSE, coverage!
"""
    sandwich!(gcm::GLMCopulaVCModel)
Calculate the sandwich estimator of the asymptotic covariance of the parameters, 
based on values `gcm.Hββ`, `gcm.HΣ`, `gcm.data[i].∇β`,
`gcm.data[i].∇Σ`, and `gcm.vcov` is updated by the sandwich 
estimator and returned.
"""
function sandwich!(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    p, m = gcm.p, gcm.m
    minv = inv(length(gcm.data))
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[          1:p,                 1:p      ] = gcm.Hβ
    gcm.Ainv[    (p + 1):(p + m),     (p + 1):(p + m)] = gcm.HΣ
    lmul!(minv, gcm.Ainv)
    # form M matrix in the sandwich formula
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇Σ)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    lmul!(minv, gcm.M)
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * inv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv 
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    gcm.vcov .*= minv
    nothing
end


"""
    sandwich!(gcm::GLMCopulaARModel)
Calculate the sandwich estimator of the asymptotic covariance of the parameters, 
based on values `gcm.Hββ`, `gcm.HΣ`, `gcm.data[i].∇β`,
`gcm.data[i].∇Σ`, and `gcm.vcov` is updated by the sandwich 
estimator and returned.
"""
function sandwich!(gcm::GLMCopulaARModel{T, D, Link}) where {T <: BlasReal, D, Link}
    p = gcm.p
    minv = inv(length(gcm.data))
    # form A matrix in the sandwich formula
    fill!(gcm.Ainv, 0.0)
    gcm.Ainv[1:p, 1:p] = gcm.Hβ
    gcm.Ainv[(p + 1) : (p + 1), (p + 1) : (p + 1)] = gcm.Hρ
    gcm.Ainv[(p + 2) : (p + 2), (p + 2) : (p + 2)] = gcm.Hσ2
    lmul!(minv, gcm.Ainv)
    # form M matrix in the sandwich formula
    fill!(gcm.M, 0.0)
    for obs in gcm.data
        copyto!(gcm.ψ, 1, obs.∇β)
        copyto!(gcm.ψ, p + 1, obs.∇ρ)
        copyto!(gcm.ψ, p + 2, obs.∇σ2)
        BLAS.syr!('U', T(1), gcm.ψ, gcm.M)
    end
    copytri!(gcm.M, 'U')
    lmul!(minv, gcm.M)
    Aeval, Aevec = eigen(Symmetric(gcm.Ainv))
    gcm.Ainv .= Aevec * inv(Diagonal(Aeval)) * Aevec
    fill!(gcm.vcov, 0.0)
    mul!(gcm.Aevec, gcm.Ainv, gcm.M) # use Avec as scratch space
    # vcov = Ainv * M * Ainv 
    mul!(gcm.vcov, gcm.Aevec, gcm.Ainv)
    gcm.vcov .*= minv
    nothing
end

"""
    coef(gcm::GLMCopulaVCModel)
Get the estimated parameter coefficients from the model.
"""
coef(gcm::GLMCopulaVCModel) = [gcm.β; gcm.Σ]

"""
    coef(gcm::GLMCopulaARModel)
Get the estimated parameter coefficients from the model.
"""
coef(gcm::GLMCopulaARModel) = [gcm.β; gcm.ρ; gcm.σ2]

"""
    stderror(gcm::GLMCopulaVCModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
stderror(gcm::GLMCopulaVCModel) = [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + gcm.m)]

"""
    stderror(gcm::GLMCopulaARModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
stderror(gcm::GLMCopulaARModel) = [sqrt(abs(gcm.vcov[i, i])) for i in 1:(gcm.p + 2)]

"""
    confint(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, level::Real) 
Get the confidence interval for each of the estimated parameters at level (default level = 95%).
"""
confint(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, level::Real) = hcat(GLMCopula.coef(gcm) + GLMCopula.stderror(gcm) * quantile(Normal(), (1. - level) / 2.), GLMCopula.coef(gcm) - GLMCopula.stderror(gcm) * quantile(Normal(), (1. - level) / 2.))

confint(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}) = confint(gcm, 0.95)

"""
    MSE(gcm::GLMCopulaVCModel, β::Vector, Σ::Vector)
Get the mean squared error of the parameters `β` and `Σ`.
"""
function MSE(gcm::GLMCopulaVCModel, β::Vector, Σ::Vector)
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseΣ = sum(abs2, gcm.Σ .- Σ) / gcm.m
    return mseβ, mseΣ
end

"""
    MSE(gcm::GLMCopulaARModel, β::Vector, ρ::Vector, σ2::Vector)
Get the mean squared error of the parameters `β` , `ρ` and `σ2`.
"""
function MSE(gcm::GLMCopulaARModel, β::Vector, ρ::Vector, σ2::Vector)
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseρ = sum(abs2, gcm.ρ .- ρ)
    mseσ2 = sum(abs2, gcm.σ2 .- σ2)
    return mseβ, mseρ, mseσ2
end

"""
    coverage!(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, trueparams::Vector, 
intervals::Matrix, curcoverage::Vector)
Find the coverage of the estimated parameters `β` and `Σ`, given the true parameters.
"""
function coverage!(gcm::Union{GLMCopulaVCModel, GLMCopulaARModel}, trueparams::Vector, 
    intervals::Matrix, curcoverage::Vector)
    copyto!(intervals, confint(gcm))
    lbs = @views intervals[:, 1]
    ubs = @views intervals[:, 2]
    map!((val, lb, ub) -> val >= lb && 
        val <= ub, curcoverage, trueparams, lbs, ubs)
    return curcoverage
end
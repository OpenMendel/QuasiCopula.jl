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
    coef(gcm::GLMCopulaVCModel)
Get the estimated parameter coefficients from the model.
"""
coef(gcm::GLMCopulaVCModel) = [gcm.β; gcm.Σ]

"""
    stderror(gcm::GLMCopulaVCModel)
Get the estimated standard errors from the asymptotic variance covariance matrix of the parameters.
"""
stderror(gcm::GLMCopulaVCModel) = [sqrt(gcm.vcov[i, i]) for i in 1:(gcm.p + gcm.m)]

"""
    confint(gcm::GLMCopulaVCModel, level::Real) 
Get the confidence interval for each of the estimated parameters at level (default level = 95%).
"""
confint(gcm::GLMCopulaVCModel, level::Real) = hcat(coef(gcm) + stderror(gcm) * quantile(Normal(), (1. - level) / 2.), coef(gcm) - stderror(gcm) * quantile(Normal(), (1. - level) / 2.))

confint(gcm::GLMCopulaVCModel) = confint(gcm, 0.95)

"""
    MSE(gcm::GLMCopulaVCModel, β::Vector, Σ::Vector)
Get the mean squared error of the parameters `β` and `Σ`.
"""
function MSE(gcm::GLMCopulaVCModel, β::Vector, Σ::Vector)
    mseβ = sum(abs2, gcm.β - β) / gcm.p
    mseΣ = sum(abs2, gcm.Σ - Σ) / gcm.m
    return mseβ, mseΣ
end

"""
    coverage!(gcm::GLMCopulaVCModel, trueparams::Vector, 
intervals::Matrix, curcoverage::Vector)
Find the coverage of the estimated parameters `β` and `Σ`, given the true parameters.
"""
function coverage!(gcm::GLMCopulaVCModel, trueparams::Vector, 
    intervals::Matrix, curcoverage::Vector)
    copyto!(intervals, confint(gcm))
    lbs = @views intervals[:, 1]
    ubs = @views intervals[:, 2]
    map!((val, lb, ub) -> val >= lb && 
        val <= ub, curcoverage, trueparams, lbs, ubs)
    return curcoverage
end
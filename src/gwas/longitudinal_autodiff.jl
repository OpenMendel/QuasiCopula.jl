function loglikelihood(
    par::AbstractVector{T}, # p+2+1 × 1 where 2 is for θ, 1 is for the SNP
    qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    ) where T
    β = [par[1:end-3]; par[end]] # nongenetic + genetic beta
    θ = par[end-2:end-1]         # vc parameters
    # allocate vector of type T
    n, p = size(qc_model.data[1].X)
    η = zeros(T, n)
    μ = zeros(T, n)
    varμ = zeros(T, n)
    res = zeros(T, n)
    storage_n = zeros(T, n)
    q = zeros(T, length(θ))
    logl = zero(T)
    for (i, gc) in enumerate(qc_model.data)
        snps = fill(z[i], size(gc.X, 1))
        X = hcat(gc.X, snps) # genetic and nongenetic covariates
        y = gc.y
        n, p = size(X)
        # update_res! step (need to avoid BLAS)
        A_mul_b!(η, X, β)
        for j in 1:gc.n
            μ[j] = GLM.linkinv(gc.link, η[j])
            varμ[j] = GLM.glmvar(gc.d, μ[j]) # Note: for negative binomial, d.r is used
            # dμ[j] = GLM.mueta(gc.link, η[j])
            # w1[j] = dμ[j] / varμ[j]
            # w2[j] = w1[j] * dμ[j]
            res[j] = y[j] - μ[j]
        end
        # standardize_res! step
        for j in eachindex(y)
            res[j] /= sqrt(varμ[j])
        end
        # std_res_differential! step (this will compute ∇resβ)
        # for i in 1:gc.p
        #     for j in 1:gc.n
        #         ∇resβ[j, i] = -sqrt(varμ[j]) * X[j, i] - (0.5 * res[j] * (1 - (2 * μ[j])) * X[j, i])
        #     end
        # end
        # update Γ
        @inbounds for k in 1:gc.m
            A_mul_b!(storage_n, gc.V[k], res)
            q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        end
        # component_loglikelihood
        for j in 1:gc.n
            logl += QuasiCopula.loglik_obs(gc.d, y[j], μ[j], one(T), one(T))
        end
        tsum = dot(θ, gc.t)
        logl += -log(1 + tsum)
        qsum  = dot(θ, q) # qsum = 0.5 r'Γr
        logl += log(1 + qsum)
    end
    return logl
end

function loglikelihood(
    par::AbstractVector{T}, # p+2+1+1, where 2 is for θ, 1 is τ, and last 1 is for the SNP
    gcm::GaussianCopulaVCModel,
    z::AbstractVector # n × 1 genotype vector
    ) where T
    β = [par[1:end-4]; par[end]] # nongenetic + genetic beta
    θ = par[end-3:end-2]         # vc parameters
    τ = par[end-1]               # dispersion for gaussian
    # allocate vector of type T
    n, p = size(gcm.data[1].X)
    μ = zeros(T, n)
    res = zeros(T, n)
    storage_n = zeros(T, n)
    q = zeros(T, length(θ))
    logl = zero(T)
    for (i, gc) in enumerate(gcm.data)
        snps = fill(z[i], size(gc.X, 1))
        X = hcat(gc.X, snps) # genetic and nongenetic covariates
        y = gc.y
        n, p = size(X)
        sqrtτ = sqrt(abs(τ))
        # update_res! step (need to avoid BLAS)
        A_mul_b!(μ, X, β)
        for j in 1:gc.n
            res[j] = y[j] - μ[j]
        end
        # standardize_res! step
        res .*= sqrtτ
        rss  = abs2(norm(res)) # RSS of standardized residual
        tsum = dot(abs.(θ), gc.t) # ben: why is there abs here?
        logl += - log(1 + tsum) - (gc.n * log(2π) -  gc.n * log(abs(τ)) + rss) / 2
        # update Γ
        @inbounds for k in 1:gc.m
            A_mul_b!(storage_n, gc.V[k], res)
            q[k] = dot(res, storage_n) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
        end
        qsum  = dot(θ, q)
        logl += log(1 + qsum)
    end
    return logl
end

# Matrix-vector multiply friendly to autodiffing
function A_mul_b!(c::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    n, p = size(A)
    fill!(c, zero(T))
    for j in 1:p, i in 1:n
        c[i] += A[i, j] * b[j]
    end
    return c
end

function GWASCopulaVCModel_autodiff(
    qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel},
    G::SnpArray;
    )
    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(β) = loglikelihood(β, qc_model, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    # some needed constants
    n, q = size(G)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    any(x -> abs(x) > 1e-3, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 1e-3, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=false)
    fullβ = [qc_model.β; qc_model.θ; 0.0]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # score test for each SNP
    pvals = zeros(T, q)
    for j in 1:q
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=false)
        Hfull = ∇²logl(fullβ)
        W = -Hfull[1:end-1, end]
        Q = -Hfull[end, end]
        R = ∇logl(fullβ)[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    return pvals
end

function GWASCopulaVCModel_autodiff(
    gcm::GaussianCopulaVCModel,
    G::SnpArray;
    )
    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(β) = loglikelihood(β, gcm, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    # some needed constants
    n, q = size(G)
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    any(x -> abs(x) > 0.005, gcm.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 0.005, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 1]), center=true, scale=false, impute=false)
    fullβ = [gcm.β; gcm.θ; gcm.τ; 0.0]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-1, 1:end-1])
    # score test for each SNP
    pvals = zeros(T, q)
    for j in 1:q
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=false)
        Hfull = ∇²logl(fullβ)
        W = -Hfull[1:end-1, end]
        Q = -Hfull[end, end]
        R = ∇logl(fullβ)[end]
        S = R * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(1), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    return pvals
end

function multivariateGWAS_autodiff(
    qc_model::MultivariateCopulaVCModel,
    G::SnpArray;
    )
    # some needed constants
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p, d = size(qc_model.B)    # dimension of fixed effects in each sample
    m = length(qc_model.θ)     # number of variance components in each sample
    s = count(x -> typeof(x) <: Normal, qc_model.vecdist) # number of nuisance parameters (only Gaussian for now)
    T = eltype(qc_model.X)
    n == length(qc_model.data) || error("sample size do not agree")

    # define autodiff likelihood, gradient, and Hessians
    autodiff_loglikelihood(vecB) = loglikelihood(vecB, qc_model, z)
    ∇logl = x -> ForwardDiff.gradient(autodiff_loglikelihood, x)
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    ∇logl! = (grad, x) -> ForwardDiff.gradient!(grad, autodiff_loglikelihood, x)
    ∇²logl! = (hess, x) -> ForwardDiff.hessian!(hess, autodiff_loglikelihood, x)

    # compute P (negative Hessian) and inv(P)
    z = convert(Vector{Float64}, @view(G[:, 2]), center=true, scale=true, impute=true)
    fullβ = [vec(qc_model.B); qc_model.θ; zeros(d)]
    Hfull = ∇²logl(fullβ)
    Pinv = inv(-Hfull[1:end-d, 1:end-d])

    # score test for each SNP
    pvals = zeros(T, q)
    grad_store = zeros(T, p*d + m + d)
    W = zeros(T, p*d + m, d)
    Q = zeros(T, d, d)
    @showprogress for j in 1:q
        # grab current SNP needed in logl (z used by autodiff grad and hess)
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=true, impute=true)

        # compute W/Q/R using in-place versions of ForwardDiff grad/hess
        ∇²logl!(Hfull, fullβ)
        copyto!(W, @view(Hfull[1:p*d+m, end-d+1:end]))
        W .*= -1
        copyto!(Q, @view(Hfull[end-d+1:end, end-d+1:end]))
        Q .*= -1
        ∇logl!(grad_store, fullβ)
        R = grad_store[end-d+1:end]

        # compute W/Q/R using not-inplace version of ForwardDiff grad/hess
        # Hfull = ∇²logl(fullβ)
        # W = -Hfull[1:p*d+m, end-d+1:end]
        # Q = -Hfull[end-d+1:end, end-d+1:end]
        # R = ∇logl(fullβ)[end-d+1:end]

        S = R' * inv(Q - W'*Pinv*W) * R
        pval = ccdf(Chisq(d), S)
        pvals[j] = pval == 0 ? 1 : pval
    end
    return pvals
end

function loglikelihood(
    par::AbstractVector{T}, # length pd+m+d, where m is number of VCs, d is for the SNP effect on d phenotypes
    qc_model::MultivariateCopulaVCModel, # fitted null model
    z::AbstractVector # n × 1 genotype vector
    ) where T
    n = length(qc_model.data)
    n == length(z) || error("Expected n == length(z)")

    # parameters
    p, d = size(qc_model.B)
    m = length(par) - p*d - d          # number of variance components
    B = reshape(par[1:p*d], p, d)      # nongenetic covariates
    γ = par[end-d+1:end]               # genetic beta
    θ = par[end-m-d+1:end-d]           # vc parameters

    # storages friendly to autodiff
    ηstore = zeros(T, d)
    μstore = zeros(T, d)
    varμstore = zeros(T, d)
    resstore = zeros(T, d)
    std_resstore = zeros(T, d)
    storage_d = zeros(T, d)
    qstore = zeros(T, m)

    logl = 0.0
    for i in 1:n
        # data for sample i
        xi = @view(qc_model.X[i, :])
        yi = @view(qc_model.Y[i, :])
        # update η, μ, res, to include effect of SNP
        At_mul_b!(ηstore, B, xi)
        ηstore .+= γ .* z[i]
        μstore .= GLM.linkinv.(qc_model.veclink, ηstore)
        varμstore .= GLM.glmvar.(qc_model.vecdist, μstore)
        resstore .= yi .- μstore
        std_resstore .= resstore ./ sqrt.(varμstore)
        # logl += loglikelihood(yi, qc, qc_model)
        # GLM loglikelihood (term 2)
        @inbounds for j in eachindex(yi)
            logl += QuasiCopula.loglik_obs(qc_model.vecdist[j], 
                yi[j], μstore[j], one(T), one(T))
        end
        # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
        tsum = dot(θ, qc_model.t) # tsum = 0.5tr(Γ)
        logl += -log(1 + tsum)
        # loglikelihood term 3 i.e. sum ln(1 + 0.5 r*Γ*r)
        @inbounds for k in 1:qc_model.m # loop over m variance components
            mul!(storage_d, qc_model.V[k], std_resstore) # storage_d = V[k] * r
            qstore[k] = dot(std_resstore, storage_d) / 2 # q[k] = 0.5 r * V[k] * r
        end
        qsum = dot(θ, qstore) # qsum = 0.5 r*Γ*r
        logl += log(1 + qsum)
    end

    return logl
end

function At_mul_b!(c::AbstractVector{T}, A::AbstractMatrix, b::AbstractVector) where T
    n, p = size(A)
    fill!(c, zero(T))
    for j in 1:p, i in 1:n
        c[j] += A[i, j] * b[i]
    end
    return c
end

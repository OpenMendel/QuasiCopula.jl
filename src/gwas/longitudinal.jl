"""
    _get_null_distribution(gcm)

Tries to guess the base distribution for an abstract Copula Model.
"""
_get_null_distribution(gcm::GaussianCopulaVCModel) = Normal()
_get_null_distribution(gcm::NBCopulaVCModel) = NegativeBinomial()
function _get_null_distribution(
    gcm::GLMCopulaVCModel
    )
    T = eltype(gcm.data[1].X)
    if eltype(gcm.d) == Bernoulli{T}
        d = Bernoulli()
    elseif eltype(gcm.d) == Poisson{T}
        d = Poisson()
    else
        error("GLMCopulaVCModel should have marginal distributions Bernoulli or Poisson but was $(eltype(gcm.d))")
    end
    return d
end

"""
    GWASCopulaVCModel(gcm::MixedCopulaVCModel, x::SnpArray)

Performs score tests for each SNP in `x`, given a fitted (null) model on the non-genetic covariates.

# Inputs
+ `gcm`: A fitted `GLMCopulaVCModel` or `NBCopulaVCModel` that includes `n` sample points and `p` non-genetic covariates.
+ `G`: A `SnpArray` (compressed `.bed/bim/fam` PLINK file) with `n` samples and `q` SNPs

# Outputs
A length `q` vector of p-values storing the score statistics for each SNP
"""
function GWASCopulaVCModel(
    gcm::Union{GLMCopulaVCModel, NBCopulaVCModel},
    G::SnpArray;
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p = length(gcm.β) # number of fixed effects in each sample
    m = length(gcm.θ) # number of variance components in each sample
    dist = _get_null_distribution(gcm)
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    any(x -> abs(x) > 1e-3, gcm.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 1e-3, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # preallocated arrays for efficiency
    z = zeros(T, n)
    W = zeros(T, p + m)
    χ2 = Chisq(1)
    pvals = zeros(T, q)
    # compute Pinv (inverse negative Hessian)
    Pinv = get_Pinv(gcm)
    # score test for each SNP
    for j in 1:q
        # sync vectors
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        Q, R = zero(T), zero(T)
        fill!(W, 0)
        # loop over each sample
        for i in 1:n
            # variables for current sample
            gc = gcm.data[i]
            d = gc.n # number of observations for current sample
            zi = fill(z[i], d)
            res = gc.res # d × 1 standardized residuals
            # update ∇resγ
            ∇resγ = zeros(T, d)
            ∇resβ = gc.∇resβ # d × p
            for k in 1:d # loop over each sample's observation
                ∇resγ[k] = update_∇resβ(dist, zi[k], res[k], gc.μ[k], gc.dμ[k], gc.varμ[k])
            end
            # form some constants 
            Γ = zeros(T, d, d)
            for k in 1:gc.m # loop over variance components
                Γ .+= gcm.θ[k] .* gc.V[k]
            end
            # denom = 1 + dot(gcm.θ, gc.q) # note dot(θ, gc.q) = qsum = 0.5 r'Γr
            denom = 1 + 0.5 * (res' * Γ * res)
            denom2 = abs2(denom)
            # calculate W
            Hβγ_i = get_Hβγ_i(gc, Γ, ∇resβ, ∇resγ, zi, gcm.β) # exact
            Hθγ_i = get_neg_Hθγ_i(gc, gcm.θ, ∇resγ) # exact
            W += vcat(Hβγ_i, Hθγ_i)
            # calculate Q
            Q += Transpose(zi) * Diagonal(gc.w2) * zi
            Q += (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2 # 2nd term
            Q -= ∇resγ' * Γ * ∇resγ / denom # 3rd term
            ek = zeros(d)
            for k in 1:d
                fill!(ek, 0)
                ek[k] = 1
                Q -= (ek' * Γ * res * dβdβ_res_ij(gc.d, gc.link, z[i], gc.η[k], gc.μ[k], gc.varμ[k], res[k])) / denom
            end
            # calculate R
            R += Transpose(zi) * Diagonal(gc.w1) * (gc.y - gc.μ) + (∇resγ' * Γ * res) / denom
        end
        # score test
        # @show R
        # @show Q
        # @show W
        # if j == 1
        #     fdsa
        # end
        S = R * inv(Q - dot(W, Pinv, W)) * R
        pvals[j] = ccdf(χ2, S)
    end
    return pvals
end

function get_Pinv(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    return inv(P)
end

function get_Pinv(qc_model::GaussianCopulaVCModel)
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hττ = 
    Hβθ = -get_Hβθ(qc_model)
    Hβτ = 
    Hθτ = 
    P = [Hββ Hβθ Hβτ; Hβθ' Hθθ Hθτ; Hβτ' Hθτ' Hττ]
    return inv(P)
end

function GWASCopulaVCModel(
    gcm::GaussianCopulaVCModel,
    G::SnpArray;
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p = length(gcm.β) # number of fixed effects in each sample
    m = length(gcm.θ) # number of variance components in each sample
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    any(x -> abs(x) > 0.005, gcm.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 0.005, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # compute P (negative Hessian) and inv(P)
    Hββ = -get_Hββ(gcm)
    Hθθ = -get_Hθθ(gcm)
    Hβθ = -get_Hβθ(gcm)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    Pinv = inv(P)
    # preallocated arrays for efficiency
    z = zeros(T, n)
    W = zeros(T, p + m)
    χ2 = Chisq(1)
    pvals = zeros(T, q)
    # score test for each SNP
    for j in 1:q
        # sync vectors
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        Q, R = zero(T), zero(T)
        fill!(W, 0)
        # accumulate precomputed quantities (todo: efficiency)
        for i in 1:n
            # variables for current sample
            gc = gcm.data[i]
            d = gc.n # number of observations for current sample
            zi = fill(z[i], d)
            res = gc.res # d × 1
            # update ∇resγ
            ∇resβ = -sqrt(gcm.τ[1]) .* gc.X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
            ∇resγ = -sqrt(gcm.τ[1]) .* fill(z[i], d)
            # calculate trailing terms (todo: efficiency)
            Γ = zeros(T, d, d)
            for k in 1:gc.m # loop over variance components
                Γ .+= gcm.θ[k] .* gc.V[k]
            end
            # denom = 1 + dot(gcm.θ, gc.q) # note dot(θ, gc.q) = qsum = 0.5 r'Γr
            denom = 1 + 0.5 * (res' * Γ * res)
            denom2 = abs2(denom)
            # calculate W
            Hβγ_i = get_Hβγ_i(gc, Γ, ∇resβ, ∇resγ, zi, gcm.β) # exact
            Hθγ_i = get_neg_Hθγ_i(gc, gcm.θ, ∇resγ) # exact
            W += vcat(Hβγ_i, Hθγ_i)
            # calculate Q
            Q += Transpose(zi) * zi
            Q += (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2 # 2nd term
            Q -= ∇resγ' * Γ * ∇resγ / denom # 3rd term
            ek = zeros(d)
            η = gc.X * gcm.β
            μ = η
            dist = Normal()
            link = IdentityLink()
            varμ = GLM.glmvar.(dist, μ)
            for k in 1:d
                fill!(ek, 0)
                ek[k] = 1
                Q -= (ek' * Γ * res * dβdβ_res_ij(dist, link, z[i], η[k], μ[k], varμ[k], res[k])) / denom
            end
            # calculate R
            R += Transpose(zi) * (gc.y - μ) + (∇resγ' * Γ * res) / denom


            # Wtrail = (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            # Qtrail = (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            # Rtrail = (∇resγ' * Γ * res) / denom
            # # third Hessian term
            # Wtrail2 = ∇resβ' * Γ * ∇resγ / denom
            # Qtrail2 = ∇resγ' * Γ * ∇resγ / denom
            # # score test variables
            # W .+= Transpose(gc.X) * zi .+ Wtrail
            # Q += Transpose(zi) * zi + Qtrail
            # # W .+= Transpose(gc.X) * zi .+ Wtrail .- Wtrail2
            # # Q += Transpose(zi) * zi + Qtrail - Qtrail2
            # R += Transpose(zi) * (gc.y - gc.X * gcm.β) + Rtrail
        end
        # score test (todo: efficiency)
        # @show W
        # @show R
        # @show Q
        # hhh
        S = R * inv(Q - dot(W, Pinv, W)) * R
        pvals[j] = ccdf(χ2, S)
    end
    return pvals
end

function get_Hθθ(qc_model)
    # use loglikelihood! function to get Hθθ. Commented out code gives the same answer
    loglikelihood!(qc_model, true, true)
    return qc_model.Hθ
    # m = length(qc_model.data[1].V) # number of variance components
    # hess_math = zeros(m, m)
    # for i in eachindex(qc_model.data)
    #     r = qc_model.data[i].res
    #     Ω = qc_model.data[i].V
    #     b = [0.5r' * Ω[k] * r for k in 1:m]
    #     c = [0.5tr(Ω[k]) for k in 1:m]
    #     hess_math += b*b' / (1 + qc_model.θ'*b)^2 - c*c' / (1 + qc_model.θ'*c)^2
    # end
    # return -hess_math
end

function get_Hβθ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    m = length(qc_model.data[1].V)  # number of variance components
    p = size(qc_model.data[1].X, 2) # number of fixed effects
    hess_math = zeros(p, m)
    for i in eachindex(qc_model.data)
        r = qc_model.data[i].res
        Ω = qc_model.data[i].V
        θ = qc_model.θ
        ∇resβ = qc_model.data[i].∇resβ
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end
# must distinguish for gaussian case because it doesn't have ∇resβ precomputed
function get_Hβθ(qc_model::GaussianCopulaVCModel)
    m = length(qc_model.data[1].V)  # number of variance components
    p = size(qc_model.data[1].X, 2) # number of fixed effects
    hess_math = zeros(p, m)
    for (i, gc) in enumerate(qc_model.data)
        r = gc.res
        Ω = gc.V
        θ = qc_model.θ
        ∇resβ = -sqrt(qc_model.τ[1]) .* gc.X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end

function get_neg_Hθγ_i(gc, θ, ∇resγ)
    m = length(gc.V)  # number of variance components
    r = gc.res
    Ω = gc.V
    b = [0.5r' * Ω[k] * r for k in 1:m]
    A = hcat([∇resγ' * Ω[k] * r for k in 1:m]...)
    # calculate Hγθ, then return Hθγ = Hγθ'
    Hγθ = (A*θ ./ (1 + θ'*b)^2) * b' - A ./ (1 + θ'*b)
    return Hγθ'
end

function get_Hβγ_i(gc::Union{GLMCopulaVCObs, NBCopulaVCObs}, Γ, ∇resβ, ∇resγ, z::AbstractVector, β) # z is a vector of SNP value (length d)
    res = gc.res
    denom = 1 + 0.5 * (res' * Γ * res)
    denom2 = abs2(denom)
    # 1st Hessian term
    Hβγ_i = Transpose(gc.X) * Diagonal(gc.w2) * z
    # 2nd Hessian term
    Hβγ_i += (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    # 3rd Hessian terms
    Hβγ_i -= ∇resβ' * Γ * ∇resγ / denom
    # 4th Hessian term
    ej = zeros(gc.n)
    η = gc.η
    μ = gc.μ
    varμ = gc.varμ
    res = gc.res
    for j in 1:gc.n
        fill!(ej, 0)
        ej[j] = 1
        xj = gc.X[j, :]
        zi = z[1]
        Hβγ_i -= dot(ej, Γ, res) * dγdβresβ_ij(gc.d, gc.link, xj, zi, η[j], μ[j], varμ[j], res[j]) / denom
    end
    return Hβγ_i
end
# need to distinguish for gaussian case because Hessian is defined differently and η is not defined
function get_Hβγ_i(gc::GaussianCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, β) # z is a vector of SNP value (length d)
    res = gc.res
    denom = 1 + 0.5 * (res' * Γ * res)
    denom2 = abs2(denom)
    # 1st Hessian term
    Hβγ_i = Transpose(gc.X) * z
    # 2nd Hessian term
    Hβγ_i += (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    # 3rd Hessian terms
    Hβγ_i -= ∇resβ' * Γ * ∇resγ / denom
    # 4th Hessian term
    ej = zeros(gc.n)
    η = gc.X * β
    μ = η
    dist = Normal()
    link = IdentityLink()
    varμ = GLM.glmvar.(dist, μ)
    res = gc.res
    for j in 1:gc.n
        fill!(ej, 0)
        ej[j] = 1
        xj = gc.X[j, :]
        zi = z[1]
        Hβγ_i -= dot(ej, Γ, res) * ∇²resβ_ij(dist, link, xj, zi, η[j], μ[j], varμ[j], res[j]) / denom
    end
    return Hβγ_i
end

function get_Hββ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)    
    # loop over samples
    for (i, gc) in enumerate(qc_model.data)
        d = gc.n # number of observations for current sample
        # GLM term
        H -= Transpose(gc.X) * Diagonal(gc.w2) * gc.X
        # 2nd term
        res = gc.res # d × 1 standardized residuals
        ∇resβ = gc.∇resβ # d × p
        Γ = zeros(T, d, d)
        for k in 1:gc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* gc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = gc.η
        μ = gc.μ
        varμ = gc.varμ
        res = gc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = gc.X[j, :]
            dist = qc_model.d[i]
            link = qc_model.link[i]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
        # @show gc.w2
        # @show res
        # @show η
        # @show μ
        # @show varμ
        # j = 1
        # @show dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])
        # fdsa
    end
    return H
end
# must distinguish for gaussian case because it doesn't have ∇resβ precomputed
function get_Hββ(qc_model::GaussianCopulaVCModel)
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)    
    # loop over samples
    for (i, gc) in enumerate(qc_model.data)
        d = gc.n # number of observations for current sample
        # GLM term
        H -= Transpose(gc.X) * gc.X
        # 2nd term
        res = gc.res # d × 1 standardized residuals
        # ∇resβ = gc.∇resβ # d × p
        ∇resβ = -sqrt(qc_model.τ[1]) .* gc.X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
        Γ = zeros(T, d, d)
        for k in 1:gc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* gc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = gc.X * qc_model.β
        μ = η
        dist = Normal()
        link = IdentityLink()
        varμ = GLM.glmvar.(dist, μ)
        res = gc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = gc.X[j, :]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
    end
    return H
end

"""
    GWASCopulaVCModel(qc_model::MixedCopulaVCModel, x::SnpArray)

Performs score tests for each SNP in `x`, given a fitted (null) model on the non-genetic covariates.

# Inputs
+ `qc_model`: A fitted `GLMCopulaVCModel` or `NBCopulaVCModel` that includes `n` sample points and `p` non-genetic covariates.
+ `G`: A `SnpArray` (compressed `.bed/bim/fam` PLINK file) with `n` samples and `q` SNPs

# Outputs
A length `q` vector of p-values storing the score statistics for each SNP
"""
function GWASCopulaVCModel(
    qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel},
    G::SnpArray;
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p = length(qc_model.β) # number of fixed effects in each sample
    m = length(qc_model.θ) # number of variance components in each sample
    maxd = maxclustersize(qc_model)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    any(x -> abs(x) > 1e-3, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 1e-3, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")
    # preallocated arrays for efficiency
    z = zeros(T, n)
    zi = zeros(T, maxd)
    W = zeros(T, p + m)
    χ2 = Chisq(1)
    pvals = zeros(T, q)
    # compute Pinv (inverse negative Hessian)
    Pinv = get_Pinv(qc_model)
    # score test for each SNP
    for j in 1:q
        # sync vectors
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        Q, R = zero(T), zero(T)
        fill!(W, 0)
        # loop over each sample
        for i in 1:n
            # variables for current sample
            qc = qc_model.data[i]
            d = qc.n # number of observations for current sample
            fill!(zi, z[i])
            # update gradient of residual with respect to β and γ
            ∇resγ = get_∇resγ(qc_model, i, @view(zi[1:d])) # d × 1
            ∇resβ = get_∇resβ(qc_model, i) # d × p
            # form some constants 
            Γ = zeros(T, d, d)
            for k in 1:qc.m # loop over variance components
                Γ .+= qc_model.θ[k] .* qc.V[k]
            end
            denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
            denom2 = abs2(denom)
            # calculate W
            update_W!(W, qc_model, i, @view(zi[1:d]), Γ, ∇resβ, ∇resγ)
            # update Q
            Q += calculate_Qi(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom, denom2)
            # update R
            R += calculate_Ri(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom)
        end
        # score test
        # @show R
        # @show Q
        # @show W
        # if j == 3
        #     fdsa
        # end
        S = R * inv(Q - dot(W, Pinv, W)) * R
        pvals[j] = ccdf(χ2, S)
    end
    return pvals
end

function update_W!(W, qc_model, i::Int, zi::AbstractVector, Γ, ∇resβ, ∇resγ)
    p, m = qc_model.p, qc_model.m
    qc = qc_model.data[i]
    Hβγ_i = get_Hβγ_i(qc, Γ, ∇resβ, ∇resγ, zi, qc_model.β) # exact
    Hθγ_i = get_neg_Hθγ_i(qc, qc_model.θ, ∇resγ) # exact
    for j in 1:p
        W[j] += Hβγ_i[j]
    end
    offset = p
    for j in 1:m
        W[offset + j] += Hθγ_i[j]
    end
    return W
end

function calculate_Ri(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resγ, denom)
    qc = qc_model.data[i]
    res = qc.res # y - Xb
    return Transpose(zi) * qc.res + (∇resγ' * Γ * res) / denom
end

function calculate_Ri(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector, Γ, ∇resγ, denom)
    qc = qc_model.data[i]
    res = qc.res # y - μ / std(σ), i.e. standardized residuals
    return Transpose(zi) * Diagonal(qc.w1) * (qc.y - qc.μ) + (∇resγ' * Γ * res) / denom
end

function calculate_Qi(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resγ, denom, denom2)
    qc = qc_model.data[i]
    res = qc.res
    d = qc_model.data[i].n # number of observation for sample i

    Qi = Transpose(zi) * zi
    Qi += (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2 # 2nd term
    Qi -= ∇resγ' * Γ * ∇resγ / denom # 3rd term
    η = qc.X * qc_model.β
    μ = η
    dist = Normal()
    link = IdentityLink()
    varμ = GLM.glmvar.(dist, μ)
    ek = zeros(d)
    for k in 1:d
        fill!(ek, 0)
        ek[k] = 1
        Qi -= (ek' * Γ * res * dβdβ_res_ij(dist, link, zi[1], η[k], μ[k], varμ[k], res[k])) / denom
    end
    return Qi
end

function calculate_Qi(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector, Γ, ∇resγ, denom, denom2)
    qc = qc_model.data[i]
    res = qc.res
    d = qc_model.data[i].n # number of observation for sample i

    Qi = Transpose(zi) * Diagonal(qc.w2) * zi
    Qi += (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2 # 2nd term
    Qi -= ∇resγ' * Γ * ∇resγ / denom # 3rd term
    ek = zeros(d)
    for k in 1:d
        fill!(ek, 0)
        ek[k] = 1
        Qi -= (ek' * Γ * res * dβdβ_res_ij(qc.d, qc.link, zi[1], qc.η[k], qc.μ[k], qc.varμ[k], res[k])) / denom
    end
    return Qi
end

function get_∇resγ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector)
    qc = qc_model.data[i]
    d = size(qc.X, 1)
    T = eltype(qc.X)
    ∇resγ = zeros(T, d)
    for k in 1:d # loop over each sample's observation
        ∇resγ[k] = update_∇res_ij(qc.d, zi[k], qc.res[k], qc.μ[k], qc.dμ[k], qc.varμ[k])
    end
    return ∇resγ
end
function get_∇resγ(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector)
    d = size(qc_model.data[i].X, 1)
    T = eltype(qc_model.data[i].X)
    ∇resγ = zeros(T, d)
    # ∇resγ = -sqrt(qc_model.τ[1]) .* fill(z[i], d) # for gaussian case
    for k in 1:d # loop over each sample's observation
        ∇resγ[k] = update_∇res_ij(Normal(), zi[k], zero(T), zero(T), zero(T), zero(T))
    end
    return ∇resγ
end

function get_∇resβ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int)
    return qc_model.data[i].∇resβ # d × p
end
function get_∇resβ(qc_model::GaussianCopulaVCModel, i::Int)
    ∇resβ = -sqrt(qc_model.τ[1]) .* qc_model.data[i].X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
    return ∇resβ
end

function get_Pinv(qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel})
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    return inv(P)
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
    for (i, qc) in enumerate(qc_model.data)
        r = qc.res
        Ω = qc.V
        θ = qc_model.θ
        ∇resβ = -sqrt(qc_model.τ[1]) .* qc.X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end

function get_neg_Hθγ_i(qc, θ, ∇resγ)
    m = length(qc.V)  # number of variance components
    r = qc.res
    Ω = qc.V
    b = [0.5r' * Ω[k] * r for k in 1:m]
    A = hcat([∇resγ' * Ω[k] * r for k in 1:m]...)
    # calculate Hγθ, then return Hθγ = Hγθ'
    Hγθ = (A*θ ./ (1 + θ'*b)^2) * b' - A ./ (1 + θ'*b)
    return Hγθ'
end

function get_Hβγ_i(qc::Union{GLMCopulaVCObs, NBCopulaVCObs}, Γ, ∇resβ, ∇resγ, z::AbstractVector, β) # z is a vector of SNP value (length d)
    res = qc.res
    denom = 1 + 0.5 * (res' * Γ * res)
    denom2 = abs2(denom)
    # 1st Hessian term
    Hβγ_i = Transpose(qc.X) * Diagonal(qc.w2) * z
    # 2nd Hessian term
    Hβγ_i += (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    # 3rd Hessian terms
    Hβγ_i -= ∇resβ' * Γ * ∇resγ / denom
    # 4th Hessian term
    ej = zeros(qc.n)
    η = qc.η
    μ = qc.μ
    varμ = qc.varμ
    res = qc.res
    for j in 1:qc.n
        fill!(ej, 0)
        ej[j] = 1
        xj = qc.X[j, :]
        zi = z[1]
        Hβγ_i -= dot(ej, Γ, res) * dγdβresβ_ij(qc.d, qc.link, xj, zi, η[j], μ[j], varμ[j], res[j]) / denom
    end
    return Hβγ_i
end
# need to distinguish for gaussian case because Hessian is defined differently and η is not defined
function get_Hβγ_i(qc::GaussianCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, β) # z is a vector of SNP value (length d)
    res = qc.res
    denom = 1 + 0.5 * (res' * Γ * res)
    denom2 = abs2(denom)
    # 1st Hessian term
    Hβγ_i = Transpose(qc.X) * z
    # 2nd Hessian term
    Hβγ_i += (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
    # 3rd Hessian terms
    Hβγ_i -= ∇resβ' * Γ * ∇resγ / denom
    # 4th Hessian term
    ej = zeros(qc.n)
    η = qc.X * β
    μ = η
    dist = Normal()
    link = IdentityLink()
    varμ = GLM.glmvar.(dist, μ)
    res = qc.res
    for j in 1:qc.n
        fill!(ej, 0)
        ej[j] = 1
        xj = qc.X[j, :]
        zi = z[1]
        Hβγ_i -= dot(ej, Γ, res) * dγdβresβ_ij(dist, link, xj, zi, η[j], μ[j], varμ[j], res[j]) / denom
    end
    return Hβγ_i
end

function get_Hββ(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    p = length(qc_model.β)
    T = eltype(qc_model.β)
    H = zeros(T, p, p)    
    # loop over samples
    for (i, qc) in enumerate(qc_model.data)
        d = qc.n # number of observations for current sample
        # GLM term
        H -= Transpose(qc.X) * Diagonal(qc.w2) * qc.X
        # 2nd term
        res = qc.res # d × 1 standardized residuals
        ∇resβ = qc.∇resβ # d × p
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = qc.η
        μ = qc.μ
        varμ = qc.varμ
        res = qc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = qc.X[j, :]
            dist = qc_model.d[i]
            link = qc_model.link[i]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
        # @show qc.w2
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
    for (i, qc) in enumerate(qc_model.data)
        d = qc.n # number of observations for current sample
        # GLM term
        H -= Transpose(qc.X) * qc.X
        # 2nd term
        res = qc.res # d × 1 standardized residuals
        # ∇resβ = qc.∇resβ # d × p
        ∇resβ = -sqrt(qc_model.τ[1]) .* qc.X # see end of 11.3.1 https://arxiv.org/abs/2205.03505
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5 * (res' * Γ * res)
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom^2
        # 3rd term
        H += (∇resβ' * Γ * ∇resβ) / denom
        # 4th term
        ej = zeros(d)
        η = qc.X * qc_model.β
        μ = η
        dist = Normal()
        link = IdentityLink()
        varμ = GLM.glmvar.(dist, μ)
        res = qc.res
        for j in 1:d
            fill!(ej, 0)
            ej[j] = 1
            xj = qc.X[j, :]
            H += (ej' * Γ * res * dβdβ_res_ij(dist, link, xj, η[j], μ[j], varμ[j], res[j])) / denom
        end
    end
    return H
end

struct Storages{T <: BlasReal}
    vec_p::Vector{T}
    vec_maxd::Vector{T}
    Γ::Matrix{T}
    denom::Vector{T}
    denom2::Vector{T}
    p_storage::Vector{T}
    p_storage2::Vector{T}
end
function storages(p::Int, maxd::Int)
    vec_p = zeros(p)
    vec_maxd = zeros(maxd)
    Γ = zeros(maxd, maxd)
    denom = zeros(1)
    denom2 = zeros(1)
    p_storage = zeros(p)
    p_storage2 = zeros(p)
    return Storages(vec_p, vec_maxd, Γ, denom, denom2, p_storage, p_storage2)
end
function Base.fill!(s::Storages, x::Number)
    fill!(s.vec_p, x)
    fill!(s.vec_maxd, x)
    fill!(s.Γ, x)
    fill!(s.denom, x)
    fill!(s.denom2, x)
end

"""
    GWASCopulaVCModel(qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel}, G::SnpArray)

Performs score tests for each SNP in `G`, given a fitted (null) model on the 
non-genetic covariates.

# Inputs
+ `qc_model`: A fitted `GaussianCopulaVCModel`, `GLMCopulaVCModel` or `NBCopulaVCModel` 
    that includes `n` sample points and `p` non-genetic covariates.
+ `G`: A `SnpArray` (compressed `.bed/bim/fam` PLINK file) with `n` samples and 
    `q` SNPs

# Outputs
A length `q` vector of p-values storing the score statistics for each SNP
"""
function GWASCopulaVCModel(
    qc_model::Union{GaussianCopulaVCModel, GLMCopulaVCModel, NBCopulaVCModel},
    G::SnpArray;
    check_grad::Bool=true
    )
    n = size(G, 1)    # number of samples with genotypes
    q = size(G, 2)    # number of SNPs in each sample
    p = length(qc_model.β) # number of fixed effects in each sample
    m = length(qc_model.θ) # number of variance components in each sample
    s = typeof(qc_model) <: GaussianCopulaVCModel ? 1 : 0 # number of nuisance parameters (only gaussian case for now)
    maxd = maxclustersize(qc_model)
    T = eltype(qc_model.data[1].X)
    n == length(qc_model.data) || error("sample size do not agree")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇β) && error("Null model gradient of beta is not zero!")
    check_grad && any(x -> abs(x) > 1e-1, qc_model.∇θ) && error("Null model gradient of variance components is not zero!")

    # preallocated arrays for efficiency
    z = zeros(T, n)
    zi = zeros(T, maxd)
    W = zeros(T, p + m + s)
    χ2 = Chisq(1)
    pvals = zeros(T, q)
    storage = storages(p, maxd)
    Wtime = 0.0
    Qtime = 0.0
    Rtime = 0.0
    Pinv = get_Pinv(qc_model) # compute Pinv (inverse negative Hessian)

    # score test for each SNP
    @showprogress for j in 1:q
        # sync vectors
        SnpArrays.copyto!(z, @view(G[:, j]), center=true, scale=false, impute=true)
        Q, R = zero(T), zero(T)
        fill!(W, zero(T))
        fill!(storage, zero(T))
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
            storage.denom[1] = denom
            storage.denom2[1] = denom2
            # calculate W
            Wtime += @elapsed update_W!(W, qc_model, i, @view(zi[1:d]), Γ, ∇resβ, ∇resγ, storage)
            # update Q
            Qtime += @elapsed Q += calculate_Qi(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom, denom2)
            # update R
            Rtime += @elapsed R += calculate_Ri(qc_model, i, @view(zi[1:d]), Γ, ∇resγ, denom)
        end
        # score test
        S = R * inv(Q - dot(W, Pinv, W)) * R
        # @show Pinv
        # @show R
        # @show Q
        # @show W
        # @show S
        # if j == 1
        #     fdsa
        # end
        pvals[j] = ccdf(χ2, S)
    end
    # @show Wtime
    # @show Qtime
    # @show Rtime
    return pvals
end

# update W expression for 1 sample
# i stands for sample i, and zi is a d-vector of SNP value
function update_W!(W, qc_model::GLMCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resβ, ∇resγ, storages::Storages)
    p, m = qc_model.p, qc_model.m
    qc = qc_model.data[i]
    Hβγ_i = get_Hβγ_i(qc, Γ, ∇resβ, ∇resγ, zi, qc_model.β, storages) # exact
    Hθγ_i = get_neg_Hθγ_i(qc, qc_model.θ, ∇resγ, storages) # exact
    for j in 1:p
        W[j] += Hβγ_i[j]
    end
    for j in 1:m
        W[p + j] += Hθγ_i[j]
    end
    return W
end

function update_W!(W, qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resβ, ∇resγ, storages::Storages)
    p, m = qc_model.p, qc_model.m
    qc = qc_model.data[i]
    Hβγ_i = get_Hβγ_i(qc, Γ, ∇resβ, ∇resγ, zi, qc_model.β, qc_model.τ[1], storages) # exact
    Hθγ_i = get_neg_Hθγ_i(qc, qc_model.θ, ∇resγ, storages) # exact
    for j in 1:p
        W[j] -= Hβγ_i[j]
    end
    for j in 1:m
        W[p + j] -= Hθγ_i[j]
    end
    W[end] -= get_Hτγ_i(qc, zi, qc_model.θ, qc_model.τ[1]) # exact
    return W
end

function calculate_Ri(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resγ, denom)
    τ = qc_model.τ[1]
    qc = qc_model.data[i]
    res = qc.y - qc.X * qc_model.β # y - Xb
    tmp = Transpose(zi) * Γ * res
    return τ * (Transpose(zi) * res - Transpose(zi) * Γ * res / denom)
end

function calculate_Ri(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel}, i::Int, zi::AbstractVector, Γ, ∇resγ, denom)
    qc = qc_model.data[i]
    res = qc.res # y - μ / std(σ), i.e. standardized residuals
    return Transpose(zi) * Diagonal(qc.w1) * (qc.y - qc.μ) + (∇resγ' * Γ * res) / denom
end

function calculate_Qi(qc_model::GaussianCopulaVCModel, i::Int, zi::AbstractVector, Γ, ∇resγ, denom, denom2)
    qc = qc_model.data[i]
    res = qc.y - qc.X*qc_model.β
    d = qc_model.data[i].n # number of observation for sample i
    τ = qc_model.τ[1]

    Qi = τ * Transpose(zi) * zi # 1st term
    Qi -= τ * Transpose(zi) * Γ * zi / denom # 2nd term
    tmp =  Transpose(zi) * Γ * res
    Qi += τ^2 * tmp * tmp' / denom2 # 3rd term
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
        dist = typeof(qc_model.d[i]) <: NegativeBinomial ? NegativeBinomial(qc_model.r[1]) : qc_model.d[i]
        Qi -= (ek' * Γ * res * dβdβ_res_ij(dist, qc.link, zi[1], qc.η[k], qc.μ[k], qc.varμ[k], res[k])) / denom
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

function get_Pinv(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    P = [Hββ Hβθ; Hβθ' Hθθ]
    return inv(P)
end

# gaussian case needs to handle tau term separately
function get_Pinv(qc_model::GaussianCopulaVCModel)
    Hββ = -get_Hββ(qc_model)
    Hθθ = -get_Hθθ(qc_model)
    Hβθ = -get_Hβθ(qc_model)
    Hβτ = -get_Hτβ(qc_model)
    Hθτ = -get_Hτθ(qc_model)
    Hττ = -get_Hττ(qc_model)
    P = [Hββ  Hβθ  Hβτ;
         Hβθ' Hθθ  Hθτ;
         Hβτ' Hθτ' Hττ]
    return inv(P)
end

function get_Hθθ(qc_model::Union{GaussianCopulaVCModel,GLMCopulaVCModel,NBCopulaVCModel})
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
        ∇resβ = -sqrt(qc_model.τ[1]) .* qc.X # see eq8 of 11.3.1 https://arxiv.org/abs/2205.03505, 2nd term is 0, and first term ∇μ = X
        b = [0.5r' * Ω[k] * r for k in 1:m]
        A = hcat([∇resβ' * Ω[k] * r for k in 1:m]...)
        hess_math += A ./ (1 + θ'*b) - (A*θ ./ (1 + θ'*b)^2) * b'
    end
    return hess_math
end

function get_neg_Hθγ_i(gc, θ, ∇resγ, storages::Storages)
    m = length(gc.V)  # number of variance components
    r = gc.res
    Ω = gc.V
    b = [0.5r' * Ω[k] * r for k in 1:m]
    A = hcat([∇resγ' * Ω[k] * r for k in 1:m]...)
    # calculate Hγθ, then return Hθγ = Hγθ'
    Hγθ = (A*θ ./ (1 + θ'*b)^2) * b' - A ./ (1 + θ'*b)
    return Hγθ'
end

function get_Hβγ_i(qc::GLMCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, β, storages::Storages) # z is a vector of SNP value (length d)
    res = qc.res
    denom = storages.denom
    denom2 = storages.denom2
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

function get_Hβγ_i(qc::GaussianCopulaVCObs, Γ, ∇resβ, ∇resγ, z::AbstractVector, β, τ, storages::Storages) # z is a vector of SNP value (length d)
    res = qc.y - qc.X * β
    denom = 1 + 0.5τ * (res' * Γ * res)
    # 1st Hessian term
    Hβγ_i = -τ * Transpose(qc.X) * z
    # 2nd Hessian term
    res = qc.y - (qc.X * β)
    Hβγ_i += τ * Transpose(qc.X) * Γ * z / denom
    # 3rd term
    tmp1 = Transpose(qc.X) * Γ * res
    tmp2 = Transpose(z) * Γ * res
    Hβγ_i -= τ^2 * tmp1 * tmp2' / denom^2
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
            dist = typeof(qc_model.d[i]) <: NegativeBinomial ? NegativeBinomial(qc_model.r[1]) : qc_model.d[i]
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
    τ = qc_model.τ[1]
    # loop over samples
    for qc in qc_model.data
        d = qc.n
        # GLM term
        H -= τ .* (Transpose(qc.X) * qc.X)
        # 2nd term
        res = qc.y - qc.X * qc_model.β
        Γ = zeros(T, d, d)
        for k in 1:qc.m # loop over variance components
            Γ .+= qc_model.θ[k] .* qc.V[k]
        end
        denom = 1 + 0.5τ * (res' * Γ * res)
        H += τ * Transpose(qc.X) * Γ * qc.X / denom
        # 3rd term
        tmp = Transpose(qc.X) * Γ * res
        H -= τ^2 * tmp * tmp' / denom^2
    end
    return H
end

# needed functions for gaussian case (zi is length d vector of SNP values)
# i stands for sample i
function get_Hτγ_i(qc::GaussianCopulaVCObs, zi::AbstractVector{T}, 
    θ::AbstractVector{T}, τ::T) where T
    Hτγ = zero(T)
    # compute sqrt(τ)z'Γ(y-Xb) (numerator of 2nd term)
    for k in 1:qc.m
        mul!(qc.storage_n, qc.V[k], qc.res) # storage_n = V[k] * res = V[k] * (y-Xb) * sqrt(τ) 
        Hτγ += θ[k] * dot(zi, qc.storage_n) # Hτγ = sqrt(τ)θ[k]z'V[k](y-Xb) = sqrt(τ)z'Γ(y-Xb)
    end
    qsum = dot(θ, qc.q) # qsum = 0.5r(β)*Γ*r(β) where r = (y-Xb) * sqrt(τ)
    denom = abs2(1 + qsum) # denom = (1 + 0.5τ(y-Xb)'Γ(y-Xb) )^2
    Hτγ /= -denom # Hτγ = -sqrt(τ)z'Γ(y-Xb) / denom
    Hτγ += dot(zi, qc.res) # Hτγ = -sqrt(τ)z'Γ(y-Xb) / denom + z'(y-Xb)sqrt(τ)
    return Hτγ / sqrt(τ)
end
function get_Hτβ(qc_model::GaussianCopulaVCModel)
    β = qc_model.β
    θ = qc_model.θ
    τ = qc_model.τ[1]
    T = eltype(β)
    Hτβ = zeros(T, length(β))
    Hτβstore = zeros(T, length(β))
    for qc in qc_model.data
        fill!(Hτβstore, 0)
        # compute Hτβstore = sqrt(τ)X'Γ(y-Xb)
        for k in 1:qc.m
            mul!(qc.storage_n, qc.V[k], qc.res) # storage_n = V[k] * res = V[k] * (y-Xb) * sqrt(τ) 
            BLAS.gemv!('T', θ[k], qc.X, qc.storage_n, one(T), Hτβstore) # Hτβstore = sqrt(τ)θ[k]X'V[k](y-Xb)
        end
        qsum = dot(θ, qc.q) # qsum = 0.5r(β)*Γ*r(β) where r = (y-Xb) * sqrt(τ)
        denom = abs2(1 + qsum) # denom = (1 + 0.5τ(y-Xb)'Γ(y-Xb) )^2
        BLAS.gemv!('T', one(T), qc.X, qc.res, -inv(denom), Hτβstore) # Hτβstore = -sqrt(τ)X'Γ(y-Xb) / denom + X'(y-Xb)sqrt(τ)
        Hτβstore ./= sqrt(τ)
        Hτβ .+= Hτβstore
    end
    return Hτβ
end
function get_Hτθ(qc_model::GaussianCopulaVCModel)
    θ = qc_model.θ
    m = length(θ)
    T = eltype(θ)
    Hτθ = zeros(T, m)
    qstore = zeros(T, m)
    for qc in qc_model.data
        copyto!(qstore, qc.q) # gc.q = 0.5r(β)*V[k]*r(β) where r = (y-Xb) * sqrt(τ)
        qstore ./= qc_model.τ[1]
        Hτθ .+= inv(1 + dot(θ, qstore)) .* qstore
    end
    return Hτθ
end
function get_Hττ(qc_model::GaussianCopulaVCModel)
    # use loglikelihood! function to get Hττ, which is called in get_Hθθ already
    # loglikelihood!(qc_model, true, true)
    return qc_model.Hτ
end
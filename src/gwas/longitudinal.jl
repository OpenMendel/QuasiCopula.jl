struct MixedCopulaVCObs{T <: BlasReal}
    # data
    y::Vector{T}    # d by 1 vector of response
    X::Matrix{T}    # d by p design matrix
    V::Vector{Matrix{T}} # vector of (known) covariances
    d::Int          # number of observations within a single sample
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇ϕ::Vector{T}   # gradient wrt ϕ
    ∇θ::Vector{T}   # gradient wrt θ2
    Hβ::Matrix{T}   # Hessian wrt β
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    Hϕ::Matrix{T}   # Hessian wrt ϕ
    res::Vector{T}  # standardized residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2 (this is variable c in VC model)
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2 (this is variable b in VC model)
    # storage_n::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    storage_d::Vector{T}
    # storage_p1::Vector{T}
    # storage_p2::Vector{T}
    # storage_np::Matrix{T}
    storage_dp::Matrix{T}
    # added_term_numerator::Matrix{T}
    # added_term2::Matrix{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function MixedCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    ) where T <: BlasReal
    d, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == d "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ = Matrix{T}(undef, d, p)
    ∇ϕ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hϕ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, d)
    t   = [tr(V[k])/2 for k in 1:m] # t is variable c in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2
    q   = Vector{T}(undef, m) # q is variable b in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2
    m1  = Vector{T}(undef, m)
    m2  = Vector{T}(undef, m)
    storage_d = Vector{T}(undef, d)
    storage_dp = Matrix{T}(undef, d, p)
    # storage_n = Vector{T}(undef, n)
    # storage_p1 = Vector{T}(undef, p)
    # storage_p2 = Vector{T}(undef, p)
    # storage_np = Matrix{T}(undef, n, p)
    # added_term_numerator = Matrix{T}(undef, n, p)
    # added_term2 = Matrix{T}(undef, p, p)
    η = Vector{T}(undef, d)
    μ = Vector{T}(undef, d)
    varμ = Vector{T}(undef, d)
    dμ = Vector{T}(undef, d)
    wt = ones(T, d)
    w1 = Vector{T}(undef, d)
    w2 = Vector{T}(undef, d)
    # constructor
    MixedCopulaVCObs{T}(y, X, V, d, p, m, ∇β, ∇resβ, ∇ϕ, ∇θ, Hβ, Hθ, Hϕ, 
        res, t, q, m1, m2, storage_d, storage_dp, η, μ, varμ, dμ, wt, w1, w2)
end

struct MixedCopulaVCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{MixedCopulaVCObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    vecdist::Vector{<:UnivariateDistribution} # length d vector of marginal distributions for each data point
    veclink::Vector{<:Link} # length d vector of link functions for each marginal distribution
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    ϕ::Vector{T}    # length d vector of dispersion parameters for each marginal; for poissona/bernoulli this should be NaN
    θ::Vector{T}    # length m vector of variance components
    # working arrays
    ∇β::Vector{T}   # gradient terms from all observations
    ∇ϕ::Vector{T}
    ∇θ::Vector{T}
    Hβ::Matrix{T}   # Hessian terms from all observations
    Hθ::Matrix{T}
    Hϕ::Matrix{T}
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    # asymptotic covariance for inference
    # Ainv::Matrix{T}
    # Aevec::Matrix{T}
    # M::Matrix{T}
    # vcov::Matrix{T}
    # ψ::Vector{T}
    # storage variables
    # storage_n::Vector{T}
    # storage_m::Vector{T}
    # storage_θ::Vector{T}
    penalized::Bool
end

function MixedCopulaVCModel(
    gcs::Vector{MixedCopulaVCObs{T}},
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each data point
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    penalized::Bool = false
    ) where T <: BlasReal
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    d       = length(vecdist)
    β       = Vector{T}(undef, p)
    ϕ       = ones(T, d)
    θ       = Vector{T}(undef, m)
    ∇β      = Vector{T}(undef, p)
    ∇ϕ      = Vector{T}(undef, d)
    ∇θ      = Vector{T}(undef, m)
    Hβ      = Matrix{T}(undef, p, p)
    Hθ      = Matrix{T}(undef, m, m)
    Hϕ      = Matrix{T}(undef, d, d)
    # Ainv    = zeros(T, p + m, p + m)
    # Aevec   = zeros(T, p + m, p + m)
    # M       = zeros(T, p + m, p + m)
    # vcov    = zeros(T, p + m, p + m)
    # ψ       = Vector{T}(undef, p + m)
    TR      = Matrix{T}(undef, n, m) # collect trace terms
    ntotal  = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        TR[i, :] .= gcs[i].t
    end
    QF = Matrix{T}(undef, n, m)
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    # storage_n = Vector{T}(undef, n)
    # storage_m = Vector{T}(undef, m)
    # storage_θ = Vector{T}(undef, m)
    MixedCopulaVCModel{T}(gcs, ntotal, p, m, vecdist, veclink, 
        β, ϕ, θ, ∇β, ∇ϕ, ∇θ, Hβ, Hθ, Hϕ, TR, QF, penalized)
end

# struct GWASCopulaVCModel{T <: BlasReal}
#     gcm::MixedCopulaVCModel{T} # fitted null model
#     G::SnpArray     # n by q (compressed) genotype matrix
#     Pinv::Matrix{T} # p by p matrix
#     W::Vector{T} # length p vector
#     Q::Vector{T} # length 1 vector
#     R::Vector{T} # length 1 vector
#     pvals::Vector{T} # length q vector of p-values for each SNP in G
# end

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
    num_Hessian_terms::Int = 2
    )
    n, q = size(G)
    p = length(gcm.β)
    dist = _get_null_distribution(gcm)
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    any(x -> abs(x) > 1e-3, gcm.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 1e-3, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # preallocated arrays for efficiency
    z = zeros(T, n)
    W = zeros(T, p)
    χ2 = Chisq(1)
    pvals = zeros(T, q)
    # compute P (negative Hessian) and inv(P)
    if num_Hessian_terms == 2
        P = -two_term_Hessian(gcm)
    elseif num_Hessian_terms == 3
        P = -three_term_Hessian(gcm)
    elseif num_Hessian_terms == 4
        P = -four_term_Hessian(gcm)
    else
        error("num_Hessian_terms should be 2, 3, or 4 but was $num_Hessian_terms")
    end
    Pinv = inv(P)
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
            # calculate trailing terms (todo: efficiency)
            Γ = zeros(T, d, d)
            for k in 1:gc.m # loop over variance components
                Γ .+= gcm.θ[k] .* gc.V[k]
            end
            # denom = 1 + dot(gcm.θ, gc.q) # note dot(θ, gc.q) = qsum = 0.5 r'Γr
            denom = 1 + 0.5 * (res' * Γ * res)
            denom2 = abs2(denom)
            Wtrail = (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            Qtrail = (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            Rtrail = (∇resγ' * Γ * res) / denom
            # third Hessian term
            Wtrail2 = ∇resβ' * Γ * ∇resγ / denom
            Qtrail2 = ∇resγ' * Γ * ∇resγ / denom
            # score test variables
            W .+= Transpose(gc.X) * Diagonal(gc.w2) * zi .+ Wtrail .- Wtrail2
            Q += Transpose(zi) * Diagonal(gc.w2) * zi + Qtrail - Qtrail2
            R += Transpose(zi) * Diagonal(gc.w1) * (gc.y - gc.μ) + Rtrail
        end
        # score test (todo: efficiency)
        S = R * inv(Q - W'*Pinv*W) * R
        pvals[j] = ccdf(χ2, S)
    end
    return pvals
end

function GWASCopulaVCModel(
    gcm::GaussianCopulaVCModel,
    G::SnpArray;
    num_Hessian_terms::Int = 2
    )
    n, q = size(G)
    p = length(gcm.β)
    T = eltype(gcm.data[1].X)
    n == length(gcm.data) || error("sample size do not agree")
    any(x -> abs(x) > 1e-3, gcm.∇β) && error("Null model gradient of beta is not zero!")
    any(x -> abs(x) > 1e-3, gcm.∇θ) && error("Null model gradient of variance components is not zero!")
    # approximate FIM by negative Hessian (todo: is the Hessian in gaussian_VC.jl actually the expect Hessian?)
    if num_Hessian_terms == 2
        P = -two_term_Hessian(gcm)
    elseif num_Hessian_terms == 3
        P = -three_term_Hessian(gcm)
    elseif num_Hessian_terms == 4
        P = -four_term_Hessian(gcm)
    else
        error("num_Hessian_terms should be 2, 3, or 4 but was $num_Hessian_terms")
    end
    Pinv = inv(P)
    # preallocated arrays for efficiency
    z = zeros(T, n)
    W = zeros(T, p)
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
            # ∇resβ = -gcm.τ[1] .* gc.X # this is wrong mathematically but empirically gives beautiful QQ plots
            # ∇resγ = -gcm.τ[1] .* fill(z[i], d)
            # calculate trailing terms (todo: efficiency)
            Γ = zeros(T, d, d)
            for k in 1:gc.m # loop over variance components
                Γ .+= gcm.θ[k] .* gc.V[k]
            end
            # denom = 1 + dot(gcm.θ, gc.q) # note dot(θ, gc.q) = qsum = 0.5 r'Γr
            denom = 1 + 0.5 * (res' * Γ * res)
            denom2 = abs2(denom)
            Wtrail = (∇resβ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            Qtrail = (∇resγ' * Γ * res) * (∇resγ' * Γ * res)' / denom2
            Rtrail = (∇resγ' * Γ * res) / denom
            # third Hessian term
            Wtrail2 = ∇resβ' * Γ * ∇resγ / denom
            Qtrail2 = ∇resγ' * Γ * ∇resγ / denom
            # score test variables
            W .+= Transpose(gc.X) * zi .+ Wtrail
            Q += Transpose(zi) * zi + Qtrail
            # W .+= Transpose(gc.X) * zi .+ Wtrail .- Wtrail2
            # Q += Transpose(zi) * zi + Qtrail - Qtrail2
            R += Transpose(zi) * (gc.y - gc.X * gcm.β) + Rtrail
        end
        # score test (todo: efficiency)
        S = R * inv(Q - W'*Pinv*W) * R
        pvals[j] = ccdf(χ2, S)
    end
    return pvals
end

# 2 term hessian from math
function two_term_Hessian(gcm::Union{GLMCopulaVCModel, NBCopulaVCModel})
    p = length(gcm.β)
    T = eltype(gcm.β)
    H = zeros(T, p, p)
    for gc in gcm.data
        d = gc.n # number of observations for current sample
        # GLM term
        H -= Transpose(gc.X) * Diagonal(gc.w2) * gc.X
        # trailing terms
        res = gc.res # d × 1 standardized residuals
        ∇resβ = gc.∇resβ # d × p
        Γ = zeros(T, d, d)
        for k in 1:gc.m # loop over variance components
            Γ .+= gcm.θ[k] .* gc.V[k]
        end
        denom = abs2(1 + 0.5 * (res' * Γ * res))
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom
    end
    return H
end

function two_term_Hessian(gcm::GaussianCopulaVCModel)
    p = length(gcm.β)
    T = eltype(gcm.β)
    H = zeros(T, p, p)
    for gc in gcm.data
        d = gc.n # number of observations for current sample
        # GLM term
        H -= Transpose(gc.X) * gc.X
        # trailing terms
        res = gc.res # d × 1 standardized residuals
        ∇resβ = -sqrt(gcm.τ[1]) .* gc.X # d × p
        Γ = zeros(T, d, d)
        for k in 1:gc.m # loop over variance components
            Γ .+= gcm.θ[k] .* gc.V[k]
        end
        denom = abs2(1 + 0.5 * (res' * Γ * res))
        H -= (∇resβ' * Γ * res) * (∇resβ' * Γ * res)' / denom
    end
    return H
end

# 3 term hessian from math
function three_term_Hessian(qc_model::Union{GLMCopulaVCModel, NBCopulaVCModel})
    loglikelihood!(qc_model, true, true)
    return qc_model.Hβ
end

# 4 term hessian using autodiff
function four_term_Hessian(qc_model)
    # to use autodiff on loglikelihood, define autodiff_loglikelihood which has only 1 argument
    autodiff_loglikelihood(β) = loglikelihood(β, qc_model)
    # call autodiff on autodiff_loglikelihood
    ∇²logl = x -> ForwardDiff.hessian(autodiff_loglikelihood, x)
    return ∇²logl(qc_model.β)
end

update_∇resβ(d::Normal, x_ji, res_j, μ_j, dμ_j, varμ_j) = -x_ji
update_∇resβ(d::Bernoulli, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -sqrt(varμ_j) * x_ji - (0.5 * res_j * (1 - 2μ_j) * x_ji)
update_∇resβ(d::Poisson, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    x_ji * (
    -(inv(sqrt(varμ_j)) + (0.5 * inv(varμ_j)) * res_j) * dμ_j
    )
update_∇resβ(d::NegativeBinomial, x_ji, res_j, μ_j, dμ_j, varμ_j) = 
    -inv(sqrt(varμ_j)) * dμ_j * x_ji - (0.5 * inv(varμ_j)) *
    res_j * (μ_j * inv(d.r) + (1 + inv(d.r) * μ_j)) * dμ_j * x_ji

# function update_∇resβ(d::Bernoulli, x_ji, res_j, μ_j, dμ_j, varμ_j)
#     # println("dμ_j = $dμ_j, varμ_j = $varμ_j") # these are equal
#     invσ = inv(sqrt(varμ_j))
#     dμdβ = dμ_j * x_ji
#     dσ2dβ = (1 - 2μ_j) * dμdβ
#     ben = -invσ*dμdβ - 0.5 * inv(varμ_j) * res_j * dσ2dβ # this should be the correct one to use??

#     sarah = -sqrt(varμ_j) * x_ji - (0.5 * res_j * (1 - 2μ_j) * x_ji) # this assumes dμ/dσ^2 = 1, which seems to be true for Bernoulli with logit link
#     empirically_desired = -varμ_j * x_ji - (0.5 * res_j * (1 - 2μ_j) * x_ji)

#     # println("ben = $ben, sarah = $sarah, empirically_desired = $empirically_desired")
#     return sarah
# end

# cannot find a good way to tweak Poisson QQ plots
# function update_∇resβ(d::Poisson, x_ji, res_j, μ_j, dμ_j, varμ_j)
#     sarah = x_ji * (-(inv(sqrt(varμ_j)) + (0.5 * inv(varμ_j)) * res_j) * dμ_j)
#     ben = (-inv(sqrt(varμ_j)) - 0.5inv(varμ_j) * res_j) * dμ_j * x_ji
#     empirically_desired = (-1 - 0.5inv(varμ_j * sqrt(varμ_j)) * res_j) * dμ_j * x_ji
#     # println("ben = $ben, sarah = $sarah, empirically_desired = $empirically_desired")
#     # rand() < 0.1 && fdsa
#     return sarah
# end

"""
    fit!(gcm::MixedCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `MixedCopulaVCModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `gcm.β`, `gcm.θ`, `gcm.ϕ` this is for Normal base.

# Arguments
- `gcm`: A `MixedCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    gcm::MixedCopulaVCModel,
    solver=Ipopt.IpoptSolver(
        print_level = 5, 
        tol = 10^-6, 
        max_iter = 1000,
        limited_memory_max_history = 6, # default value
        accept_after_max_steps = 4,
        warm_start_init_point="yes", 
        hessian_approximation = "limited-memory")
    )
    initialize_model!(gcm)
    npar = gcm.p + gcm.m # todo: optimize ϕ
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    offset = gcm.p + 1
    for k in 1:gcm.m # variance components must be >0
        lb[offset] = 0
        offset += 1
    end
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, gcm)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, gcm)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(gcm, MathProgBase.getsolution(optm))
    loglikelihood!(gcm, true, false)
    return optm
end

"""
    modelpar_to_optimpar!(par, gcm)

Translate model parameters in `gcm` to optimization variables in `par` for Normal base.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    gcm :: MixedCopulaVCModel
    )
    # β
    copyto!(par, gcm.β)
    # variance components
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.θ[k]
        offset += 1
    end
    # par[offset] = gcm.ϕ[1] # todo
    par
end

"""
    optimpar_to_modelpar_quasi!(gcm, par)

Translate optimization variables in `par` to the model parameters in `gcm`.
"""
function optimpar_to_modelpar!(
    gcm :: MixedCopulaVCModel,
    par :: Vector
    )
    # β
    copyto!(gcm.β, 1, par, 1, gcm.p)
    # variance components # todo: dispatch to CS/VC/AR variance models
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset += 1
    end
    # gcm.ϕ[1] = par[offset] # todo
    gcm
end

function MathProgBase.initialize(
    gcm::MixedCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(gcm::MixedCopulaVCModel) = [:Grad, :Hess]

function MathProgBase.eval_f(
    gcm :: MixedCopulaVCModel,
    par :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
    gcm  :: MixedCopulaVCModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(gcm, par)
    obj = loglikelihood!(gcm, true, false)
    # gradient wrt β
    copyto!(grad, gcm.∇β)
    # gradient wrt variance comps
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        grad[offset] = gcm.∇θ[k]
        offset += 1
    end
    # grad[offset] = gcm.∇ϕ[1] # todo
    obj
end

MathProgBase.eval_g(gcm::MixedCopulaVCModel, g, par) = nothing
MathProgBase.jac_structure(gcm::MixedCopulaVCModel) = Int[], Int[]
MathProgBase.eval_jac_g(gcm::MixedCopulaVCModel, J, par) = nothing

function MathProgBase.hesslag_structure(gcm::MixedCopulaVCModel)
    m◺ = ◺(gcm.m)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    arr2 = Vector{Int}(undef, ◺(gcm.p) + m◺ + 1)
    # Hββ block
    idx = 1
    for j in 1:gcm.p
        for i in j:gcm.p
            arr1[idx] = i
            arr2[idx] = j
            idx += 1
        end
    end
    # variance components
    for j in 1:gcm.m
        for i in 1:j
            arr1[idx] = gcm.p + i
            arr2[idx] = gcm.p + j
            idx += 1
        end
    end
    arr1[idx] = gcm.p + gcm.m + 1
    arr2[idx] = gcm.p + gcm.m + 1
    return (arr1, arr2)
end

function MathProgBase.eval_hesslag(
    gcm :: MixedCopulaVCModel,
    H   :: Vector{T},
    par :: Vector{T},
    σ   :: T,
    μ   :: Vector{T}
    )where {T <: BlasReal}
    optimpar_to_modelpar!(gcm, par)
    loglikelihood!(gcm, true, true)
    # Hβ block
    idx = 1
    @inbounds for j in 1:gcm.p, i in 1:j
        H[idx] = gcm.Hβ[i, j]
        idx += 1
    end
    # Haa block
    @inbounds for j in 1:gcm.m, i in 1:j
        H[idx] = gcm.Hθ[i, j]
        idx += 1
    end
    # H[idx] = gcm.Hϕ[1, 1] # todo
    # lmul!(σ, H)
    H .*= σ
end

"""
    initialize_model!(gcm)

# todo
"""
function initialize_model!(gcm::MixedCopulaVCModel{T}) where T <: BlasReal
    fill!(gcm.β, 0)
    fill!(gcm.ϕ, 1)
    fill!(gcm.θ, 1.0)
    return nothing
end

function loglikelihood!(
    gc::MixedCopulaVCObs{T},
    β::Vector{T},
    ϕ::Vector{T}, # dispersion parameters for each marginal distributions
    θ::Vector{T}, # variance components
    vecdist::Vector{<:UnivariateDistribution},
    veclink::Vector{<:Link},
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where T <: BlasReal
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇ϕ, 0)
        fill!(gc.∇θ, 0)
    end
    if needhess
        fill!(gc.Hβ, 0)
        fill!(gc.Hθ, 0)
        fill!(gc.Hϕ, 0)
    end
    # update residuals and its gradient
    update_res!(gc, β, vecdist, veclink)
    standardize_res!(gc, ϕ)
    std_res_differential!(gc, vecdist) # compute ∇resβ

    # loglikelihood term 2 i.e. sum sum ln(f_ij | β)
    logl = QuasiCopula.component_loglikelihood(gc, vecdist)
    # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
    tsum = dot(θ, gc.t) # tsum = 0.5tr(Γ)
    logl += -log(1 + tsum)
    # update Γ before computing logl term3 (todo: multiple dispatch to handle VC/AR/CS)
    @inbounds for k in 1:gc.m # loop over m variance components
        mul!(gc.storage_d, gc.V[k], gc.res) # storage_d = V[k] * r
        if needgrad
            BLAS.gemv!('T', θ[k], gc.∇resβ, gc.storage_d, 1.0, gc.∇β) # ∇β = ∇r'Γr
        end
        gc.q[k] = dot(gc.res, gc.storage_d) / 2 # q[k] = 0.5 r' * V[k] * r (update variable b for variance component model)
    end
    # loglikelihood term 3 i.e. sum ln(1 + 0.5 r'Γr)
    qsum = dot(θ, gc.q) # qsum = 0.5 r'Γr
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * dot(θ, θ)
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum) # inv1pq = 1 / (1 + 0.5r'Γr)
        if needhess
            # approximate Hessian of β
            mul!(gc.storage_dp, Diagonal(gc.w2), gc.X)
            BLAS.gemm!('T', 'N', -one(T), gc.X, gc.storage_dp, zero(T), gc.Hβ) # Hβ = -Xi'*Diagonal(W2)*Xi
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # Hβ = -Xi'*Diagonal(W2)*Xi - ∇β*∇β' / (1 + 0.5r'Γr)^2
            copytri!(gc.Hβ, 'L') # syrk! above only lower triangular
            # println(gc.Hβ)
            # println(-1 .* (transpose(gc.X)*Diagonal(gc.w2)*gc.X) - abs2(inv1pq) .* (gc.∇β * transpose(gc.∇β)))
            # Hessian of vc vector (todo: are these correct?)
            inv1pt = inv(1 + tsum) # inv1pt = 1 / (1 + 0.5tr(Γ))
            gc.m1 .= gc.q
            gc.m1 .*= inv1pq # m1[k] = 0.5 r' * V[k] * r / (1 + 0.5r'Γr)
            gc.m2 .= gc.t
            gc.m2 .*= inv1pt
            BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            copytri!(gc.Hθ, 'U')
            # Hessian of ϕ (todo)
            # gc.Hϕ[1, 1] = - abs2(qsum * inv1pq / ϕ)
        end
        # compute (y-μ)*(dg/varμ) for remaining parts of ∇β
        gc.storage_d .= gc.w1 .* (gc.y .- gc.μ)
        BLAS.gemv!('T', one(T), gc.X, gc.storage_d, inv1pq, gc.∇β) # ∇β = X'*Diagonal(dg/varμ)*(y-μ) + ∇r'Γr/(1 + 0.5r'Γr)
        # gc.∇β .*= sqrtϕ # todo: how does ϕ get involved here?
        # gc.∇ϕ  .= (gc.n - rss + 2qsum * inv1pq) / 2ϕ # todo: deal with ϕ
        gc.∇θ .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
        if penalized
            gc.∇θ .-= θ
        end
    end
    # output
    logl
end

function loglikelihood!(
    gcm::MixedCopulaVCModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇ϕ, 0)
        fill!(gcm.∇θ, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0)
        fill!(gcm.Hϕ, 0)
        fill!(gcm.Hθ, 0)
    end
    logl = zeros(T, Threads.nthreads())
    Threads.@threads for i in eachindex(gcm.data)
        @inbounds logl[Threads.threadid()] += loglikelihood!(gcm.data[i], gcm.β,
            gcm.ϕ, gcm.θ, gcm.vecdist, gcm.veclink, needgrad, needhess; 
            penalized = gcm.penalized)
    end
    @inbounds for i in eachindex(gcm.data)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇ϕ .+= gcm.data[i].∇ϕ
            gcm.∇θ .+= gcm.data[i].∇θ
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hϕ .+= gcm.data[i].Hϕ
            gcm.Hθ .+= gcm.data[i].Hθ
        end
    end
    return sum(logl)
end

function update_res!(
    gc::MixedCopulaVCObs,
    β::Vector,
    vecdist::Vector{<:UnivariateDistribution},
    veclink::Vector{<:Link}
    )
    mul!(gc.η, gc.X, β)
    @inbounds for i in 1:gc.d
        gc.μ[i] = GLM.linkinv(veclink[i], gc.η[i])
        gc.varμ[i] = GLM.glmvar(vecdist[i], gc.μ[i]) # Note: for negative binomial, d.r is used
        gc.dμ[i] = GLM.mueta(veclink[i], gc.η[i])
        gc.w1[i] = gc.dμ[i] / gc.varμ[i]
        gc.w2[i] = gc.w1[i] * gc.dμ[i]
        gc.res[i] = gc.y[i] - gc.μ[i]
    end
    return gc.res
end

# todo: when j is Gaussian, should we divide by ϕ[j]?
function standardize_res!(gc::MixedCopulaVCObs, ϕ::AbstractVector)
    @turbo for j in eachindex(gc.y)
        gc.res[j] /= sqrt(gc.varμ[j])
    end
end

function std_res_differential!(
    gc::MixedCopulaVCObs,
    vecdist::Vector{<:UnivariateDistribution},
    )
    fill!(gc.∇resβ, 0.0)
    @inbounds for i in 1:gc.p, j in 1:gc.d
        gc.∇resβ[j, i] = update_∇resβ(vecdist[j], gc.X[j, i], gc.res[j], gc.μ[j], gc.dμ[j], gc.varμ[j])
    end
    return nothing
end

function component_loglikelihood(
    gc::MixedCopulaVCObs{T},
    vecdist::Vector{<:UnivariateDistribution},
    ) where T <: BlasReal
    logl = zero(T)
    @inbounds for j in 1:gc.d
        logl += QuasiCopula.loglik_obs(vecdist[j], gc.y[j], gc.μ[j], gc.wt[j], one(T))
    end
    logl
end
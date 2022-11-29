# struct holding intermediate arrays for a given sample
struct MultivariateCopulaVCObs{T <: BlasReal}
    η::Vector{T} # η = B'x (linear predictor value of current sample)
    μ::Vector{T} # μ = linkinv(link, η) (mean of current sample)
    res::Vector{T} # res[i] = yᵢ - μᵢ (residual)
    std_res::Vector{T} # std_res[i] = (yᵢ - μᵢ) / σᵢ (standardized residual)
    dμ::Vector{T} # intermediate GLM quantity
    varμ::Vector{T} # intermediate GLM quantity
    w1::Vector{T} # intermediate GLM quantity
    w2::Vector{T} # intermediate GLM quantity
    ∇resβ::Matrix{T} # gradient of standardized residual with respect to beta
    q::Vector{T} # q[k] = res_i' * V_i[k] * res_i / 2 (this is variable b in VC model, see sec 6.2 of QuasiCopula paper)
end

struct MultivariateCopulaVCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    V::Vector{Matrix{T}} # length m vector of d × d matrices
    vecdist::Vector{<:UnivariateDistribution} # length d vector of marginal distributions for each phenotype
    veclink::Vector{<:Link} # length d vector of link functions for each phenotype's marginal distribution
    data::Vector{MultivariateCopulaVCObs{T}}
    # data dimension
    n::Int # sample size
    d::Int # number of phenotypes per sample
    p::Int # number of (non-genetic) covariates per sample
    m::Int # number of variance components
    # parameters
    B::Matrix{T}    # p × d matrix of mean regression coefficients, Y = XB
    θ::Vector{T}    # length m vector of variance components
    ϕ::Vector{T}    # nuisance parameters for each phenotype
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2 (this is variable c in VC model)
    # working arrays
    Γ::Matrix{T}      # d × d covariance matrix, in VC model this is θ[1]*V[1] + ... + θ[m]*V[m]
    ∇vecB::Vector{T}  # length pd vector, its the gradient of vec(B) 
    ∇θ::Vector{T}     # length m vector, gradient of variance components
    # ∇ϕ::Vector{T}
    HvecB::Matrix{T}  # pd × pd matrix of Hessian
    Hθ::Matrix{T}     # m × m matrix of Hessian for variance components
    # Hϕ::Matrix{T}
    penalized::Bool
end

function MultivariateCopulaVCModel(
    Y::Matrix{T},
    X::Matrix{T},
    V::Union{Matrix{T}, Vector{Matrix{T}}}, # variance component matrices of the phenotypes
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each phenotype
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    penalized = false
    ) where T <: BlasReal
    n, d = size(Y)
    p = size(X, 2)
    m = typeof(V) <: Matrix ? 1 : length(V)
    n == size(X, 1) || error("Number of samples in Y and X mismatch")
    # initialize variables
    B = zeros(T, p, d)
    θ = zeros(T, m)
    ϕ = zeros(T, d)
    Γ = zeros(T, d, d)
    ∇vecB = zeros(T, p*d)
    ∇θ = zeros(T, m)
    HvecB = zeros(T, p*d, p*d)
    Hθ = zeros(T, m, m)
    t = [tr(V[k])/2 for k in 1:m] # t is variable c in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2. Because all Vs are the same in multivariate analysis, all samples share the same t
    # construct MultivariateCopulaVCObs that hold intermediate variables for each sample
    data = MultivariateCopulaVCObs{T}[]
    for _ in 1:n
        η = zeros(T, d)
        μ = zeros(T, d)
        res = zeros(T, d)
        std_res = zeros(T, d)
        dμ = zeros(T, d)
        varμ = zeros(T, d)
        w1 = zeros(T, d)
        w2 = zeros(T, d)
        ∇resβ = zeros(T, p, d)
        q = zeros(T, m) # q is variable b in f(θ) = sum ln(1 + θ'b) - sum ln(1 + θ'c) in section 6.2
        obs = MultivariateCopulaVCObs(η, μ, res, std_res, dμ, varμ, w1, w2, ∇resβ, q)
        push!(data, obs)
    end
    # change type of variables to match struct
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    typeof(V) <: Matrix && (V = [V])
    return MultivariateCopulaVCModel(
        Y, X, V, vecdist, veclink, data,
        n, d, p, m,
        B, θ, ϕ, t, 
        Γ, ∇vecB, ∇θ, HvecB, Hθ,
        penalized
    )
end

"""
    fit!(qc_model::MultivariateCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `MultivariateCopulaVCModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `qc_model.β`, `qc_model.θ`, `qc_model.ϕ` this is for Normal base.

# Arguments
- `qc_model`: A `MultivariateCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    qc_model::MultivariateCopulaVCModel,
    solver=Ipopt.IpoptSolver(
        print_level = 5, 
        tol = 10^-6, 
        max_iter = 1000,
        limited_memory_max_history = 50, 
        accept_after_max_steps = 4,
        warm_start_init_point="yes", 
        hessian_approximation = "limited-memory")
    )
    p, d, m = qc_model.p, qc_model.d, qc_model.m
    initialize_model!(qc_model)
    npar = p * d + m
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    offset = p*d + 1
    for k in 1:m # variance components must be >0
        lb[offset] = 0
        offset += 1
    end
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Max, qc_model)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, qc_model)
    MathProgBase.setwarmstart!(optm, par0)
    # optimize
    MathProgBase.optimize!(optm)
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(qc_model, MathProgBase.getsolution(optm))
    loglikelihood!(qc_model, true, false)
end

"""
    modelpar_to_optimpar!(par, qc_model)

Translate model parameters in `qc_model` to optimization variables in `par` for Normal base.
"""
function modelpar_to_optimpar!(
    par :: Vector,
    qc_model :: MultivariateCopulaVCModel
    )
    # β
    copyto!(par, qc_model.B)
    # variance components
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        par[offset] = qc_model.θ[k]
        offset += 1
    end
    # par[offset] = qc_model.ϕ[1] # todo
    par
end

"""
    optimpar_to_modelpar_quasi!(qc_model, par)

Translate optimization variables in `par` to the model parameters in `qc_model`.
"""
function optimpar_to_modelpar!(
    qc_model :: MultivariateCopulaVCModel,
    par :: Vector
    )
    # β
    copyto!(qc_model.B, 1, par, 1, qc_model.p * qc_model.d)
    # variance components
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        qc_model.θ[k] = par[offset]
        offset += 1
    end
    # qc_model.ϕ[1] = par[offset] # todo
    qc_model
end

function MathProgBase.initialize(
    qc_model::MultivariateCopulaVCModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(qc_model::MultivariateCopulaVCModel) = [:Grad]

function MathProgBase.eval_f(
    qc_model :: MultivariateCopulaVCModel,
    par :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    loglikelihood!(qc_model, false, false) # don't need gradient here
end

function MathProgBase.eval_grad_f(
    qc_model  :: MultivariateCopulaVCModel,
    grad :: Vector,
    par  :: Vector
    )
    optimpar_to_modelpar!(qc_model, par)
    obj = loglikelihood!(qc_model, true, false)
    # gradient wrt β
    copyto!(grad, qc_model.∇vecB)
    # gradient wrt variance comps
    offset = qc_model.p * qc_model.d + 1
    @inbounds for k in 1:qc_model.m
        grad[offset] = qc_model.∇θ[k]
        offset += 1
    end
    # grad[offset] = qc_model.∇ϕ[1] # todo
    obj
end

MathProgBase.eval_g(qc_model::MultivariateCopulaVCModel, g, par) = nothing
MathProgBase.jac_structure(qc_model::MultivariateCopulaVCModel) = Int[], Int[]
MathProgBase.eval_jac_g(qc_model::MultivariateCopulaVCModel, J, par) = nothing

# function MathProgBase.hesslag_structure(qc_model::MultivariateCopulaVCModel)
#     m◺ = ◺(qc_model.m)
#     # we work on the upper triangular part of the Hessian
#     arr1 = Vector{Int}(undef, ◺(qc_model.p) + m◺ + 1)
#     arr2 = Vector{Int}(undef, ◺(qc_model.p) + m◺ + 1)
#     # Hββ block
#     idx = 1
#     for j in 1:qc_model.p
#         for i in j:qc_model.p
#             arr1[idx] = i
#             arr2[idx] = j
#             idx += 1
#         end
#     end
#     # variance components
#     for j in 1:qc_model.m
#         for i in 1:j
#             arr1[idx] = qc_model.p + i
#             arr2[idx] = qc_model.p + j
#             idx += 1
#         end
#     end
#     arr1[idx] = qc_model.p + qc_model.m + 1
#     arr2[idx] = qc_model.p + qc_model.m + 1
#     return (arr1, arr2)
# end

# function MathProgBase.eval_hesslag(
#     qc_model :: MultivariateCopulaVCModel,
#     H   :: Vector{T},
#     par :: Vector{T},
#     σ   :: T,
#     μ   :: Vector{T}
#     )where {T <: BlasReal}
#     optimpar_to_modelpar!(qc_model, par)
#     loglikelihood!(qc_model, true, true)
#     # Hβ block
#     idx = 1
#     @inbounds for j in 1:qc_model.p, i in 1:j
#         H[idx] = qc_model.Hβ[i, j]
#         idx += 1
#     end
#     # Haa block
#     @inbounds for j in 1:qc_model.m, i in 1:j
#         H[idx] = qc_model.Hθ[i, j]
#         idx += 1
#     end
#     # H[idx] = qc_model.Hϕ[1, 1] # todo
#     # lmul!(σ, H)
#     H .*= σ
# end

"""
    initialize_model!(qc_model)

# todo
"""
function initialize_model!(qc_model::MultivariateCopulaVCModel)
    fill!(qc_model.B, 0)
    # fill!(qc_model.ϕ, 1)
    fill!(qc_model.θ, 1.0)
    return nothing
end

function loglikelihood!(
    qc_model::MultivariateCopulaVCModel{T},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where T <: BlasReal
    if needgrad
        fill!(qc_model.∇vecB, 0)
        fill!(qc_model.∇θ, 0)
    end
    if needhess
        fill!(qc_model.HvecB, 0)
        fill!(qc_model.Hθ, 0)
    end
    logl = zero(T)
    for i in 1:qc_model.n
        logl += loglikelihood!(qc_model, i, needgrad, needhess)
        if needgrad
            qc_model.∇β .+= qc_model.data[i].∇vecB
            qc_model.∇θ .+= qc_model.data[i].∇θ
        end
        if needhess
            qc_model.Hβ .+= qc_model.data[i].HvecB
            qc_model.Hθ .+= qc_model.data[i].Hθ
        end
    end
    return logl
end

# evaluates the loglikelihood for sample i
function loglikelihood!(
    qc_model::MultivariateCopulaVCModel,
    i::Int,
    needgrad::Bool = false,
    needhess::Bool = false;
    ) where T <: BlasReal
    # update residuals and its gradient
    update_res!(qc_model, i)
    std_res_differential!(qc_model, i) # compute ∇resβ
    # loglikelihood term 2 i.e. sum sum ln(f_ij | β)
    logl = QuasiCopula.component_loglikelihood(qc_model, i)
    fdsa
    # loglikelihood term 1 i.e. -sum ln(1 + 0.5tr(Γ(θ)))
    tsum = dot(θ, gc.t) # tsum = 0.5tr(Γ)
    logl += -log(1 + tsum)
    # update Γ before computing logl term3 (todo: multiple dispatch to handle VC/AR/CS)
    @inbounds for k in 1:gc.m # loop over m variance components
        mul!(gc.storage_d, gc.V[k], gc.res) # storage_d = V[k] * r
        if needgrad
            BLAS.gemv!('T', θ[k], gc.∇resβ, gc.storage_d, 1.0, gc.∇β) # ∇β = ∇r'Γr
        end
        gc.q[k] = dot(gc.res, gc.storage_d) / 2 # q[k] = 0.5 r' * V[k] * r
    end
    # loglikelihood term 3 i.e. sum ln(1 + 0.5 r'Γr)
    qsum = dot(θ, gc.q) # qsum = 0.5 r'Γr
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if qc_model.penalized
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

function update_res!(
    qc_model::MultivariateCopulaVCModel,
    i::Int
    ) where T <: BlasReal
    # data for sample i
    xi = @view(qc_model.X[i, :])
    yi = @view(qc_model.Y[i, :])
    vecdist = qc_model.vecdist
    veclink = qc_model.veclink
    obs = qc_model.data[i]
    # update necessary quantities
    mul!(obs.η, qc_model.B', xi)
    @inbounds for j in eachindex(xi)
        obs.μ[j] = GLM.linkinv(veclink[j], obs.η[j])
        obs.varμ[j] = GLM.glmvar(vecdist[j], obs.μ[j]) # Note: for negative binomial, d.r is used
        obs.dμ[j] = GLM.mueta(veclink[j], obs.η[j])
        obs.w1[j] = obs.dμ[j] / obs.varμ[j]
        obs.w2[j] = obs.w1[j] * obs.dμ[j]
        obs.res[j] = yi[j] - obs.μ[j]
        obs.std_res[j] = obs.res[j] / sqrt(obs.varμ[j]) # todo: when j is Gaussian, should we divide by ϕ[j]?
    end
    return nothing
end

function std_res_differential!(
    qc_model::MultivariateCopulaVCModel,
    i::Int
    )
    obs = qc_model.data[i]
    p, d = size(obs.∇resβ) # p = number of covariates, d = number of phenotypes
    @inbounds for k in 1:p, j in 1:d
        obs.∇resβ[k, j] = update_∇resβ(qc_model.vecdist[j], qc_model.X[i, k], 
            obs.std_res[j], obs.μ[j], obs.dμ[j], obs.varμ[j])
    end
    return nothing
end

function component_loglikelihood(
    qc_model::MultivariateCopulaVCModel{T},
    i::Int
    ) where T <: BlasReal
    y = @view(qc_model.Y[i, :])
    obs = qc_model.data[i]
    logl = zero(T)
    @inbounds for j in eachindex(y)
        logl += QuasiCopula.loglik_obs(qc_model.vecdist[j], y[j], obs.μ[j], one(T), one(T))
    end
    return logl
end


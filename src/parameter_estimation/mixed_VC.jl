struct MixedCopulaVCObs{T <: BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}} # vector of (known) covariances
    n::Int          # number of observations within a single sample (note this is "d" in Quasi-Copula paper)
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
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    # storage_n::Vector{T}
    # m1::Vector{T}
    # m2::Vector{T}
    # storage_p1::Vector{T}
    # storage_p2::Vector{T}
    # storage_np::Matrix{T}
    # storage_pp::Matrix{T}
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
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ = Matrix{T}(undef, n, p)
    ∇ϕ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hϕ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    # storage_n = Vector{T}(undef, n)
    # m1        = Vector{T}(undef, m)
    # m2        = Vector{T}(undef, m)
    # storage_p1 = Vector{T}(undef, p)
    # storage_p2 = Vector{T}(undef, p)
    # storage_np = Matrix{T}(undef, n, p)
    # storage_pp = Matrix{T}(undef, p, p)
    # added_term_numerator = Matrix{T}(undef, n, p)
    # added_term2 = Matrix{T}(undef, p, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    wt = Vector{T}(undef, n)
    fill!(wt, one(T))
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    MixedCopulaVCObs{T}(y, X, V, n, p, m, 
        ∇β, ∇resβ, ∇ϕ, ∇θ, Hβ, Hθ, Hϕ, res, t, q, xtx, η, μ, varμ, dμ, wt, w1, w2)
end

struct MixedCopulaVCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{MixedCopulaVCObs{T}}
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    vecdist::Vector{UnivariateDistribution} # vector of marginal distributions for each data point
    veclink::Vector{Link} # vector of link functions for each marginal distribution
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    ϕ::Vector{T}    # dispersion parameters for each marginal; for poissona/bernoulli this should be NaN
    θ::Vector{T}    # variance components
    # working arrays
    ∇β::Vector{T}   # gradient terms from all observations
    ∇ϕ::Vector{T}
    ∇θ::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
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
    vecdist::Vector{UnivariateDistribution}, # vector of marginal distributions for each data point
    veclink::Vector{Link}; # vector of link functions for each marginal distribution
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
    XtX     = zeros(T, p, p) # sum_i xi'xi
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
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] .= gcs[i].t
    end
    QF = Matrix{T}(undef, n, m)
    # storage_n = Vector{T}(undef, n)
    # storage_m = Vector{T}(undef, m)
    # storage_θ = Vector{T}(undef, m)
    MixedCopulaVCModel{T}(gcs, ntotal, p, m, vecdist, veclink, 
        β, ϕ, θ, ∇β, ∇ϕ, ∇θ, XtX, Hβ, Hθ, Hϕ, TR, QF, penalized)
end

"""
    fit_quasi!(gcm::MixedCopulaVCModel, solver=Ipopt.IpoptSolver)

Fit an `MixedCopulaVCModel` object by MLE using a nonlinear programming solver. Start point
should be provided in `gcm.β`, `gcm.θ`, `gcm.ϕ` this is for Normal base.

# Arguments
- `gcm`: A `MixedCopulaVCModel` model object.
- `solver`: Specified solver to use. By default we use IPOPT with 100 quas-newton iterations with convergence tolerance 10^-6.
    (default `solver = Ipopt.IpoptSolver(print_level=3, max_iter = 100, tol = 10^-6, limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")`)
"""
function fit!(
    gcm::MixedCopulaVCModel,
    solver=Ipopt.IpoptSolver(print_level = 3, tol = 10^-6, max_iter = 100,
    limited_memory_max_history = 20, warm_start_init_point="yes", hessian_approximation = "limited-memory")
    )
    initialize_model!(gcm)
    npar = gcm.p + gcm.m + 1
    optm = MathProgBase.NonlinearModel(solver)
    # set lower bounds and upper bounds of parameters
    lb   = fill(-Inf, npar)
    ub   = fill( Inf, npar)
    offset = gcm.p + 1
    for k in 1:gcm.m + 1
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
    # gcm
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
    # L
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        par[offset] = gcm.θ[k]
        offset += 1
    end
    par[offset] = gcm.ϕ[1]
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
    offset = gcm.p + 1
    @inbounds for k in 1:gcm.m
        gcm.θ[k] = par[offset]
        offset   += 1
    end
    gcm.ϕ[1] = par[offset]
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
    grad[offset] = gcm.∇ϕ[1]
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
    H[idx] = gcm.Hϕ[1, 1]
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
    vecdist::Vector{UnivariateDistribution},
    veclink::Vector{Link},
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where T <: BlasReal
    # n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        fill!(gc.∇ϕ, 0)
        fill!(gc.∇θ, 0)
    end
    needhess && fill!(gc.Hβ, 0)
    # update residuals and its gradient
    update_res!(gc, β, vecdist, veclink)
    standardize_res!(gc, ϕ)
    std_res_differential!(gc, vecdist) # compute ∇resβ

    println("Reached here")
    fdsa

    rss  = abs2(norm(gc.res)) # RSS of standardized residual
    tsum = dot(θ, gc.t)
    logl = - log(1 + tsum) - (gc.n * log(2π) -  gc.n * log(abs(ϕ)) + rss) / 2
    @inbounds for k in 1:gc.m
        mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
        if needgrad # ∇β stores X'*Γ*res (standardized residual)
            BLAS.gemv!('T', θ[k], gc.X, gc.storage_n, one(T), gc.∇β)
        end
        gc.q[k] = dot(gc.res, gc.storage_n) / 2
    end
    qsum  = dot(θ, gc.q)
    logl += log(1 + qsum)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * dot(θ, θ)
    end
    # gradient
    if needgrad
        inv1pq = inv(1 + qsum)
        if needhess
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, one(T), gc.Hβ) # only lower triangular
            gc.Hϕ[1, 1] = - abs2(qsum * inv1pq / ϕ)
            # # hessian of vc vector use with fit_newton_normal.jl
            inv1pt = inv(1 + tsum)
            gc.m1 .= gc.q
            gc.m1 .*= inv1pq
            gc.m2 .= gc.t
            gc.m2 .*= inv1pt
            # hessian for vc
            fill!(gc.Hθ, 0.0)
            BLAS.syr!('U', one(T), gc.m2, gc.Hθ)
            BLAS.syr!('U', -one(T), gc.m1, gc.Hθ)
            copytri!(gc.Hθ, 'U')
        end
        BLAS.gemv!('T', one(T), gc.X, gc.res, -inv1pq, gc.∇β)
        gc.∇β .*= sqrtϕ
        gc.∇ϕ  .= (gc.n - rss + 2qsum * inv1pq) / 2ϕ
        gc.∇θ  .= inv1pq .* gc.q .- inv(1 + tsum) .* gc.t
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
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
        fill!(gcm.∇ϕ, 0)
        fill!(gcm.∇θ, 0)
    end
    if needhess
        # todo
        # gcm.Hβ .= - gcm.XtX
        # gcm.Hϕ .= - gcm.ntotal / 2abs2(gcm.ϕ[1])
    end
    logl = zeros(Threads.nthreads())
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
    needhess && error("todo")
    return sum(logl)
end

function update_res!(
    gc::MixedCopulaVCObs,
    β::Vector,
    vecdist::Vector{UnivariateDistribution},
    veclink::Vector{Link}
    )
    mul!(gc.η, gc.X, β)
    @turbo for i in 1:gc.n
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
    gc::MixedCopulaVCObs{T},
    vecdist::Vector{UnivariateDistribution},
    ) where T
    fill!(gc.∇resβ, 0.0)
    for j in 1:gc.n # loop over each marginal distributions
        d = typeof(vecdist[j])
        if d <: Normal
            gc.∇resβ[j, :] .= @view(gc.X[j, :])
            gc.∇resβ[j, :] .*= -one(T)
        elseif d <: Bernoulli
            @inbounds for i in 1:gc.p
                gc.∇resβ[j, i] = -sqrt(gc.varμ[j]) * gc.X[j, i] - 
                    (0.5 * gc.res[j] * (1 - (2 * gc.μ[j])) * gc.X[j, i])
            end
        elseif d <: Poisson
            @inbounds for i in 1:gc.p
                gc.∇resβ[j, i] = gc.X[j, i]
                gc.∇resβ[j, i] *= -(inv(sqrt(gc.varμ[j])) + 
                    (0.5 * inv(gc.varμ[j])) * gc.res[j]) * gc.dμ[j]
            end
        elseif d <: NegativeBinomal
            @inbounds for i in 1:gc.p
                gc.∇resβ[j, i] = -inv(sqrt(gc.varμ[j])) * gc.dμ[j] * 
                    gc.X[j, i] - (0.5 * inv(gc.varμ[j])) * gc.res[j] * 
                    (gc.μ[j] * inv(gc.d.r) + (1 + inv(gc.d.r) * gc.μ[j])) *
                    gc.dμ[j] * gc.X[j, i]
            end
        else
            error("Marginal distribution $d not supported! Sorry!")
        end
    end
    return nothing
end

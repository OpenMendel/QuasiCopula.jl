mutable struct Mixed_VCObs{T <: BlasReal}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}} # vector of (known) covariances
    n::Int
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇θ::Vector{T}   # gradient wrt θ2
    Hβ::Matrix{T}   # Hessian wrt β
    Hθ::Matrix{T}   # Hessian wrt variance components θ
    Hτ::Matrix{T}   # Hessian wrt τ
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

function Mixed_VCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}}
    ) where T <: BlasReal
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    Hθ  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
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
    Mixed_VCObs{T}(y, X, V, n, p, m, 
        ∇β, ∇resβ, ∇τ, ∇θ, Hβ, Hθ, Hτ, res, t, q, xtx, η, μ, varμ, dμ, wt, w1, w2)
end

"""
    Mixed_VCModel
    Mixed_VCModel(gcs)

Bivariate Mixed Poisson, Bernoulli variance component model, which contains a vector of
`Mixed_VCObs` as data, model parameters, and working arrays.
"""
struct Mixed_VCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{Mixed_VCObs{T}}
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
    ∇τ::Vector{T}
    ∇θ::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}   # Hessian terms from all observations
    Hθ::Matrix{T}
    Hτ::Matrix{T}
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

function Mixed_VCModel(
    gcs::Vector{Mixed_VCObs{T}},
    vecdist::Vector{UnivariateDistribution}, # vector of marginal distributions for each data point
    veclink::Vector{Link}; # vector of link functions for each marginal distribution
    penalized::Bool = false
    ) where T <: BlasReal
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β       = Vector{T}(undef, p)
    τ       = [1.0]
    θ       = Vector{T}(undef, m)
    ∇β      = Vector{T}(undef, p)
    ∇τ      = Vector{T}(undef, 1)
    ∇θ      = Vector{T}(undef, m)
    XtX     = zeros(T, p, p) # sum_i xi'xi
    Hβ      = Matrix{T}(undef, p, p)
    Hθ      = Matrix{T}(undef, m, m)
    Hτ      = Matrix{T}(undef, 1, 1)
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
    Mixed_VCModel{T}(gcs, ntotal, p, m, vecdist, veclink, 
        β, τ, θ, ∇β, ∇τ, ∇θ, XtX, Hβ, Hθ, Hτ, TR, QF, penalized)
end

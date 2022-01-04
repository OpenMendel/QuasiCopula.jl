@reexport using Distributions
mutable struct Poisson_Bernoulli_VCObs{T <: BlasReal, VD, VL}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇μβ::Matrix{T}
    ∇σ2β::Matrix{T}
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇τ::Vector{T}   # gradient wrt τ
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β
    HΣ::Matrix{T}   # Hessian wrt variance components Σ
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    m1::Vector{T}
    m2::Vector{T}
    storage_p1::Vector{T}
    storage_p2::Vector{T}
    storage_np::Matrix{T}
    storage_pp::Matrix{T}
    added_term_numerator::Matrix{T}
    added_term2::Matrix{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    # d::D            # distribution()
    vecd::VD
    veclink::VL      # link function ()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function Poisson_Bernoulli_VCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    vecd::VD,
    veclink::VL) where {T <: BlasReal, VD, VL}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇μβ = Matrix{T}(undef, n, p)
    ∇σ2β = Matrix{T}(undef, n, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    HΣ  = Matrix{T}(undef, m, m)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    m1        = Vector{T}(undef, m)
    m2        = Vector{T}(undef, m)
    storage_p1 = Vector{T}(undef, p)
    storage_p2 = Vector{T}(undef, p)
    storage_np = Matrix{T}(undef, n, p)
    storage_pp = Matrix{T}(undef, p, p)
    added_term_numerator = Matrix{T}(undef, n, p)
    added_term2 = Matrix{T}(undef, p, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    wt = Vector{T}(undef, n)
    fill!(wt, one(T))
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    Poisson_Bernoulli_VCObs{T, VD, VL}(y, X, V, ∇β, ∇μβ, ∇σ2β, ∇resβ, ∇τ, ∇Σ, Hβ, HΣ,
        Hτ, res, t, q, xtx, storage_n, m1, m2, storage_p1, storage_p2, storage_np, storage_pp, added_term_numerator, added_term2, η, μ, varμ, dμ, vecd, veclink, wt, w1, w2)
end

"""
Poisson_Bernoulli_VCModel
Poisson_Bernoulli_VCModel(gcs)
Bivariate Mixed Poisson, Bernoulli variance component model, which contains a vector of
`Poisson_Bernoulli_VCObs` as data, model parameters, and working arrays.
"""
struct Poisson_Bernoulli_VCModel{T <: BlasReal, VD, VL} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}
    Y1total::T
    Y2total::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    θ::Vector{T}
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    ∇θ::Vector{T}   # overall gradient for beta and variance components vector Σ
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    HΣ::Matrix{T}
    Hτ::Matrix{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    TR::Matrix{T}         # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}         # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    vecd::Vector{VD}
    veclink::Vector{VL}
end


function Poisson_Bernoulli_VCModel(gcs::Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}) where {T <: BlasReal, VD, VL}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β       = Vector{T}(undef, p)
    τ       = [1.0]
    Σ       = Vector{T}(undef, m)
    θ       = Vector{T}(undef, m + p)
    ∇β      = Vector{T}(undef, p)
    ∇τ      = Vector{T}(undef, 1)
    ∇Σ      = Vector{T}(undef, m)
    ∇θ      = Vector{T}(undef, m + p)
    XtX     = zeros(T, p, p) # sum_i xi'xi
    Hβ      = Matrix{T}(undef, p, p)
    HΣ      = Matrix{T}(undef, m, m)
    Hτ      = Matrix{T}(undef, 1, 1)
    Ainv    = zeros(T, p + m, p + m)
    Aevec   = zeros(T, p + m, p + m)
    M       = zeros(T, p + m, p + m)
    vcov    = zeros(T, p + m, p + m)
    ψ       = Vector{T}(undef, p + m)
    TR      = Matrix{T}(undef, n, m) # collect trace terms
    Y1total  = 0.0
    Y2total  = 0.0
    ntotal  = 0.0
    vecd       = Vector{VD}(undef, n)
    veclink    = Vector{VL}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Y1total  += sum(gcs[i].y[1])
        Y2total  += sum(gcs[i].y[2])
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
        vecd[i] = gcs[i].vecd
        veclink[i] = gcs[i].veclink
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    Poisson_Bernoulli_VCModel{T, VD, VL}(gcs, Y1total, Y2total, ntotal, p, m, β, τ, Σ, θ,
        ∇β, ∇τ, ∇Σ, ∇θ, XtX, Hβ, HΣ, Hτ, Ainv, Aevec, M, vcov, ψ, TR, QF,
        storage_n, storage_m, storage_Σ, vecd, veclink)
end
#
# """
#     update_res!(gc, β)
# Update the residual vector according to `β` given link functions and distributions.
# """
# function update_res!(
#    gc::Poisson_Bernoulli_VCObs{T, VD, VL},
#    β::Vector{T}) where {T <: BlasReal, VD, VL}
#    mul!(gc.η, gc.X, β)
#    @inbounds for i in 1:length(gc.y)
#        gc.μ[i] = GLM.linkinv(gc.veclink[i], gc.η[i])
#        gc.varμ[i] = GLM.glmvar(gc.vecd[i], gc.μ[i]) # Note: for negative binomial, d.r is used
#        gc.dμ[i] = GLM.mueta(gc.veclink[i], gc.η[i])
#        gc.w1[i] = gc.dμ[i] / gc.varμ[i]
#        gc.w2[i] = gc.w1[i] * gc.dμ[i]
#        gc.res[i] = gc.y[i] - gc.μ[i]
#    end
#    return gc.res
# end
#
# function standardize_res!(
#     gc::Poisson_Bernoulli_VCObs{T, VD, VL},
#     ) where {T <: BlasReal, VD, VL}
#     @inbounds for j in eachindex(gc.y)
#         σinv = inv(sqrt(gc.varμ[j]))
#         gc.res[j] *= σinv
#     end
# end
#
#
# #
# """
# glm_score_statistic(gc, β, τ)
#
# Get gradient and hessian of beta to for a single independent vector of observations.
# """
# function glm_score_statistic(gc::Poisson_Bernoulli_VCObs{T, VD, VL},
#   β::Vector{T}, τ::T) where {T<: BlasReal, VD, VL}
#    fill!(gc.∇β, 0.0)
#    fill!(gc.Hβ, 0.0)
#    update_res!(gc, β)
#    gc.∇β .= glm_gradient(gc)
#    gc.Hβ .= GLMCopula.glm_hessian(gc)
#    gc
# end

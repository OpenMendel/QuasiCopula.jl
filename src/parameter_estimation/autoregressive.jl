export GLMCopulaARObs, GLMCopulaARModel, get_AR_cov, get_∇ARV, get_∇2ARV
struct GLMCopulaARObs{T <: BlasReal, D, Link}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Matrix{T}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇μβ::Matrix{T}
    ∇σ2β::Matrix{T}
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇ρ::Vector{T}
    ∇σ2::Vector{T}
    Hβ::Matrix{T}   # Hessian wrt β
    Hρ::Matrix{T}   # Hessian wrt ρ
    Hσ2::Matrix{T}   # Hessian wrt ρ
    res::Vector{T}  # residual vector res_i
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    storage_p1::Vector{T}
    storage_np::Matrix{T}
    storage_pp::Matrix{T}
    added_term_numerator::Matrix{T}
    added_term2::Matrix{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    link::Link      # link function ()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function GLMCopulaARObs(
    y::Vector{T},
    X::Matrix{T},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p = size(X, 1), size(X, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    V = Matrix{T}(undef, n, n)
    ∇β  = Vector{T}(undef, p)
    ∇μβ = Matrix{T}(undef, n, p)
    ∇σ2β = Matrix{T}(undef, n, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    storage_p1 = Vector{T}(undef, p)
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
    GLMCopulaARObs{T, D, Link}(y, X, V, ∇β, ∇μβ, ∇σ2β, ∇resβ, ∇ρ, ∇σ2, Hβ, Hρ, Hσ2,
       res, xtx, storage_n, storage_p1, storage_np, storage_pp, added_term_numerator, added_term2,
        η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
GLMCopulaARModel
GLMCopulaARModel(gcs)
Gaussian copula autoregressive model, which contains a vector of
`GLMCopulaARObs` as data, model parameters, and working arrays.
"""
struct GLMCopulaARModel{T <: BlasReal, D, Link} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{GLMCopulaARObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    ρ::Vector{T}            # autocorrelation parameter
    σ2::Vector{T}           # autoregressive noise parameter
    # working arrays
    ∇β::Vector{T}   # gradient of beta from all observations
    ∇ρ::Vector{T}           # gradient of rho from all observations
    ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hρ::Matrix{T}    # Hessian from all observations
    Hσ2::Matrix{T}    # Hessian from all observations
    storage_n::Vector{T}
    d::Vector{D}
    link::Vector{Link}
end

function GLMCopulaARModel(gcs::Vector{GLMCopulaARObs{T, D, Link}}) where {T <: BlasReal, D, Link}
    n, p = length(gcs), size(gcs[1].X, 2)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    ρ = [1.0]
    σ2 = [1.0]
    ∇β  = Vector{T}(undef, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Ytotal = 0.0
    ntotal = 0.0
    d = Vector{D}(undef, n)
    link = Vector{Link}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        d[i] = gcs[i].d
        link[i] = gcs[i].link
    end
    storage_n = Vector{T}(undef, n)
    GLMCopulaARModel{T, D, Link}(gcs, Ytotal, ntotal, p, β, τ, ρ, σ2,
        ∇β, ∇ρ, ∇σ2, XtX, Hβ, Hρ, Hσ2,
        storage_n, d, link)
end

"""
    get_AR_cov(n, ρ, σ2, V)
Forms the AR(1) covariance structure given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
"""
function get_AR_cov(n, ρ, σ2, V)
    @inbounds for i in 1:n
        V[i, i] = 1.0
        @inbounds for j in i+1:n
            V[i, j] = ρ^(j-i)
            V[j, i] = V[i, j]
        end
    end
    V
end

"""
    get_∇ARV(n, ρ, σ2, V)
Forms the first derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
"""
function get_∇ARV(n, ρ, σ2, ∇ARV)
    @inbounds for i in 1:n
        ∇ARV[i, i] = 0.0
        @inbounds for j in i+1:n
            ∇ARV[i, j] = (j-i)* ρ^(j-i-1)
            ∇ARV[j, i] = ∇ARV[i, j]
        end
    end
    ∇ARV
end

"""
    get_∇A2RV(n, ρ, σ2, V)
Forms the second derivative of AR(1) covariance structure wrt to ρ, given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
"""
function get_∇2ARV(n, ρ, σ2, ∇2ARV)
    @inbounds for i in 1:n
        ∇2ARV[i, i] = 0.0
        @inbounds for j in i+1:n
            ∇2ARV[i, j] = (j-i)*(j-i-1)* ρ^(j-i-2)
            ∇2ARV[j, i] = ∇2ARV[i, j]
        end
    end
    ∇2ARV
end

function loglikelihood!(
    gc::GLMCopulaARObs{T, D, Link},
    β::Vector{T},
    ρ::T,
    σ2::T,
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D, Link}
    n, p = size(gc.X, 1), size(gc.X, 2)
    needgrad = needgrad || needhess
    if needgrad
      fill!(gc.∇β, 0)
    end
    needhess && fill!(gc.Hβ, 0)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    standardize_res!(gc)
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # form V
    V = zeros(n, n)
    V .= get_AR_cov(n, ρ, σ2, V)

    #evaluate copula loglikelihood
    mul!(gc.storage_n, V, gc.res) # storage_n = V[k] * res
    if needgrad
        BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    end
    q = dot(gc.res, gc.storage_n) * 0.5 * σ2

    # loglikelihood
    logl = GLMCopula.component_loglikelihood(gc)
    logl -= log(1 + 0.5 * n * σ2)
    logl += log(1 + q)
    if needgrad
      inv1pq = inv(1 + q)
      if needhess
          BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 0.0, gc.Hβ) # only lower triangular
          fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
          fill!(gc.added_term2, 0.0) # fill hessian with 0
          mul!(gc.added_term_numerator, V, gc.∇resβ) # storage_n = V[k] * res
          BLAS.gemm!('T', 'N', σ2, gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
          gc.added_term2 .*= inv1pq
          gc.Hβ .+= gc.added_term2
          gc.Hβ .+= GLMCopula.glm_hessian(gc, β)
      end
      gc.∇β .= gc.∇β .* inv1pq
      gc.∇β .+= GLMCopula.glm_gradient(gc, β, 1.0)
    end
    logl
end

function loglikelihood!(
    gcm::GLMCopulaARModel{T, D, Link},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D, Link}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0)
    end
    if needhess
        fill!(gcm.Hβ, 0)
    end
    @inbounds for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
        end
    end
    logl
  end
  
  
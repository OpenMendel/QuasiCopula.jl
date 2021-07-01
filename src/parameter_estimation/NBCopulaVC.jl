export NBCopulaVCObs, NBCopulaVCModel
### first structures
struct NBCopulaVCObs{T <: BlasReal, D, Link}
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
    ∇r::Vector{T}   # gradient wrt r (NB)
    Hβ::Matrix{T}   # Hessian wrt β
    HΣ::Matrix{T}   # Hessian wrt variance components Σ
    Hr::Matrix{T}   # Hessian wrt r (NB)
    Hτ::Matrix{T}   # Hessian wrt τ
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
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
    d::D            # distribution()
    link::Link      # link function ()
    wt::Vector{T}   # weights wt for GLM.jl
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function NBCopulaVCObs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇μβ = Matrix{T}(undef, n, p)
    ∇σ2β = Matrix{T}(undef, n, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    ∇r  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    HΣ  = Matrix{T}(undef, m, m)
    Hr  = Matrix{T}(undef, 1, 1)
    Hτ  = Matrix{T}(undef, 1, 1)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
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
    NBCopulaVCObs{T, D, Link}(y, X, V, ∇β, ∇μβ, ∇σ2β, ∇resβ, ∇τ, ∇Σ, ∇r, Hβ, HΣ, Hr,
        Hτ, res, t, q, xtx, storage_n, storage_p1, storage_p2, storage_np, storage_pp, added_term_numerator, added_term2, η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
NBCopulaVCModel
NBCopulaVCModel(gcs)
Negative Binomial copula variance component model, which contains a vector of
`NBCopulaVCObs` as data, model parameters, and working arrays.
"""
struct NBCopulaVCModel{T <: BlasReal, D, Link} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{NBCopulaVCObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    r::Vector{T}    # r parameter for negative binomial
    θ::Vector{T}    # all parameters, beta vector, variance components vector and r
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    ∇τ::Vector{T}
    ∇Σ::Vector{T}
    ∇r::Vector{T}
    ∇θ::Vector{T}   # overall gradient for beta and variance components vector Σ
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    HΣ::Matrix{T}
    Hr::Matrix{T}
    Hτ::Matrix{T}
    TR::Matrix{T}         # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}         # n-by-m matrix with qik = res_i' Vi[k] res_i
    hess1::Matrix{T}      # holds transpose(gcm.QF) * Diagonal(gcm.storage_n) required for outer product in hessian term 1 
    hess2::Matrix{T}      # holds transpose(gcm.TR) * Diagonal(gcm.storage_n2) required for outer product in hessian term 2 
    storage_n::Vector{T}
    storage_n2::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
end

function NBCopulaVCModel(gcs::Vector{NBCopulaVCObs{T, D, Link}}) where {T <: BlasReal, D, Link}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    τ   = [1.0]
    Σ   = Vector{T}(undef, m)
    r   = [1.0]
    θ   = Vector{T}(undef, m + p)
    ∇β  = Vector{T}(undef, p)
    ∇τ  = Vector{T}(undef, 1)
    ∇Σ  = Vector{T}(undef, m)
    ∇r  = Vector{T}(undef, 1)
    ∇θ  = Vector{T}(undef, m + p)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    HΣ  = Matrix{T}(undef, m, m)
    Hr  = Matrix{T}(undef, 1, 1)
    Hτ  = Matrix{T}(undef, 1, 1)
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0.0
    ntotal = 0.0
    d = Vector{D}(undef, n)
    link = Vector{Link}(undef, n)
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
        d[i] = gcs[i].d
        link[i] = gcs[i].link
    end
    QF        = Matrix{T}(undef, n, m)
    hess1     = Matrix{T}(undef, m, n)
    hess2     = Matrix{T}(undef, m, n)
    storage_n = Vector{T}(undef, n)
    storage_n2 = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    NBCopulaVCModel{T, D, Link}(gcs, Ytotal, ntotal, p, m, β, τ, Σ, r, θ,
        ∇β, ∇τ, ∇Σ, ∇r, ∇θ, XtX, Hβ, HΣ, Hr, Hτ, TR, QF, hess1, hess2,
        storage_n, storage_n2, storage_m, storage_Σ, d, link)
end

### loglikelihood functions
"""
    component_loglikelihood!(gc::NBCopulaVCObs{T, D, Link}, r)
Calculates the loglikelihood of observing `y` given parameters for `μ` and `r` for Negative Binomial distribution using the GLM.jl package.
"""
function component_loglikelihood(gc::NBCopulaVCObs{T, D, Link}, r::T) where {T <: BlasReal, D<:NegativeBinomial{T}, Link}
    logl = zero(T)
    @inbounds for j in 1:length(gc.y)
        logl += logpdf(D(r, r/(gc.μ[j] + r)), gc.y[j])
    end
    logl
end


"""
    loglikelihood!(gc::NBCopulaVCObs{T, D, Link}, β, τ, Σ, r)
Calculates the loglikelihood of observing `y` given parameters for `β`, `τ = 1`, `Σ`, and `r`for Negative Binomial distribution.
"""
function loglikelihood!(
  gc::NBCopulaVCObs{T, D, Link},
  β::Vector{T},
  τ::T, # inverse of linear regression variance
  Σ::Vector{T},
  r::T,
  needgrad::Bool = false,
  needhess::Bool = false
  ) where {T <: BlasReal, D, Link}
  n, p, m = size(gc.X, 1), size(gc.X, 2), length(gc.V)
  needgrad = needgrad || needhess
  if needgrad
      fill!(gc.∇β, 0)
      fill!(gc.∇τ, 0)
      fill!(gc.∇Σ, 0) 
  end
  needhess && fill!(gc.Hβ, 0)
  fill!(gc.∇β, 0.0)
  update_res!(gc, β)
  standardize_res!(gc)
  fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
  std_res_differential!(gc) # this will compute ∇resβ

  # evaluate copula loglikelihood
  @inbounds for k in 1:m
      mul!(gc.storage_n, gc.V[k], gc.res) # storage_n = V[k] * res
      if needgrad
          BLAS.gemv!('T', Σ[k], gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
      end
      gc.q[k] = dot(gc.res, gc.storage_n) / 2
  end
  # loglikelihood
  logl = GLMCopula.component_loglikelihood(gc, r)
  tsum = dot(Σ, gc.t)
  logl += -log(1 + tsum)
  qsum  = dot(Σ, gc.q)
  logl += log(1 + qsum)
  
  if needgrad
      inv1pq = inv(1 + qsum)

       # add gradient with respect to r
       # gc.∇r .= 

      if needhess
          BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 1.0, gc.Hβ) # only lower triangular
          # does adding this term to the approximation of the hessian violate negative semidefinite properties?
          fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
          fill!(gc.added_term2, 0.0) # fill hessian with 0
          @inbounds for k in 1:m
              mul!(gc.added_term_numerator, gc.V[k], gc.∇resβ) # storage_n = V[k] * res
              BLAS.gemm!('T', 'N', Σ[k], gc.∇resβ, gc.added_term_numerator, one(T), gc.added_term2)
          end
          gc.added_term2 .*= inv1pq
          gc.Hβ .+= gc.added_term2
          gc.Hβ .+= GLMCopula.glm_hessian(gc, β)

          # add hessian for r
          # gc.Hr .= 
      end
      gc.storage_p2 .= gc.∇β .* inv1pq
      gc.∇β .= GLMCopula.glm_gradient(gc, β, τ)
      gc.∇β .+= gc.storage_p2
  end
  logl
end

function loglikelihood!(
  gcm::NBCopulaVCModel{T, D, Link},
  needgrad::Bool = false,
  needhess::Bool = false
  ) where {T <: BlasReal, D, Link}
  logl = zero(T)
  if needgrad
      fill!(gcm.∇β, 0.0)
      fill!(gcm.∇Σ, 0.0)
      fill!(gcm.∇r, 0.0)
  end
  if needhess
      fill!(gcm.Hβ, 0.0)
      fill!(gcm.HΣ, 0.0)
      fill!(gcm.Hr, 0.0)
  end
  @inbounds for i in eachindex(gcm.data)
      logl += loglikelihood!(gcm.data[i], gcm.β, gcm.τ[1], gcm.Σ, gcm.r[1], needgrad, needhess)
      if needgrad
          gcm.∇β .+= gcm.data[i].∇β
          # uncomment this
          # gcm.∇r .+= gcm.data[i].∇r
      end
      if needhess
          gcm.Hβ .+= gcm.data[i].Hβ
          # uncomment this
          # gcm.Hr .+= gcm.data[i].Hr
      end
  end
    if needgrad
        gcm.∇Σ .= update_∇Σ!(gcm)
    end
    if needhess
        gcm.HΣ .= update_HΣ!(gcm)
    end
  logl
end

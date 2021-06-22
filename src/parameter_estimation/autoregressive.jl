export GLMCopulaARObs, GLMCopulaARModel, get_AR_cov, get_∇ARV, get_∇2ARV
struct GLMCopulaARObs{T <: BlasReal, D, Link}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Matrix{T}
    # working arrays
    ∇ARV::Matrix{T}
    ∇2ARV::Matrix{T}
    ∇β::Vector{T}   # gradient wrt β
    ∇μβ::Matrix{T}
    ∇σ2β::Matrix{T}
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇ρ::Vector{T}
    ∇σ2::Vector{T}
    Hβ::Matrix{T}   # Hessian wrt β
    Hρ::Matrix{T}   # Hessian wrt ρ
    Hσ2::Matrix{T}   # Hessian wrt ρ
    Hρσ2::Matrix{T}  # Hessian wrt ρ, σ2
    Hβσ2::Vector{T}  # Hessian wrt β and σ2
    # Hβρ::Vector{T}
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
    ∇ARV = Matrix{T}(undef, n, n)
    ∇2ARV = Matrix{T}(undef, n, n)
    ∇β  = Vector{T}(undef, p)
    ∇μβ = Matrix{T}(undef, n, p)
    ∇σ2β = Matrix{T}(undef, n, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    # Hβρ = Vector{T}(undef, p)
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
    GLMCopulaARObs{T, D, Link}(y, X, V, ∇ARV, ∇2ARV, ∇β, ∇μβ, ∇σ2β, ∇resβ, ∇ρ, ∇σ2, Hβ, Hρ, Hσ2, Hρσ2, Hβσ2,# Hβρ,
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
    θ::Vector{T}   # all parameters
    # working arrays
    ∇β::Vector{T}   # gradient of beta from all observations
    ∇ρ::Vector{T}           # gradient of rho from all observations
    ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
    ∇θ::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hρ::Matrix{T}    # Hessian from all observations
    Hσ2::Matrix{T}    # Hessian from all observations
    Hρσ2::Matrix{T}
    Hβσ2::Vector{T}
    # Hβρ::Vector{T}
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
    θ = Vector{T}(undef, p + 2)
    ∇β  = Vector{T}(undef, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    ∇θ = Vector{T}(undef, p + 2)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    # Hβρ = Vector{T}(undef, p)
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
    GLMCopulaARModel{T, D, Link}(gcs, Ytotal, ntotal, p, β, τ, ρ, σ2, θ,
        ∇β, ∇ρ, ∇σ2, ∇θ, XtX, Hβ, Hρ, Hσ2, Hρσ2, Hβσ2, # Hβρ,
        storage_n, d, link)
end

"""
    get_AR_cov(n, ρ, σ2, V)
Forms the AR(1) covariance structure given n (size of cluster), ρ (correlation parameter), σ2 (noise parameter)
"""
function get_AR_cov(n, ρ, σ2, V)
    fill!(V, 1.0)
    @inbounds for i in 1:n
        @inbounds for j in i+1:n
            power = j - i
            @inbounds for k in 1:power
                V[i, j] *= ρ
            end
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
            power = j - i - 1
            ∇ARV[i, j] = (j-i)
            @inbounds for k in 1:power
                ∇ARV[i, j] *= ρ
            end
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
            pw = (j-i-1)
            if pw == 0
                ∇2ARV[i, j] = 0.0
            else
                ∇2ARV[i, j] = (pw + 1)*(pw)
                @inbounds for k in 1:pw - 1
                    ∇2ARV[i, j] *= ρ
                end
            end
            ∇2ARV[j, i] = ∇2ARV[i, j]
        end
    end
    ∇2ARV
end

# one that works but is slow
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
        gc.∇ARV .= get_∇ARV(n, ρ, σ2, gc.∇ARV)
    end
    needhess && fill!(gc.Hβ, 0)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    standardize_res!(gc)
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # form V
    gc.V .= get_AR_cov(n, ρ, σ2, gc.V)

    #evaluate copula loglikelihood
    mul!(gc.storage_n, gc.V, gc.res) # storage_n = V[k] * res

    if needgrad
        BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    end
    q = dot(gc.res, gc.storage_n)

    # GLMCopula.storage_n!(gc, ρ, n)
    # if needgrad
    #     BLAS.gemv!('T', σ2, gc.∇resβ, gc.storage_n, 1.0, gc.∇β) # stores ∇resβ*Γ*res (standardized residual)
    # end
    # q = dot(gc.res, gc.storage_n)
    
    # @show q
    c1 = 1 + 0.5 * n * σ2
    c2 = 1 + 0.5 * σ2 * q
    # loglikelihood
    logl = GLMCopula.component_loglikelihood(gc)
    logl += -log(c1)
    # @show logl
    logl += log(c2)
    # @show logl
    if needgrad
        inv1pq = inv(c2)
        # gradient with respect to rho
        mul!(gc.storage_n, gc.∇ARV, gc.res) # storage_n = ∇ARV * res
        q2 = dot(gc.res, gc.storage_n) # 
        # gc.∇ρ .= inv(c2) * 0.5 * σ2 * transpose(gc.res) * gc.∇ARV * gc.res
        gc.∇ρ .= inv(c2) * 0.5 * σ2 * q2

        # gradient with respect to sigma2
        gc.∇σ2 .= -0.5 * n * inv(c1) .+ inv(c2) * 0.5 * q
      if needhess
            gc.∇2ARV .= get_∇2ARV(n, ρ, σ2, gc.∇2ARV)
            mul!(gc.storage_n, gc.∇2ARV, gc.res) # storage_n = ∇ARV * res
            q3 = dot(gc.res, gc.storage_n) # 
            # hessian for rho
            gc.Hρ .= 0.5 * σ2 * (inv(c2) * q3 - inv(c2)^2 * 0.5 * σ2 * q2^2)
            
            # hessian for sigma2
            gc.Hσ2 .= 0.25 * n^2 * inv(c1)^2 - inv(c2)^2 * (0.25 * q^2)

            # hessian cross term for rho and sigma2
            gc.Hρσ2 .= 0.5 * q2 * inv1pq - 0.25 * σ2 * q * q2 * inv1pq^2

            # hessian cross term for beta and sigma2
            gc.Hβσ2 .= inv1pq * gc.∇β - 0.5 * q * inv1pq^2 * σ2 * gc.∇β

            #  hessian cross term for beta and rho
            # gc.Hβρ .= inv1pq * σ2 * transpose(gc.∇resβ) * gc.∇ARV * gc.res - 0.5 * σ2^2 * inv1pq^2 * q2 * transpose(gc.∇resβ) * gc.V * gc.res
            
            BLAS.syrk!('L', 'N', -abs2(inv1pq), gc.∇β, 0.0, gc.Hβ) # only lower triangular
            fill!(gc.added_term_numerator, 0.0) # fill gradient with 0
            fill!(gc.added_term2, 0.0) # fill hessian with 0
            # gc.V .= get_AR_cov(n, ρ, σ2, gc.V)
            mul!(gc.added_term_numerator, gc.V, gc.∇resβ) # storage_n = V[k] * res
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
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇ρ, 0.0)
        fill!(gcm.∇σ2, 0.0)
    end
    if needhess
        fill!(gcm.Hβ, 0.0)
        fill!(gcm.Hρ, 0.0)
        fill!(gcm.Hσ2, 0.0)
        fill!(gcm.Hρσ2, 0.0)
        # fill!(gcm.Hβσ2, 0.0)
        # @show gcm.Hβσ2
        # fill!(gcm.Hβρ, 0)
    end
    @inbounds for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.ρ[1], gcm.σ2[1], needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇ρ .+= gcm.data[i].∇ρ
            gcm.∇σ2 .+= gcm.data[i].∇σ2
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hρ .+= gcm.data[i].Hρ
            gcm.Hσ2 .+= gcm.data[i].Hσ2
            gcm.Hρσ2 .+= gcm.data[i].Hρσ2
            gcm.Hβσ2 .+= gcm.data[i].Hβσ2
            # gcm.Hβρ .+= gcm.data[i].Hβρ
        end
    end
    logl
end
  
  
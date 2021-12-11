export NBCopulaARObs, NBCopulaARModel
### first structures
mutable struct NBCopulaARObs{T <: BlasReal, D, Link} # d changes, so must be mutable
    # data
    n::T
    p::T
    y::Vector{T}
    X::Matrix{T}
    V::Matrix{T}
    vec::Vector{T}
    # working arrays
    ∇ARV::Matrix{T}
    ∇2ARV::Matrix{T}
    ∇β::Vector{T}   # gradient wrt β
    ∇μβ::Matrix{T}
    ∇σ2β::Matrix{T}
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇ρ::Vector{T}
    ∇σ2::Vector{T}
    ∇r::Vector{T}   # gradient wrt r (NB)
    Hβ::Matrix{T}   # Hessian wrt β
    Hρ::Matrix{T}   # Hessian wrt ρ
    Hr::Matrix{T}   # Hessian wrt r (NB)
    Hσ2::Matrix{T}   # Hessian wrt ρ
    Hρσ2::Matrix{T}  # Hessian wrt ρ, σ2
    Hβσ2::Vector{T}  # Hessian wrt β and σ2
    # Hβρ::Vector{T}
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    xtx::Matrix{T}  # Xi'Xi
    storage_n::Vector{T}
    storage_n2::Vector{T}
    storage_n3::Vector{T}
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

function NBCopulaARObs(
    y::Vector{T},
    X::Matrix{T},
    d::D,
    link::Link) where {T <: BlasReal, D, Link}
    n, p = size(X, 1), size(X, 2)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    V = ones(T, n, n)
    vec = Vector{T}(undef, n)
    ∇ARV = Matrix{T}(undef, n, n)
    ∇2ARV = Matrix{T}(undef, n, n)
    ∇β  = Vector{T}(undef, p)
    ∇μβ = Matrix{T}(undef, n, p)
    ∇σ2β = Matrix{T}(undef, n, p)
    ∇resβ  = Matrix{T}(undef, n, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    ∇r  = Vector{T}(undef, 1)
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hr  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    # Hβρ = Vector{T}(undef, p)
    res = Vector{T}(undef, n)
    t   = [tr(V)/2]
    q   = Vector{T}(undef, 1)
    xtx = transpose(X) * X
    storage_n = Vector{T}(undef, n)
    storage_n2 = Vector{T}(undef, n)
    storage_n3 = Vector{T}(undef, n)
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
    NBCopulaARObs{T, D, Link}(n, p, y, X, V, vec, ∇ARV, ∇2ARV, ∇β, ∇μβ, ∇σ2β,
        ∇resβ, ∇ρ, ∇σ2, ∇r, Hβ, Hρ, Hr, Hσ2, Hρσ2, Hβσ2, # Hβρ,
        res, t, q, xtx, storage_n, storage_n2, storage_n3, storage_p1, storage_np,
        storage_pp, added_term_numerator, added_term2,
        η, μ, varμ, dμ, d, link, wt, w1, w2)
end

"""
    NBCopulaARModel(gcs)

Negative Binomial copula variance component model, which contains a vector of
`NBCopulaARObs` as data, model parameters, and working arrays.
"""
struct NBCopulaARModel{T <: BlasReal, D, Link} <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{NBCopulaARObs{T, D, Link}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    τ::Vector{T}    # inverse of linear regression variance parameter (IS THIS NEEDED??)
    ρ::Vector{T}            # autocorrelation parameter
    σ2::Vector{T}           # autoregressive noise parameter
    Σ::Vector{T}
    r::Vector{T}   # r parameter for negative binomial
    θ::Vector{T}   # all parameters
    # working arrays
    ∇β::Vector{T}   # gradient of beta from all observations
    ∇ρ::Vector{T}           # gradient of rho from all observations
    ∇σ2::Vector{T}          # gradient of sigmasquared from all observations
    ∇r::Vector{T}
    ∇θ::Vector{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    Hβ::Matrix{T}    # Hessian from all observations
    Hρ::Matrix{T}    # Hessian from all observations
    Hσ2::Matrix{T}    # Hessian from all observations
    Hr::Matrix{T}
    Hρσ2::Matrix{T}
    Hβσ2::Vector{T}
    Ainv::Matrix{T}
    Aevec::Matrix{T}
    M::Matrix{T}
    vcov::Matrix{T}
    ψ::Vector{T}
    # Hβρ::Vector{T}
    TR::Matrix{T}
    QF::Matrix{T}         # n-by-1 matrix with qik = res_i' Vi res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::Vector{D}
    link::Vector{Link}
end

function NBCopulaARModel(gcs::Vector{NBCopulaARObs{T, D, Link}}) where {T <: BlasReal, D, Link}
    n, p = length(gcs), size(gcs[1].X, 2)
    β   = Vector{T}(undef, p)
    τ   = [one(T)]
    ρ = [one(T)]
    σ2 = [one(T)]
    Σ = [one(T)]
    r = [one(T)]
    θ = Vector{T}(undef, p + 2)
    ∇β  = Vector{T}(undef, p)
    ∇ρ  = Vector{T}(undef, 1)
    ∇σ2  = Vector{T}(undef, 1)
    ∇r  = Vector{T}(undef, 1)
    ∇θ = Vector{T}(undef, p + 2)
    XtX = zeros(T, p, p) # sum_i xi'xi
    Hβ  = Matrix{T}(undef, p, p)
    Hρ  = Matrix{T}(undef, 1, 1)
    Hσ2  = Matrix{T}(undef, 1, 1)
    Hr  = Matrix{T}(undef, 1, 1)
    Hρσ2 = Matrix{T}(undef, 1, 1)
    Hβσ2 = zeros(T, p)
    Ainv    = zeros(T, p + 3, p + 3)
    Aevec   = zeros(T, p + 3, p + 3)
    M       = zeros(T, p + 3, p + 3)
    vcov    = zeros(T, p + 3, p + 3)
    ψ       = Vector{T}(undef, p + 3)
    # Hβρ = Vector{T}(undef, p)
    TR  = Matrix{T}(undef, n, 1) # collect trace terms
    QF  = Matrix{T}(undef, n, 1)
    Ytotal = zero(T)
    ntotal = 0
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
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, 1)
    storage_Σ = Vector{T}(undef, 1)
    NBCopulaARModel{T, D, Link}(gcs, Ytotal, ntotal, p, β, τ, ρ, σ2, Σ, r, θ,
        ∇β, ∇ρ, ∇σ2, ∇r, ∇θ, XtX, Hβ, Hρ, Hσ2, Hr, Hρσ2, Hβσ2, Ainv, Aevec,  M, vcov, ψ,
        TR, QF, storage_n, storage_m, storage_Σ, d, link)
end

function loglikelihood!(
    gc::NBCopulaARObs{T, D, Link},
    β::Vector{T},
    ρ::T,
    σ2::T,
    r::T,
    needgrad::Bool = false,
    needhess::Bool = false;
    penalized::Bool = false
    ) where {T <: BlasReal, D, Link}
    n, p = size(gc.X, 1), size(gc.X, 2)
    needgrad = needgrad || needhess
    if needgrad
        fill!(gc.∇β, 0)
        # gc.∇ARV .= get_∇ARV(n, ρ, σ2, gc.∇ARV)
        get_∇V!(ρ, gc)
    end
    needhess && fill!(gc.Hβ, 0)
    fill!(gc.∇β, 0.0)
    update_res!(gc, β)
    standardize_res!(gc)
    fill!(gc.∇resβ, 0.0) # fill gradient of residual vector with 0
    std_res_differential!(gc) # this will compute ∇resβ

    # form V
    # gc.V .= get_AR_cov(n, ρ, σ2, gc.V)
    get_V!(ρ, gc)

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
    logl = GLMCopula.component_loglikelihood(gc, r)
    logl += -log(c1)
    # @show logl
    logl += log(c2)
    # add L2 ridge penalty
    if penalized
        logl -= 0.5 * (σ2)^2
    end
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
        if penalized
            gc.∇σ2 .-= σ2
        end

        # gradient with respect to r
        gc.∇r .= nb_first_derivative(gc, ρ, σ2, r)

      if needhess
            # gc.∇2ARV .= get_∇2ARV(n, ρ, σ2, gc.∇2ARV)
            get_∇2V!(ρ, gc)
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
            # hessian for r
            gc.Hr .= nb_second_derivative(gc, ρ, σ2, r)
      end
      gc.∇β .= gc.∇β .* inv1pq
      gc.res .= gc.y .- gc.μ
      gc.∇β .+= GLMCopula.glm_gradient(gc, β, 1.0)
    end
    logl
end


function loglikelihood!(
    gcm::NBCopulaARModel{T, D, Link},
    needgrad::Bool = false,
    needhess::Bool = false
    ) where {T <: BlasReal, D, Link}
    logl = zero(T)
    if needgrad
        fill!(gcm.∇β, 0.0)
        fill!(gcm.∇ρ, 0.0)
        fill!(gcm.∇σ2, 0.0)
        fill!(gcm.∇r, 0.0)
    end
    if needhess
        fill!(gcm.Hβ, 0.0)
        fill!(gcm.Hρ, 0.0)
        fill!(gcm.Hσ2, 0.0)
        fill!(gcm.Hρσ2, 0.0)
        fill!(gcm.Hβσ2, 0.0)
        fill!(gcm.Hr, 0.0)
    end
    @inbounds for i in eachindex(gcm.data)
        logl += loglikelihood!(gcm.data[i], gcm.β, gcm.ρ[1], gcm.σ2[1], gcm.r[1], needgrad, needhess)
        if needgrad
            gcm.∇β .+= gcm.data[i].∇β
            gcm.∇ρ .+= gcm.data[i].∇ρ
            gcm.∇σ2 .+= gcm.data[i].∇σ2
            gcm.∇r .+= gcm.data[i].∇r
        end
        if needhess
            gcm.Hβ .+= gcm.data[i].Hβ
            gcm.Hρ .+= gcm.data[i].Hρ
            gcm.Hσ2 .+= gcm.data[i].Hσ2
            gcm.Hρσ2 .+= gcm.data[i].Hρσ2
            gcm.Hβσ2 .+= gcm.data[i].Hβσ2
            gcm.Hr .+= gcm.data[i].Hr
        end
    end
    logl
end

"""
1st derivative of loglikelihood of a sample with ρ and σ2 being the AR(1) parameterization of the covariance matrix.
"""
function nb_first_derivative(gc::NBCopulaARObs, ρ::T, σ2::T, r::Number) where T <: BlasReal
    s = zero(T)
    # 2nd term of logl
    for j in eachindex(gc.y)
        yi, μi = gc.y[j], gc.μ[j]
        s += -(yi+r)/(μi+r) - log(μi+r) + 1 + log(r) + digamma(r+yi) - digamma(r)
    end
    # 3rd term of logl
    resid = gc.res # res = inv(D)(y - μ)
    get_V!(ρ, gc)
    Γ = σ2 * gc.V
    η = gc.η
    D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
    dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
    dresid = -inv(D)*dD*resid
    s += resid'*Γ*dresid / (1 + 0.5resid'*Γ*resid)
    return s
end

"""
2nd derivative of loglikelihood of a sample with ρ and σ2 being the AR(1) parameterization of the covariance matrix.
"""
function nb_second_derivative(gc::NBCopulaARObs, ρ::T, σ2::T, r::Number) where T <: BlasReal
    s = zero(T)
    # 2nd term of logl
    for j in eachindex(gc.y)
        yi, μi = gc.y[j], gc.μ[j]
        s += (yi+r)/(μi+r)^2 - 2/(μi+r) + 1/r + trigamma(r+yi) - trigamma(r)
    end
    # 3rd term of logl
    ## set gc.V .= AR1(ρ)
    get_V!(ρ, gc)
    # Γ = σ2 * AR1(ρ)
    Γ = σ2 * gc.V
    η = gc.η
    D = Diagonal([sqrt(exp(η[j])*(exp(η[j])+r) / r) for j in 1:length(η)])
    dD = Diagonal([-exp(2η[i]) / (2r^1.5 * sqrt(exp(η[i])*(exp(η[i])+r))) for i in 1:length(η)])
    d2D = Diagonal([(exp(3η[i]) / (4r^1.5 * (exp(η[i])*(exp(η[i])+r))^(1.5))) +
        (3exp(2η[i]) / (4r^(2.5)*sqrt(exp(η[i])*(exp(η[i])+r)))) for i in 1:length(η)])
    resid = gc.res
    dresid = -inv(D)*dD*resid
    d2resid = (2inv(D)*dD*inv(D)*dD - inv(D)*d2D)*resid
    denom = 1 + 0.5resid'*Γ*resid
    term1 = (resid'*Γ*dresid / denom)^2
    term2 = dresid'*Γ*dresid / denom
    term3 = resid'*Γ*d2resid / denom
    s += -term1 + term2 + term3
    return s
end
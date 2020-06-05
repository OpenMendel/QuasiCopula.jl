export glm_VCobs, glm_VCModel
#  form data object to store data and intermediate fields
struct glm_VCobs{T <: BlasReal, D}
    # data
    y::Vector{T}
    X::Matrix{T}
    V::Vector{Matrix{T}}
    # working arrays
    ∇β::Vector{T}   # gradient wrt β
    ∇resβ::Matrix{T}# residual gradient matrix d/dβ_p res_ij (each observation has a gradient of residual is px1)
    ∇Σ::Vector{T}   # gradient wrt σ2
    Hβ::Matrix{T}   # Hessian wrt β
    xtx::Matrix{T}  # Xi'Xi
    #xtw2x::Matrix{T}# Xi'W2iXi where W2i = Diagonal(mueta(link, Xi*B))^2/Var(mu_i)
    res::Vector{T}  # residual vector res_i
    t::Vector{T}    # t[k] = tr(V_i[k]) / 2
    q::Vector{T}    # q[k] = res_i' * V_i[k] * res_i / 2
    storage_n::Vector{T} # storage_n = V_i[k] * res_i
    storage_p::Vector{T}
    η::Vector{T}    # η = Xβ systematic component
    μ::Vector{T}    # μ(β) = ginv(Xβ) # inverse link of the systematic component
    varμ::Vector{T} # v(μ_i) # variance as a function of the mean
    dμ::Vector{T}   # derivative of μ
    d::D            # distribution()
    w1::Vector{T}   # working weights in the gradient = dμ/v(μ)
    w2::Vector{T}   # working weights in the information matrix = dμ^2/v(μ)
end

function glm_VCobs(
    y::Vector{T},
    X::Matrix{T},
    V::Vector{Matrix{T}},
    d::D
    ) where {T <: BlasReal, D}
    n, p, m = size(X, 1), size(X, 2), length(V)
    @assert length(y) == n "length(y) should be equal to size(X, 1)"
    # working arrays
    ∇β  = Vector{T}(undef, p)
    ∇resβ  = Matrix{T}(undef, p, n)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    xtx = transpose(X) * X
    #xtw2x = Matrix{T}(undef, p, n)
    res = Vector{T}(undef, n)
    t   = [tr(V[k])/2 for k in 1:m]
    q   = Vector{T}(undef, m)
    storage_n = Vector{T}(undef, n)
    storage_p = Vector{T}(undef, p)
    η = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    varμ = Vector{T}(undef, n)
    dμ = Vector{T}(undef, n)
    w1 = Vector{T}(undef, n)
    w2 = Vector{T}(undef, n)
    # constructor
    glm_VCobs{T, D}(y, X, V, ∇β, ∇resβ, ∇Σ, Hβ, xtx, res,
      t, q, storage_n, storage_p, η, μ, varμ, dμ, d, w1, w2)
end

#  form model object to store data and intermediate fields
struct glm_VCModel{T <: BlasReal, D}  <: MathProgBase.AbstractNLPEvaluator
    # data
    data::Vector{glm_VCobs{T, D}}
    Ytotal::T
    ntotal::Int     # total number of singleton observations
    p::Int          # number of mean parameters in linear regression
    m::Int          # number of variance components
    # parameters
    β::Vector{T}    # p-vector of mean regression coefficients
    Σ::Vector{T}    # m-vector: [σ12, ..., σm2]
    # working arrays
    ∇β::Vector{T}   # gradient from all observations
    # ∇τ::Vector{T}
    ∇Σ::Vector{T}
    Hβ::Matrix{T}    # Hessian from all observations
    # Hτ::Matrix{T}
    XtX::Matrix{T}  # X'X = sum_i Xi'Xi
    # XtW2X::Matrix{T} # X'W2X = sum_i Xi'W2iXi
    TR::Matrix{T}   # n-by-m matrix with tik = tr(Vi[k]) / 2
    QF::Matrix{T}   # n-by-m matrix with qik = res_i' Vi[k] res_i
    storage_n::Vector{T}
    storage_m::Vector{T}
    storage_Σ::Vector{T}
    d::D
end

function glm_VCModel(gcs::Vector{glm_VCobs{T, D}}) where {T <: BlasReal, D}
    n, p, m = length(gcs), size(gcs[1].X, 2), length(gcs[1].V)
    β   = Vector{T}(undef, p)
    Σ   = Vector{T}(undef, m)
    ∇β  = Vector{T}(undef, p)
    ∇Σ  = Vector{T}(undef, m)
    Hβ  = Matrix{T}(undef, p, p)
    XtX = zeros(T, p, p) # sum_i xi'xi
    # XtW2X = zeros(T, p, p)
    TR  = Matrix{T}(undef, n, m) # collect trace terms
    Ytotal = 0
    ntotal = 0
    for i in eachindex(gcs)
        ntotal  += length(gcs[i].y)
        Ytotal  += sum(gcs[i].y)
        BLAS.axpy!(one(T), gcs[i].xtx, XtX)
        TR[i, :] = gcs[i].t
    end
    QF        = Matrix{T}(undef, n, m)
    storage_n = Vector{T}(undef, n)
    storage_m = Vector{T}(undef, m)
    storage_Σ = Vector{T}(undef, m)
    glm_VCModel{T, D}(gcs, Ytotal, ntotal, p, m, β, Σ,
        ∇β, ∇Σ, Hβ, XtX, TR, QF,
        storage_n, storage_m, storage_Σ, gcs[1].d)
end

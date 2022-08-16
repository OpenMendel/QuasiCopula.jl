struct MultivariateCopulaVCModel{T <: BlasReal} <: MathProgBase.AbstractNLPEvaluator
    # data
    Y::Matrix{T}    # n × d matrix of phenotypes, each row is a sample phenotype
    X::Matrix{T}    # n × p matrix of non-genetic covariates, each row is a sample covariate
    V::Vector{Matrix{T}} # length m vector of d × d matrices
    vecdist::Vector{<:UnivariateDistribution} # length d vector of marginal distributions for each phenotype
    veclink::Vector{<:Link} # length d vector of link functions for each phenotype's marginal distribution
    # parameters
    B::Matrix{T}    # p × d matrix of mean regression coefficients, Y = XB
    θ::Vector{T}    # length m vector of variance components
    # working arrays
    Γ::Matrix{T}      # d × d covariance matrix, in VC model this is θ[1]*V[1] + ... + θ[m]*V[m]
    ∇vecB::Vector{T}  # length pd vector, its the gradient of vec(B) 
    ∇θ::Vector{T}     # length m vector, gradient of variance components
    HvecB::Matrix{T}  # pd × pd matrix of Hessian
    Hθ::Matrix{T}     # m × m matrix of Hessian for variance components
end

function MultivariateCopulaVCModel(
    Y::Matrix{T},
    X::Matrix{T},
    V::Union{Matrix{T}, Vector{Matrix{T}}}, # variance component matrices
    vecdist::Union{Vector{<:UnivariateDistribution}, Vector{UnionAll}}, # vector of marginal distributions for each data point
    veclink::Vector{<:Link}; # vector of link functions for each marginal distribution
    ) where T <: BlasReal
    n, d = size(Y)
    p = size(X, 2)
    m = typeof(V) <: Matrix ? 1 : length(V)
    n == size(X, 1) || error("Number of samples in Y and X mismatch")
    m == (typeof(V) <: Matrix ? 1 : length(V)) || 
        error("Number of variance components should be equal to size(Y, 2)")
    # initialize variables
    B = zeros(T, p, d)
    θ = zeros(T, m)
    Γ = zeros(T, d, d)
    ∇vecB = zeros(T, p*d)
    ∇θ = zeros(T, m)
    HvecB = zeros(T, p*d, p*d)
    Hθ = zeros(T, m, m)
    # change type of variables to match struct
    if typeof(vecdist) <: Vector{UnionAll}
        vecdist = [vecdist[j]() for j in 1:d]
    end
    typeof(V) <: Matrix && (V = [V])
    return MultivariateCopulaVCModel(
        Y, X, V, 
        vecdist, veclink,
        B, θ, Γ, 
        ∇vecB, ∇θ, HvecB, Hθ
    )
end


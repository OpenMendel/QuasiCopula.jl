export CS_model

"""
    CS_model(df, y, grouping, covariates, d, link; penalized = penalized)

Form the compound symmetric (CS) model for regression with the specified base distribution (d) and link function (link).

# Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `covariates`: Covariate names of interest as a vector of `Symbol`s.
    Each variable name must be present in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

# Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance parameter for the CS structured covariance.
    One can turn this option on by specifying `penalized = true` to add this penalty for numerical stability. (default `penalized = false`).
"""
function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaCSObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GaussianCopulaCSObs(Y, X)
    end
    gcm = GaussianCopulaCSModel(gcs; penalized = penalized);
    return gcm
end

function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaCSObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GLMCopulaCSObs(Y, X, d, link)
    end
    gcm = GLMCopulaCSModel(gcs; penalized = penalized);
    return gcm
end

function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaCSObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = NBCopulaCSObs(Y, X, d, link)
    end
    gcm = NBCopulaCSModel(gcs; penalized = penalized);
    return gcm
end

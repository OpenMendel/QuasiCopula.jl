export AR_model

"""
    AR_model(df, y, grouping, covariates, d, link; penalized = penalized)

Form the autoregressive (AR(1)) model for regression with the specified base distribution (d) and link function (link).

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
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance parameter for the AR(1) structured covariance.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).
"""
function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaARObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GaussianCopulaARObs(Y, X)
    end
    gcm = GaussianCopulaARModel(gcs; penalized = penalized);
    return gcm
end

function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaARObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GLMCopulaARObs(Y, X, d, link)
    end
    gcm = GLMCopulaARModel(gcs; penalized = penalized);
    return gcm
end

function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaARObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = NBCopulaARObs(Y, X, d, link)
    end
    gcm = NBCopulaARModel(gcs; penalized = penalized);
    return gcm
end

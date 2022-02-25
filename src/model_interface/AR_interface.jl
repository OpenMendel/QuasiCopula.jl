export AR_model

"""
    AR_model(df, y, grouping, covariates, d, link)
Form the autoregressive (AR(1)) model with named dataframe (df), outcome variable name of interest(y) as a symbol,
grouping variable name of interest as a symbol (grouping), covariate names of interest as a vector of symbols (covariates),
base distribution (d), and link function (link).
"""
function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaARObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GaussianCopulaARObs(Y, X)
    end
    gcm = GaussianCopulaARModel(gcs);
    return gcm
end

function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaARObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GLMCopulaARObs(Y, X, d, link)
    end
    gcm = GLMCopulaARModel(gcs);
    return gcm
end

function AR_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaARObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = NBCopulaARObs(Y, X, d, link)
    end
    gcm = NBCopulaARModel(gcs);
    return gcm
end

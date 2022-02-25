export CS_model

"""
    CS_model(df, y, grouping, covariates, d, link)
Form the compound symmetric (CS) model with named dataframe (df), outcome variable name of interest(y) as a symbol,
grouping variable name of interest as a symbol (grouping), covariate names of interest as a vector of symbols (covariates),
base distribution (d), and link function (link).
"""
function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaCSObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GaussianCopulaCSObs(Y, X)
    end
    gcm = GaussianCopulaCSModel(gcs);
    return gcm
end

function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaCSObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GLMCopulaCSObs(Y, X, d, link)
    end
    gcm = GLMCopulaCSModel(gcs);
    return gcm
end

function CS_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaCSObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(ni, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = NBCopulaCSObs(Y, X, d, link)
    end
    gcm = NBCopulaCSModel(gcs);
    return gcm
end

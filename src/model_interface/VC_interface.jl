export VC_model

"""
    VC_model(df, y, grouping, covariates, d, link)
Form the variance component model (VCM) with named dataframe (df), outcome variable name of interest(y) as a symbol,
grouping variable name of interest as a symbol (grouping), covariate names of interest as a vector of symbols (covariates),
base distribution (d), and link function (link). By default this will form a random intercept model with V_i = ones(d_i, d_i)
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaVCObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        V = [ones(di, di)]
        gcs[i] = GaussianCopulaVCObs(Y, X, V)
    end
    gcm = GaussianCopulaVCModel(gcs)
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        V = [ones(di, di)]
        gcs[i] = GLMCopulaVCObs(Y, X, V, d, link)
    end
    gcm = GLMCopulaVCModel(gcs)
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        V = [ones(di, di)]
        gcs[i] = NBCopulaVCObs(Y, X, V, d, link)
    end
    gcm = NBCopulaVCModel(gcs)
    return gcm
end

"""
    VC_model(df, y, grouping, covariates, V, d, link)
Form the variance component model (VCM) with named dataframe (df), outcome variable name of interest(y) as a symbol,
grouping variable name of interest as a symbol (grouping), covariate names of interest as a vector of symbols (covariates),
Vector of Vector of PSD Covariance Matrices (V),
base distribution (d), and link function (link).
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaVCObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GaussianCopulaVCObs(Y, X, V[i])
    end
    gcm = GaussianCopulaVCModel(gcs)
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = GLMCopulaVCObs(Y, X, V[i], d, link)
    end
    gcm = GLMCopulaVCModel(gcs)
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        @inbounds for i in 1:length(covariates)
            U = Float64.(df[gidx, covariates[i]])
            X = hcat(X, U)
        end
        gcs[i] = NBCopulaVCObs(Y, X, V[i], d, link)
    end
    gcm = NBCopulaVCModel(gcs)
    return gcm
end

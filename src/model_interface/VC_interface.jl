export VC_model
"""
    VC_model(df, y, grouping, d, link)

Form the variance component model (VCM) for intercept only regression with a random intercept covariance matrix and the specified base distribution (d) and link function (link).

# Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

# Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance components vector.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaVCObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        V = [ones(di, di)]
        gcs[i] = GaussianCopulaVCObs(Y, X, V)
    end
    gcm = GaussianCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        V = [ones(di, di)]
        gcs[i] = GLMCopulaVCObs(Y, X, V, d, link)
    end
    gcm = GLMCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        V = [ones(di, di)]
        gcs[i] = NBCopulaVCObs(Y, X, V, d, link)
    end
    gcm = NBCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

"""
    VC_model(df, y, grouping, covariates, d, link)

Form the variance component model (VCM) for regression with a random intercept covariance matrix and the specified base distribution (d) and link function (link).

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
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance components vector.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
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
    gcm = GaussianCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
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
    gcm = GLMCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
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
    gcm = NBCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

"""
    VC_model(df, y, grouping, covariates, V, d, link)

Form the variance component model (VCM) for intercept only regression with the specified base distribution (d) and link function (link).

# Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `V`: Vector of Vector of Positive Semi-Definite (PSD) Covariance Matrices. `V` is of length n, where n is the number of groups/clusters.
    Each element of `V` is also a `Vector`, but of length m. Here m is the number of variance components.
    Each element of `V` is a `Vector` of d_i x d_i PSD covariance matrices under the VCM framework,
    where d_i is the cluster size of the ith cluster, which may vary for each cluster of observations i in [1, n].
    Each of these dimensions must match that specified in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

# Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance components vector.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GaussianCopulaVCObs{Float64}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        gcs[i] = GaussianCopulaVCObs(Y, X, V[i])
    end
    gcm = GaussianCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{GLMCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        gcs[i] = GLMCopulaVCObs(Y, X, V[i], d, link)
    end
    gcm = GLMCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
    groups = unique(df[!, grouping])
    n = length(groups)
    gcs = Vector{NBCopulaVCObs{Float64, D, Link}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        di = count(gidx)
        Y = Float64.(df[gidx, y])
        X = ones(di, 1)
        gcs[i] = NBCopulaVCObs(Y, X, V[i], d, link)
    end
    gcm = NBCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

"""
    VC_model(df, y, grouping, covariates, V, d, link)

Form the variance component model (VCM) for regression with the specified base distribution (d) and link function (link).

# Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `covariates`: Covariate names of interest as a vector of `Symbol`s.
    Each variable name must be present in `df`.
- `V`: Vector of Vector of Positive Semi-Definite (PSD) Covariance Matrices. `V` is of length n, where n is the number of groups/clusters.
    Each element of `V` is also a `Vector`, but of length m. Here m is the number of variance components.
    Each element of `V` is a `Vector` of d_i x d_i PSD covariance matrices under the VCM framework,
    where d_i is the cluster size of the ith cluster, which may vary for each cluster of observations i in [1, n].
    Each of these dimensions must match that specified in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

# Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance components vector.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).
"""
function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Normal, Link}
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
    gcm = GaussianCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:Union{Poisson, Bernoulli}, Link}
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
    gcm = GLMCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

function VC_model(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol,
    covariates::Vector{Symbol},
    V::Vector{Vector{Matrix{Float64}}},
    d::D,
    link::Link;
    penalized::Bool = false) where {D<:NegativeBinomial, Link}
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
    gcm = NBCopulaVCModel(gcs; penalized = penalized);
    return gcm
end

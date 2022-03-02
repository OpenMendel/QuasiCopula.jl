using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
using LinearAlgebra: BlasReal, copytri!

function __get_distribution(dist::Type{D}, μ, r) where D <: UnivariateDistribution
    return dist(r, μ)
end

p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(1234)
βtrue = rand(Uniform(-0.2, 0.2), p_fixed)
Σtrue = [0.01]
rtrue = 10.0

# #simulation parameters
samplesize = 10000
ni = 5 # number of observations per individual

d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, samplesize)

Γ = Σtrue[1] * ones(ni, ni) + 0.00000000000001 * Matrix(I, ni, ni)
a = collect(1:samplesize)
group = [repeat([a[i]], ni) for i in 1:samplesize]
groupstack = vcat(group...)
Xstack = []
Ystack = []
Random.seed!(12345)
X_samplesize = [randn(ni, p_fixed - 1) for i in 1:samplesize]
for i in 1:samplesize
    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
    X = [ones(ni) X_samplesize[i]]
    η = X * βtrue
    V = [ones(ni, ni)]
    # generate mvn response
    mvn_d = MvNormal(η, Γ)
    mvn_η = rand(mvn_d)
    μ = GLM.linkinv.(link, mvn_η)
    p_nb = rtrue ./ (μ .+ rtrue)
    y = Float64.(rand.(__get_distribution.(D, p_nb, rtrue)))
    # add to data
    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
    push!(Xstack, X)
    push!(Ystack, y)
end

# form NBCopulaVCModel
gcm = NBCopulaVCModel(gcs);
fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter=100, tol = 1e-6)
@show fittime
@show gcm.β
@show gcm.θ
@show gcm.r
@show gcm.∇β
@show gcm.∇θ
@show gcm.∇r

function get_CI(gcm)
    loglikelihood!(gcm, true, true)
    vcov!(gcm)
    GLMCopula.confint(gcm)
end
get_CI(gcm)

# form glmm
Xstack = [vcat(Xstack...)][1]
Ystack = [vcat(Ystack...)][1]
# p = 3
df = DataFrame(Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))

using RCall

@rput df
R"""
    library("lme4")
    ptm <- proc.time()
    m.nb <- glmer.nb(Y ~ 1 + X2 + X3 + (1|group), data = df, verbose = TRUE)
    # Stop the clock
    proc.time() - ptm
"""

### Show estimates
R"""
    m.nb
"""

# Show estimated r from lme4: glmer.nb
R"""
    getME(m.nb, "glmer.nb.theta")
"""

### Show confidence intervals from lme4: glmer.nb
R"""
    ptm <- proc.time()
    confint(m.nb)
    # Stop the clock
    proc.time() - ptm
"""

# using glmmTMB
R"""
    library("glmmTMB")
    ptm <- proc.time()
    m.glmmtmb_nb <- glmmTMB(Y ~ 1 + X2 + X3 + (1|group), data = df, family=nbinom2)
    # Stop the clock
    proc.time() - ptm
"""

### Show estimates
R"""
    m.glmmtmb_nb
"""

# Show estimated r from glmmTMB
R"""
    sigma(m.glmmtmb_nb)
"""

### Show confidence intervals from glmmTMB
R"""
    ptm <- proc.time()
    confint(m.glmmtmb_nb, full = TRUE)
    # Stop the clock
    proc.time() - ptm
"""

using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
using LinearAlgebra: BlasReal, copytri!

# multi-threading results with this machine
versioninfo()
# Julia Version 1.6.2
# Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
# Platform Info:
#   OS: macOS (x86_64-apple-darwin18.7.0)
#   CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
#   WORD_SIZE: 64
#   LIBM: libopenlibm
#   LLVM: libLLVM-11.0.1 (ORCJIT, skylake)

BLAS.set_num_threads(1)
Threads.nthreads()

function __get_distribution(dist::Type{D}, μ, r) where D <: UnivariateDistribution
    return dist(r, μ)
end

p_fixed = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
Random.seed!(1234)
βtrue = rand(Uniform(-0.2, 0.2), p_fixed)
θtrue = [0.01]
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

Γ = θtrue[1] * ones(ni, ni) + 0.00000000000001 * Matrix(I, ni, ni)
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
fittime = @elapsed GLMCopula.fit!(gcm)
@show fittime # 1.768249684 seconds
@show gcm.β
# 0.032738347307332966
# 0.10609650651774433
# 0.026158048751589738
@show gcm.θ
#  0.00703047401415037
@show gcm.r
# 10.001819287469104
@show gcm.∇β
@show gcm.∇θ
@show gcm.∇r

function get_CI(gcm)
    loglikelihood!(gcm, true, true)
    vcov!(gcm)
    GLMCopula.confint(gcm)
end
get_CI(gcm)

@time get_CI(gcm)
# 0.045099 seconds
# 0.0233753    0.0421014
# 0.100997     0.111196
# 0.0172389    0.0350772
# 9.09354     10.9101
# 0.00332143   0.0107395

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
#    user  system elapsed
# 70.575   1.744  72.415

### Show estimates
R"""
    m.nb
"""
# Random effects:
#  Groups Name        Std.Dev.
#  group  (Intercept) 0.09233
# Number of obs: 50000, groups:  group, 10000
# Fixed Effects:
# (Intercept)           X2           X3
#     0.03214      0.10575      0.02603

# Show estimated r from lme4: glmer.nb
R"""
    getME(m.nb, "glmer.nb.theta")
"""
# 10.14696

### Show confidence intervals from lme4: glmer.nb
R"""
    ptm <- proc.time()
    confint(m.nb)
    # Stop the clock
    proc.time() - ptm
"""
# user  system elapsed
# 149.092   4.184 153.412

# using glmmTMB
R"""
    library("glmmTMB")
    ptm <- proc.time()
    m.glmmtmb_nb <- glmmTMB(Y ~ 1 + X2 + X3 + (1|group), data = df, family=nbinom2)
    # Stop the clock
    proc.time() - ptm
"""
# user  system elapsed
# 101.985   0.997 103.195

### Show estimates
R"""
    m.glmmtmb_nb
"""
# Conditional model:
#  Groups Name        Std.Dev.
#  group  (Intercept) 0.08967
#
# Number of obs: 50000 / Conditional model: group, 10000
#
# Dispersion parameter for nbinom2 family (): 10.1
#
# Fixed Effects:
#
# Conditional model:
# (Intercept)           X2           X3
#     0.03276      0.10579      0.02604

# Show estimated r from glmmTMB
R"""
    sigma(m.glmmtmb_nb)
"""
# 10.10115

### Show confidence intervals from glmmTMB
R"""
    ptm <- proc.time()
    confint(m.glmmtmb_nb, full = TRUE)
    # Stop the clock
    proc.time() - ptm
"""
# user  system elapsed
# 0.512   0.032   0.547

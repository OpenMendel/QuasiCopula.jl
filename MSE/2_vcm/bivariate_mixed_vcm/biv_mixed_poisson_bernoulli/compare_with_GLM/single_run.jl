using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

p = 3    # number of fixed effects, including intercept
m = 1    # number of variance components
# true parameter values
βtrue = [0.2; 0.1; -0.05; 0.2; 0.1; -0.1]
Σtrue = [0.1]

# generate data
intervals = zeros(2 * p + m, 2) #hold intervals
curcoverage = zeros(2 * p + m) #hold current coverage resutls
trueparams = [βtrue; Σtrue] #hold true parameters

#simulation parameters
# samplesizes = [100; 1000; 10000]
samplesize = 10000
ni = 2

currentind = 1
d1 = Poisson()
d2 = Bernoulli()
vecdist = [d1, d2]

link1 = LogLink()
link2 = LogitLink()
veclink = [link1, link2]

T = Float64
VD = typeof(vecdist)
VL = typeof(veclink)

Random.seed!(1234)
Xstack = [ones(samplesize) randn(samplesize, p - 1)]

Ystack = []

gcs = Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}(undef, samplesize)
Γ = Σtrue[1] * ones(ni, ni)
vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)

for i in 1:samplesize
    xi = Xstack[i, :]
    X = [transpose(xi) zeros(size(transpose(xi))); zeros(size(transpose(xi))) transpose(xi)]
    η = X * βtrue
    μ = zeros(ni)
    for j in 1:ni
        μ[j] = GLM.linkinv(veclink[j], η[j])
        vecd[j] = typeof(vecdist[j])(μ[j])
    end
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    # simuate single vector y
    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    rand(nonmixed_multivariate_dist, y, res)
    V = [ones(ni, ni)]
    push!(Ystack, y)
    gcs[i] = Poisson_Bernoulli_VCObs(y, xi, V, vecdist, veclink)
end

gcm = Poisson_Bernoulli_VCModel(gcs);


fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-7, limited_memory_max_history = 12, accept_after_max_steps = 1, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
# coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
@show mseβ
@show mseΣ

# N10k
# vc = 0.1
# beta
# 0.19683256272278854
# 0.09480307627780372
# -0.03655334972255244
#
# 0.20451894655376088
# 0.10635688606099081
# -0.08407303584850163

# ci
# 0.181275    0.212391
# 0.0878555   0.101751
# -0.0431862  -0.0299205
#
# 0.178689    0.230349
# 0.0753035   0.13741
# -0.118724   -0.0494217
#
# 0.0940878   0.118651

### fit using glm

Y = transpose(hcat(Ystack...))
Data = hcat(Xstack, Y)
df = DataFrame(Y1 = Data[:, 4], Y2 = Data[:, 5], X1 = Data[:, 2], X2 = Data[:, 3])

poisson_glm = glm(@formula(Y1 ~ 1 + X1 + X2), df, Poisson(), LogLink())
# ────────────────────────────────────────────────────────────────────────────
#                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
# ────────────────────────────────────────────────────────────────────────────
# (Intercept)   0.235627   0.00891075  26.44    <1e-99   0.218162    0.253091
# X1            0.0922597  0.00887493  10.40    <1e-24   0.0748651   0.109654
# X2           -0.0351364  0.00893941  -3.93    <1e-04  -0.0526574  -0.0176155
# ────────────────────────────────────────────────────────────────────────────

# poisson_glm.model.pp.beta0

bernoulli_glm = glm(@formula(Y2 ~ 1 + X1 + X2), df, Bernoulli(), LogitLink())
# ────────────────────────────────────────────────────────────────────────────
#                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
# ────────────────────────────────────────────────────────────────────────────
# (Intercept)   0.194727    0.0201368   9.67    <1e-21   0.15526     0.234195
# X1            0.101355    0.0201591   5.03    <1e-06   0.0618437   0.140866
# X2           -0.0799781   0.0203455  -3.93    <1e-04  -0.119854   -0.0401017
# ────────────────────────────────────────────────────────────────────────────
# bernoulli_glm.model.pp.beta0

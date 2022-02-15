using Revise
using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns
using LinearAlgebra: BlasReal, copytri!

using GLM, DataFrames, DelimitedFiles, Statistics

d1 = Poisson()
d2 = Bernoulli()
vecdist = [d1, d2]

link1 = LogLink()
link2 = LogitLink()
veclink = [link1, link2]

T = Float64
VD = typeof(vecdist)
VL = typeof(veclink)

p = 3    # number of fixed effects, including intercept
Random.seed!(12345)
# β1true = rand(Uniform(-0.2, 0.2), p)
β1true = rand(p)
Random.seed!(1234)
β2true = rand(Uniform(-0.2, 0.2), p)
βtrue = [β1true; β2true]

vc = 0.1
ni = 2
Γ = vc * ones(ni, ni)

# sample size
m = 10_000
gcs = Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}(undef, m)
vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)

for i in 1:m
    xi = [1.0 randn(p - 1)...]
    X = [xi zeros(size(xi)); zeros(size(xi)) xi]
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
    gcs[i] = Poisson_Bernoulli_VCObs(y, X, V, vecdist, veclink)
end

# form VarLmmModel
gcm = Poisson_Bernoulli_VCModel(gcs);

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 12, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))

@show fittime
@show gcm.β
@show gcm.Σ
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)
# mse and time under our model
intervals = zeros(2 * p + 1, 2) #hold intervals
curcoverage = zeros(2 * p + 1) #hold current coverage resutls
trueparams = [βtrue; vc] #hold true parameters

coverage!(gcm, trueparams, intervals, curcoverage)
mseβ, mseΣ = MSE(gcm, βtrue, [vc])
@show mseβ
@show mseΣ

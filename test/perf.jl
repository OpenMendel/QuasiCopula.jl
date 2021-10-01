module PerfTest

using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM
using BenchmarkTools, Profile, Distributions, DataFrames, CategoricalArrays

Random.seed!(123346264234)
n = 20
Σ = [0.1; 0.1]
Γ = Σ[1] * ones(n, n) + Σ[2] * Matrix(I, n, n)

p = 3
β = ones(p)
X = [ones(n) randn(n, p - 1)]
η = X * β
μ = exp.(η) ./ (1 .+ exp.(η))
vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Bernoulli(μ[i])
end

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
nsample = 10000
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

d = Bernoulli()
link = LogitLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    V = [ones(n, n), Matrix(I, n, n)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end

gcm = GLMCopulaVCModel(gcs)

function profiling_everything(gcm)
    initialize_model!(gcm)
    @time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-6, mu_strategy = "adaptive",  mu_oracle = "loqo", hessian_approximation = "limited-memory")) 
end

@info "profile..."
Profile.clear()
# @profile loglikelihood!($gcm, true, true)
@profile profiling_everything(gcm)
Profile.print()
end

# @info "Fit with MixedModels..."

# Ystack = vcat(Y_nsample...)
# a = collect(1:nsample)
# group = [repeat([a[i]], n) for i in 1:nsample]
# groupstack = vcat(group...)
# Xstack = repeat(X, nsample)

# df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
# using MixedModels
# form = @formula(Y ~ 1 + X2 + X3 + (1|group));
# @time gm1 = fit(MixedModel, form, df, Poisson())
# display(gm1)
# @show gm1.β

# @info "Get MSE under GLMM..."
# level = 0.95
# p = 3
# m = 1
# @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
# @show GLMM_mse = [sum(abs2, gm1.β - β) / p, sum(abs2, (gm1.θ.^2) - Σ) / m]

# @info "Get MSE under our model..."
# @show GLMCopula.MSE(gcm, β, Σ)

# @info "benchmarking..."
# gcm.β .= β
# gcm.Σ[1] = 0.1
# bm = @benchmark loglikelihood!($gcm, true, true)
# display(bm); println()

# n = 5
# N = 10k 
# 31 iterations at 2.243225 seconds (825.94 k allocations: 41.718 MiB, 1.10% gc time)
# estimated beta = [0.9959693049116686, 0.9957353472570317, 1.0038515925669134]; true beta value= [1.0, 1.0, 1.0]
# estimated variance component 1 = 0.10228927061489634; true variance component 1 = 0.1
# GLMCopula.confint(gcm) = [0.9913336851192123 1.0006049247041249; 0.9900386135812697 1.0014320809327937; 1.0020412940292578 1.005661891104569; 0.09160230016817568 0.11297624106161701]# GLMCopula.MSE(gcm, β, Σ) = (2.2180113893697107e-5, 2.4723246153639047e-5)
# GLMCopula.MSE(gcm, β, Σ) = (1.6422843738236074e-5, 5.2407599482278585e-6)

# 28.667554 seconds (66.62 M allocations: 5.007 GiB, 4.24% gc time)
# glmmci = [1.0063601936388564 1.0192401567779497; 0.9798310930562129 0.9929028821238144; 0.9958806902146504 1.0032513804256842]
# glmm_mse = [0.00011663061269333358, 0.009396394176828126]
##############################################################################

# n = 10
# N = 10k 
# 18 iterations at 2.150277 seconds (825.77 k allocations: 41.710 MiB)
# estimated beta = [1.0059885151579946, 0.9978301028543147, 0.9949039648489529]; true beta value= [1.0, 1.0, 1.0]
# estimated variance component 1 = 0.09502775240423016; true variance component 1 = 0.1
# GLMCopula.confint(gcm) = [1.0041939639924797 1.0077830663235094; 0.9961149122338488 0.9995452934747806; 0.9911254760480566 0.9986824536498493; 0.08732673111580914 0.10272877369265117]# GLMCopula.MSE(gcm, β, Σ) = (4.318074964275062e-7, 1.32740321502985e-5)
# GLMCopula.MSE(gcm, β, Σ) = (2.2180113893697107e-5, 2.4723246153639047e-5)

# 28.909152 seconds (66.64 M allocations: 5.026 GiB, 3.70% gc time)
# glmmci = [1.0103022534875203 1.0193373507812258; 0.9898019561214898 0.9947389213476826; 0.9827198120556369 0.9913573519381971]
# glmm_mse = [0.00014912366976946624, 0.008487842912329333]
##############################################################################

# n = 25
# N = 10k 
# 23 iterations at 3.723728 seconds (825.75 k allocations: 41.709 MiB)
# estimated beta = [0.9985351923442617, 1.0005405218211443, 1.0007877824336908]; true beta value= [1.0, 1.0, 1.0]
# estimated variance component 1 = 0.09972376763232019; true variance component 1 = 0.1
# GLMCopula.confint(gcm) = [0.9973447589733082 0.9997256257152152; 0.9997580339501962 1.0013230096920924; 0.9974346358156974 1.0041409290516843; 0.09277658247597277 0.1066709527886676]
# GLMCopula.MSE(gcm, β, Σ) = (1.0194754900915073e-6, 7.630432095399839e-8)

# 33.761396 seconds (66.79 M allocations: 5.089 GiB, 2.37% gc time)
# glmmci = [1.0043752895527218 1.01122830994936; 0.9863228214468255 0.9919832488079193; 0.9855975884959534 0.9928724659348505]
# glmm_mse = [9.813645511883213e-5, 0.008195055931679294]
##############################################################################

# n = 50
# N = 10k 
# 33 iterations at 13.206931 seconds (825.93 k allocations: 41.718 MiB)
# estimated beta = [1.0006990445573276, 1.0003479289205113, 0.9991719271394225]; true beta value= [1.0, 1.0, 1.0]
# estimated variance component 1 = 0.10364335451888758; true variance component 1 = 0.1
# GLMCopula.confint(gcm) = [0.9998993347983607 1.0014987543162945; 0.9994459914717769 1.0012498663692455; 0.9983442274939252 0.9999996267849198; 0.0955793706346555 0.11170733840311965]
# GLMCopula.MSE(gcm, β, Σ) = (4.318074964275062e-7, 1.32740321502985e-5)

# 38.770442 seconds (67.04 M allocations: 5.194 GiB, 2.62% gc time)
# glmmci = [1.003346466578153 1.0078454964301011; 0.9961130148042211 0.998695539723851; 0.9951666622454416 0.9975664505473067]
# glmm_mse = [1.7084899312420798e-5, 0.00926565322167157]

####################################################################################
# n = 100 
# N = 10k
# 20 iterations at 25.400764 seconds (825.78 k allocations: 41.710 MiB, 0.12% gc time)
# estimated beta = [1.0011993599516098, 0.9999387615161686, 0.9996963879676932]; true beta value= [1.0, 1.0, 1.0]
# estimated variance component 1 = 0.10135967080327069; true variance component 1 = 0.1
# GLMCopula.confint(gcm) = [1.0011581945993393 1.0012405253038803; 0.9995678377450825 1.0003096852872546; 0.9992888515632405 1.000103924372146; 0.09124198634581605 0.11147735526072532]
# GLMCopula.MSE(gcm, β, Σ) = (5.114649038629638e-7, 1.8487046932667347e-6)

# 68.452931 seconds (67.55 M allocations: 5.406 GiB, 1.61% gc time)
# glmmci = [1.0028048693297908 1.0057852297804566; 0.9967315484962164 0.9989331294207203; 0.9980603150060715 0.9990926910793477]
# glmm_mse = [8.390849553119179e-6, 0.009697991719652185]

using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

p  = 3    # number of fixed effects, including intercept
m  = 1    # number of variance components
# true parameter values
βtrue = ones(p)
Σtrue = [0.5]
rtrue = 100.0
# p = 0.7
# μ = r * (1 - p) * inv(p)
# true beta
# β_true = log(μ)

# generate data
intervals = zeros(p + m + 1, 2) #hold intervals
curcoverage = zeros(p + m + 1) #hold current coverage resutls
trueparams = [βtrue; rtrue; Σtrue] #hold true parameters

#simulation parameters
samplesizes = [1000; 10000; 25000; 50000]
ns = [5; 10; 20; 50]
nsims = 10

#storage for our results
βMseResults = ones(nsims * length(ns) * length(samplesizes))
ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
rMseResults = ones(nsims * length(ns) *  length(samplesizes))
βrΣcoverage = Matrix{Float64}(undef, p + m + 1, nsims * length(ns) * length(samplesizes))
fittimes = zeros(nsims * length(ns) * length(samplesizes))

#storage for glmm results
βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
ΣMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
rMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

# solver = KNITRO.KnitroSolver(outlev=0)
solver = Ipopt.IpoptSolver(print_level = 5)

st = time()
currentind = 1
d = NegativeBinomial()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64

for t in 1:length(samplesizes)
    m = samplesizes[t]
    gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, m)
    for k in 1:length(ns)
        ni = ns[k] # number of observations per individual
        y = Vector{Float64}(undef, ni)
        res = Vector{Float64}(undef, ni)
        β = ones(p)
        X = [ones(ni) randn(ni, p - 1)]
        η = X * β
        μ = exp.(η)
        p = rtrue ./ (μ .+ rtrue)
        vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

        vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]

        Γ = Σtrue[1] * ones(ni, ni)
        nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
        for j in 1:nsims
            println("rep $j obs per person $ni samplesize $m")
            Random.seed!(123 * j + 100 * k)
            @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, m)
            Ystack = vcat(Y_nsample...)
            @show length(Ystack)
            a = collect(1:m)
            group = [repeat([a[i]], ni) for i in 1:m]
            groupstack = vcat(group...)
            Xstack = repeat(X, m)
            df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
            # df = (Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
            form = @formula(Y ~ 1 + X2 + X3 + (1|group));

            gcs = Vector{NBCopulaVCObs{T, D, Link}}(undef, m)
            for i in 1:m
                y = Float64.(Y_nsample[i])
                V = [ones(ni, ni)]
                gcs[i] = NBCopulaVCObs(y, X, V, d, link)
            end

            # form VarLmmModel
            gcm = NBCopulaVCModel(gcs);
            fittime = NaN
            initialize_model!(gcm)
            @show gcm.β
            @show gcm.Σ
            @show gcm.r
            # try
                fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter=100)
                # fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, hessian_approximation = "limited-memory"))
                loglikelihood!(gcm, true, true)
                @show gcm.θ
                @show gcm.∇θ
                sandwich!(gcm)
                @show GLMCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mser, mseΣ = MSE(gcm, βtrue, rtrue, Σtrue)
                @show mseβ
                @show mser
                @show mseΣ
                #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                global currentind
                @views copyto!(βrΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                ΣMseResults[currentind] = mseΣ
                rMseResults[currentind] = mser
                fittimes[currentind] = fittime
                # glmm
                # fit glmm
                @info "Fit with MixedModels..."
                fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, NegativeBinomial(), LogLink(); nAGQ = 25)
                display(gm1)
                @show gm1.β
                # mse and time under glmm
                @info "Get MSE under GLMM..."
                level = 0.95
                p = 3
                @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, gm1.σ^2 - inv(rtrue)), sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]
                # glmm
                βMseResults_GLMM[currentind] = GLMM_mse[1]
                rMseResults_GLMM[currentind] = GLMM_mse[2]
                ΣMseResults_GLMM[currentind] = GLMM_mse[3]
                fittimes_GLMM[currentind] = fittime_GLMM
                currentind += 1
        end
    end
end
en = time()

@show en - st #seconds
@info "writing to file..."
ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
writedlm("new/mse_beta_" * ftail, βMseResults, ',')
writedlm("new/mse_Sigma_" * ftail, ΣMseResults, ',')
writedlm("new/fittimes_" * ftail, fittimes, ',')

writedlm("new/beta_sigma_coverage_" * ftail, βΣcoverage, ',')

# glmm
writedlm("new/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
writedlm("new/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
writedlm("new/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')

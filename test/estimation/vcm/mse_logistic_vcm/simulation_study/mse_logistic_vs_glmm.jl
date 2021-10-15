using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function runtest()
    p  = 3    # number of fixed effects, including intercept
    m  = 1    # number of variance components
    # true parameter values
    βtrue = ones(p)
    Σtrue = [0.5]

    # generate data
    intervals = zeros(p + m, 2) #hold intervals
    curcoverage = zeros(p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [1000; 10000; 25000; 50000]
    ns = [5; 10; 20; 50]
    nsims = 50

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βΣcoverage = Matrix{Float64}(undef, p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
    fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

    # solver = KNITRO.KnitroSolver(outlev=0)
    solver = Ipopt.IpoptSolver(print_level = 5)

    st = time()
    currentind = 1
    d = Bernoulli()
    link = LogitLink()
    D = typeof(d)
    Link = typeof(link)
    T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            y = Vector{Float64}(undef, ni)
            res = Vector{Float64}(undef, ni)
            β = ones(p)
            Random.seed!(123 * k)
            X = [ones(ni) randn(ni, p - 1)]
            η = X * β
            μ = exp.(η) ./ (1 .+ exp.(η))
            vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

            for i in 1:length(μ)
            vecd[i] = Bernoulli(μ[i])
            end

            Γ = Σtrue[1] * ones(ni, ni)
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Random.seed!(123 * k + j)
                @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, m)
                Ystack = vcat(Y_nsample...)
                @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = repeat(X, m)
                # df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
                df = (Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
                form = @formula(Y ~ 1 + X2 + X3 + (1|group));

                gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
                for i in 1:m
                    y = Float64.(Y_nsample[i])
                    V = [ones(ni, ni)]
                    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
                end

                # form VarLmmModel
                gcm = GLMCopulaVCModel(gcs);
                fittime = NaN
                # initialize_model!(gcm)
                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 25, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
                @show gcm.β
                @show gcm.Σ
                @show gcm.θ
                @show gcm.∇θ
                loglikelihood!(gcm, true, true)
                sandwich!(gcm)
                @show GLMCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
                @show mseβ
                @show mseΣ
                #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                # global currentind
                @views copyto!(βΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                ΣMseResults[currentind] = mseΣ
                fittimes[currentind] = fittime
                try
                    # glmm
                    # fit glmm
                    @info "Fit with MixedModels..."
                    # fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(); nAGQ = 7)
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(), contrasts = Dict(:group => Grouping()))
                    display(gm1)
                    @show gm1.β
                    # mse and time under glmm
                    @info "Get MSE under GLMM..."
                    level = 0.95
                    p = 3
                    @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    ΣMseResults_GLMM[currentind] = GLMM_mse[2]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                catch
                    println("random seed is $(123 * k + j), rep $j obs per person $ni samplesize $m ")
                    # βMseResults[currentind] = NaN
                    # ΣMseResults[currentind] = NaN
                    # fittimes[currentind] = NaN
                    # glmm
                    βMseResults_GLMM[currentind] = NaN
                    ΣMseResults_GLMM[currentind] = NaN
                    fittimes_GLMM[currentind] = NaN
                    currentind += 1
                end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_logistic_vcm$(nsims)reps_sim.csv"
    writedlm("bernoulli/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bernoulli/mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("bernoulli/fittimes_" * ftail, fittimes, ',')

    writedlm("bernoulli/beta_sigma_coverage_" * ftail, βΣcoverage, ',')

    # glmm
    writedlm("bernoulli/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("bernoulli/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    writedlm("bernoulli/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

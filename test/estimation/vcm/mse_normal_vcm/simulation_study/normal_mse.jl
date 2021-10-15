using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, Statistics
import StatsBase: sem

function normal_mse(samplesize, clustersize)
    p = 2    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    βtrue = ones(p)
    Σtrue = [0.5]

    # generate data
    intervals = zeros(p + m + 1, 2) #hold intervals
    curcoverage = zeros(p + m + 1) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    # #simulation parameters
    # samplesizes = [1000]
    # ns = [5]
    # # ns = [5; 10; 20; 50]
    nsims = 50
    samplesizes = [samplesize]
    ns = [clustersize]

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    τMseResults = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βτΣcoverage = Matrix{Float64}(undef, p + m + 1, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    τMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
    fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

    # solver = KNITRO.KnitroSolver(outlev=0)
    solver = Ipopt.IpoptSolver(print_level = 5)

    st = time()
    currentind = 1
    d = Normal()
    link = IdentityLink()
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
            Random.seed!(1234 * k)
            X = [ones(ni) randn(ni, p - 1)]
            μ = X * β
            σ2 = 0.1
            vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
            for i in 1:length(μ)
                vecd[i] = Normal(μ[i], σ2)
            end

            Γ = Σtrue[1] * ones(ni, ni)
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            for j in 1:nsims
                Random.seed!(1234 * k + j)
                @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, m)
                Ystack = vcat(Y_nsample...)
                @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = repeat(X, m)
                df = (Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], group = string.(groupstack))
                # df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
                form = @formula(Y ~ 1 + X2 + (1|group));

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
                # @show gcm.β
                # @show gcm.τ
                # fill!(gcm.Σ, 1.0)
                # update_Σ!(gcm)
                # @show gcm.Σ
                # try
                    fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
                    @show fittime
                    @show gcm.β
                    @show gcm.Σ
                    @show gcm.τ
                    @show gcm.θ
                    @show gcm.∇θ
                    loglikelihood!(gcm, true, true)
                    sandwich!(gcm)
                    @show GLMCopula.confint(gcm)
                    # mse and time under our model
                    coverage!(gcm, trueparams, intervals, curcoverage)
                    mseβ, mseτ, mseΣ = MSE(gcm, βtrue, 0.1, Σtrue)
                    @show mseβ
                    @show mseτ
                    @show mseΣ
                    #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                    # global currentind
                    @views copyto!(βτΣcoverage[:, currentind], curcoverage)
                    βMseResults[currentind] = mseβ
                    τMseResults[currentind] = mseτ
                    ΣMseResults[currentind] = mseΣ
                    fittimes[currentind] = fittime
                    # glmm
                    # fit glmm
                    @info "Fit with MixedModels..."
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Normal(), contrasts = Dict(:group => Grouping()))
                    @show fittime_GLMM
                    display(gm1)
                    @show gm1.β
                    # mse and time under glmm
                    @info "Get MSE under GLMM..."
                    level = 0.95
                    p = 3
                    @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, gm1.σ .- 0.1), sum(abs2, (gm1.σs[1][1]^2) .- Σtrue[1]) / 1]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    τMseResults_GLMM[currentind] = GLMM_mse[2]
                    ΣMseResults_GLMM[currentind] = GLMM_mse[3]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                # catch
                #     println("random seed is $(123 + k * j), rep $j obs per person $ni samplesize $m ")
                #     βMseResults[currentind] = NaN
                #     ΣMseResults[currentind] = NaN
                #     τMseResults[currentind] = NaN
                #     fittimes[currentind] = NaN
                #     # glmm
                #     βMseResults_GLMM[currentind] = NaN
                #     τMseResults_GLMM[currentind] = NaN
                #     ΣMseResults_GLMM[currentind] = NaN
                #     fittimes_GLMM[currentind] = NaN
                #     currentind += 1
                # end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "normal_vcm$(nsims)reps_sim$(samplesize)_$(clustersize).csv"
    writedlm("normal_vcm/mse_beta_multivariate_" * ftail, βMseResults, ',')
    writedlm("normal_vcm/mse_Sigma_multivariate_" * ftail, ΣMseResults, ',')
    writedlm("normal_vcm/fittimes_multivariate_" * ftail, fittimes, ',')

    writedlm("normal_vcm/beta_tau_sigma_coverage_" * ftail, βτΣcoverage, ',')

    # glmm
    writedlm("normal_vcm/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("normal_vcm/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    writedlm("normal_vcm/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
# samplesize = parse(Int, ARGS[1])
# clustersize = parse(Int, ARGS[2])

samplesize = 1000
clustersize = 2
normal_mse(samplesize, clustersize)
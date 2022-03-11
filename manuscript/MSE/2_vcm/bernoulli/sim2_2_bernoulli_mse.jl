using QuasiCopula, LinearAlgebra, GLM, DelimitedFiles, MixedModels
using Random
using DataFrames

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end

BLAS.set_num_threads(1)
Threads.nthreads()

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    Random.seed!(12345)
    # try next
    βtrue = rand(Uniform(-0.2, 0.2), p)
    θtrue = [0.05]

    # generate data
    intervals = zeros(p + m, 2) #hold intervals
    curcoverage = zeros(p + m) #hold current coverage resutls
    trueparams = [βtrue; θtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    θMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βθcoverage = Matrix{Float64}(undef, p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    θMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
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
            Γ = θtrue[1] * ones(ni, ni) + 0.00000000001 * Matrix(I, ni, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                # Ystack = vcat(Y_nsample...)
                # @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = []
                Ystack = []
                for i in 1:m
                    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    X = [ones(ni) randn(ni, p - 1)]
                    η = X * βtrue
                    V = [ones(ni, ni)]
                    # generate mvn response
                    mvn_d = MvNormal(η, Γ)
                    mvn_η = rand(mvn_d)
                    μ = GLM.linkinv.(link, mvn_η)
                    y = Float64.(rand.(__get_distribution.(D, μ)))
                    # add to data
                    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
                    push!(Xstack, X)
                    push!(Ystack, y)
                end

                # form VarLmmModel
                gcm = GLMCopulaVCModel(gcs);
                fittime = NaN

                # form glmm
                Xstack = [vcat(Xstack...)][1]
                Ystack = [vcat(Ystack...)][1]
                # p = 3
                df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
                form = @formula(Y ~ 1 + X2 + X3 + (1|group));

                fittime = @elapsed QuasiCopula.fit!(gcm)
                @show fittime
                @show gcm.β
                @show gcm.θ
                @show gcm.∇β
                @show gcm.∇θ
                loglikelihood!(gcm, true, true)
                vcov!(gcm)
                @show QuasiCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseθ = MSE(gcm, βtrue, θtrue)
                @show mseβ
                @show mseθ
                @views copyto!(βθcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                θMseResults[currentind] = mseθ
                fittimes[currentind] = fittime
                try
                    # glmm
                    # fit glmm
                    @info "Fit with MixedModels..."
                    # fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(); nAGQ = 25)
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(), contrasts = Dict(:group => Grouping()); nAGQ = 25)
                    @show fittime_GLMM
                    display(gm1)
                    @show gm1.β
                    # mse and time under glmm
                    @info "Get MSE under GLMM..."
                    level = 0.95
                    p = 3
                    @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, (gm1.θ.^2) .- θtrue[1]) / 1]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    θMseResults_GLMM[currentind] = GLMM_mse[2]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                catch
                    # println("random seed is $(1000000000 * t + 10000000 * j + 1000000 * k), rep $j obs per person $ni samplesize $m ")
                    # glmm
                    βMseResults_GLMM[currentind] = NaN
                    θMseResults_GLMM[currentind] = NaN
                    fittimes_GLMM[currentind] = NaN
                    currentind += 1
                    # continue
                end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_bernoulli_vcm$(nsims)reps_sim.csv"

    # make sure bernoulli_vc2_2 is a directory
    isdir("bernoulli_vc2_2") || mkdir("bernoulli_vc2_2")

    writedlm("bernoulli_vc2_2/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bernoulli_vc2_2/mse_theta_" * ftail, θMseResults, ',')
    writedlm("bernoulli_vc2_2/fittimes_" * ftail, fittimes, ',')

    writedlm("bernoulli_vc2_2/beta_theta_coverage_" * ftail, βθcoverage, ',')

#     # glmm
    writedlm("bernoulli_vc2_2/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("bernoulli_vc2_2/mse_theta_GLMM_" * ftail, θMseResults_GLMM, ',')
    writedlm("bernoulli_vc2_2/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

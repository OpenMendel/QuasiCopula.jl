using QuasiCopula, LinearAlgebra, DelimitedFiles, GLM, MixedModels
using Random
using DataFrames

BLAS.set_num_threads(1)
Threads.nthreads()

function runtest()
    p  = 3    # number of fixed effects, including intercept
    m  = 1    # number of variance components
    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-0.2, 0.2), p)
    θtrue = [0.1]

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
    d = Poisson()
    link = LogLink()
    D = typeof(d)
    Link = typeof(link)
    T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            Γ = θtrue[1] * ones(ni, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = []
                Ystack = []
                for i in 1:m
                    X = [ones(ni) randn(ni, p - 1)]
                    η = X * βtrue
                    μ = exp.(η)
                    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
                    for i in 1:ni
                        vecd[i] = Poisson(μ[i])
                    end
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
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
                # p = 2
                # df = (Y = Ystack, X2 = Xstack[:, 2], group = string.(groupstack))
                # form = @formula(Y ~ 1 + X2 + (1|group));

                    # fittime = @elapsed QuasiCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "exact"))
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
                    #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                    # global currentind
                    @views copyto!(βθcoverage[:, currentind], curcoverage)
                    βMseResults[currentind] = mseβ
                    θMseResults[currentind] = mseθ
                    fittimes[currentind] = fittime
                    # glmm
                    # fit glmm
                try
                    @info "Fit with MixedModels..."
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Poisson(), contrasts = Dict(:group => Grouping()); nAGQ = 25)
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
                    println("error occured with MixedModels.jl, random seed = $(1000000000 * t + 10000000 * j + 1000000 * k), rep $j obs per person $ni samplesize $m ")
#                     βMseResults[currentind] = NaN
#                     θMseResults[currentind] = NaN
#                     fittimes[currentind] = NaN
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
    ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"

    # make sure poisson_vc1 is a directory
    isdir("poisson_vc1") || mkdir("poisson_vc1")

    writedlm("poisson_vc1/mse_beta_" * ftail, βMseResults, ',')
    writedlm("poisson_vc1/mse_theta_" * ftail, θMseResults, ',')
    writedlm("poisson_vc1/fittimes_" * ftail, fittimes, ',')

    writedlm("poisson_vc1/beta_theta_coverage_" * ftail, βθcoverage, ',')

#     # glmm
    writedlm("poisson_vc1/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("poisson_vc1/mse_theta_GLMM_" * ftail, θMseResults_GLMM, ',')
    writedlm("poisson_vc1/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

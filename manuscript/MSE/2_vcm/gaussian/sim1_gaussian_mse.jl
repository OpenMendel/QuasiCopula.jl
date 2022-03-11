using QuasiCopula, LinearAlgebra, DelimitedFiles, MixedModels
using Random
using DataFrames

BLAS.set_num_threads(1)
Threads.nthreads()

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-2, 2), p)
    θtrue = [0.1]
    τtrue = 100.0
    σ2 = inv(τtrue)
    σ = sqrt(σ2)

    # generate data
    intervals = zeros(p + m + 1, 2) #hold intervals
    curcoverage = zeros(p + m + 1) #hold current coverage resutls
    trueparams = [βtrue; θtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

   #storage for our results
   βMseResults = ones(nsims * length(ns) * length(samplesizes))
   τMseResults = ones(nsims * length(ns) * length(samplesizes))
   θMseResults = ones(nsims * length(ns) *  length(samplesizes))
   βτθcoverage = Matrix{Float64}(undef, p + m + 1, nsims * length(ns) * length(samplesizes))
   fittimes = zeros(nsims * length(ns) * length(samplesizes))

   #storage for glmm results
   βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
   τMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
   θMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
   fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

   # solver = KNITRO.KnitroSolver(outlev=0)
   solver = Ipopt.IpoptSolver(print_level = 5)

   st = time()
   currentind = 1
   T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            Γ = θtrue[1] * ones(ni, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                # Ystack = vcat(Y_nsample...)
                # @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = []
                Ystack = []
                # df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))

                gcs = Vector{GaussianCopulaVCObs{T}}(undef, m)
                for i in 1:m
                    X = [ones(ni) randn(ni, p - 1)]
                    μ = X * βtrue
                    vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
                    for i in 1:length(μ)
                        vecd[i] = Normal(μ[i], σ)
                    end
                    Γ = θtrue[1] * ones(ni, ni)
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
                    gcs[i] = GaussianCopulaVCObs(y, X, V)
                    push!(Xstack, X)
                    push!(Ystack, y)
                end

                # form NBCopulaVCModel
                gcm = GaussianCopulaVCModel(gcs);
                fittime = NaN

                # form glmm
                Xstack = [vcat(Xstack...)][1]
                Ystack = [vcat(Ystack...)][1]
                # p = 3
                df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
                form = @formula(Y ~ 1 + X2 + X3 + (1|group));
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show fittime
                    @show gcm.β
                    @show gcm.θ
                    @show gcm.τ
                    @show gcm.∇β
                    @show gcm.∇θ
                    @show gcm.∇τ
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm)
                    @show QuasiCopula.confint(gcm)
                    # mse and time under our model
                    coverage!(gcm, trueparams, intervals, curcoverage)
                    mseβ, mseτ, mseθ = MSE(gcm, βtrue, τtrue[1], θtrue)
                    @show mseβ
                    @show mseτ
                    @show mseθ
                    #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                    # global currentind
                    @views copyto!(βτθcoverage[:, currentind], curcoverage)
                    βMseResults[currentind] = mseβ
                    τMseResults[currentind] = mseτ
                    θMseResults[currentind] = mseθ
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
                    # @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, gm1.σ .- σ), sum(abs2, (gm1.σs[1][1]^2 / gm1.σ^2) - θtrue[1])]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    τMseResults_GLMM[currentind] = GLMM_mse[2]
                    θMseResults_GLMM[currentind] = GLMM_mse[3]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                catch
                    # println("random seed is $(1000000000 * t + 10000000 * j + 1000000 * k ), rep $j obs per person $ni samplesize $m ")
                    # ours
                    βMseResults[currentind] = NaN
                    τMseResults[currentind] = NaN
                    θMseResults[currentind] = NaN
                    fittimes[currentind] = NaN
                    @views copyto!(βτθcoverage[:, currentind], NaN)
                    # glmm
                    βMseResults_GLMM[currentind] = NaN
                    τMseResults[currentind] = NaN
                    θMseResults_GLMM[currentind] = NaN
                    fittimes_GLMM[currentind] = NaN
                    currentind += 1
                end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "normal$(nsims)reps_sim.csv"

    # make sure bernoulli_vc1 is a directory
    isdir("gaussian_vc1") || mkdir("gaussian_vc1")

    writedlm("gaussian_vc1/mse_beta_multivariate_" * ftail, βMseResults, ',')
    writedlm("gaussian_vc1/mse_theta_multivariate_" * ftail, θMseResults, ',')
    writedlm("gaussian_vc1/mse_tau_multivariate_" * ftail, τMseResults, ',')
    writedlm("gaussian_vc1/fittimes_multivariate_" * ftail, fittimes, ',')

    writedlm("gaussian_vc1/beta_theta_tau_coverage_" * ftail, βτθcoverage, ',')

    # glmm
    writedlm("gaussian_vc1/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("gaussian_vc1/mse_theta_GLMM_" * ftail, θMseResults_GLMM, ',')
    writedlm("gaussian_vc1/mse_tau_GLMM_" * ftail, τMseResults_GLMM, ',')
    writedlm("gaussian_vc1/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

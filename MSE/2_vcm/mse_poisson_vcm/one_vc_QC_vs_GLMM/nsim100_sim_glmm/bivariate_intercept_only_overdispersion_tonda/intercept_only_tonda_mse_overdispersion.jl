using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end

function runtest()
    p = 1    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    βtrue = [log(5.0)]
    Σtrue = [0.05]

    # generate data
    intervals = zeros(p + m, 2) #hold intervals
    curcoverage = zeros(p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    # samplesizes = [10000]
    samplesizes = [100; 1000; 10000]
    ns = [2]
    # ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

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
            Γ = Σtrue[1] *  Matrix(I, ni, ni)
            # Γ = Σtrue[1] * ones(ni, ni) + 0.000000000000001 * Matrix(I, ni, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                # Ystack = vcat(Y_nsample...)
                # @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Ystack = []
                for i in 1:m
                    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    X = ones(ni, 1)
                    η = X * βtrue
                    # V = [ones(ni, ni)]
                    V = [Float64.(Matrix(I, ni, ni))]
                    # generate mvn response
                    mvn_d = MvNormal(η, Γ)
                    mvn_η = rand(mvn_d)
                    μ = GLM.linkinv.(link, mvn_η)
                    y = Float64.(rand.(__get_distribution.(D, μ)))
                    # add to data
                    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
                    push!(Ystack, y)
                end

                # form VarLmmModel
                gcm = GLMCopulaVCModel(gcs);
                fittime = NaN

                # form glmm
                Ystack = [vcat(Ystack...)][1]
                # p = 1
                df = (Y = Ystack, group = string.(groupstack))
                form = @formula(Y ~ 1 + (1|group));

                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-7, limited_memory_max_history = 20, accept_after_max_steps = 5, hessian_approximation = "limited-memory"))
                @show fittime
                @show gcm.β
                @show gcm.Σ
                @show gcm.∇β
                @show gcm.∇Σ
                loglikelihood!(gcm, true, true)
                vcov!(gcm)
                @show GLMCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
                @show mseβ
                @show mseΣ
                @views copyto!(βΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                ΣMseResults[currentind] = mseΣ
                fittimes[currentind] = fittime
                currentind += 1
               #  try
               #      # glmm
               #      # fit glmm
               #      @info "Fit with MixedModels..."
               #      # fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(); nAGQ = 25)
               #      fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Poisson(), contrasts = Dict(:group => Grouping()); nAGQ = 25)
               #      @show fittime_GLMM
               #      display(gm1)
               #      @show gm1.β
               #      # mse and time under glmm
               #      @info "Get MSE under GLMM..."
               #      level = 0.95
               #      p = 1
               #      @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
               #      @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]
               #      # glmm
               #      βMseResults_GLMM[currentind] = GLMM_mse[1]
               #      ΣMseResults_GLMM[currentind] = GLMM_mse[2]
               #      fittimes_GLMM[currentind] = fittime_GLMM
               #      currentind += 1
               #  catch
               #      # println("random seed is $(1000000000 * t + 10000000 * j + 1000000 * k), rep $j obs per person $ni samplesize $m ")
               #      # glmm
               #      βMseResults_GLMM[currentind] = NaN
               #      ΣMseResults_GLMM[currentind] = NaN
               #      fittimes_GLMM[currentind] = NaN
               #      currentind += 1
               #      # continue
               # end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
    writedlm("sim_glmm_vs_ours_overdispersion/mse_beta_" * ftail, βMseResults, ',')
    writedlm("sim_glmm_vs_ours_overdispersion/mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("sim_glmm_vs_ours_overdispersion/fittimes_" * ftail, fittimes, ',')

    writedlm("sim_glmm_vs_ours_overdispersion/beta_sigma_coverage_" * ftail, βΣcoverage, ',')

#     # glmm
    # writedlm("sim_glmm_vs_ours_random_int/poisson_sim_glmm/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    # writedlm("sim_glmm_vs_ours_random_int/poisson_sim_glmm/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    # writedlm("sim_glmm_vs_ours_random_int/poisson_sim_glmm/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()
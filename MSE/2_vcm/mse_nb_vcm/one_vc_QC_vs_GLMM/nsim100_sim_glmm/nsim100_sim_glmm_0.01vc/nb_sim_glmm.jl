using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem
import GLMCopula.initialize_model!

function __get_distribution(dist::Type{D}, μ) where D <: UnivariateDistribution
    return dist(μ)
end

function __get_distribution(dist::Type{D}, μ, r) where D <: UnivariateDistribution
    return dist(r, μ)
end

function initialize_model!(
    gcm::NBCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}

    # initial guess for r = 1
    fill!(gcm.r, 1)
    fill!(gcm.τ, 1)
    fill!(gcm.β, 0)

    println("initializing variance components using MM-Algorithm")
    fill!(gcm.Σ, 1)
    update_Σ!(gcm)
    nothing
end

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    Random.seed!(1234)
    βtrue = rand(Uniform(-0.2, 0.2), p)
    Σtrue = [0.01]
    rtrue = 10.0

    # generate data
    intervals = zeros(p + m + 1, 2) #hold intervals
    curcoverage = zeros(p + m + 1) #hold current coverage resutls
    trueparams = [βtrue; rtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

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
            Γ = Σtrue[1] * ones(ni, ni) + 0.00000000001 * Matrix(I, ni, ni)
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
                    p_nb = rtrue ./ (μ .+ rtrue)
                    y = Float64.(rand.(__get_distribution.(D, p_nb, rtrue)))
                    # add to data
                    gcs[i] = NBCopulaVCObs(y, X, V, d, link)
                    push!(Xstack, X)
                    push!(Ystack, y)
                end

                # form NBCopulaVCModel
                gcm = NBCopulaVCModel(gcs);
                fittime = NaN

                # form glmm
                Xstack = [vcat(Xstack...)][1]
                Ystack = [vcat(Ystack...)][1]
                # p = 3
                df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
                form = @formula(Y ~ 1 + X2 + X3 + (1|group));

                fittime = @elapsed GLMCopula.fit!(gcm, maxBlockIter=30, tol=1e-8)
                @show fittime
                @show gcm.β
                @show gcm.Σ
                @show gcm.r
                loglikelihood!(gcm, true, true)
                vcov!(gcm)
                @show GLMCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mser, mseΣ = MSE(gcm, βtrue, rtrue, Σtrue)
                @show mseβ
                @show mser
                @show mseΣ
                #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                # global currentind
                @views copyto!(βrΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                ΣMseResults[currentind] = mseΣ
                rMseResults[currentind] = mser
                fittimes[currentind] = fittime
                try
                    # glmm
                    # fit glmm
                    @info "Fit with MixedModels..."
                    # fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Bernoulli(); nAGQ = 25)
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, NegativeBinomial(rtrue), LogLink(), contrasts = Dict(:group => Grouping()); nAGQ = 25)
                    @show fittime_GLMM
                    display(gm1)
                    @show gm1.β
                    # mse and time under glmm
                    @info "Get MSE under GLMM..."
                    level = 0.95
                    @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, inv(gm1.σ^2) - rtrue), sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    rMseResults_GLMM[currentind] = GLMM_mse[2]
                    ΣMseResults_GLMM[currentind] = GLMM_mse[3]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                catch
                    println("random seed is $(1000000000 * t + 10000000 * j + 1000000 * k ), rep $j obs per person $ni samplesize $m ")
                    # glmm
                    βMseResults_GLMM[currentind] = NaN
                    rMseResults_GLMM[currentind] = NaN
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
    ftail = "multivariate_NB_vcm$(nsims)reps_sim.csv"
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_beta_" * ftail, βMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_r_" * ftail, rMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/fittimes_" * ftail, fittimes, ',')

    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/beta_r_sigma_coverage_" * ftail, βrΣcoverage, ',')

    # glmm
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/mse_r_GLMM" * ftail, rMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/nb_sim_glmm/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

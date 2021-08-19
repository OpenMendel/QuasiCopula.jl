using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics, RCall, Printf
import StatsBase: sem

# run 100 simulations, starting at nsims (e.g. if nsims = 1, we run 1:100 for nsims)
function run_simulations(
    p::Int, # number of fixed effects, including intercept
    m::Int, # number of variance components
    samplesize::Int,
    ni::Int, # number of observations per individual
    nsims::Int # number of repeats
    )
    # make new directory for current simulation
    dir = "n$samplesize.ni$ni.sims$nsims"
    isdir(dir) ? (return nothing) : mkdir(dir)
    cd(dir)

    # true parameter values
    βtrue = ones(p)
    Σtrue = [0.1]

    # generate data
    intervals = zeros(p + m, 2) #hold intervals
    curcoverage = zeros(p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    # storage for results (nsims simulations!)
    βMseResults = ones(nsims)
    ΣMseResults = ones(nsims)
    βΣcoverage = Matrix{Float64}(undef, p + m, nsims)
    fittimes = zeros(nsims)

    #storage for glmm results
    βMseResults_GLMM = ones(nsims)
    ΣMseResults_GLMM = ones(nsims)
    fittimes_GLMM = zeros(nsims)

    # solver = KNITRO.KnitroSolver(outlev=0)
    solver = Ipopt.IpoptSolver(print_level = 5)

    st = time()
    d = Poisson()
    link = LogLink()
    D = typeof(d)
    Link = typeof(link)
    T = Float64

    y = Vector{Float64}(undef, ni)
    res = Vector{Float64}(undef, ni)
    β = ones(p)
    X = [ones(ni) randn(ni, p - 1)]
    η = X * β
    μ = exp.(η)
    vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

    for i in 1:length(μ)
        vecd[i] = Poisson(μ[i])
    end

    Γ = Σtrue[1] * ones(ni, ni)
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

    for j in 1:nsims
        println("rep $j obs per person $ni samplesize $samplesize")
        Random.seed!(j + nsims)
        @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, samplesize)
        # for glmm fit we need the dataframe stacking over the samplesize
        Ystack = vcat(Y_nsample...)
        a = collect(1:samplesize)
        group = [repeat([a[i]], ni) for i in 1:samplesize]
        groupstack = vcat(group...)
        Xstack = repeat(X, samplesize)
        df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
        form = @formula(Y ~ 1 + X2 + X3 + (1|group));
        gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, samplesize)
        for i in 1:samplesize
            y = Float64.(Y_nsample[i])
            V = [ones(ni, ni)]
            gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
        end

        # form VarLmmModel
        gcm = GLMCopulaVCModel(gcs)
        fittime = NaN
        initialize_model!(gcm)
        @show gcm.β
        @show gcm.Σ
        try
            # quasi copula fit
            fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 1, max_iter = 150, tol = 10^-5, hessian_approximation = "limited-memory"))
            @show gcm.θ
            @show gcm.∇θ
            loglikelihood!(gcm, true, true)
            sandwich!(gcm)
            @show GLMCopula.confint(gcm)
            # glmm fit
            @info "Fit with MixedModels..."
            fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Poisson())
            display(gm1)
            @show gm1.β
            # mse and time under glmm
            @info "Get MSE under GLMM..."
            level = 0.95
            p = 3
            @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
            @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, (gm1.θ.^2) .- Σtrue[1]) / 1]
            # glmm
            βMseResults_GLMM[j] = GLMM_mse[1]
            ΣMseResults_GLMM[j] = GLMM_mse[2]
            fittimes_GLMM[j] = fittime_GLMM
        catch
            println("rep $j ni obs = $ni , samplesize = $samplesize had an error")
            βMseResults[j] = NaN
            ΣMseResults[j] = NaN
            βΣcoverage[:, j] .= NaN
            fittimes[j] = NaN

            βMseResults_GLMM[j] = NaN
            ΣMseResults_GLMM[j] = NaN
            fittimes_GLMM[j] = NaN
            continue
        end
        coverage!(gcm, trueparams, intervals, curcoverage)
        mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
        @show mseβ
        @show mseΣ
        #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
        @views copyto!(βΣcoverage[:, j], curcoverage)
        βMseResults[j] = mseβ
        ΣMseResults[j] = mseΣ
        fittimes[j] = fittime
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
    writedlm("mse_beta_" * ftail, βMseResults, ',')
    writedlm("mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("fittimes" * ftail, fittimes, ',')
    writedlm("beta_sigma_coverage_5betas_" * ftail, βΣcoverage, ',')

    # glmm
    writedlm("mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    writedlm("fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
    # return to previous folder
    cd("../")
end

p = parse(Int, ARGS[1])
m = parse(Int, ARGS[2])
samplesize = parse(Int, ARGS[3])
ni = parse(Int, ARGS[4])
nsims = parse(Int, ARGS[5])

run_simulations(p, m, samplesize, ni, nsims)

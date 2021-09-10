using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, StatsFuns, Distributions, DataFrames
function run_test()
    p = 3    # number of fixed effects, including intercept
    m = 2    # number of variance components
    # true parameter values
    βtrue = ones(p)
    Σtrue = [0.1; 0.1]

    # generate data
    intervals = zeros(p + m, 2) #hold intervals
    curcoverage = zeros(p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [1000; 10000; 50000]
    ns = [5; 10; 20; 50]
    nsims = 5

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βΣcoverage = Matrix{Float64}(undef, p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

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
            X = [ones(ni) randn(ni, p - 1)]
            η = X * βtrue
            μ = exp.(η) ./ (1 .+ exp.(η))
            vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))
        
            for i in 1:length(μ)
            vecd[i] = Bernoulli(μ[i])
            end

            Γ = Σtrue[1] * ones(ni, ni) + Σtrue[2] * Matrix(I, ni, ni)
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, m)
                gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
                for i in 1:m
                    y = Float64.(Y_nsample[i])
                    V = [ones(ni, ni), Matrix(I, ni, ni)]
                    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
                end
                
                # form model
                gcm = GLMCopulaVCModel(gcs);
                fittime = NaN
                initialize_model!(gcm)
                @show gcm.β
                @show gcm.Σ
                loglikelihood!(gcm, true, false)
                try 
                    fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5,  max_iter = 100, tol = 10^-5, limited_memory_max_history = 25, hessian_approximation = "limited-memory"))
                    @show gcm.θ
                    @show gcm.∇θ
                    loglikelihood!(gcm, true, true)
                    sandwich!(gcm)
                    @show GLMCopula.confint(gcm)
                catch
                    println("rep $j ni obs = $ni , samplesize = $m had an error")
                    βMseResults[currentind] = NaN
                    ΣMseResults[currentind] = NaN
                    βΣcoverage[:, currentind] .= NaN
                    fittimes[currentind] = NaN

                    currentind += 1
                    continue
                end

                # mse and time under our model    
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
                @show mseβ
                @show mseΣ
                #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
                global currentind
                @views copyto!(βΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                ΣMseResults[currentind] = mseΣ
                fittimes[currentind] = fittime
                    
                currentind += 1
            end
        end
    end 
    en = time()

    using Random, DataFrames, DelimitedFiles, Statistics, RCall, Printf
    import StatsBase: sem

    @show en - st #seconds 
    @info "writing to file..."
    ftail = "multivariate_logistic_vcm$(nsims)reps_sim.csv"
    writedlm("mse_beta_" * ftail, βMseResults, ',')
    writedlm("mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("fittimes_" * ftail, fittimes, ',')
    writedlm("beta_sigma_coverage_" * ftail, βΣcoverage, ',')
end
run_test()
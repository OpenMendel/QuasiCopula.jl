using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    #   βtrue = ones(p)
    Random.seed!(12345)
    # β1true = rand(Uniform(-0.2, 0.2), p)
    β1true = rand(p)
    Random.seed!(1234)
    β2true = rand(Uniform(-0.2, 0.2), p)
    βtrue = [β1true; β2true]
    Σtrue = [0.1]

    # generate data
    intervals = zeros(2 * p + m, 2) #hold intervals
    curcoverage = zeros(2 * p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2]
    nsims = 5

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βΣcoverage = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
    fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

    # solver = KNITRO.KnitroSolver(outlev=0)
    solver = Ipopt.IpoptSolver(print_level = 5)

    st = time()
    currentind = 1
    d1 = Poisson()
    d2 = Bernoulli()
    vecdist = [d1, d2]

    link1 = LogLink()
    link2 = LogitLink()
    veclink = [link1, link2]

    T = Float64
    VD = typeof(vecdist)
    VL = typeof(veclink)

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{Poisson_Bernoulli_VCObs{T, VD, VL}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            Γ = Σtrue[1] * ones(ni, ni)
            vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                for i in 1:m
                    xi = [1.0 randn(p - 1)...]
                    X = [xi zeros(size(xi)); zeros(size(xi)) xi]
                    η = X * βtrue
                    μ = zeros(ni)
                    for j in 1:ni
                        μ[j] = GLM.linkinv(veclink[j], η[j])
                        vecd[j] = typeof(vecdist[j])(μ[j])
                    end
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
                    gcs[i] = Poisson_Bernoulli_VCObs(y, X, V, vecdist, veclink)
                end

                gcm = Poisson_Bernoulli_VCModel(gcs);


                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 12, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
                @show fittime
                @show gcm.β
                @show gcm.Σ
                @show gcm.θ
                @show gcm.∇θ
                loglikelihood!(gcm, true, true)
                vcov!(gcm)
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
                currentind += 1
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "bivariate_mixed_poisson_bernoulli_vcm$(nsims)reps_sim.csv"
    writedlm("biv_mixed_poisson_bernoulli/mse_beta_" * ftail, βMseResults, ',')
    writedlm("biv_mixed_poisson_bernoulli/mse_Sigma_" * ftail, ΣMseResults, ',')
    writedlm("biv_mixed_poisson_bernoulli/fittimes_" * ftail, fittimes, ',')

    writedlm("biv_mixed_poisson_bernoulli/beta_sigma_coverage_" * ftail, βΣcoverage, ',')
end
runtest()

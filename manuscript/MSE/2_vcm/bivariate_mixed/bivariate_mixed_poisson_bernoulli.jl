using QuasiCopula, LinearAlgebra, GLM, DelimitedFiles
using Random
using DataFrames

BLAS.set_num_threads(1)
Threads.nthreads()

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-0.2, 0.2), 2 * p)
    θtrue = [0.5]

    # generate data
    intervals = zeros(2 * p + m, 2) #hold intervals
    curcoverage = zeros(2 * p + m) #hold current coverage resutls
    trueparams = [βtrue; θtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    θMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βθcoverage = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
    θMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
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
            Γ = θtrue[1] * ones(ni, ni)
            vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Xstack = [ones(m) randn(m, p - 1)]
                for i in 1:m
                    xi = Xstack[i, :]
                    X = [transpose(xi) zeros(size(transpose(xi))); zeros(size(transpose(xi))) transpose(xi)]
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
                    gcs[i] = Poisson_Bernoulli_VCObs(y, xi, V, vecdist, veclink)
                end

                gcm = Poisson_Bernoulli_VCModel(gcs);


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
                currentind += 1
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "bivariate_mixed_poisson_bernoulli_vcm$(nsims)reps_sim.csv"

    # make sure bivariate_mixed_vc is a directory
    isdir("bivariate_mixed_vc") || mkdir("bivariate_mixed_vc")

    writedlm("bivariate_mixed_vc/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bivariate_mixed_vc/mse_theta_" * ftail, θMseResults, ',')
    writedlm("bivariate_mixed_vc/fittimes_" * ftail, fittimes, ',')

    writedlm("bivariate_mixed_vc/beta_theta_coverage_" * ftail, βθcoverage, ',')
end
runtest()

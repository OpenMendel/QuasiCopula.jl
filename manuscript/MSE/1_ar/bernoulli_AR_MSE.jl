using QuasiCopula, LinearAlgebra, GLM, DelimitedFiles
using Random, ToeplitzMatrices
using DataFrames

BLAS.set_num_threads(1)
Threads.nthreads()

function run_test()
    p = 3    # number of fixed effects, including intercept

    # true parameter values
    Random.seed!(12345)
    # try next
    βtrue = rand(Uniform(-2, 2), p)
    σ2true = [0.5]
    ρtrue = [0.5]

    function get_V(ρ, n)
        vec = zeros(n)
        vec[1] = 1.0
        for i in 2:n
            vec[i] = vec[i - 1] * ρ
        end
        V = ToeplitzMatrices.SymmetricToeplitz(vec)
        V
    end

    # generate data
    intervals = zeros(p + 2, 2) #hold intervals
    curcoverage = zeros(p + 2) #hold current coverage resutls
    trueparams = [βtrue; ρtrue; σ2true] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    σ2MseResults = ones(nsims * length(ns) * length(samplesizes))
    ρMseResults = ones(nsims * length(ns) * length(samplesizes))
    βρσ2coverage = Matrix{Float64}(undef, p + 2, nsims * length(ns) * length(samplesizes))
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
        gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)

            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                for i in 1:m
                    X = [ones(ni) randn(ni, p - 1)]
                    η = X * βtrue
                    μ = exp.(η) ./ (1 .+ exp.(η))
                    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
                    for i in 1:ni
                        vecd[i] = Bernoulli(μ[i])
                    end
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
                    gcs[i] = GLMCopulaARObs(y, X, d, link)
                end

                # form model
                gcm = GLMCopulaARModel(gcs);
                 try
                    #fittime = @elapsed QuasiCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, warm_start_init_point="yes", tol = 10^-8, limited_memory_max_history = 20, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
                    fittime = @elapsed QuasiCopula.fit!(gcm) #, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, warm_start_init_point="yes", accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
                    # fittime = @elapsed QuasiCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
                    @show fittime
                    @show gcm.β
                    @show gcm.σ2
                    @show gcm.ρ
                    @show gcm.∇β
                    @show gcm.∇σ2
                    @show gcm.∇ρ
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm)
                    @show QuasiCopula.confint(gcm)

                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseρ, mseσ2 = MSE(gcm, βtrue, ρtrue, σ2true)
                @show mseβ
                @show mseσ2
                @show mseρ
                # global currentind
                @views copyto!(βρσ2coverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                σ2MseResults[currentind] = mseσ2
                ρMseResults[currentind] = mseρ
                fittimes[currentind] = fittime
                currentind += 1

                catch
                    println("rep $j ni obs = $ni , samplesize = $m had an error")
                    βMseResults[currentind] = NaN
                    σ2MseResults[currentind] = NaN
                    ρMseResults[currentind] = NaN
                    βρσ2coverage[:, currentind] .= NaN
                    fittimes[currentind] = NaN
                    currentind += 1
                 end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_bernoulli_AR$(nsims)reps_sim.csv"

    # make sure bernoulli_ar is a directory
    isdir("bernoulli_ar") || mkdir("bernoulli_ar")

    writedlm("bernoulli_ar/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bernoulli_ar/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("bernoulli_ar/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("bernoulli_ar/fittimes_" * ftail, fittimes, ',')

    writedlm("bernoulli_ar/beta_rho_sigma_coverage_" * ftail, βρσ2coverage, ',')
end

run_test()

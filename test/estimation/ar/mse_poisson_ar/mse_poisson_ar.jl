using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, StatsFuns, Distributions, DataFrames, ToeplitzMatrices

function run_test()
    p = 3    # number of fixed effects, including intercept

    # true parameter values
    βtrue = ones(p)
    σ2true = [0.1]
    ρtrue = [0.9]

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
    trueparams = [βtrue; σ2true; ρtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 50

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    σ2MseResults = ones(nsims * length(ns) * length(samplesizes))
    ρMseResults = ones(nsims * length(ns) * length(samplesizes))
    βρσ2coverage = Matrix{Float64}(undef, p + 2, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

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
        gcs = Vector{GLMCopulaARObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)

            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Y_nsample = []
                for i in 1:m
                    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
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
                    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
                    gcs[i] = GLMCopulaARObs(y, X, d, link)
                    push!(Y_nsample, y)
                end

                # form model
                gcm = GLMCopulaARModel(gcs);
                fittime = NaN
                initialize_model!(gcm)
                @show gcm.β
                @show gcm.ρ
                @show gcm.σ2

                ### now sigma2 is initialized now we need rho
                Y_1 = [Y_nsample[i][1] for i in 1:m]
                Y_2 = [Y_nsample[i][2] for i in 1:m]

                update_rho!(gcm, Y_1, Y_2)
                @show gcm.ρ
                @show gcm.σ2
                try
                    fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, hessian_approximation = "limited-memory"))
                    # fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
                    @show fittime
                    @show gcm.θ
                    @show gcm.∇θ
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm) 
                    @show GLMCopula.confint(gcm)
                    # mse and time under our model
                    coverage!(gcm, trueparams, intervals, curcoverage)
                    mseβ, mseσ2, mseρ = MSE(gcm, βtrue, ρtrue, σ2true)
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
                    println("random seed is $(1000000000 * t + 10000000 * j + 1000000 * k), rep $j obs per person $ni samplesize $m ")
                    βMseResults[currentind] = NaN
                    σ2MseResults[currentind] = NaN
                    ρMseResults[currentind] = NaN
                    fittimes[currentind] = NaN
                    currentind += 1
               end

            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_poisson_AR$(nsims)reps_sim.csv"
    writedlm("poisson_ar/mse_beta_" * ftail, βMseResults, ',')
    writedlm("poisson_ar/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("poisson_ar/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("poisson_ar/fittimes_" * ftail, fittimes, ',')

    writedlm("poisson_ar/beta_sigma_coverage_" * ftail, βρσ2coverage, ',')
end

run_test()
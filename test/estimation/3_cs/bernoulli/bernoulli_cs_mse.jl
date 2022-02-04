using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions, StatsFuns, Distributions, DataFrames, ToeplitzMatrices

function run_test()
    p = 3   # number of fixed effects, including intercept

    # true parameter values
    Random.seed!(1234)
    βtrue = rand(Uniform(-0.2, 0.2), p)
    # σ2true = [0.5]
    # ρtrue = [0.9]
    # βtrue = 0.1 * ones(p)
    σ2true = [0.2]
    ρtrue = [0.5]

    function get_V(ρ, n)
        vec = zeros(n)
        vec[1] = 1.0
        for i in 2:n
            vec[i] = ρ
        end
        V = ToeplitzMatrices.SymmetricToeplitz(vec)
        V
    end
    # generate data
    intervals = zeros(p + 2, 2) #hold intervals
    curcoverage = zeros(p + 2) #hold current coverage resutls
    trueparams = [βtrue; ρtrue; σ2true] #hold true parameters

    #simulation parameters
    samplesize = [10000]
    # samplesizes = [100; 1000; 10000]
    # ns = [2; 5; 10; 15; 20; 25]
    ns = [5]
    nsims = 1
    # nsims = 100

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
        gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)

            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Ystack = []
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
                    push!(Ystack, y)
                    V = [ones(ni, ni)]
                    gcs[i] = GLMCopulaCSObs(y, X, d, link)
                end

                # form model
                gcm = GLMCopulaCSModel(gcs);
                fittime = NaN
                initialize_model!(gcm)
                @show gcm.β
                @show gcm.ρ
                @show gcm.σ2

                try
                    # fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, limited_memory_max_history = 20, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
                    fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-8, accept_after_max_steps = 2, limited_memory_max_history = 20, warm_start_init_point="yes", mu_strategy = "adaptive", mu_oracle = "probing", hessian_approximation = "limited-memory"))
                    @show fittime
                    @show gcm.θ
                    @show gcm.∇θ
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm)
                    @show GLMCopula.confint(gcm)
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
    ftail = "multivariate_poisson_CS$(nsims)reps_sim.csv"
    writedlm("mse_poisson_cs/mse_beta_" * ftail, βMseResults, ',')
    writedlm("mse_poisson_cs/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("mse_poisson_cs/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("mse_poisson_cs/fittimes_" * ftail, fittimes, ',')

    writedlm("mse_poisson_cs/beta_sigma_coverage_" * ftail, βρσ2coverage, ',')
end

run_test()

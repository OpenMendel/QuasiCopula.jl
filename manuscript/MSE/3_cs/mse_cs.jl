using QuasiCopula, LinearAlgebra, GLM, DelimitedFiles
using Random, ToeplitzMatrices
using DataFrames

BLAS.set_num_threads(1)
Threads.nthreads()

function run_test_pcs()
    p = 3    # number of fixed effects, including intercept

    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-2, 2), p)
    σ2true = [0.5]
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
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    σ2MseResults = ones(nsims * length(ns) * length(samplesizes))
    ρMseResults = ones(nsims * length(ns) * length(samplesizes))
    βρσ2coverage = Matrix{Float64}(undef, p + 2, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    st = time()
    currentind = 1
    d = Poisson()
    link = LogLink()
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
                Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k)
                X_samplesize = [randn(ni, p - 1) for i in 1:m]
                for i in 1:m
                    X = [ones(ni) X_samplesize[i]]
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
                    rand(nonmixed_multivariate_dist, y, res)
                    V = [ones(ni, ni)]
                    gcs[i] = GLMCopulaCSObs(y, X, d, link)
                end

                # form model
                gcm = GLMCopulaCSModel(gcs);
                fittime = NaN
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show fittime
                    @show gcm.β
                    @show gcm.ρ
                    @show gcm.σ2
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

    # make sure poisson_cs is a directory
    isdir("poisson_cs") || mkdir("poisson_cs")

    writedlm("poisson_cs/mse_beta_" * ftail, βMseResults, ',')
    writedlm("poisson_cs/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("poisson_cs/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("poisson_cs/fittimes_" * ftail, fittimes, ',')

    writedlm("poisson_cs/beta_rho_sigma_coverage_" * ftail, βρσ2coverage, ',')
end

run_test_pcs()

function run_test_bcs()
    p = 3   # number of fixed effects, including intercept

    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-2, 2), p)
    σ2true = [0.5]
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
        gcs = Vector{GLMCopulaCSObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)

            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Ystack = []
                Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k)
                X_samplesize = [randn(ni, p - 1) for i in 1:m]
                for i in 1:m
                    X = [ones(ni) X_samplesize[i]]
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
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show fittime
                    @show gcm.β
                    @show gcm.ρ
                    @show gcm.σ2
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
    ftail = "multivariate_bernoulli_CS$(nsims)reps_sim.csv"
    # make sure bernoulli_cs is a directory
    isdir("bernoulli_cs") || mkdir("bernoulli_cs")

    writedlm("bernoulli_cs/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bernoulli_cs/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("bernoulli_cs/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("bernoulli_cs/fittimes_" * ftail, fittimes, ',')

    writedlm("bernoulli_cs/beta_rho_sigma_coverage_" * ftail, βρσ2coverage, ',')
end

run_test_bcs()

function run_test_gcs()
    p = 3    # number of fixed effects, including intercept

    # true parameter values
    Random.seed!(12345)
    # try next
    βtrue = rand(Uniform(-2, 2), p)
    σ2true = [0.5]
    ρtrue = [0.5]
    τtrue = 100.0
    σ2 = inv(τtrue)
    σ = sqrt(inv(τtrue))

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
    intervals = zeros(p + 3, 2) #hold intervals
    curcoverage = zeros(p + 3) #hold current coverage resutls
    trueparams = [βtrue; ρtrue; σ2true; τtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    σ2MseResults = ones(nsims * length(ns) * length(samplesizes))
    ρMseResults = ones(nsims * length(ns) * length(samplesizes))
    τMseResults = ones(nsims * length(ns) * length(samplesizes))

    βρσ2τcoverage = Matrix{Float64}(undef, p + 3, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    solver = Ipopt.IpoptSolver(print_level = 5)

    st = time()
    currentind = 1
    T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{GaussianCopulaCSObs{T}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)
            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k)
                X_samplesize = [randn(ni, p - 1) for i in 1:m]
                for i in 1:m
                    X = [ones(ni) X_samplesize[i]]
                    μ = X * βtrue
                    vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
                    for i in 1:length(μ)
                        vecd[i] = Normal(μ[i], σ)
                    end
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    gcs[i] = GaussianCopulaCSObs(y, X)
                end

                # form model
                gcm = GaussianCopulaCSModel(gcs);
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show gcm.β
                    @show gcm.σ2
                    @show gcm.ρ
                    @show gcm.∇β
                    @show gcm.∇σ2
                    @show gcm.∇ρ
                    @show gcm.τ
                    @show gcm.∇τ
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm)
                    @show QuasiCopula.confint(gcm)

                    # mse and time under our model
                    coverage!(gcm, trueparams, intervals, curcoverage)
                    mseβ, mseρ, mseσ2, mseτ = MSE(gcm, βtrue, τtrue, ρtrue, σ2true)
                    @show mseβ
                    @show mseτ
                    @show mseσ2
                    @show mseρ
                    # global currentind
                    @views copyto!(βρσ2τcoverage[:, currentind], curcoverage)
                    βMseResults[currentind] = mseβ
                    τMseResults[currentind] = mseτ
                    σ2MseResults[currentind] = mseσ2
                    ρMseResults[currentind] = mseρ
                    fittimes[currentind] = fittime
                    currentind += 1

                catch
                    println("rep $j ni obs = $ni , samplesize = $m had an error")
                    βMseResults[currentind] = NaN
                    τMseResults[currentind] = NaN
                    σ2MseResults[currentind] = NaN
                    ρMseResults[currentind] = NaN
                    βρσ2τcoverage[:, currentind] .= NaN
                    fittimes[currentind] = NaN
                    currentind += 1
                 end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_normal_CS$(nsims)reps_sim.csv"

    # make sure gaussian_cs is a directory
    isdir("gaussian_cs") || mkdir("gaussian_cs")

    writedlm("gaussian_cs/mse_beta_" * ftail, βMseResults, ',')
    writedlm("gaussian_cs/mse_tau_" * ftail, τMseResults, ',')
    writedlm("gaussian_cs/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("gaussian_cs/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("gaussian_cs/fittimes_" * ftail, fittimes, ',')

    writedlm("gaussian_cs/beta_rho_sigma_coverage_" * ftail, βρσ2τcoverage, ',')
end

run_test_gcs()

function run_test_nbcs()
    p_fixed = 3    # number of fixed effects, including intercept
    # true parameter values
    Random.seed!(12345)
    βtrue = rand(Uniform(-2, 2), p_fixed)
    rtrue = 10.0
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
    intervals = zeros(p_fixed + 3, 2) #hold intervals
    curcoverage = zeros(p_fixed + 3) #hold current coverage resutls
    trueparams = [βtrue; ρtrue; σ2true; rtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    σ2MseResults = ones(nsims * length(ns) * length(samplesizes))
    ρMseResults = ones(nsims * length(ns) * length(samplesizes))
    rMseResults = ones(nsims * length(ns) * length(samplesizes))

    βρσ2rcoverage = Matrix{Float64}(undef, p_fixed + 3, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    st = time()
    currentind = 1
    d = NegativeBinomial()
    link = LogLink()
    D = typeof(d)
    Link = typeof(link)
    T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{NBCopulaCSObs{T, D, Link}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = get_V(ρtrue[1], ni)
            # true Gamma
            Γ = σ2true[1] * V

            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Y_nsample = []
                Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k)
                X_samplesize = [randn(ni, p_fixed - 1) for i in 1:m]
                for i in 1:m
                    X = [ones(ni) X_samplesize[i]]
                    η = X * βtrue
                    μ = exp.(η)
                    p = rtrue ./ (μ .+ rtrue)
                    vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
                    vecd = [NegativeBinomial(rtrue, p[i]) for i in 1:ni]
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                    # simuate single vector y
                    y = Vector{Float64}(undef, ni)
                    res = Vector{Float64}(undef, ni)
                    rand(nonmixed_multivariate_dist, y, res)
                    gcs[i] = NBCopulaCSObs(y, X, d,link)
                    push!(Y_nsample, y)
                end

                # form model
                gcm = NBCopulaCSModel(gcs);
                fittime = NaN
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show fittime
                    @show gcm.β
                    @show gcm.σ2
                    @show gcm.ρ
                    @show gcm.∇β
                    @show gcm.∇σ2
                    @show gcm.∇ρ
                    @show gcm.r
                    @show gcm.∇r
                    loglikelihood!(gcm, true, true)
                    vcov!(gcm)
                    @show QuasiCopula.confint(gcm)

                    # mse and time under our model
                    coverage!(gcm, trueparams, intervals, curcoverage)
                    mseβ, mseρ, mseσ2, mser = MSE(gcm, βtrue, ρtrue, σ2true, rtrue)
                    @show mseβ
                    @show mser
                    @show mseσ2
                    @show mseρ
                    # global currentind
                    @views copyto!(βρσ2rcoverage[:, currentind], curcoverage)
                    βMseResults[currentind] = mseβ
                    rMseResults[currentind] = mser
                    σ2MseResults[currentind] = mseσ2
                    ρMseResults[currentind] = mseρ
                    fittimes[currentind] = fittime
                    currentind += 1

                catch
                    println("rep $j ni obs = $ni , samplesize = $m had an error")
                    βMseResults[currentind] = NaN
                    rMseResults[currentind] = NaN
                    σ2MseResults[currentind] = NaN
                    ρMseResults[currentind] = NaN
                    βρσ2rcoverage[:, currentind] .= NaN
                    fittimes[currentind] = NaN
                    currentind += 1
                 end
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "multivariate_nb_CS$(nsims)reps_sim.csv"

    # make sure nb_cs is a directory
    isdir("nb_cs") || mkdir("nb_cs")

    writedlm("nb_cs/mse_beta_" * ftail, βMseResults, ',')
    writedlm("nb_cs/mse_r_" * ftail, rMseResults, ',')
    writedlm("nb_cs/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("nb_cs/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("nb_cs/fittimes_" * ftail, fittimes, ',')

    writedlm("nb_cs/beta_rho_sigma_coverage_" * ftail, βρσ2rcoverage, ',')
end
run_test_nbcs()

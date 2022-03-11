using QuasiCopula, LinearAlgebra, DelimitedFiles
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
    τtrue = 100.0
    σ2 = inv(τtrue)
    σ = sqrt(σ2)

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
        gcs = Vector{GaussianCopulaARObs{T}}(undef, m)
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
                    gcs[i] = GaussianCopulaARObs(y, X)
                end

                # form model
                gcm = GaussianCopulaARModel(gcs);
                try
                    fittime = @elapsed QuasiCopula.fit!(gcm)
                    @show fittime
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
    ftail = "multivariate_normal_AR$(nsims)reps_sim.csv"

    # make sure gaussian_ar is a directory
    isdir("gaussian_ar") || mkdir("gaussian_ar")

    writedlm("gaussian_ar/mse_beta_" * ftail, βMseResults, ',')
    writedlm("gaussian_ar/mse_tau_" * ftail, τMseResults, ',')
    writedlm("gaussian_ar/mse_sigma_" * ftail, σ2MseResults, ',')
    writedlm("gaussian_ar/mse_rho_" * ftail, ρMseResults, ',')
    writedlm("gaussian_ar/fittimes_" * ftail, fittimes, ',')

    writedlm("gaussian_ar/beta_rho_sigma_coverage_" * ftail, βρσ2τcoverage, ',')
end

run_test()

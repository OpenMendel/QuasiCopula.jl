using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem
import GLMCopula.MSE

function MSE(gcm::GaussianCopulaVCModel{T}, β::Vector, invτ::T, Σ::Vector) where {T <: BlasReal}
    mseβ = sum(abs2, gcm.β .- β) / gcm.p
    mseτ = sum(abs2, inv.(gcm.τ ./ (gcm.Σ .* inv.(gcm.τ))) .- invτ)
    mseΣ = sum(abs2, gcm.Σ .* inv.(gcm.τ) .- Σ) / gcm.m
    return mseβ, mseτ, mseΣ
end

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    Random.seed!(1234)
    # βtrue = rand(Uniform(-0.2, 0.2), p)
    βtrue = ones(p)
    Σtrue = [0.1]
    τtrue = 100.0
    σ2 = inv(τtrue)
    σ = sqrt(σ2)

    # generate data
    intervals = zeros(p + m + 1, 2) #hold intervals
    curcoverage = zeros(p + m + 1) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2; 5; 10; 15; 20; 25]
    nsims = 50

   #storage for our results
   βMseResults = ones(nsims * length(ns) * length(samplesizes))
   τMseResults = ones(nsims * length(ns) * length(samplesizes))
   ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
   βτΣcoverage = Matrix{Float64}(undef, p + m + 1, nsims * length(ns) * length(samplesizes))
   fittimes = zeros(nsims * length(ns) * length(samplesizes))

   #storage for glmm results
   βMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
   τMseResults_GLMM = ones(nsims * length(ns) * length(samplesizes))
   ΣMseResults_GLMM = ones(nsims * length(ns) *  length(samplesizes))
   fittimes_GLMM = zeros(nsims * length(ns) * length(samplesizes))

   # solver = KNITRO.KnitroSolver(outlev=0)
   solver = Ipopt.IpoptSolver(print_level = 5)

   st = time()
   currentind = 1
   T = Float64

    for t in 1:length(samplesizes)
        m = samplesizes[t]
        gcs = Vector{GaussianCopulaVCObs{T}}(undef, m)
        for k in 1:length(ns)
            ni = ns[k] # number of observations per individual
            V = [ones(ni, ni)]
            Γ = Σtrue[1] * ones(ni, ni) + σ2 * Matrix(I, ni, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")

                # Ystack = vcat(Y_nsample...)
                # @show length(Ystack)
                a = collect(1:m)
                group = [repeat([a[i]], ni) for i in 1:m]
                groupstack = vcat(group...)
                Xstack = []
                Ystack = []
                # df = DataFrame(Y = Ystack, X1 = Xstack[:, 1], X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = CategoricalArray(groupstack))
                for i in 1:m
                    # Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k + i)
                    X = [ones(ni) randn(ni, p - 1)]
                    μ = X * βtrue
                    vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))
                    for i in 1:length(μ)
                        vecd[i] = Normal(μ[i], σ)
                    end
                    # generate mvn response
                    mvn_d = MvNormal(μ, Γ)
                    y =  Float64.(rand(mvn_d))
                    # add to data
                    gcs[i] = GaussianCopulaVCObs(y, X, V)
                    push!(Xstack, X)
                    push!(Ystack, y)
                end

                # form VarLmmModel
                gcm = GaussianCopulaVCModel(gcs);
                fittime = NaN

                # form glmm
                Xstack = [vcat(Xstack...)][1]
                Ystack = [vcat(Ystack...)][1]
                # p = 3
                df = (Y = Ystack, X2 = Xstack[:, 2], X3 = Xstack[:, 3], group = string.(groupstack))
                form = @formula(Y ~ 1 + X2 + X3 + (1|group));

                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-7, hessian_approximation = "limited-memory"))
                @show fittime
                @show gcm.β
                @show gcm.Σ
                @show gcm.τ
                @show gcm.∇β
                @show gcm.∇Σ
                @show gcm.∇τ
                loglikelihood!(gcm, true, true)
                vcov!(gcm)
                @show GLMCopula.confint(gcm)
                # mse and time under our model
                coverage!(gcm, trueparams, intervals, curcoverage)
                mseβ, mseτ, mseΣ = MSE(gcm, βtrue, σ2, Σtrue)
                @show mseβ
                @show mseτ
                @show mseΣ
                @views copyto!(βτΣcoverage[:, currentind], curcoverage)
                βMseResults[currentind] = mseβ
                τMseResults[currentind] = mseτ
                ΣMseResults[currentind] = mseΣ
                fittimes[currentind] = fittime
                try
                    # glmm
                    # fit glmm
                    @info "Fit with MixedModels..."
                    fittime_GLMM = @elapsed gm1 = fit(MixedModel, form, df, Normal(), contrasts = Dict(:group => Grouping()))
                    @show fittime_GLMM
                    display(gm1)
                    @show gm1.β
                    # mse and time under glmm
                    @info "Get MSE under GLMM..."
                    level = 0.95
                    p = 3
                    # @show GLMM_CI_β = hcat(gm1.β + MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.), gm1.β - MixedModels.stderror(gm1) * quantile(Normal(), (1. - level) / 2.))
                    @show GLMM_mse = [sum(abs2, gm1.β .- βtrue) / p, sum(abs2, gm1.σ^2 .- σ2), sum(abs2, (gm1.σs[1][1]^2) .- Σtrue[1]) / 1]
                    # glmm
                    βMseResults_GLMM[currentind] = GLMM_mse[1]
                    τMseResults_GLMM[currentind] = GLMM_mse[2]
                    ΣMseResults_GLMM[currentind] = GLMM_mse[3]
                    fittimes_GLMM[currentind] = fittime_GLMM
                    currentind += 1
                catch
                    βMseResults[currentind] = NaN
                    τMseResults[currentind] = NaN
                    ΣMseResults[currentind] = NaN
                    fittimes[currentind] = NaN
                    @views copyto!(βτΣcoverage[:, currentind], NaN)
                    # glmm
                    βMseResults_GLMM[currentind] = NaN
                    τMseResults[currentind] = NaN
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
    ftail = "normal$(nsims)reps_sim.csv"
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_beta_multivariate_" * ftail, βMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_Sigma_multivariate_" * ftail, ΣMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_tau_multivariate_" * ftail, τMseResults, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/fittimes_multivariate_" * ftail, fittimes, ',')

    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/beta_sigma_tau_coverage_" * ftail, βτΣcoverage, ',')

    # glmm
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_beta_GLMM_" * ftail, βMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_Sigma_GLMM_" * ftail, ΣMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/mse_tau_GLMM_" * ftail, τMseResults_GLMM, ',')
    writedlm("sim_glmm_vs_ours_random_int/normal_sim_lmm/fittimes_GLMM_" * ftail, fittimes_GLMM, ',')
end
runtest()

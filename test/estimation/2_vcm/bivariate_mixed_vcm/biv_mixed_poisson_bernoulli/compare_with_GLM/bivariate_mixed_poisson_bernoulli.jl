using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM, MixedModels, CategoricalArrays
using Random, Roots, SpecialFunctions
using DataFrames, DelimitedFiles, Statistics
import StatsBase: sem

function coverage_GLMs!(trueparams, intervals, curcoverage)
    lbs = @views intervals[:, 1]
    ubs = @views intervals[:, 2]
    map!((val, lb, ub) -> val >= lb &&
        val <= ub, curcoverage, trueparams, lbs, ubs)
    return curcoverage
end

function fit_GLM_twice!(Ystack, Xstack, glm_betas, intervals_GLM_1, intervals_GLM_2)
    Y = transpose(hcat(Ystack...))
    Data = hcat(Xstack, Y)
    df = DataFrame(Y1 = Data[:, 4], Y2 = Data[:, 5], X1 = Data[:, 2], X2 = Data[:, 3])
    poisson_glm = glm(@formula(Y1 ~ 1 + X1 + X2), df, Poisson(), LogLink());
    copyto!(intervals_GLM_1, GLM.confint(poisson_glm))
    bernoulli_glm = glm(@formula(Y2 ~ 1 + X1 + X2), df, Bernoulli(), LogitLink());
    copyto!(intervals_GLM_2, GLM.confint(bernoulli_glm))
    copyto!(glm_betas, [poisson_glm.model.pp.beta0; bernoulli_glm.model.pp.beta0])
    nothing
end


function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    # true parameter values
    βtrue = [0.2; 0.1; -0.05; 0.2; 0.1; -0.1]
    Σtrue = [0.5]

    glm_betas = zeros(length(βtrue))

    # generate data
    intervals = zeros(2 * p + m, 2) #hold intervals
    intervals_GLM_1 = zeros(p, 2) #hold intervals
    intervals_GLM_2 = zeros(p, 2) #hold intervals
    curcoverage = zeros(2 * p + m) #hold current coverage resutls
    curcoverage_GLM_1 = zeros(p) #hold current coverage resutls
    curcoverage_GLM_2 = zeros(p) #hold current coverage resutls
    curcoverag_GLM = zeros(2 * p + m) #hold current coverage resutls
    trueparams = [βtrue; Σtrue] #hold true parameters

    #simulation parameters
    samplesizes = [100; 1000; 10000]
    ns = [2]
    nsims = 20

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βΣcoverage = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLM = ones(nsims * length(ns) * length(samplesizes))
    ΣMseResults_GLM = ones(nsims * length(ns) *  length(samplesizes))
    βΣcoverage_GLM = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
    fittimes_GLM = zeros(nsims * length(ns) * length(samplesizes))

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
                Xstack = [ones(m) randn(m, p - 1)]
                Ystack = []
                for i in 1:m
                    # xi = [1.0 randn(p - 1)...]
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
                    push!(Ystack, y)
                    gcs[i] = Poisson_Bernoulli_VCObs(y, xi, V, vecdist, veclink)
                end

                gcm = Poisson_Bernoulli_VCModel(gcs);

                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 7, accept_after_max_steps = 2, hessian_approximation = "limited-memory"))
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

                ### fit using glm
                # poisson_glm = glm(@formula(Y1 ~ 1 + X1 + X2), df, Poisson(), LogLink());
                # bernoulli_glm = glm(@formula(Y2 ~ 1 + X1 + X2), df, Bernoulli(), LogitLink());

                fittime_GLM = @elapsed fit_GLM_twice!(df, glm_betas, intervals_GLM_1, intervals_GLM_2)
                fittimes_GLM[currentind] = fittime_GLM
                # glm_betas = [poisson_glm.model.pp.beta0; bernoulli_glm.model.pp.beta0]
                GLM_betaMSE = sum(abs2, glm_betas .- βtrue) / p
                GLM_thetaMSE = Σtrue[1]^2
                βMseResults_GLM[currentind] = GLM_betaMSE
                ΣMseResults_GLM[currentind] = GLM_thetaMSE


                coverage_GLMs!(trueparams[1:p], intervals_GLM_1, curcoverage_GLM_1)
                coverage_GLMs!(trueparams[p+1:p+p], intervals_GLM_2, curcoverage_GLM_2)
                curcoverag_GLM = [curcoverage_GLM_1; curcoverage_GLM_2; 0.0]
                @views copyto!(βΣcoverage_GLM[:, currentind], curcoverag_GLM)

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

    # glm
    writedlm("biv_mixed_poisson_bernoulli/mse_beta_GLM_" * ftail, βMseResults_GLM, ',')
    writedlm("biv_mixed_poisson_bernoulli/mse_Sigma_GLM_" * ftail, ΣMseResults_GLM, ',')
    writedlm("biv_mixed_poisson_bernoulli/beta_sigma_coverage_GLM_" * ftail, βΣcoverage_GLM, ',')
    writedlm("biv_mixed_poisson_bernoulli/fittimes_GLM_" * ftail, fittimes_GLM, ',')


end
runtest()

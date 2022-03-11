using QuasiCopula, LinearAlgebra, GLM, DelimitedFiles
using Random
using DataFrames

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

BLAS.set_num_threads(1)
Threads.nthreads()

function runtest()
    p = 3    # number of fixed effects, including intercept
    m = 1    # number of variance components
    Random.seed!(12345)
    βtrue = rand(Uniform(-0.2, 0.2), 2 * p)
    θtrue = [0.5]

    glm_betas = zeros(length(βtrue))

    # generate data
    intervals = zeros(2 * p + m, 2) #hold intervals
    intervals_GLM_1 = zeros(p, 2) #hold intervals
    intervals_GLM_2 = zeros(p, 2) #hold intervals
    curcoverage = zeros(2 * p + m) #hold current coverage resutls
    curcoverage_GLM_1 = zeros(p) #hold current coverage resutls
    curcoverage_GLM_2 = zeros(p) #hold current coverage resutls
    curcoverag_GLM = zeros(2 * p + m) #hold current coverage resutls
    trueparams = [βtrue; θtrue] #hold true parameters

    #simulation parameters
    # samplesizes = [10000]
    samplesizes = [100; 1000; 10000]
    ns = [2]
    nsims = 100

    #storage for our results
    βMseResults = ones(nsims * length(ns) * length(samplesizes))
    θMseResults = ones(nsims * length(ns) *  length(samplesizes))
    βθcoverage = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
    fittimes = zeros(nsims * length(ns) * length(samplesizes))

    #storage for glmm results
    βMseResults_GLM = ones(nsims * length(ns) * length(samplesizes))
    θMseResults_GLM = ones(nsims * length(ns) *  length(samplesizes))
    βθcoverage_GLM = Matrix{Float64}(undef, 2 * p + m, nsims * length(ns) * length(samplesizes))
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
            Γ = θtrue[1] * ones(ni, ni)
            vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
            for j in 1:nsims
                println("rep $j obs per person $ni samplesize $m")
                Random.seed!(1000000000 * t + 10000000 * j + 1000000 * k)
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

                ### fit using glm
                fittime_GLM = @elapsed fit_GLM_twice!(Ystack, Xstack, glm_betas, intervals_GLM_1, intervals_GLM_2)
                @show fittime_GLM
                fittimes_GLM[currentind] = fittime_GLM
                # glm_betas = [poisson_glm.model.pp.beta0; bernoulli_glm.model.pp.beta0]
                GLM_betaMSE = sum(abs2, glm_betas .- βtrue) / p
                GLM_thetaMSE = θtrue[1]^2
                βMseResults_GLM[currentind] = GLM_betaMSE
                θMseResults_GLM[currentind] = GLM_thetaMSE


                coverage_GLMs!(trueparams[1:p], intervals_GLM_1, curcoverage_GLM_1)
                coverage_GLMs!(trueparams[p+1:p+p], intervals_GLM_2, curcoverage_GLM_2)
                curcoverag_GLM = [curcoverage_GLM_1; curcoverage_GLM_2; 0.0]
                @views copyto!(βθcoverage_GLM[:, currentind], curcoverag_GLM)

                currentind += 1
            end
        end
    end
    en = time()

    @show en - st #seconds
    @info "writing to file..."
    ftail = "bivariate_mixed_poisson_bernoulli_vcm$(nsims)reps_sim.csv"

    # make sure bivariate_mixed_vc_vs_glm is a directory
    isdir("bivariate_mixed_vc_vs_glm") || mkdir("bivariate_mixed_vc_vs_glm")

    writedlm("bivariate_mixed_vc_vs_glm/mse_beta_" * ftail, βMseResults, ',')
    writedlm("bivariate_mixed_vc_vs_glm/mse_theta_" * ftail, θMseResults, ',')
    writedlm("bivariate_mixed_vc_vs_glm/fittimes_" * ftail, fittimes, ',')

    writedlm("bivariate_mixed_vc_vs_glm/beta_theta_coverage_" * ftail, βθcoverage, ',')

    # glm
    writedlm("bivariate_mixed_vc_vs_glm/mse_beta_GLM_" * ftail, βMseResults_GLM, ',')
    writedlm("bivariate_mixed_vc_vs_glm/mse_theta_GLM_" * ftail, θMseResults_GLM, ',')
    writedlm("bivariate_mixed_vc_vs_glm/beta_theta_coverage_GLM_" * ftail, βθcoverage_GLM, ',')
    writedlm("bivariate_mixed_vc_vs_glm/fittimes_GLM_" * ftail, fittimes_GLM, ',')


end
runtest()

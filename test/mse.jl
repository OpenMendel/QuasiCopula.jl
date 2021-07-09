module mse
using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM
using Random, Roots, SpecialFunctions

# some notes: July 7 2021
# I made a new function that works and gives the confidence intervals for one fit. 
# Next need to make this run nsim times and make figure
"""
    sandwich!(gcm::GLMCopulaVCModel)
Calculate the sandwich estimator of the asymptotic covariance of the parameters, 
based on values `gcm.Hββ`, `gcm.HΣ`, `gcm.data[i].∇β`,
`gcm.data[i].∇Σ`, and `gcm.vcov` is updated by the sandwich 
estimator and returned.
"""
function sandwich!(gcm::GLMCopulaVCModel{T, D, Link}) where {T <: BlasReal, D, Link}
    p, m = gcm.p, gcm.m
    minv = inv(length(gcm.data))
    # form A matrix in the sandwich formula
    Ainv = zeros(p + m, p + m)
    Ainv[          1:p,                 1:p      ] = gcm.Hβ
    Ainv[    (p + 1):(p + m),     (p + 1):(p + m)] = gcm.HΣ
    # form M matrix in the sandwich formula
    M = gcm.∇θ * transpose(gcm.∇θ)
    vcov = zeros(p + m, p + m)
    vcov = Ainv * M * Ainv
    vcov
end

coef(gcm::GLMCopulaVCModel) = [gcm.β; gcm.Σ]
stderror(gcm::GLMCopulaVCModel, vcov) = [sqrt(vcov[i, i]) for i in 1:(gcm.p + gcm.m)]

confint(gcm::GLMCopulaVCModel, vcov, level::Real) = hcat(coef(gcm) + stderror(gcm, vcov) * quantile(Normal(), (1. - level) / 2.), coef(gcm) - stderror(gcm, vcov) * quantile(Normal(), (1. - level) / 2.))

confint(gcm::GLMCopulaVCModel, vcov) = confint(gcm, vcov, 0.95)

# The following code runs the simulation and saves the results.
function MSE(gcm::GLMCopulaVCModel, β::Vector, Σ::Vector)
    mseβ = sum(abs2, gcm.β - β) / gcm.p
    mseΣ = sum(abs2, gcm.Σ - Σ) / gcm.m
    return mseβ, mseΣ
end

function coverage!(gcm::GLMCopulaVCModel, vcov, trueparams::Vector, 
    intervals::Matrix, curcoverage::Vector)
    copyto!(intervals, confint(gcm, vcov))
    lbs = @views intervals[:, 1]
    ubs = @views intervals[:, 2]
    map!((val, lb, ub) -> val >= lb && 
        val <= ub, curcoverage, trueparams, lbs, ubs)
    return curcoverage
end

# dimensions
@show Threads.nthreads()
p  = 3    # number of fixed effects, including intercept
m  = 2    # number of variance components
# true parameter values
βtrue = [1.0; 0.1; 0.5]
Σtrue = [0.1; 0.1]

# generate data
intervals = zeros(p + m, 2) #hold intervals
curcoverage = zeros(p + m) #hold current coverage resutls
trueparams = [βtrue; Σtrue] #hold true parameters

#simulation parameters
samplesizes = collect(10000:10000:30000)
ns = [5; 10] # ; 50; 100; 1000]
nsims = 2

#storage for results
βMseResults = ones(nsims * length(ns) * length(samplesizes))
ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
βΣcoverage = Matrix{Float64}(undef, p + m, nsims * length(ns) * length(samplesizes))
fittimes = zeros(nsims * length(ns) * length(samplesizes))
# solver = KNITRO.KnitroSolver(outlev=0)
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
    gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
    for k in 1:length(ns)
        ni = ns[k] # number of observations per individual
        y = Vector{Float64}(undef, ni)
        for j in 1:nsims
            println("rep $j obs per person $ni samplesize $m")
            Random.seed!(j + 100000k + 1000t)
            for i in 1:m
                # first column intercept, remaining entries iid std normal
                X = Matrix{Float64}(undef, ni, p)
                X[:, 1] .= 1
                @views randn!(X[:, 2:p])

                η = X * βtrue
                μ = exp.(η)
                vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))
                for i in 1:length(μ)
                    vecd[i] = Poisson(μ[i])
                end
                # first column intercept, remaining entries iid std normal
                # set up covariance matrix
                V = [ones(ni, ni), Matrix(I, ni, ni)]
                Γ = Σtrue[1] * V[1] + Σtrue[2] * V[2]
                # generate y
                y = Vector{Float64}(undef, ni)
                res = Vector{Float64}(undef, ni)
                nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
                y .= Float64.(rand(nonmixed_multivariate_dist, y, res))
                # form a VarLmmObs instance
                gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
            end
            # form VarLmmModel
            gcm = GLMCopulaVCModel(gcs);
            vcov = sandwich!(gcm)
            fittime = NaN
            initialize_model!(gcm)
            @show gcm.β
            @show gcm.Σ
            loglikelihood!(gcm, true, true)

            try 
                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
                confint(gcm, vcov)
            catch
                println("rep $j ni obs = $ni , samplesize = $m had an error")
                try 
                    fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "limited-memory"))
                        confint(gcm, vcov)
                catch
                    println("rep $j ni obs = $ni , samplesize = $m had a second error")
                    try
                        fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"), init = initialize_model!(gcm), parallel = false)
                        confint(gcm)
                    catch
                        println("rep $j ni obs = $ni , samplesize = $m had a third error")
                        βMseResults[currentind] = NaN
                        ΣMseResults[currentind] = NaN
                        βΣcoverage[:, currentind] .= NaN
                        fittimes[currentind] = NaN
                        currentind += 1
                        continue
                    end
                end
            end
            # coverage!(gcm, vcov, trueparams, intervals, curcoverage)
            mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
            @show mseβ
            @show mseΣ
            #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
            # global currentind
            # @views copyto!(βΣcoverage[:, currentind], curcoverage)
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
ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
writedlm("test/result_files/mse_beta_" * ftail, βMseResults, ',')
writedlm("test/result_files/mse_Sigma_" * ftail, ΣMseResults, ',')
writedlm("test/result_files/fittimes" * ftail, fittimes, ',')

#### 
using Random, DataFrames, DelimitedFiles, Statistics, RCall, Printf
import StatsBase: sem
ENV["COLUMNS"]=1000

βMseresult = vec(readdlm("test/result_files/mse_beta_multivariate_poisson_vcm2reps_sim.csv", ','))
ΣMseresult = vec(readdlm("test/result_files/mse_Sigma_multivariate_poisson_vcm2reps_sim.csv", ','))
fittimes = vec(readdlm("test/result_files/fittimesmultivariate_poisson_vcm2reps_sim.csv", ','))

end
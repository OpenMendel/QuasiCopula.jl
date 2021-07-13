module mse
using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM
using Random, Roots, SpecialFunctions

# some notes: July 7 2021
# I made a new function that works and gives the confidence intervals for one fit. 
# Next need to make this run nsim times and make figures 

using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns, GLM
using BenchmarkTools, Profile, Distributions

Random.seed!(1234)
n = 5
variance_component_1 = 0.1
variance_component_2 = 0.1
Γ = variance_component_1 * ones(n, n) + variance_component_2 * Matrix(I, n, n)

dist = Poisson
p = 3
β = ones(p)
X = [ones(n) randn(n, p - 1)]
η = X * β
μ = exp.(η)
vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Poisson(μ[i])
end

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
nsample = 10_000
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

d = Poisson()
link = LogLink()
D = typeof(d)
Link = typeof(link)
T = Float64
gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    V = [ones(n, n), Matrix(I, n, n)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
end
gcm = GLMCopulaVCModel(gcs)
initialize_model!(gcm)
@show gcm.β
@show gcm.Σ
loglikelihood!(gcm, true, true)
# use quasi-newton
@time GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
println("estimated beta = $(gcm.β); true beta value= $β")
println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated variance component 2 = $(gcm.Σ[2]); true variance component 2 = $variance_component_2");

@info "get standard errors"

sandwich!(gcm)
@show GLMCopula.confint(gcm)
@show gcm.θ

# p  = 1    # number of fixed effects, including intercept
# m  = 2    # number of variance components
# # true parameter values
# βtrue = [log(5)]
# Σtrue = [0.1; 0.1]

# # generate data
# intervals = zeros(p + m, 2) #hold intervals
# curcoverage = zeros(p + m) #hold current coverage resutls
# trueparams = [βtrue; Σtrue] #hold true parameters

# #simulation parameters
# # samplesizes = collect(10000:50000:100000)
# samplesizes = [10000; 20000]
# ns = [2; 5]
# nsims = 5

# #storage for results
# βMseResults = ones(nsims * length(ns) * length(samplesizes))
# ΣMseResults = ones(nsims * length(ns) *  length(samplesizes))
# βΣcoverage = Matrix{Float64}(undef, p + m, nsims * length(ns) * length(samplesizes))
# fittimes = zeros(nsims * length(ns) * length(samplesizes))
# # solver = KNITRO.KnitroSolver(outlev=0)
# solver = Ipopt.IpoptSolver(print_level = 5)

# st = time()
# currentind = 1
# d = Poisson()
# link = LogLink()
# D = typeof(d)
# Link = typeof(link)
# T = Float64

# for t in 1:length(samplesizes)
#     m = samplesizes[t]
#     gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
#     for k in 1:length(ns)
#         ni = ns[k] # number of observations per individual
#         y = Vector{Float64}(undef, ni)
#         res = Vector{Float64}(undef, ni)
#         for j in 1:nsims
#             println("rep $j obs per person $ni samplesize $m")
#             Random.seed!(j + 100000k + 1000t)
#             for i in 1:m
#                 # first column intercept, remaining entries iid std normal
#                 X = ones(ni, 1)
#                 vecd = Vector{DiscreteUnivariateDistribution}(undef, ni)
#                 for l in 1:ni
#                     vecd[l] = Poisson(5)
#                 end
#                 # first column intercept, remaining entries iid std normal
#                 # set up covariance matrix
#                 V = [ones(ni, ni), Matrix(I, ni, ni)]
#                 Γ = Σtrue[1] * V[1] + Σtrue[2] * V[2]
#                 # generate y
#                 nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
#                 y .= Float64.(rand(nonmixed_multivariate_dist, y, res))
#                 # form a GLMCopulaVCObs instance
#                 gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
#             end
#             # form VarLmmModel
#             gcm = GLMCopulaVCModel(gcs);
#             fittime = NaN
#             initialize_model!(gcm)
#             @show gcm.β
#             @show gcm.Σ
#             try 
#                 fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 1, max_iter = 100, hessian_approximation = "limited-memory"))
#                 @show gcm.θ
#                 loglikelihood!(gcm, true, true)
#                 sandwich!(gcm)
#                 @show GLMCopula.confint(gcm)
#             catch
#                 println("rep $j ni obs = $ni , samplesize = $m had an error")
#                 βMseResults[currentind] = NaN
#                 ΣMseResults[currentind] = NaN
#                 βΣcoverage[:, currentind] .= NaN
#                 fittimes[currentind] = NaN
#                 currentind += 1
#                 continue
#             end
#             coverage!(gcm, trueparams, intervals, curcoverage)
#             mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
#             @show mseβ
#             @show mseΣ
#             #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
#             global currentind
#             @views copyto!(βΣcoverage[:, currentind], curcoverage)
#             βMseResults[currentind] = mseβ
#             ΣMseResults[currentind] = mseΣ
#             fittimes[currentind] = fittime
#             currentind += 1
#         end
#     end
# end 
# en = time()

# using Random, DataFrames, DelimitedFiles, Statistics, RCall, Printf
# import StatsBase: sem

# @show en - st #seconds 
# @info "writing to file..."
# ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
# writedlm("test/result_files/mse_beta_" * ftail, βMseResults, ',')
# writedlm("test/result_files/mse_Sigma_" * ftail, ΣMseResults, ',')
# writedlm("test/result_files/fittimes" * ftail, fittimes, ',')

# writedlm("test/result_files/beta_tau_coverage_5betas_" * ftail, βΣcoverage, ',')

# ENV["COLUMNS"]=1000
# @info "reading in the files with results"
# βMseresult = vec(readdlm("test/result_files/mse_beta_multivariate_poisson_vcm$(nsims)reps_sim.csv", ','))
# ΣMseresult = vec(readdlm("test/result_files/mse_Sigma_multivariate_poisson_vcm$(nsims)reps_sim.csv", ','))
# fittimes = vec(readdlm("test/result_files/fittimesmultivariate_poisson_vcm$(nsims)reps_sim.csv", ','))

# #simulation parameters
# @info "making results table"
# sample_sizes = repeat(string.(samplesizes), inner = nsims * length(ns))
# obs_sizes = repeat(string.(ns), inner = nsims, outer = length(samplesizes))

# msedf = DataFrame(βmse = βMseresult, Σmse = ΣMseresult, 
#     samplesize = sample_sizes, obssize = obs_sizes, fittimes = fittimes)
# timedf = combine(groupby(msedf, [:samplesize, :obssize]), :fittimes => mean => :fittime,
#     :fittimes => x -> (std(x)/sqrt(length(x))))
# rename!(timedf, Dict(:fittimes_function => "se"))
# timedf[!, :nobs] = Meta.parse.(timedf[!, :samplesize]) .* Meta.parse.(timedf[!, :obssize])
# timedf[!, :fitmin] = timedf[!, :fittime] - timedf[!, :se]
# timedf[!, :fitmax] = timedf[!, :fittime] + timedf[!, :se]
# timedf[!, :perobsratio] = timedf[!, :fittime] ./ timedf[!, :nobs]
# timedf

# ########### plot the runtimes
# using RCall
# @rput timedf

# R"""
# library(ggplot2)
# timedf$obssize <- factor(timedf$obssize, levels = c('2', '5'))
# timedf$samplesize <- factor(timedf$samplesize, levels = c('10000', '20000'))

# fittime_1 = ggplot(timedf, aes(x=samplesize, y=fittime, group=obssize, color=obssize)) + 
#   geom_line() +
#   geom_point()+
#   geom_errorbar(aes(ymin=fitmin, ymax=fitmax), width=0.5, alpha = 0.8, position=position_dodge(0.005)) + 
#   theme(legend.position=c(0.15,0.85), legend.key = element_blank(), axis.text.x = element_text(angle=0, size=13),
#         axis.text.y = element_text(angle=0, size=13), axis.title.x = element_text(size = 17), 
#         axis.title.y = element_text(size = 17), legend.title = element_text(size = 14),
#         #axis.ticks = element_blank(),
#         panel.grid.major = element_blank(), legend.text=element_text(size=13),
#         panel.border = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), 
#         axis.line = element_line(color = 'black',size=0.3), plot.title = element_text(hjust = 0.5)) + 
#    scale_color_manual(values = c("#c85f55",
# "#a964bf",
# "#8db352",
# "#fa7300",
# "#05aec0")) +
#   labs(x = "Number of Individuals", y = "Fit time (seconds)", color = "Obs per Individual")
# """

# using RCall
# @rput timedf

# R"""
# library(scales)
# library(ggplot2)
# timedf$obssize <- factor(timedf$obssize, levels = c('2', '5'))
# timedf$samplesize <- factor(timedf$samplesize, levels = c('10000', '20000'))

# fancy_scientific <- function(l) {
#      # turn in to character string in scientific notation
#      l <- format(l, scientific = TRUE)
#      # quote the part before the exponent to keep all the digits
#      l <- gsub("^(.*)e", "'\\1'e", l)
#      # turn the 'e+' into plotmath format
#      l <- gsub("e", "%*%10^", l)
#      # return this as an expression
#      parse(text=l)
# }

# fittimeperobs = ggplot(timedf, aes(x=nobs, y=perobsratio)) + 
#   geom_line() +
#   geom_point()+
# #  geom_errorbar(aes(ymin=fitmin, ymax=fitmax), width=0.5, alpha = 0.8, position=position_dodge(0.005)) + 
#   theme(legend.position=c(0.15,0.8), legend.key = element_blank(), axis.text.x = element_text(angle=0, size=13),
#         axis.text.y = element_text(angle=0, size=13), axis.title.x = element_text(size = 17), 
#         axis.title.y = element_text(size = 17), legend.title = element_text(size = 14),
#         #axis.ticks = element_blank(),
#         panel.grid.major = element_blank(), legend.text=element_text(size=11),
#         panel.border = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), 
#         axis.line = element_line(color = 'black',size=0.3), plot.title = element_text(hjust = 0.5)) + 
#   labs(x = "Total Number of Observations", y = "Fit time per observation (seconds)", color = "Obs per Individual") +
# #scale_x_log10(breaks = 10^seq(0, 7, 1), labels=trans_format("log10", math_format(10^.x)))# + #, limit=c(10^0, 10^7))
# scale_x_continuous(breaks = seq(0, 6000000, 1000000), labels= fancy_scientific) +
# scale_y_continuous(breaks = c(10^-5, 2 * 10^-5, 3 * 10^-5, 4 * 10^-5, 5 * 10^-5, 6 * 10^-5), labels= fancy_scientific)

# """

# ######  supplementary table s1
# using Random, DataFrames, DelimitedFiles, Statistics
# import StatsBase: sem
# ENV["COLUMNS"]=800

# βΣcoverage = readdlm("test/result_files/beta_tau_coverage_5betas_multivariate_poisson_vcm5reps_sim.csv", ',')
# samplesizes = collect(1000:1000:6000)
# ns = [10; 25; 50; 100; 1000]
# nsims = 1000

# covdf = DataFrame(transpose(βτcoverage))
# rename!(covdf, Symbol.([["β$i" for i in 1:p]; ["τ$i" for i in 1:l]]))
# covdf[!, :samplesize] = sample_sizes
# covdf[!, :obssize] = obs_sizes
# first(covdf, 10)

# row_stats = [[(mean(col), sem(col)) for col = eachcol(d[!, 1:end-2])] for d = groupby(covdf, [:samplesize; :obssize])]
# df = DataFrame(row_stats)
# ss_obs = unique("m: " .* sample_sizes .* "  ni: " .* obs_sizes)
# rename!(df, Symbol.(ss_obs))
# covdfdisplay = DataFrame([[names(df)]; collect.(eachrow(df))], 
#     [:people_obs; Symbol.([["β$i" for i in 1:p]; ["τ$i" for i in 1:l]])])
# deletecols!(covdfdisplay, p + 2)
# covdfdisplay

end
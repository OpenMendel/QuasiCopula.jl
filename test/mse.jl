module mse
using GLMCopula, DelimitedFiles, LinearAlgebra, Random, GLM
using Random, Roots, SpecialFunctions

p  = 1    # number of fixed effects, including intercept
m  = 2    # number of variance components
# true parameter values
βtrue = ones(p)
Σtrue = [0.1; 0.1]

# generate data
intervals = zeros(p + m, 2) #hold intervals
curcoverage = zeros(p + m) #hold current coverage resutls
trueparams = [βtrue; Σtrue] #hold true parameters

#simulation parameters
samplesizes = [10000; 50000; 100000]
# samplesizes = collect(10000:20000:100000)
ns = [5; 10; 20; 50]
nsims = 10

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
        res = Vector{Float64}(undef, ni)
        for j in 1:nsims
            println("rep $j obs per person $ni samplesize $m")
            Random.seed!(j + 100000k + 1000t)
            β = ones(p)
            X = [ones(ni) randn(ni, p - 1)]
            η = X * β
            μ = exp.(η)
            vecd = Vector{DiscreteUnivariateDistribution}(undef, length(μ))
    
            for i in 1:length(μ)
                vecd[i] = Poisson(μ[i])
            end
            Random.seed!(j + 100000i)
            Γ = Σtrue[1] * ones(ni, ni) + Σtrue[2] * Matrix(I, ni, ni)
            nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
            @time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, m)
    
            gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, m)
            for i in 1:m
                y = Float64.(Y_nsample[i])
                V = [ones(ni, ni), Matrix(I, ni, ni)]
                gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
            end
            
            # form VarLmmModel
            gcm = GLMCopulaVCModel(gcs);
            fittime = NaN
            initialize_model!(gcm)
            @show gcm.β
            @show gcm.Σ
            try 
                fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 1, max_iter = 300, tol = 10^-6, hessian_approximation = "exact"))
                @show gcm.θ
                @show gcm.∇θ
                loglikelihood!(gcm, true, true)
                sandwich!(gcm)
                @show GLMCopula.confint(gcm)
            catch
                println("rep $j ni obs = $ni , samplesize = $m had an error")
                βMseResults[currentind] = NaN
                ΣMseResults[currentind] = NaN
                βΣcoverage[:, currentind] .= NaN
                fittimes[currentind] = NaN
                currentind += 1
                continue
            end
            coverage!(gcm, trueparams, intervals, curcoverage)
            mseβ, mseΣ = MSE(gcm, βtrue, Σtrue)
            @show mseβ
            @show mseΣ
            #index = Int(nsims * length(ns) * (t - 1) + nsims * (k - 1) + j)
            global currentind
            @views copyto!(βΣcoverage[:, currentind], curcoverage)
            βMseResults[currentind] = mseβ
            ΣMseResults[currentind] = mseΣ
            fittimes[currentind] = fittime
            currentind += 1
        end
    end
end 
en = time()

using Random, DataFrames, DelimitedFiles, Statistics, RCall, Printf
import StatsBase: sem

@show en - st #seconds 
@info "writing to file..."
ftail = "multivariate_poisson_vcm$(nsims)reps_sim.csv"
writedlm("test/result_files/mse_beta_" * ftail, βMseResults, ',')
writedlm("test/result_files/mse_Sigma_" * ftail, ΣMseResults, ',')
writedlm("test/result_files/fittimes" * ftail, fittimes, ',')

writedlm("test/result_files/beta_tau_coverage_5betas_" * ftail, βΣcoverage, ',')

ENV["COLUMNS"]=1000
@info "reading in the files with results"
βMseresult = vec(readdlm("test/result_files/mse_beta_multivariate_poisson_vcm$(nsims)reps_sim.csv", ','))
ΣMseresult = vec(readdlm("test/result_files/mse_Sigma_multivariate_poisson_vcm$(nsims)reps_sim.csv", ','))
fittimes = vec(readdlm("test/result_files/fittimesmultivariate_poisson_vcm$(nsims)reps_sim.csv", ','))

#simulation parameters
@info "making results table"
sample_sizes = repeat(string.(samplesizes), inner = nsims * length(ns))
obs_sizes = repeat(string.(ns), inner = nsims, outer = length(samplesizes))

msedf = DataFrame(βmse = βMseresult, Σmse = ΣMseresult, 
    samplesize = sample_sizes, obssize = obs_sizes, fittimes = fittimes)
timedf = combine(groupby(msedf, [:samplesize, :obssize]), :fittimes => mean => :fittime,
    :fittimes => x -> (std(x)/sqrt(length(x))))
rename!(timedf, Dict(:fittimes_function => "se"))
timedf[!, :nobs] = Meta.parse.(timedf[!, :samplesize]) .* Meta.parse.(timedf[!, :obssize])
timedf[!, :fitmin] = timedf[!, :fittime] - timedf[!, :se]
timedf[!, :fitmax] = timedf[!, :fittime] + timedf[!, :se]
timedf[!, :perobsratio] = timedf[!, :fittime] ./ timedf[!, :nobs]
timedf

########### plot the runtimes
using RCall
@rput timedf

R"""
library(ggplot2)
timedf$obssize = factor(timedf$obssize, levels = c('5', '10', '20', '50'))
timedf$samplesize = factor(timedf$samplesize, levels = c('10000', '50000'))

fittime_1 = ggplot(timedf, aes(x=samplesize, y=fittime, group=obssize, color=obssize)) + 
  geom_line() +
  geom_point()+
  geom_errorbar(aes(ymin=fitmin, ymax=fitmax), width=0.5, alpha = 0.8, position=position_dodge(0.005)) + 
  theme(legend.position=c(0.15,0.85), legend.key = element_blank(), axis.text.x = element_text(angle=0, size=13),
        axis.text.y = element_text(angle=0, size=13), axis.title.x = element_text(size = 17), 
        axis.title.y = element_text(size = 17), legend.title = element_text(size = 14),
        #axis.ticks = element_blank(),
        panel.grid.major = element_blank(), legend.text=element_text(size=13),
        panel.border = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), 
        axis.line = element_line(color = 'black',size=0.3), plot.title = element_text(hjust = 0.5)) + 
   scale_color_manual(values = c("#c85f55",
"#a964bf",
"#8db352",
"#fa7300",
"#05aec0")) +
  labs(x = "Number of Individuals", y = "Fit time (seconds)", color = "Obs per Individual")
"""

using RCall
@rput timedf

R"""
library(scales)
library(ggplot2)
timedf$obssize <- factor(timedf$obssize, levels = c('5', '10', '20', '50'))
timedf$samplesize <- factor(timedf$samplesize, levels = c('10000', '50000'))

fancy_scientific <- function(l) {
     # turn in to character string in scientific notation
     l <- format(l, scientific = TRUE)
     # quote the part before the exponent to keep all the digits
     l <- gsub("^(.*)e", "'\\1'e", l)
     # turn the 'e+' into plotmath format
     l <- gsub("e", "%*%10^", l)
     # return this as an expression
     parse(text=l)
}

fittimeperobs = ggplot(timedf, aes(x=nobs, y=perobsratio)) + 
  geom_line() +
  geom_point()+
#  geom_errorbar(aes(ymin=fitmin, ymax=fitmax), width=0.5, alpha = 0.8, position=position_dodge(0.005)) + 
  theme(legend.position=c(0.15,0.8), legend.key = element_blank(), axis.text.x = element_text(angle=0, size=13),
        axis.text.y = element_text(angle=0, size=13), axis.title.x = element_text(size = 17), 
        axis.title.y = element_text(size = 17), legend.title = element_text(size = 14),
        #axis.ticks = element_blank(),
        panel.grid.major = element_blank(), legend.text=element_text(size=11),
        panel.border = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), 
        axis.line = element_line(color = 'black',size=0.3), plot.title = element_text(hjust = 0.5)) + 
  labs(x = "Total Number of Observations", y = "Fit time per observation (seconds)", color = "Obs per Individual") +
#scale_x_log10(breaks = 10^seq(0, 7, 1), labels=trans_format("log10", math_format(10^.x)))# + #, limit=c(10^0, 10^7))
scale_x_continuous(breaks = seq(0, 6000000, 1000000), labels= fancy_scientific) +
scale_y_continuous(breaks = c(10^-5, 2 * 10^-5, 3 * 10^-5, 4 * 10^-5, 5 * 10^-5, 6 * 10^-5), labels= fancy_scientific)

"""

######  supplementary table s1
using Random, DataFrames, DelimitedFiles, Statistics
import StatsBase: sem
ENV["COLUMNS"]=800

βΣcoverage = readdlm("test/result_files/beta_tau_coverage_5betas_multivariate_poisson_vcm5reps_sim.csv", ',')
# samplesizes = collect(1000:1000:6000)
# ns = [10; 25; 50; 100; 1000]
# nsims = 1000

m  = 2

covdf = DataFrame(Matrix(transpose(βΣcoverage)), :auto)
rename!(covdf, Symbol.([["β$i" for i in 1:p]; ["Σ$i" for i in 1:m]]))
covdf[!, :samplesize] = sample_sizes
covdf[!, :obssize] = obs_sizes
first(covdf, 10)

row_stats = [[(mean(col), sem(col)) for col = eachcol(d[!, 1:end-2])] for d = groupby(covdf, [:samplesize; :obssize])]
df = DataFrame(row_stats, :auto)
ss_obs = unique("N: " .* sample_sizes .* "  ni: " .* obs_sizes)
rename!(df, Symbol.(ss_obs))
covdfdisplay = DataFrame([[names(df)]; collect.(eachrow(df))], 
    [:people_obs; Symbol.([["β$i" for i in 1:p]; ["Σ$i" for i in 1:m]])])
# deletecols!(covdfdisplay, p + 2)
covdfdisplay


#### 
#import data and reorganize to create figure
using Random, DataFrames, DelimitedFiles, Statistics, RCall, Printf
import StatsBase: sem
ENV["COLUMNS"]=1000

βMseresultpoisson = vec(readdlm("test/result_files/mse_beta_multivariate_poisson_vcm5reps_sim.csv", ','))
ΣMseresultpoisson = vec(readdlm("test/result_files/mse_Sigma_multivariate_poisson_vcm5reps_sim.csv", ','))
βΣcoveragepoisson = readdlm("test/result_files/beta_tau_coverage_5betas_multivariate_poisson_vcm5reps_sim.csv", ',')


samplesizesrobust = samplesizes
nsrobust = ns
nsimsrobust = nsims

# βMseresultnorm = vec(readdlm("result_files/mse_beta_normal_normal_lognormal_1000reps_sim.csv", ','))
# τMseresultnorm = vec(readdlm("result_files/mse_tau_normal_normal_lognormal_1000reps_sim.csv", ','))
# ΣMseresultnorm = vec(readdlm("result_files/mse_Sigma_normal_normal_lognormal_1000reps_sim.csv", ','))
# βτcoveragenorm = readdlm("result_files/beta_tau_coverage_5betas_normal_normal_lognormal_1000reps_sim.csv", ',')


# samplesizesnorm = collect(1000:1000:6000)
# nsnorm = [10; 25; 50; 100; 1000]
# nsimsnorm = 1000

p = 3; m = 2
sample_sizesrobust = repeat(string.(samplesizesrobust), inner = nsimsrobust * length(nsrobust))
obs_sizesrobust = repeat(string.(nsrobust), inner = nsimsrobust, outer = length(samplesizesrobust))
msedfrobust = DataFrame(βmse = βMseresultpoisson, Σmse = ΣMseresultpoisson, 
    samplesize = sample_sizesrobust, obssize = obs_sizesrobust)

# p = 5; l = 5
# sample_sizesnorm = repeat(string.(samplesizesnorm), inner = nsimsnorm * length(nsnorm))
# obs_sizesnorm = repeat(string.(nsnorm), inner = nsimsnorm, outer = length(samplesizesnorm))
# msedfnorm = DataFrame(βmse = βMseresultnorm, τmse = τMseresultnorm, Σγmse = ΣMseresultnorm, 
#     samplesize = sample_sizesnorm, obssize = obs_sizesnorm)

#rename to make parsing easier for R.

msedfrobustR = deepcopy(msedfrobust)
rename!(msedfrobustR, ["betamse"
    "Sigmamse"
    "samplesize"
 "obssize"]);

# msedfnormR = deepcopy(msedfnorm)
# rename!(msedfnormR, ["betamse"
#     "taumse"
#     "Sigmamse"
#     "samplesize"
#  "obssize"]);

mses = [msedfrobustR[!, :betamse]; msedfrobustR[!, :Sigmamse]]
    # msedfnormR[!, :betamse]; msedfnormR[!, :taumse]; msedfnormR[!, :Sigmamse]]
obssize = collect([repeat(msedfrobustR[!, :obssize], 2)]...) # ; repeat(msedfnormR[!, :obssize], 3)]

samplesize = collect([repeat(msedfrobustR[!, :samplesize], 2)]...) # ; repeat(msedfnormR[!, :samplesize], 3)]

parameters = collect([repeat(string.([:beta, :Sigma]), inner = nsimsrobust * length(nsrobust) * length(samplesizesrobust))]...)
# ; repeat(string.([:beta, :tau, :Sigma]), inner = nsimsnorm * length(nsnorm) * length(samplesizesnorm))]
robust = collect([repeat(["Poisson with LogLink"], 2 * nsimsrobust * length(nsrobust) * length(samplesizesrobust))]...) # ;
# repeat(["Normal Normal Log-Normal"], 3 * nsimsnorm * length(nsnorm) * length(samplesizesnorm))];

msedfR = DataFrame(mse = mses, obssize = obssize, samplesize = samplesize,
    parameters = parameters, robust = robust)

#5 of the 30,000 simulations did not converge, filter out
msedfR = filter(x -> !isnan(x.mse), msedfR)
# additionally, there are 5 where an error was not caught so it did not try a differnt solver
msedfR = filter(x -> x.mse < 40000, msedfR)


### working on figure next
@rput msedfR

R"""
install.packages("facetscales")
library(scales)
library(ggplot2)
# library(facetscales)
library(data.table)

msedfR = data.table(msedfR)

msedfR[parameters == "beta",y_min := 10^-8]
msedfR[parameters == "beta",y_max := 10^1]
msedfR[parameters == "Sigma",y_min := 10^-5]
msedfR[parameters == "Sigma",y_max := 10^1]

#msedfR[parameters == "beta",y_min := 10^-8]
#msedfR[parameters == "beta",y_max := 10^-2]
#msedfR[parameters == "tau",y_min := 10^-8]
#msedfR[parameters == "tau",y_max := 10^-2]
#msedfR[parameters == "Sigma",y_min := 10^-5]
#msedfR[parameters == "Sigma",y_max := 10^-2]


msedfR$obssize = factor(msedfR$obssize, levels = c('5', '10', '20', '50'))
msedfR$samplesize = factor(msedfR$samplesize, levels = c('10000', '50000'))
msedfR$parameters = factor(msedfR$parameters, levels = c('beta', 'Sigma'), labels = c(beta = expression(hat(bold(beta))), Sigma = expression(hat(bold(Sigma))[bold(gamma)])))
msedfR$robust = factor(msedfR$robust, levels = c('Poisson with LogLink'),
    labels = c(expression(paste("Poisson with LogLink")))) # , expression(paste("MvT Gamma Inverse-Gamma"))))


#mseplot <- ggplot(msedfR[msedfR$mse < 10^0, ], aes(x=samplesize, y=mse, fill=obssize)) + 
mseplot = ggplot(msedfR[msedfR$mse < 10^3, ], aes(x=samplesize, y=mse, fill=obssize)) + 
  #geom_boxplot(outlier.size = 0.0, outlier.alpha = 0) +
  geom_boxplot(outlier.size = 0.25) +
#    geom_violin() +
    facet_grid(parameters ~ robust, labeller = label_parsed, scales = "free_y") +
  theme(legend.position="right", legend.key = element_blank(), axis.text.x = element_text(angle=0, size=11),
        axis.text.y = element_text(angle=0, size=12), axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15), legend.title = element_text(size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        legend.text=element_text(size=10), panel.background = element_rect(fill = NA, color = "black"),
        #panel.background = element_blank(), #panel.border = element_blank(),
        axis.line = element_line(color = 'black',size=0.3), plot.title = element_text(hjust = 0.5),
        strip.background = element_rect(colour="black", fill="white"), strip.text.y = element_text(size=15, face="bold"),
        strip.text.x = element_text(size=15)) + 
  scale_fill_manual(values = c("#c85f55",
"#a964bf",
"#8db352",
"#fa7300",
"#05aec0")) +
#scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x)) +
#scale_y_log10(breaks = 10^(-8:8), limit=c(10^-8, 10^2)) + 
scale_y_log10(breaks = 10^seq(-10, 10, 2), labels=trans_format("log10", math_format(10^.x))) + #, limit=c(10^-8, 10^2)) +
  labs(x = "Number of Individuals", y = "MSE of Parameter Estimates", fill = "Obs per Individual") +
geom_blank(aes(y = y_max)) + 
geom_blank(aes(y = y_min)) 

"""


end
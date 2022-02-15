using CSV, GLMCopula, GLM, DataFrames
df = CSV.read("survey_api_bivariate_binary.csv", DataFrame)
# https://cran.r-project.org/web/packages/ggeffects/vignettes/practical_logisticmixedmodel.html
groups = unique(df[!, :subject])
n, p, m = length(groups), 3, 1
d = Bernoulli()
link = ProbitLink()
D = typeof(d)
L = typeof(link)
T = Float64

gcs = Vector{GLMCopulaARObs{T, D, L}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = df[!, :subject] .== grp
    ni = count(gidx)
    y = Float64.(df[gidx, :score])
    cov1 = Float64.(df[gidx, :colgrad])
    xi = [1.0 cov1[1]]
    X = [xi zeros(size(xi)); zeros(size(xi)) xi]
    # V = [ones(ni, ni)]
    gcs[i] = GLMCopulaARObs(y, X, d, link)
end
gcm = GLMCopulaARModel(gcs);

initialize_model!(gcm)
@show gcm.β
@show gcm.ρ
@show gcm.σ2

# turn penalized on in logl autoregressive.jl

fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, limited_memory_max_history = 20, accept_after_max_steps = 1, hessian_approximation = "limited-memory"))
# fittime = @elapsed GLMCopula.fit!(gcm, IpoptSolver(print_level = 5, max_iter = 100, tol = 10^-5, hessian_approximation = "limited-memory"))
@show fittime
@show gcm.θ
@show gcm.∇θ
loglikelihood!(gcm, true, true)
vcov!(gcm)
@show GLMCopula.confint(gcm)

using CSV
#### 
dietox = CSV.read("dietox_pig_normal_randomintercept.csv")

## first get loglikelihood under the null (intercept only model)
groups = unique(dietox[!, :Pig])
n, p, m = length(groups), 1, 2
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dietox[!, :Pig] .== grp
    ni = count(gidx)
    y = Float64.(dietox[gidx, :Weight])
    start = Float64.(dietox[gidx, :Start])
    X = ones(ni, 1)
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
@time initialize_model!(gcm);
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
@time update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;

@show loglikelihood!(gcm, true, true)
# fit model using NLP on profiled loglikelihood
@info "MLE:"
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))

@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 20, hessian_approximation = "exact"))

@show gcm.β
@show gcm.τ
@show gcm.Σ
logl_null = loglikelihood!(gcm, true, true)

##### now get loglikelihood under the alternative model
groups = unique(dietox[!, :Pig])
n, p, m = length(groups), 1, 2
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
for (i, grp) in enumerate(groups)
    gidx = dietox[!, :Pig] .== grp
    ni = count(gidx)
    y = Float64.(dietox[gidx, :Weight])
    start = Float64.(dietox[gidx, :Start])
    X = [ones(ni) start]
    V = [ones(ni, ni)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
initialize_model!(gcm);
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;


@show loglikelihood!(gcm, true, true)
# fit model using NLP on profiled loglikelihood
@info "MLE:"
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))

@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 20, hessian_approximation = "exact"))

@show gcm.β
@show gcm.τ
@show gcm.Σ
logl_alternative = loglikelihood!(gcm, true, true)

lrt = 2 * (logl_alternative - logl_null)
# 40.031774014139046
pval = ccdf(Chisq(1), lrt)
# 2.49865242799851e-10
## 
#### 
df = CSV.read("ahs_random_noise_only.csv")
n, p, m = size(df, 1), 1, 2
d = Poisson()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
for i in 1:n
    y = Float64.([df[i, :Ndoc]; df[i, :Nmed]])
    # Time = Float64.(dietox[gidx, :Time])
    Income = Float64.([df[i, :income]; df[i, :income]])
    Age = Float64.([df[i, :age], df[i, :age]])
    Intercept = ones(2)
    X = [Intercept Income Age]
    V = [[1.0 1.0; 1.0 1.0], [1.0 0.0; 0.0 1.0]]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

# initialize β and τ from least square solution
@info "Initial point:"
initialize_model!(gcm);
@show gcm.β
# update σ2 and τ from β using the MM algorithm
fill!(gcm.Σ, 1)
# update_Σ!(gcm, 500, 1e-6, GurobiSolver(OutputFlag=0), true)
update_Σ!(gcm)
@show gcm.τ
@show gcm.Σ;


@show loglikelihood!(gcm, true, true)
# fit model using NLP on profiled loglikelihood
@info "MLE:"
# @time GLMCopula.fit!(gcm, IpoptSolver(print_level=5))
@time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))

# @time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))

@show gcm.β
@show gcm.τ
@show gcm.Σ
loglikelihood!(gcm, true, true)

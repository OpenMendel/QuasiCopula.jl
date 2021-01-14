using CSV
#### 
# GMMAT package in R, single vector outcome first logistic and then normal
# Unit: milliseconds
#                                                                                                                                                              expr
# min       lq     mean   median      uq      max neval
# 708.8659 736.6811 759.3661 757.6034 778.465 824.1027   100
# (Intercept)          age          sex 
#  0.472081189 -0.006818634 -0.086444746
β_GMMAT = [0.472081189; -0.006818634; -0.086444746]

df = CSV.read("GMMAT_example_pheno.csv")
kinship = CSV.read("GMMAT_example_pheno_GRM.csv")

n, p, m = 1, 3, 1
d = Bernoulli()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
y = Float64.(df[:, :disease])
cov1 = Float64.(df[:, :age])
cov2 = Float64.(df[:, :sex])
X = [ones(400, 1) cov1 cov2]
V = [Matrix(kinship[:, 2:end])]
gcs[1] = GLMCopulaVCObs(y, X, V, d)
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

@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
# 8 iterations 0.235414 seconds (86.68 k allocations: 9.564 MiB) ours is roughly 3 times faster
@show gcm.β
@show gcm.τ
@show gcm.Σ
logl_alternative = loglikelihood!(gcm, true, true)

### mvn
#  obj1 <- glmmkin(trait ~ age + sex, data = pheno, kins = kins,      id = "id", family = gaussian(link = "identity"), method = "REML",      method.optim = "AI")
# min       lq     mean   median       uq      max neval
# 425.6147 433.7323 441.7185 441.5609 447.2029 471.0377   100

n, p, m = 1, 3, 1
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
y = Float64.(df[:, :trait])
cov1 = Float64.(df[:, :age])
cov2 = Float64.(df[:, :sex])
X = [ones(400, 1) cov1 cov2]
V = [Matrix(kinship[:, 2:end])]
gcs[1] = GLMCopulaVCObs(y, X, V, d)
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

@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
# 7 iterations 0.240158 seconds (12.26 k allocations: 1.579 MiB) ours is roughly twice as fast
@show gcm.β
@show gcm.τ
@show gcm.Σ
logl_alternative = loglikelihood!(gcm, true, true)


## simulating to test using kinship for normal vector outcome 
@testset "Generate 10 independent multivariate normal vectors and then fit the model to test for the correct random intercepts and mean. " begin
Random.seed!(12345)
n = 400
variance_component_1 = 0.8
Γ = variance_component_1 * Matrix(kinship[:, 2:end])

cov1 = Float64.(df[:, :age])
cov2 = Float64.(df[:, :sex])
X = [ones(400, 1) cov1 cov2]
β = [3.75;  0.04;  0.42]
η = X * β
μ = η
noise_sd = 0.05
vecd = Vector{ContinuousUnivariateDistribution}(undef, length(μ))

for i in 1:length(μ)
    vecd[i] = Normal(μ[i], noise_sd)
end

nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)

Y = Vector{Float64}(undef, n)
res = Vector{Float64}(undef, n)
rand(nonmixed_multivariate_dist, Y, res)

@time rand(nonmixed_multivariate_dist, Y, res)

#### 
function simulate_nobs_independent_vectors(
    multivariate_distribution::Union{NonMixedMultivariateDistribution, MultivariateMix},
    n_obs::Integer)
    dimension = length(multivariate_distribution.vecd)
    Y = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    res = [Vector{Float64}(undef, dimension) for i in 1:n_obs]
    for i in 1:n_obs
        rand(multivariate_distribution, Y[i], res[i])
    end
    Y
end

nsample = 100
@info "sample $nsample independent vectors for the mvn distribution"
# compile
Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)
Random.seed!(12345)
@time Y_nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, nsample)

####

dim = 400
p, m = 1, 2
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, nsample)
for i in 1:nsample
    y = Float64.(Y_nsample[i])
    cov1 = Float64.(df[:, :age])
    cov2 = Float64.(df[:, :sex])
    X = [ones(400, 1) cov1 cov2]
    V = [Matrix(kinship[:, 2:end])]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)

GLMCopula.loglikelihood!(gcm, true, true)
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 500, hessian_approximation = "exact"))
# @time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, derivative_test = "first-order", hessian_approximation = "limited-memory"))

# check default ipopt quasi newton 
# then go back and check the hessian
GLMCopula.loglikelihood!(gcm, true, true)

println("estimated mean = $(gcm.β)); true mean value= $β")
println("estimated variance component = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
println("estimated noise variance = $(inv.(gcm.τ[1])); true noise variance = $(noise_sd^2)")


# variance component is not being estimated correctly when there is a kinship matrix, but can it be a nuisance parameter?

using CSV
df = CSV.read("GMMAT_example_pheno_normal.csv")
kinship = CSV.read("GMMAT_example_pheno_GRM.csv")

# estimates using the GMMAT package
β_GMMAT = [9.5621494;  0.5239570;   0.9176407 ]
## 
# obj1 <- glmmkin(y1 ~ x1 + x2, data = pheno, kins = kins, id = "id",      family = gaussian(link = "identity")) 
# min       lq     mean   median      uq      max neval
# 465.3032 483.569 497.6846 494.881 509.6123 559.5517   100

nsample = 1
dim = 400
p, m = 1, 2
d = Normal()
D = typeof(d)
gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, nsample)
for i in 1:nsample
    y = Float64.(df[:, :y1])
    cov1 = Float64.(df[:, :x1])
    cov2 = Float64.(df[:, :x2])
    X = [ones(400, 1) cov1 cov2]
    V = [Matrix(kinship[:, 2:end]), Matrix(I, dim, dim)]
    gcs[i] = GLMCopulaVCObs(y, X, V, d)
end
gcm = GLMCopulaVCModel(gcs);

initialize_model!(gcm)
@show gcm.β

fill!(gcm.Σ, 1.0)
update_Σ!(gcm)

GLMCopula.loglikelihood!(gcm, true, true)
# @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
@time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 500, hessian_approximation = "exact"))
# @time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, derivative_test = "first-order", hessian_approximation = "limited-memory"))
# 7 iterations 0.441992 seconds (61.38 k allocations: 4.117 MiB)
# check default ipopt quasi newton 
# then go back and check the hessian
GLMCopula.loglikelihood!(gcm, true, true)

println("estimated mean = $(gcm.β)); true mean = [10; 0.5; 0.8]; GMMAT package estimated mean = $β_GMMAT")
println("estimated variance component 1= $(gcm.Σ[1]); true variance component 1= 0.5; GMMAT estimated variance component 1 = 0.2277995")
println("estimated noise variance = $(inv.(gcm.τ[1])); true noise variance = $0.2; GMMAT estimated variance component 1 = 0.394363683")

end
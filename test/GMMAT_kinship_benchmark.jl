using CSV
using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

#### 
# GMMAT package in R, single vector outcome first logistic and then normal
# Unit: milliseconds
#                                                                                                                                                              expr
# min       lq     mean   median      uq      max neval
# 708.8659 736.6811 759.3661 757.6034 778.465 824.1027   100
# (Intercept)          age          sex 
#  0.472081189 -0.006818634 -0.086444746
# β_GMMAT = [0.472081189; -0.006818634; -0.086444746]

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

# β = [  0.38211856171039454,
#       -0.006737933268545437,
#       -0.08589569160997726]

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
#   0.3930374072758513
#  -0.00675502832790577
#  -0.0863285713022923
@show gcm.τ
@show gcm.Σ
logl = loglikelihood!(gcm, true, true)

### mvn
#  obj1 <- glmmkin(trait ~ age + sex, data = pheno, kins = kins, id = "id", family = gaussian(link = "identity"), method = "REML",      method.optim = "AI")
# min       lq     mean   median       uq      max neval
# 425.6147 433.7323 441.7185 441.5609 447.2029 471.0377   100

#β_GMMAT = [ 3.74899225;  0.03469714; 0.42159220 ]

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
#β = [3.76285440621963,
# 0.0337675153871073,
# 0.4215465759637164] 
#
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
logl = loglikelihood!(gcm, true, true)

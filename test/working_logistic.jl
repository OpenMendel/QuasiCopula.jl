println()
@info "Testing Logistic Regression on Ken's test data using glm_regress as starting point, then using GLM.jl for loglikelihood and solvers for updating beta"

using Convex, LinearAlgebra, MathProgBase, Reexport, GLM, Statistics, GLMCopula, Test
using LinearAlgebra: BlasReal, copytri!
# @reexport using Ipopt
# @reexport using NLopt

# # Ken's test data
# # I want to form the model using the test data Ken has, each obs object will be the same.
function create_gcm_logistic(n_groups, dist)
           n, p, m = n_groups, 1, 1
           D = typeof(dist)
           gcs = Vector{GLMCopulaVCObs{Float64, D}}(undef, n)
           #quantity = zeros(n)
           for i in 1:n_groups
             X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75,
                 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50];
             y = [0., 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1];
             X = [ones(size(X,1)) X];
             ni = length(y)
             V = [ones(ni, ni)]
             gcs[i] = GLMCopulaVCObs(y, X, V, dist)
           end
           gcms = GLMCopulaVCModel(Vector(gcs))
           return gcms
       end

logistic_model = create_gcm_logistic(2, Bernoulli());
logistic_β = [ -4.077713431087562
                 1.5046454283733053]
copyto!(logistic_model.β, logistic_β)
@show loglikelihood!(logistic_model, true, false) #≈  -8.029878464344675*2
@show logistic_model.∇β
@show logistic_model.∇Σ

# fit model using NLP on profiled loglikelihood
@info "MLE:"
@time GLMCopula.fit!(logistic_model, IpoptSolver(print_level=5))
@show logistic_model.β
@show logistic_model.Σ
@show loglikelihood!(logistic_model, true, false)
@show logistic_model.∇β
@show logistic_model.∇Σ

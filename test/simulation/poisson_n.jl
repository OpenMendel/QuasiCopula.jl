using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

@testset "Generate 10,000 independent bivariate poisson vectors and then fit the model to test for the correct random intercepts and mean. " begin
#### with single variance component 
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


function run_analysis(Y_Nsample, N, n, d)
    d = Poisson()
    link = LogLink()
    D = typeof(d)
    Link = typeof(link)
    T = Float64
    gcs = Vector{GLMCopulaVCObs{T, D, Link}}(undef, N)
    for i in 1:N
        y = Float64.(Y_Nsample[i])
        X = ones(n, 1)
        V = [ones(n, n)]
        gcs[i] = GLMCopulaVCObs(y, X, V, d, link)
    end
    gcm = GLMCopulaVCModel(gcs);
    
    initialize_model!(gcm)
    @show gcm.β
    
    fill!(gcm.Σ, 1.0)
    update_Σ!(gcm)
    
    GLMCopula.loglikelihood!(gcm, true, true)
    
    # @time GLMCopula.fit2!(gcm, IpoptSolver(print_level = 5, derivative_test = "first-order"))
    @time fit2!(gcm, IpoptSolver(print_level = 5, max_iter = 100, hessian_approximation = "exact"))
    #print results
    println("estimated mean = $(exp.(gcm.β)[1]); true mean value= $mean")
    println("estimated variance component 1 = $(gcm.Σ[1]); true variance component 1 = $variance_component_1")
    return exp.(gcm.β[1]), gcm.Σ[1]
end




function poisson_for_n_N2(N, n, variance_component_1, mean, dist, seed)
    df = []
    estimated_values = zeros(length(N), 2)
    # make the simulated data
    Γ = variance_component_1 * ones(n, n)
    vecd = [dist(mean) for i in 1:n]
    nonmixed_multivariate_dist = NonMixedMultivariateDistribution(vecd, Γ)
    for k in 1:length(N)
        Random.seed!(seed)
        Y_Nsample = simulate_nobs_independent_vectors(nonmixed_multivariate_dist, N[k])
        # run the analysis
        d = vecd[1]
        estimated_values[k, :] .= run_analysis(Y_Nsample, N[k], n, d);
    end
    push!(df, estimated_values)
    return df[1]
end

# sample size
N = [1000, 10_000, 100_000]
# N = [1000]
# observations per subject
n = [2, 5, 10, 20, 50, 100]

variance_component_1 = 0.1 

mean = 5

dist = Poisson

dfsol1 = poisson_for_n_N2(N[1], n[2], variance_component_1, mean, dist, 1234)
# n = 5 
#  5.03176  0.0921773 (24 iterations, 0.5 seconds)
#  5.01852  0.0918741 (28 iterations, 6 seconds)
#  5.00543  0.0994946 (25 iterations, 65 seconds)

dfsol2 = poisson_for_n_N2(N[1], n[3], variance_component_1, mean, dist, 12345)
# n = 10 
# 5.03749  0.106864 (23 iterations, 0.5 seconds)
# 5.01297  0.0998973 (21 iterations, 6.9 seconds)
# 5.00244  0.10011 (32 iterations, 132 seconds)

dfsol3 = poisson_for_n_N2(N[1], n[4], variance_component_1, mean, dist, 12345)
# n = 20 
# 5.01121  0.124562 (30 iterations, 1.3 seconds)
# 5.00258  0.0966973 (23 iterations, 13.78 seconds)
# 5.00224  0.100382 (22 iterations, 158.91 seconds)

dfsol4 = poisson_for_n_N2(N[1], n[5], variance_component_1, mean, dist, 12345)
# n = 50
# 5.0045   0.0937558 (28 iterations, 3.52 seconds)
# 4.99823  0.100744 (26 iterations, 36.23 seconds)
# 5.00169  0.100146 (31 iterations, 494.6 seconds)

dfsol5 = poisson_for_n_N2(N, n[6], variance_component_1, mean, dist, 12345)
# # n = 100
# 5.00334  0.0949427 (59 iterations, 17.4 seconds)


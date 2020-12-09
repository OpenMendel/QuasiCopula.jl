using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

### Poisson ### 
@testset "Poisson(θ) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(1234)
Γ = rand(5, 5)
dist = Poisson(5)

# test if the proper c0, c1, c2 constants are stored.
d_pois = marginal_pdf_constants(Γ, dist)
@test d_pois.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_pois.d)^2 * inv(var(d_pois.d)) * 0.5 * Γ[1,1])
@test d_pois.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_pois.d) * inv(var(d_pois.d)))
@test d_pois.c2 == 0.5 * Γ[1, 1] * (inv(var(d_pois.d)))

μ = 5
max_value = 25
pmf = pmf_copula(max_value, d_pois)
reordered_k, reordered_pmf = reorder_pmf(pmf, μ)
sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000
@info "sample $nsample points for the $dist distribution"
s = Vector{Float64}(undef, nsample)
discrete_rand!(max_value, d_pois, μ, s) # compile 
@time discrete_rand!(max_value, d_pois, μ, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(mean(d_pois))")
println("sample var = $(Statistics.var(s)); theoretical var = $(var(d_pois))")
end
# n = 10,0000 ::: time = 0.019234 seconds
# sample mean = 5.1773; theoretical mean = 5.186521822457729
# sample var = 7.017566466646667; theoretical var = 7.016949656782039

# ### Binomial ### 
@testset "Binomial(n, p) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
Γ = rand(5, 5)
n = 30
p = 0.1
dist = Binomial(n, p)

# test if the proper c0, c1, c2 constants are stored.
d_binomial = marginal_pdf_constants(Γ, dist)
@test d_binomial.c ≈ inv(1 + 0.5 * tr(Γ))
@test d_binomial.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_binomial.d)^2 * inv(var(d_binomial.d)) * 0.5 * Γ[1,1])
@test d_binomial.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_binomial.d) * inv(var(d_binomial.d)))
@test d_binomial.c2 == 0.5 * Γ[1, 1] * (inv(var(d_binomial.d)))

μ = mean(dist)
max_value = 30
pmf = pmf_copula(max_value, d_binomial)
reordered_k, reordered_pmf = reorder_pmf(pmf, μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution using the Bisection method."
s = Vector{Float64}(undef, nsample)
discrete_rand!(max_value, d_binomial, μ, s) # compile 
@time discrete_rand!(max_value, d_binomial, μ, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d_binomial))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d_binomial))")
end

# n = 10,0000 ::: time = 0.038302 seconds (40.00 k allocations: 12.817 MiB, 23.83% gc time)
# sample mean = 3.129; theoretical mean = 3.122671204329908
# sample var = 3.561715171517152; theoretical var = 3.583518347344844

# ### Geometric p = 0.2 ### 
@testset "Geometric(p = 0.2) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
Γ = rand(5, 5)
p = 0.2
dist = Geometric(p)

# test if the proper c0, c1, c2 constants are stored.
d_geometric = marginal_pdf_constants(Γ, dist)
@test d_geometric.c ≈ inv(1 + 0.5 * tr(Γ))
@test d_geometric.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_geometric.d)^2 * inv(var(d_geometric.d)) * 0.5 * Γ[1,1])
@test d_geometric.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_geometric.d) * inv(var(d_geometric.d)))
@test d_geometric.c2 == 0.5 * Γ[1, 1] * (inv(var(d_geometric.d)))

μ = mean(dist)
max_value = 200
pmf = pmf_copula(max_value, d_geometric)
reordered_k, reordered_pmf = reorder_pmf(pmf, μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Float64}(undef, nsample)
discrete_rand!(max_value, d_geometric, μ, s) # compile 
@time discrete_rand!(max_value, d_geometric, μ, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d_geometric))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d_geometric))")
end

# n = 10,0000 ::: time = 0.032615 seconds (40.00 k allocations: 34.180 MiB, 21.35% gc time)
# sample mean = 5.3313; theoretical mean = 5.380051048711484
# sample var = 42.31337164716471; theoretical var = 42.78303897434462


# ### Geometric p = 0.5### 
@testset "Geometric(p = 0.5) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
Γ = rand(5, 5)
p = 0.5
dist = Geometric(p)

# test if the proper c0, c1, c2 constants are stored.
d_geometric = marginal_pdf_constants(Γ, dist)
@test d_geometric.c ≈ inv(1 + 0.5 * tr(Γ))
@test d_geometric.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_geometric.d)^2 * inv(var(d_geometric.d)) * 0.5 * Γ[1,1])
@test d_geometric.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_geometric.d) * inv(var(d_geometric.d)))
@test d_geometric.c2 == 0.5 * Γ[1, 1] * (inv(var(d_geometric.d)))

μ = mean(dist)
max_value = 100
pmf = pmf_copula(max_value, d_geometric)
reordered_k, reordered_pmf = reorder_pmf(pmf, μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Float64}(undef, nsample)
discrete_rand!(max_value, d_geometric, μ, s) # compile 
@time discrete_rand!(max_value, d_geometric, μ, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d_geometric))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d_geometric))")
end

# n = 10,0000 ::: time = 0.025624 seconds (40.00 k allocations: 34.180 MiB, 33.37% gc time)
# sample mean = 1.4506; theoretical mean = 1.4600170162371608
# sample var = 4.338793519351937; theoretical var = 4.395147436782838

# ### Negative Binomial r = 5, p = 0.5 ### 
@testset "NegativeBinomial(r = 5, p = 0.5) * (c0 + c1 * x + c2 * x^2);" begin
Random.seed!(123)
Γ = rand(5, 5)
r, p = 5, 0.5
dist = NegativeBinomial(r, p)

# test if the proper c0, c1, c2 constants are stored.
d_nb = marginal_pdf_constants(Γ, dist)
@test d_nb.c ≈ inv(1 + 0.5 * tr(Γ))
@test d_nb.c0 == 1 + 0.5tr(Γ[2:end, 2:end]) + (mean(d_nb.d)^2 * inv(var(d_nb.d)) * 0.5 * Γ[1,1])
@test d_nb.c1 == 0.5 * Γ[1, 1] * (-2 * mean(d_nb.d) * inv(var(d_nb.d)))
@test d_nb.c2 == 0.5 * Γ[1, 1] * (inv(var(d_nb.d)))

μ = mean(dist)
max_value = 50
pmf = pmf_copula(max_value, d_nb)
reordered_k, reordered_pmf = reorder_pmf(pmf, μ)
@test sum(reordered_pmf) ≈ 1

###
Random.seed!(1234)
nsample = 10_000 #
@info "sample $nsample points for the $dist distribution."
s = Vector{Float64}(undef, nsample)
discrete_rand!(max_value, d_nb, μ, s) # compile 
@time discrete_rand!(max_value, d_nb, μ, s) # get time
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d_nb))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d_nb))")
end

# n = 10,0000 ::: time = 0.142361 seconds (40.00 k allocations: 18.921 MiB, 57.42% gc time)
# sample mean = 5.3987; theoretical mean = 5.4600170162371615
# sample var = 14.753013611361133; theoretical var = 14.848571523381025
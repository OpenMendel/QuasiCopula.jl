module RandTest

using GLMCopula, Random, Statistics, Test, LinearAlgebra, StatsFuns

# Normal
@testset "Normal(0,1) * (1 + 0.5 x^2)" begin ## where x = mean(d)
d = ContinuousUnivariateCopula(Normal(), 1.0, 0.0, 0.5)
@test d.c ≈ 2/3 
@test mean(d) == 0
@test var(d) ≈ 5/3
@test minimum(d) == -Inf
@test maximum(d) == Inf
@test cdf(d, -Inf) == 0
@test cdf(d, 0) == 0.5
@test cdf(d, Inf) == 1

Random.seed!(123)
nsample = 1_000_000
@info "sample $nsample points"
s = Vector{Float64}(undef, nsample)
rand!(d, s) # compile
@time rand!(d, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d))")

# sample mean = -0.0023492396407176915; theoretical mean = 0.0
# sample var = 1.6680321316234767; theoretical var = 1.6666666666666665
end

# GAMMA
# when α = 1.0 = θ
@testset "Gamma(α = 1.0, θ = 1.0) * (c0 + c1 * x + c2 * x^2)" begin ## where x = mean(d)
α, θ = (1.0, 1.0)
dist = Gamma(α, θ)
μ = mean(dist)
σ2 = var(dist)

Random.seed!(1234)
Γ = rand(5, 5)
μ = mean(dist)
σ2 = var(dist)
c_0 = μ^2 * inv(σ2)
c_1 = -2μ * inv(σ2)
c_2 = inv(σ2)

c = inv(1 + 0.5*tr(Γ))
c0 = 1 + 0.5 * tr(Γ[2:end, 2:end]) + 0.5 * Γ[1, 1] * c_0
c1 = 0.5 * Γ[1, 1] * c_1 # * (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) # note this is negative 
c2 = 0.5 * Γ[1, 1] * c_2 # * (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

d1 = ContinuousUnivariateCopula(dist, c0, c1, c2)
@test d1.c == c
a1 = (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α))
a2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

@test d1.c * (d1.c0 + a1 * d1.c1 + a2 * d1.c2) == 1.0
@test minimum(d1) == 0
@test maximum(d1) == Inf
@test cdf(d1, 0) == 0
@test cdf(d1, Inf) == 1

### 
Random.seed!(123)
nsample = 1000 # 0
@info "sample $nsample points"
s = Vector{Float64}(undef, nsample)
rand!(d1, s) # compile
@time rand!(d1, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d1))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d1))")
end

# sample mean = 1.3058253300320248; theoretical mean = 1.316327062152223
# sample var = 2.1461815984533175; theoretical var = 2.165245438359036

# when α = 2.5, θ = 1.0
@testset "Gamma(α = 2.5, θ = 1.0) * (c0 + c1 * x + c2 * x^2)" begin ## where x = mean(d)
α, θ = (2.5, 1.0)
dist = Gamma(α, θ)
μ = mean(dist)
σ2 = var(dist)

Random.seed!(1234)
Γ = rand(5, 5)
μ = mean(dist)
σ2 = var(dist)
c_0 = μ^2 * inv(σ2)
c_1 = -2μ * inv(σ2)
c_2 = inv(σ2)

c = inv(1 + 0.5*tr(Γ))
c0 = 1 + 0.5 * tr(Γ[2:end, 2:end]) + 0.5 * Γ[1, 1] * c_0
c1 = 0.5 * Γ[1, 1] * c_1 # * (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) # note this is negative 
c2 = 0.5 * Γ[1, 1] * c_2 # * (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

d1 = ContinuousUnivariateCopula(dist, c0, c1, c2)
@test d1.c ≈ c
a1 = (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α))
a2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

@test d1.c * (d1.c0 + a1 * d1.c1 + a2 * d1.c2) == 1.0
@test minimum(d1) == 0
@test maximum(d1) == Inf
@test cdf(d1, 0) == 0
@test cdf(d1, Inf) == 1

### 
Random.seed!(123)
nsample = 1_000 #_000
@info "sample $nsample points"
s = Vector{Float64}(undef, nsample)
rand!(d1, s) # compile
@time rand!(d1, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d1))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d1))")
end

# sample mean = 2.8034635122593636; theoretical mean = 2.8163270621522227
# sample var = 4.10035387850464; theoretical var = 4.139736031587371

## when there is a lot of noise there is some deviation from truth 
@testset "Gamma(α = 3.0, θ = 2.0) * (c0 + c1 * x + c2 * x^2); Lots of nosie ==> deviation" begin ## where x = mean(d)
α, θ = (3.0, 2.0)
dist = Gamma(α, θ)
μ = mean(dist)
σ2 = var(dist)

Random.seed!(1234)
Γ = rand(5, 5)
μ = mean(dist)
σ2 = var(dist)
c_0 = μ^2 * inv(σ2)
c_1 = -2μ * inv(σ2)
c_2 = inv(σ2)

c = inv(1 + 0.5*tr(Γ))
c0 = 1 + 0.5 * tr(Γ[2:end, 2:end]) + 0.5 * Γ[1, 1] * c_0
c1 = 0.5 * Γ[1, 1] * c_1 # * (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α)) # note this is negative 
c2 = 0.5 * Γ[1, 1] * c_2 # * (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

d1 = ContinuousUnivariateCopula(dist, c0, c1, c2)
@test d1.c == c
a1 = (StatsFuns.gamma(α + 1)/ StatsFuns.gamma(α))
a2 = (StatsFuns.gamma(α + 2)/ StatsFuns.gamma(α))

@test d1.c * (d1.c0 + a1 * d1.c1 + a2 * d1.c2) == 1.0
@test minimum(d1) == 0
@test maximum(d1) == Inf
@test cdf(d1, 0) == 0
@test cdf(d1, Inf) == 1

### 
Random.seed!(123)
nsample = 1_000 #_000
@info "sample $nsample points"
s = Vector{Float64}(undef, nsample)
rand!(d1, s) # compile
@time rand!(d1, s)
println("sample mean = $(Statistics.mean(s)); theoretical mean = $(GLMCopula.mean(d1))")
println("sample var = $(Statistics.var(s)); theoretical var = $(GLMCopula.var(d1))")
end
# sample mean = 5.663051756167577; theoretical mean = 6.632654124304445
# sample var = 11.845482796775215; theoretical var = 19.191598250653946
end

# 1. Next, we will check benchmarking for normal vs gamma
# using BenchmarkTools
# @benchmark rand!($d1, $s)

# 2. And then we will derive, code up and test the exponential and beta base densities. 

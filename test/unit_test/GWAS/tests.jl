using QuasiCopula
using BenchmarkTools
using GLM
using Distributions
using Test
using LinearAlgebra

@testset "utilities" begin
    # ∇²μ_j vs ∇²μ_j! (p × p)
    p = 10
    storage = zeros(p, p)
    xj = randn(p)
    l = GLM.LogitLink()
    ηj = randn()
    c = rand()
    QuasiCopula.∇²μ_j!(storage, l, ηj, xj, c)
    @test all(c .* QuasiCopula.∇²μ_j(l, ηj, xj) .≈ storage)

    # ∇²μ_j vs ∇²μ_j! (p × 1)
    storage = zeros(p)
    xj = randn(p)
    zj = randn()
    l = GLM.LogitLink()
    ηj = randn()
    c = rand()
    QuasiCopula.∇²μ_j!(storage, l, ηj, xj, zj, c)
    @test all(c .* QuasiCopula.∇²μ_j(l, ηj, xj, zj) .≈ storage)

    # ∇²σ²_j vs ∇²σ²_j! (p × p)
    p = 10
    storage = zeros(p, p)
    xj = randn(p)
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    ηj = randn()
    μj = rand()
    c = rand()
    QuasiCopula.∇²σ²_j!(storage, d, l, xj, μj, ηj, c)
    @test all(c .* QuasiCopula.∇²σ²_j(d, l, xj, μj, ηj) .≈ storage)

    # ∇²σ²_j vs ∇²σ²_j! (p × 1)
    p = 10
    storage = zeros(p)
    xj = randn(p)
    zj = randn()
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    ηj = randn()
    μj = rand()
    c = randn()
    QuasiCopula.∇²σ²_j!(storage, d, l, xj, μj, ηj, zj, c)
    @test all(c .* QuasiCopula.∇²σ²_j(d, l, xj, μj, ηj, zj) .≈ storage)

    # dγdβresβ_ij vs dγdβresβ_ij! (later should be efficient)
    n = 100
    p = 100
    d = GLM.Bernoulli()
    l = GLM.LogitLink()
    xj = randn(p)
    z = randn()
    η_j = randn()
    μ_j = randn()
    varμ_j = rand()
    res_j = randn()
    maxd = 10
    storage = QuasiCopula.storages(p, maxd, 2)
    W = zeros(p)
    c = randn()
    QuasiCopula.dγdβresβ_ij!(W, d, l, xj, z, η_j, μ_j, varμ_j, res_j, c, storage)
    @test all(-c .* QuasiCopula.dγdβresβ_ij(d, l, xj, z, η_j, μ_j, varμ_j, res_j) .≈ W)
    b = @benchmark QuasiCopula.dγdβresβ_ij!($W, $d, $l, $xj, $z, $η_j, $μ_j, $varμ_j, $res_j, $c, $storage)
    @test b.allocs == 0
    @test b.memory == 0

    # get_Hβγ_i vs get_Hβγ_i! (later should be efficient)
    p = 15
    maxd = 10
    m = 2
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        d_min=5, d_max=maxd, m=m, p=p)
    i = 1
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = QuasiCopula.get_∇resγ(qc_model, i, z) # d × 1
    storages = QuasiCopula.storages(p, maxd, m)
    denom = 1 + dot(qc_model.θ, qc.q) # same as denom = 1 + 0.5 * (res' * Γ * res), since dot(θ, qc.q) = qsum = 0.5 r'Γr
    denom2 = abs2(denom)
    storages.denom[1] = denom
    storages.denom2[1] = denom2
    W = zeros(p)
    QuasiCopula.get_Hβγ_i!(W, qc, Γ, qc.∇resβ, ∇resγ, z, storages)
    @test all(QuasiCopula.get_Hβγ_i(qc, Γ, qc.∇resβ, ∇resγ, z, storages) .≈ W)
    b = @benchmark QuasiCopula.get_Hβγ_i!($W, $qc, $Γ, $(qc.∇resβ), $∇resγ, $z, $storages)
    @test b.allocs == 0
    @test b.memory == 0

    # get_neg_Hθγ_i! vs get_neg_Hθγ_i (later should be efficient)
    p = 15
    maxd = 10
    m = 2
    qc_model, G, βtrue, θtrue, γtrue, τtrue = simulate_longitudinal_traits(
        d_min=5, d_max=maxd, m=m, p=p)
    i = 1
    qc = qc_model.data[i]
    Γ = qc.V[1] * θtrue[1] + qc.V[2] * θtrue[2]
    z = randn(qc.n)
    ∇resγ = QuasiCopula.get_∇resγ(qc_model, i, z) # d × 1
    storages = QuasiCopula.storages(p, maxd, m)
    W = zeros(m)
    QuasiCopula.get_neg_Hθγ_i!(W, qc, θtrue, ∇resγ, storages)
    @test all(QuasiCopula.get_neg_Hθγ_i(qc, θtrue, ∇resγ, storages) .≈ W)
    b = @benchmark QuasiCopula.get_neg_Hθγ_i!($W, $qc, $θtrue, $∇resγ, $storages)
    @test b.allocs == 0
    @test b.memory == 0

end

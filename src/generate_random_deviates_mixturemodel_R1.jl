"""
generate_R1_mixture_Normal
generate_R1_mixture_Normal()
Let R1~N(0, 1) and create first vector of residuals R_1 as a mixture of 3 distributions with mixing probabilities.
"""
function generate_R1_mixture_Normal(d::Distributions.Distribution)
    csamplers = map(sampler, d.components)
    psampler = sampler(d.prior)
    random_deviate = csamplers[rand(psampler)]
    
    if typeof(random_deviate) == Normal{Float64}
        println("using standard normal")
        return rand(random_deviate)
    else
        println("if chi (3), one is positive and one is negative with equal probabilty")
        return rand([-1, 1]) * rand(random_deviate)
    end
end
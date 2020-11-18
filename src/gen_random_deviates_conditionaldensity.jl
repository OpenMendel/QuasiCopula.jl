### joint and conditional density

function joint_density_value(density::D, res::Vector{T}) where {T <: BlasReal, D <: Distributions.UnivariateDistribution}  
    pdf_vector = pdf.(density, res)
    joint_pdf = 1.0
    for i in 1:length(pdf_vector)
        joint_pdf = pdf_vector[i] * joint_pdf
    end
    return joint_pdf
 end
 
 # this function will get the cross terms for s, and all the cross terms up to s if all = true; used in marginal density of i in S
 function crossterm_res(res::Vector{T}, s::Integer, Γ::Matrix{T}; all = false) where {T<: BlasReal}
    results = []
    if s == 1
        return 0.0
    elseif s > 1
        if all == true
            for i in 2:s
                for j in 1:i - 1
                    push!(results, res[i] * sum(res[j] * Γ[i, j]))
                end
            end
        else
            for j in 1:s - 1
                push!(results, res[s] * sum(res[j] * Γ[s, j]))
            end
        end
    end
    return results
 end

###
# now we want to look at the way the root finding package works, and define the conditional density of Ri | R1, ..., Ri-1 as a function of x = R_i.

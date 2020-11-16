### joint and conditional density

function joint_density_value(density, res)
    pdf_vector = pdf(density, res)
    joint_pdf = 1.0
    for i in 1:length(pdf_vector)
        joint_pdf = pdf_vector[i] * joint_pdf
    end
    return joint_pdf
 end
 
 # this function will get the cross terms for s, and all the cross terms up to s if all = true; used in marginal density of i in S
 function crossterm_res(res, s; all = false)
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

## for s = 3 check the conditional recursive formula ############################################################################################################################################
# term1_s3 = 1 + 0.5 * transpose(res[1:s-1]) * Γ[1:s-1, 1:s-1] * res[1:s-1] +  0.5 * tr(Γ[s:end, s:end])

# t1 = 1 + 0.5 * (Γ[1, 1] * res[1]^2 + Γ[2, 2] * res[2]^2) + res[1] * res[2] * Γ[1, 2] + 0.5 * tr(Γ[s:end, s:end])
# @test term1_s3 == t1
# # 2.9808259645359545

# term2_s3 = sum(crossterm_res(res, s))
# # 0.27113634223316985

# term3_s3 = (0.5 * Γ[s, s] * (res[s]^2 - 1))
# # -0.4617767542199328

# conditional_r3_r12 = inv(term1_s3) * pdf(d, res[s]) * (term1_s3 + term2_s3 + term3_s3)
# # 0.35942343153750456

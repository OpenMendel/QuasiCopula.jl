using CSV, DataFrames, GLMCopula, LinearAlgebra, GLM, RCall, RData, RDatasets
using Test
# we will use this example dataset to make sure the model interface is working with CS structure
R"""
    library("gcmr")
    data("epilepsy", package = "gcmr")
"""
@rget epilepsy;

df = epilepsy
y = :counts
grouping = :id
covariates = [:visit, :trt]
d = Poisson()
link = LogLink()

# forming CS model with Poisson base
Poisson_CS_model = CS_model(df, y, grouping, covariates, d, link);

@test typeof(Poisson_CS_model) == GLMCopulaCSModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_CS_model.penalized == false

# forming CS model with Poisson base with L2 ridge penalty
Poisson_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(Poisson_CS_model_penalized) == GLMCopulaCSModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_CS_model_penalized.penalized == true

#### nb ####
d = NegativeBinomial()
link = LogLink()
nb_CS_model = CS_model(df, y, grouping, covariates, d, link);

@test typeof(nb_CS_model) == NBCopulaCSModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_CS_model.penalized == false

# forming CS model with nb base with L2 ridge penalty
nb_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(nb_CS_model_penalized) == NBCopulaCSModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_CS_model_penalized.penalized == true

#### gaussian ####
d = Normal()
link = IdentityLink()
gaussian_CS_model = CS_model(df, y, grouping, covariates, d, link);

@test typeof(gaussian_CS_model) == GaussianCopulaCSModel{Float64}
@test gaussian_CS_model.penalized == false

# forming CS model with gaussian base with L2 ridge penalty
gaussian_CS_model_penalized = CS_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(gaussian_CS_model_penalized) == GaussianCopulaCSModel{Float64}
@test gaussian_CS_model_penalized.penalized == true

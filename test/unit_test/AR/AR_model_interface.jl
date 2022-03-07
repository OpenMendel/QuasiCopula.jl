using CSV, DataFrames, GLMCopula, LinearAlgebra, GLM, RCall, RData, RDatasets
using Test
# we will use this example dataset to make sure the model interface is working with autoregressive structure
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

#### POISSON ####
# forming AR(1) model with Poisson base
Poisson_AR_model = AR_model(df, y, grouping, covariates, d, link);

@test typeof(Poisson_AR_model) == GLMCopulaARModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_AR_model.penalized == false

# forming AR(1) model with Poisson base with L2 ridge penalty
Poisson_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(Poisson_AR_model_penalized) == GLMCopulaARModel{Float64, Poisson{Float64}, LogLink}
@test Poisson_AR_model_penalized.penalized == true

#### nb ####
d = NegativeBinomial()
link = LogLink()
nb_AR_model = AR_model(df, y, grouping, covariates, d, link);

@test typeof(nb_AR_model) == NBCopulaARModel{Float64, NegativeBinomial{Float64}, LogLink}
@test Poisson_AR_model.penalized == false

# forming AR(1) model with nb base with L2 ridge penalty
nb_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(nb_AR_model_penalized) == NBCopulaARModel{Float64, NegativeBinomial{Float64}, LogLink}
@test nb_AR_model_penalized.penalized == true

#### gaussian ####
d = Normal()
link = IdentityLink()
gaussian_AR_model = AR_model(df, y, grouping, covariates, d, link);

@test typeof(gaussian_AR_model) == GaussianCopulaARModel{Float64}
@test gaussian_AR_model.penalized == false

# forming AR(1) model with gaussian base with L2 ridge penalty
gaussian_AR_model_penalized = AR_model(df, y, grouping, covariates, d, link; penalized = true);

@test typeof(gaussian_AR_model_penalized) == GaussianCopulaARModel{Float64}
@test gaussian_AR_model_penalized.penalized == true

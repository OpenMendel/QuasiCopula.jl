# VC Covariance

In this notebook we will demonstrate how to fit a variance component model (VCM) with Poisson base on an example dataset, "Mmmec", from the `mlmRev` R package. We will access the data using the `RDatasets` Julia package. For these examples, the variance components parameterization of the covariance matrix for the $i^{th}$ cluster is given by $$\mathbf{\Gamma_i}(\boldsymbol{\theta}) = \sum_{k = 1}^m \theta_k * \mathbf{V_{ik}}$$ 

where $m$ is the number of variance components, which are arranged in a vector $\boldsymbol{\theta} = \{\theta_1, ..., \theta_m \}$ for estimation. 

### Table of Contents:
* [Example 1: Single VCM (mlmRev:Mmmec)](#Example-1:-Single-VCM-(Random-Intercept-Model))
* [Example 2: Multiple VCM (mlmRev:Mmmec)](#Example-2:-Multiple-VCM)



```julia
versioninfo()
```

    Julia Version 1.6.2
    Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin18.7.0)
      CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-11.0.1 (ORCJIT, skylake)



```julia
using CSV, DataFrames, GLMCopula, LinearAlgebra, GLM, RCall, RData, RDatasets
```

## Example 1: Single VCM (Random Intercept Model)

We first demonstrate how to fit the model with Poisson base and a single variance component using the "Mmmec" dataset from the "mlmRev" package in R. 

By default, the `VC_model` function will construct a random intercept model using a single variance component. That is, it will parameterize $\boldsymbol{\Gamma_i}(\boldsymbol{\theta})$ for each cluster $i$ with cluster size ${d_i}$ as follows:

$$\mathbf{\Gamma_i}(\boldsymbol{\theta}) = \theta_1 * \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t$$



```julia
Mmmec = dataset("mlmRev", "Mmmec");
```

Let's take a preview of the first 20 lines of the Mmmec dataset.


```julia
@show Mmmec[1:20, :];
```

    Mmmec[1:20, :] = 20×6 DataFrame
     Row │ Nation     Region  County  Deaths  Expected  UVB
         │ Cat…       Cat…    Cat…    Int32   Float64   Float64
    ─────┼──────────────────────────────────────────────────────
       1 │ Belgium    1       1           79   51.222   -2.9057
       2 │ Belgium    2       2           80   79.956   -3.2075
       3 │ Belgium    2       3           51   46.5169  -2.8038
       4 │ Belgium    2       4           43   55.053   -3.0069
       5 │ Belgium    2       5           89   67.758   -3.0069
       6 │ Belgium    2       6           19   35.976   -3.4175
       7 │ Belgium    3       7           19   13.28    -2.6671
       8 │ Belgium    3       8           15   66.5579  -2.6671
       9 │ Belgium    3       9           33   50.969   -3.1222
      10 │ Belgium    3       10           9   11.171   -2.4852
      11 │ Belgium    3       11          12   19.683   -2.5293
      12 │ W.Germany  4       12         156  108.04    -1.1375
      13 │ W.Germany  4       13         110   73.692   -1.3977
      14 │ W.Germany  4       14          77   57.098   -0.4386
      15 │ W.Germany  4       15          56   46.622   -1.0249
      16 │ W.Germany  5       16         220  112.61    -0.5033
      17 │ W.Germany  5       17          46   30.334   -1.4609
      18 │ W.Germany  5       18          47   29.973   -1.8956
      19 │ W.Germany  5       19          50   32.027   -2.5541
      20 │ W.Germany  5       20          90   46.521   -1.9671


#### Forming the Model

To form the model, we give it the following arguments:

- named dataframe
- outcome variable name of interest as a symbol
- grouping variable name of interest as a symbol
- covariate names of interest as a vector of symbols
- base distribution
- link function



```julia
df = Mmmec
y = :Deaths
grouping = :Region
covariates = [:UVB]
d = Poisson()
link = LogLink()

Poisson_VC_model = VC_model(df, y, grouping, covariates, d, link);
```

Fit the model. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
GLMCopula.fit!(Poisson_VC_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    gcm.β = [3.25512456536309, -0.07968965522100831]
    initializing variance components using MM-Algorithm
    gcm.θ = [1.0]
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    Total number of variables............................:        3
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 56
    
                                       (scaled)                 (unscaled)
    Objective...............:   1.5124244154849350e+03    5.8200416853931556e+03
    Dual infeasibility......:   4.3338249104395330e-07    1.6677224579098036e-06
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.0909090909090901e-08    3.4983215905203376e-07
    Overall NLP error.......:   4.3338249104395330e-07    1.6677224579098036e-06
    
    
    Number of objective function evaluations             = 77
    Number of objective gradient evaluations             = 57
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 56
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.078
    Total CPU secs in NLP function evaluations           =      0.041
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_VC_model.β
@show Poisson_VC_model.θ;
```

    Poisson_VC_model.β = [3.2954590224335103, -0.07184274041348349]
    Poisson_VC_model.θ = [50.20211632159837]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Poisson_VC_model, false, false);
```

    loglikelihood!(Poisson_VC_model, false, false) = -5820.041685393156


## Example 2: Multiple VCM

Next we demonstrate how to fit the model with Poisson base and two variance components using the "Mmmec" dataset from the "mlmRev" package in R. 

To specify our own positive semi-definite covariance matrices, we need to make sure the dimensions match that of each cluster size $d_i$. To illustrate, we will add an additional variance component proportional to the Identity matrix to the random intercept model above to help capture overdispersion. More explicitly, I will make $\mathbf{V_{i1}} = \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t$ and $\mathbf{V_{i2}} = \mathbf{I_{d_i}}$ to parameterize $\mathbf{\Gamma_i}(\boldsymbol{\theta})$ follows:

$$\mathbf{\Gamma_i}(\boldsymbol{\theta}) = \theta_1 * \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t + \theta_2 * \mathbf{I_{d_i}}$$


```julia
function make_Vs(
    df::DataFrame,
    y::Symbol,
    grouping::Symbol)
    groups = unique(df[!, grouping])
    n = length(groups)
    V = Vector{Vector{Matrix{Float64}}}(undef, n)
    for (i, grp) in enumerate(groups)
        gidx = df[!, grouping] .== grp
        ni = count(gidx)
        V[i] = [ones(ni, ni), Matrix(I, ni, ni)]
    end
    V
end
```




    make_Vs (generic function with 1 method)




```julia
V = make_Vs(df, y, grouping);
```

#### Forming the Model

To form the model, we give it the following arguments:

- named dataframe
- outcome variable name of interest as a symbol
- grouping variable name of interest as a symbol
- covariate names of interest as a vector of symbols
- Vector of Vector of PSD Covariance Matrices
- base distribution
- link function



```julia
Poisson_2VC_model = VC_model(df, y, grouping, covariates, V, d, link);
```

Fit the model. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
GLMCopula.fit!(Poisson_2VC_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    gcm.β = [3.25512456536309, -0.07968965522100831]
    initializing variance components using MM-Algorithm
    gcm.θ = [1.0, 1.0]
    Total number of variables............................:        4
                         variables with only lower bounds:        2
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 47
    
                                       (scaled)                 (unscaled)
    Objective...............:   1.8794436243388843e+03    5.8110834414104884e+03
    Dual infeasibility......:   4.8397116120453827e-07    1.4963985961458093e-06
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.0909090909090531e-08    2.8108335148457850e-07
    Overall NLP error.......:   4.8397116120453827e-07    1.4963985961458093e-06
    
    
    Number of objective function evaluations             = 48
    Number of objective gradient evaluations             = 48
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 47
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.021
    Total CPU secs in NLP function evaluations           =      0.011
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_2VC_model.β
@show Poisson_2VC_model.θ;
```

    Poisson_2VC_model.β = [3.2673642555712115, -0.07503633126910883]
    Poisson_2VC_model.θ = [441832.1877029869, 160836.08322258486]


Calculate the loglikelihood at the maximum


```julia
@show loglikelihood!(Poisson_2VC_model, false, false);
```

    loglikelihood!(Poisson_2VC_model, false, false) = -5811.083441410488



```julia

```

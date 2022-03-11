# Compound Symmetric (CS) Covariance

In this notebook we will show how to form the quasi-copula model with Bernoulli base distribution, Logit Link function and compound symmetric (CS) structured covariance on the `respiratory` dataset provided in the `geepack` R package. 

### Table of Contents:
* [Example 1: Intercept Only CS Model](#Example-1:-Intercept-Only-CS-Model)
* [Example 2: CS Model with Covariates](#Example-2:-CS-Model-with-Covariates)
* [Example 3: CS Model with Covariates + L2 Penalty (optional)](#Example-3:-CS-Model-with-Covariates-L2-penalty-(optional))

For these examples, we have $n$ independent clusters indexed by $i$. 

Under the CS parameterization of the covariance matrix, the $i^{th}$ cluster with cluster size $d_i$, has covariance matrix $\mathbf{\Gamma_i}$ that takes the form: 

$$\mathbf{\Gamma_i}(\rho, \sigma^2) =  \sigma^2 * \Big[ \rho * \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t + (1 - \rho) * \mathbf{I_{d_i}} \Big]$$


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
using DataFrames, QuasiCopula, LinearAlgebra, GLM, RCall
```


```julia
R"""
    data(respiratory, package="geepack")
    respiratory_df <- respiratory[order(respiratory$id),]
"""

@rget respiratory_df;
```

Let's take a preview of the first 10 lines of the respiratory dataset in long format.


```julia
@show respiratory_df[1:10, :];
```

    respiratory_df[1:10, :] = 10×8 DataFrame
     Row │ center  id     treat  sex   age    baseline  visit  outcome
         │ Int64   Int64  Cat…   Cat…  Int64  Int64     Int64  Int64
    ─────┼─────────────────────────────────────────────────────────────
       1 │      1      1  P      M        46         0      1        0
       2 │      1      1  P      M        46         0      2        0
       3 │      1      1  P      M        46         0      3        0
       4 │      1      1  P      M        46         0      4        0
       5 │      2      1  P      F        39         0      1        0
       6 │      2      1  P      F        39         0      2        0
       7 │      2      1  P      F        39         0      3        0
       8 │      2      1  P      F        39         0      4        0
       9 │      1      2  P      M        28         0      1        0
      10 │      1      2  P      M        28         0      2        0


## Forming the Models

We can form the CS model for regression with following arguments:

##### Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `covariates`: Covariate names of interest as a vector of `Symbol`s.
    Each variable name must be present in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.
- `link`: Canonical `Link` function of the base distribution specified in `d`, from `GLM.jl`.

##### Optional Arguments
- `penalized`: Boolean to specify whether or not to add an L2 Ridge penalty on the variance parameter for the AR(1) structured covariance.
    One can put true (e.g. `penalized = true`) to add this penalty for numerical stability (default `penalized = false`).

### Example 1: Intercept Only CS Model

We can form the CS model with intercept only by excluding the `covariates` argument.


```julia
df = respiratory_df
y = :outcome
grouping = :id
d = Bernoulli()
link = LogitLink()

Bernoulli_CS_model = CS_model(df, y, grouping, d, link)
```




    Quasi-Copula Compound Symmetric CS Model
      * base distribution: Bernoulli
      * link function: LogitLink
      * number of clusters: 56
      * cluster size min, max: 4, 8
      * number of fixed effects: 1




### Example 2: CS Model with Covariates

We can form the CS model with covariates by including the `covariates` argument.


```julia
covariates = [:center, :age, :baseline]

Bernoulli_CS_model = CS_model(df, y, grouping, covariates, d, link)
```




    Quasi-Copula Compound Symmetric CS Model
      * base distribution: Bernoulli
      * link function: LogitLink
      * number of clusters: 56
      * cluster size min, max: 4, 8
      * number of fixed effects: 4




### Example 3: CS Model with Covariates + L2 penalty (optional)

We can form the same CS model from Example 2 with the optional argument for adding the L2 penalty on the variance parameter in the CS parameterization of Gamma.


```julia
Bernoulli_CS_model = CS_model(df, y, grouping, covariates, d, link; penalized = true)
```




    Quasi-Copula Compound Symmetric CS Model
      * base distribution: Bernoulli
      * link function: LogitLink
      * number of clusters: 56
      * cluster size min, max: 4, 8
      * number of fixed effects: 4
      * L2 ridge penalty on CS variance parameter: true



## Fitting the model

Let's show how to fit the model on the model from example 3. By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
QuasiCopula.fit!(Bernoulli_CS_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    Total number of variables............................:        6
                         variables with only lower bounds:        1
                    variables with lower and upper bounds:        1
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 23
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.5027875108223640e+02    2.5027875108223640e+02
    Dual infeasibility......:   4.5418531868790524e-07    4.5418531868790524e-07
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   1.0000038536138807e-11    1.0000038536138807e-11
    Overall NLP error.......:   4.5418531868790524e-07    4.5418531868790524e-07
    
    
    Number of objective function evaluations             = 39
    Number of objective gradient evaluations             = 24
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =     12.516
    Total CPU secs in NLP function evaluations           =      0.010
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Bernoulli_CS_model.β
@show Bernoulli_CS_model.σ2
@show Bernoulli_CS_model.ρ;
```

    Bernoulli_CS_model.β = [-0.834645421736292, 0.8091668556560436, -0.024759691500116324, 1.9286662077720407]
    Bernoulli_CS_model.σ2 = [0.1703805055662562]
    Bernoulli_CS_model.ρ = [1.0]


Calculate the loglikelihood at the maximum


```julia
logl(Bernoulli_CS_model)
```




    -250.2787511004698



Get asymptotic confidence intervals at the MLE's


```julia
get_CI(Bernoulli_CS_model)
```




    6×2 Matrix{Float64}:
     -1.07019    -0.599097
      0.771294    0.847039
     -0.0339459  -0.0155735
      1.81842     2.03892
      0.46641     1.53359
     -0.743613    1.08437



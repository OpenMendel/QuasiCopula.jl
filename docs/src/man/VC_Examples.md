# VC Covariance

In this notebook we will demonstrate how to fit a variance component model (VCM) with Poisson base on the `Mmmec` dataset from the `mlmRev` R package. We will access the data using the `RDatasets` Julia package. 

### Table of Contents:
* [Example 1: Intercept Only Random Intercept Model](#Example-1:-Intercept-Only-Random-Intercept-Model)
* [Example 2: Random Intercept Model with Covariates](#Example-2:-Random-Intercept-Model-with-Covariates)
* [Example 2: Multiple VC Model with Covariates](#Example-3:-Multiple-VC-Model-with-Covariates)

For these examples, we have $n$ independent clusters indexed by $i$. 

Under the VC parameterization of the covariance matrix, the $i^{th}$ cluster with cluster size $d_i$, has covariance matrix $\mathbf{\Gamma_i}$ that takes the form: 

$$\mathbf{\Gamma_i}(\boldsymbol{\theta}) = \sum_{k = 1}^m \theta_k * \mathbf{V_{ik}}$$ 

* where $m$ is the number of variance components, which are arranged in a vector $\boldsymbol{\theta} = \{\theta_1, ..., \theta_m \}$ for estimation

* and $\mathbf{V_{ik}}, k \in [1, m]$ are symmetric, positive semi-definite matrices of dimension $d_i \times d_i$ provided by the user.


```julia
versioninfo()
```

    Julia Version 1.7.2
    Commit bf53498635 (2022-02-06 15:21 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin19.5.0)
      CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-12.0.1 (ORCJIT, skylake)



```julia
using DataFrames, QuasiCopula, LinearAlgebra, GLM, RDatasets
```


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


## Forming the Models

We can form the VC models for regression with following arguments:

##### Arguments
- `df`: A named `DataFrame`
- `y`: Ouctcome variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `grouping`: Grouping or Clustering variable name of interest, specified as a `Symbol`.
    This variable name must be present in `df`.
- `covariates`: Covariate names of interest as a vector of `Symbol`s.
    Each variable name must be present in `df`.
- `V`: Vector of Vector of Positive Semi-Definite (PSD) Covariance Matrices. `V` is of length n, where n is the number of groups/clusters.
    Each element of `V` is also a `Vector`, but of length m. Here m is the number of variance components.
    Each element of `V` is a `Vector` of d_i x d_i PSD covariance matrices under the VCM framework,
    where d_i is the cluster size of the ith cluster, which may vary for each cluster of observations i in [1, n].
    Each of these dimensions must match that specified in `df`.
- `d`: Base `Distribution` of outcome from `Distributions.jl`.

### Example 1: Intercept Only Random Intercept Model

We first demonstrate how to form the intercept only random intercept model with Poisson base. 

We can form the random intercept VC model with intercept only by excluding the `covariates` and `V` arguments. 

By default, the `VC_model` function will construct a random intercept model using a single variance component. That is, it will parameterize $\boldsymbol{\Gamma_i}(\boldsymbol{\theta})$ for each cluster $i$ with cluster size ${d_i}$ as follows:

$$\mathbf{\Gamma_i}(\boldsymbol{\theta}) = \theta_1 * \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t$$


```julia
df = Mmmec
y = :Deaths
grouping = :Region
d = Poisson()
link = LogLink()

Poisson_VC_model = VC_model(df, y, grouping, d, link)
```




    Quasi-Copula Variance Component Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 78
      * cluster size min, max: 1, 13
      * number of variance components: 1
      * number of fixed effects: 1



### Example 2: Random Intercept Model with Covariates

We can form the random intercept model with covariates by including the `covariates` argument to the model in Example 1.


```julia
covariates = [:UVB]

Poisson_VC_model = VC_model(df, y, grouping, covariates, d, link)
```




    Quasi-Copula Variance Component Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 78
      * cluster size min, max: 1, 13
      * number of variance components: 1
      * number of fixed effects: 2



### Example 3: Multiple VC Model with Covariates

Next we demonstrate how to form the model with Poisson base and two variance components.

To specify our own positive semi-definite covariance matrices, `V_i = [V_i1, V_i2]`, we need to make sure the dimensions match that of each cluster size $d_i$, for each independent cluster $i \in [1, n]$. 

To illustrate, we will add an additional variance component proportional to the Identity matrix to the random intercept model above to help capture overdispersion. More explicitly, I will make $\mathbf{V_{i1}} = \mathbf{1_{d_i}} \mathbf{1_{d_i}}^t$ and $\mathbf{V_{i2}} = \mathbf{I_{d_i}}$ to parameterize $\mathbf{\Gamma_i}(\boldsymbol{\theta})$ follows:

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

V = make_Vs(df, y, grouping);
```


```julia
Poisson_VC_model = VC_model(df, y, grouping, covariates, V, d, link)
```




    Quasi-Copula Variance Component Model
      * base distribution: Poisson
      * link function: LogLink
      * number of clusters: 78
      * cluster size min, max: 1, 13
      * number of variance components: 2
      * number of fixed effects: 2



## Fitting the model

By default, we limit the maximum number of Quasi-Newton iterations to 100, and set the convergence tolerance to $10^{-6}.$ 


```julia
QuasiCopula.fit!(Poisson_VC_model);
```

    initializing β using Newton's Algorithm under Independence Assumption
    gcm.β = [3.25512456536309, -0.07968965522100831]
    initializing variance components using MM-Algorithm
    gcm.θ = [1.0, 1.0]
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    Total number of variables............................:        4
                         variables with only lower bounds:        2
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    
    Number of Iterations....: 73
    
                                       (scaled)                 (unscaled)
    Objective...............:   1.8794436356381009e+03    5.8110834763467265e+03
    Dual infeasibility......:   2.0173863100686985e-07    6.2375907580054106e-07
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   9.9999999999999978e-12    3.0919168663303757e-11
    Overall NLP error.......:   2.0173863100686985e-07    6.2375907580054106e-07
    
    
    Number of objective function evaluations             = 274
    Number of objective gradient evaluations             = 74
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 0
    Total CPU secs in IPOPT (w/o function evaluations)   =      3.180
    Total CPU secs in NLP function evaluations           =      0.038
    
    EXIT: Optimal Solution Found.


We can take a look at the MLE's


```julia
@show Poisson_VC_model.β
@show Poisson_VC_model.θ;
```

    Poisson_VC_model.β = [3.267366389079128, -0.07503868222380021]
    Poisson_VC_model.θ = [14855.181986628666, 5411.298363976766]


Calculate the loglikelihood at the maximum


```julia
logl(Poisson_VC_model)
```




    -5811.0834763467265



Get asymptotic confidence intervals at the MLE's


```julia
get_CI(Poisson_VC_model)
```




    4×2 Matrix{Float64}:
      3.09211     3.44262
     -0.111956   -0.0381218
     -9.59795e5   9.89505e5
     -3.52785e5   3.63607e5



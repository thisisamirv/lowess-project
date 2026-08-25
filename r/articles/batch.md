# Batch Mode (Lowess)

> **See also:** [Choosing an Execution
> Mode](https://thisisamirv.github.io/lowess-project/r/articles/adapter-choice.md)
> for a comparison of all three modes.

The default mode. Processes the entire dataset at once and supports all
features.

![Gap handling](../reference/figures/gap_handling.svg)

Gap handling

## When to Use

- Dataset fits comfortably in memory
- Need confidence/prediction intervals
- Need cross-validation
- One-shot analysis

## Parameters

| Parameter              | Default | Description             |
|------------------------|---------|-------------------------|
| `fraction`             | 0.67    | Neighbourhood size      |
| `iterations`           | 3       | Robustness iterations   |
| `confidence_intervals` | `NULL`  | CI coverage (e.g. 0.95) |
| `prediction_intervals` | `NULL`  | PI coverage (e.g. 0.95) |
| `parallel`             | `FALSE` | Enable CPU parallelism  |
| `backend`              | `"cpu"` | `"cpu"` or `"gpu"`      |

## Example

``` r

library(rfastlowess)
set.seed(42)
x <- seq(0, 2 * pi, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.3)

model <- Lowess(
    fraction = 0.5,
    iterations = 3,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,
    return_diagnostics = TRUE
)
result <- fit(model, x, y)

cat("Confidence lower bounds:\n")
#> Confidence lower bounds:
print(result$confidence_lower)
#>   [1]  0.402912806  0.444425371  0.422964886  0.424915481  0.433441883
#>   [6]  0.446929963  0.457679203  0.460112021  0.503347796  0.476936803
#>  [11]  0.508467445  0.575237027  0.528723036  0.519214195  0.530327602
#>  [16]  0.557746593  0.548293232  0.620499438  0.610611587  0.636836532
#>  [21]  0.574779503  0.588079172  0.583440472  0.649965404  0.705346779
#>  [26]  0.573558526  0.571724435  0.566052600  0.580009989  0.537916583
#>  [31]  0.554940183  0.551396997  0.551839318  0.478585919  0.484934281
#>  [36]  0.450798591  0.415028558  0.389982680  0.405612481  0.336616301
#>  [41]  0.306209221  0.264875040  0.246130697  0.192691651  0.166265570
#>  [46]  0.117829369  0.076952391  0.059458319 -0.007620676 -0.046040786
#>  [51] -0.091271794 -0.123734424 -0.155686707 -0.215585454 -0.257481664
#>  [56] -0.299136467 -0.339823494 -0.378562970 -0.333851372 -0.458325202
#>  [61] -0.485519279 -0.531178489 -0.569305373 -0.599106397 -0.606963060
#>  [66] -0.661662984 -0.687227029 -0.714555460 -0.735973378 -0.753598831
#>  [71] -0.720614877 -0.767806703 -0.789504958 -0.751148996 -0.773300911
#>  [76] -0.802878896 -0.804292857 -0.799280589 -0.754409589 -0.736875309
#>  [81] -0.773122205 -0.761792936 -0.744189930 -0.722750878 -0.663413089
#>  [86] -0.688968767 -0.653976063 -0.627282024 -0.600788991 -0.568507847
#>  [91] -0.521750850 -0.492030543 -0.460702506 -0.406396679 -0.371907161
#>  [96] -0.344340162 -0.305085025 -0.264666341 -0.247019614 -0.203001699
cat("Confidence upper bounds:\n")
#> Confidence upper bounds:
print(result$confidence_upper)
#>   [1]  0.54166355  0.52001237  0.56246539  0.58266084  0.59725030  0.60773573
#>   [7]  0.62172140  0.64481765  0.62802315  0.68189806  0.67873672  0.64102362
#>  [13]  0.71664953  0.75448571  0.77010616  0.76725791  0.79878487  0.74593967
#>  [19]  0.77227333  0.75954061  0.83191842  0.82542677  0.83309601  0.76562380
#>  [25]  0.70534678  0.82824896  0.81722409  0.80603541  0.77130084  0.78905597
#>  [31]  0.74445466  0.71727226  0.68279219  0.71854782  0.67085631  0.65958302
#>  [37]  0.64534351  0.61530697  0.53957380  0.54375970  0.50509551  0.47350188
#>  [43]  0.41603254  0.39044743  0.33551058  0.30074916  0.25732106  0.18998448
#>  [49]  0.17192246  0.12495629  0.08464427  0.03158521 -0.02163454 -0.04609485
#>  [55] -0.08751076 -0.12812421 -0.16879326 -0.21032377 -0.33385137 -0.28627512
#>  [61] -0.33365675 -0.35988377 -0.39045015 -0.42567271 -0.47873757 -0.48043475
#>  [67] -0.50626119 -0.52490908 -0.54384033 -0.56065236 -0.62201028 -0.59737545
#>  [73] -0.59271548 -0.64296661 -0.62795121 -0.60091924 -0.59736114 -0.59526085
#>  [79] -0.62790574 -0.62807473 -0.56934679 -0.55321459 -0.53856942 -0.52297251
#>  [85] -0.54027560 -0.46769448 -0.45078137 -0.42104518 -0.38712180 -0.35553728
#>  [91] -0.33546335 -0.29624951 -0.25750939 -0.24160246 -0.20657384 -0.16587739
#>  [97] -0.13837350 -0.11376134 -0.06828002 -0.05109043
cat("R²:", result$diagnostics$r_squared, "\n")
#> R²: 0.7841212
```

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] rfastlowess_3.0.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.31    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```

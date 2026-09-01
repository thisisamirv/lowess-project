# Benchmarks

## CPU Benchmarks

Speedup relative to R’s
[`stats::lowess`](https://rdrr.io/r/stats/lowess.html) (higher is
better):

| Category | R baseline | rfastlowess Serial | rfastlowess Parallel |
|----|----|----|----|
| **Clustered** | 2.34 ms | 2.0× | **2.5×** |
| **Constant Y** | 1.81 ms | 1.7× | **3.2×** |
| **Extreme Outliers** | 5.81 ms | 1.5× | **2.6×** |
| **Financial** (500–5K) | 0.65 ms | **2.0×** | 1.4× |
| **Fraction** (0.05–0.67) | 3.8 ms | 1.6× | **3.2×** |
| **Genomic** (1K–100K) | 11.2 ms | 2.2× | **2.4×** |
| **High Noise** | 7.08 ms | 1.5× | **3.6×** |
| **Iterations** (0–10) | 3.0 ms | 1.9× | **2.7×** |
| **Large** (50K, delta=0) | 5805.90 ms | 1.9× | **5.5×** |
| **Large** (50K, delta=auto) | 14.46 ms | 1.7× | **2.2×** |
| **Large** (50K, 10 iter) | 31694.32 ms | 3.5× | **9.7×** |
| **Large** (20K, fraction=0.67) | 12627.11 ms | 3.6× | **17.4×** |
| **Scale** (1K–10K) | 1.6 ms | 1.5× | **1.6×** |
| **Scientific** (500–5K) | 0.9 ms | 1.4× | 1.4× |

*The R column shows the average time across scenarios in multi-scenario
categories. Speedups are averages across the same range.*

------------------------------------------------------------------------

## GPU Backend

For large batch datasets, the GPU backend can outperform CPU-parallel
execution. The crossover point depends on `fraction × n`:

| Scenario               | CPU-Parallel | GPU    | Speedup  |
|------------------------|--------------|--------|----------|
| n = 1M, fraction = 0.5 | 1.24 s       | 187 ms | **6.6×** |

At `fraction = 0.5`, GPU overtakes CPU around n ≥ 50K; at smaller
fractions, around n ≥ 100K–250K. See the benchmarks README in the source
repository for the full sweep and transfer-overhead breakdown.

------------------------------------------------------------------------

## Reproducing Benchmarks

The benchmarks compare `rfastlowess` against
[`stats::lowess`](https://rdrr.io/r/stats/lowess.html) using the
`microbenchmark` package.

``` r

# Install benchmark dependency
# install.packages("microbenchmark")

library(rfastlowess)
library(microbenchmark)

set.seed(42)
n <- 5000
x <- seq(0, 10, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.3)

mb <- microbenchmark(
    stats_lowess = stats::lowess(x, y, f = 0.67),
    rfastlowess_serial = {
        m <- Lowess(fraction = 0.67)
        fit(m, x, y)
    },
    rfastlowess_parallel = {
        m <- Lowess(fraction = 0.67, parallel = TRUE)
        fit(m, x, y)
    },
    times = 50
)
cat("Benchmark results:\n")
print(mb)
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
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.31    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```

# Installation

## From R-universe (recommended)

Pre-built binaries — no Rust compiler required.

``` r

install.packages("rfastlowess", repos = "https://thisisamirv.r-universe.dev")
```

## From conda-forge

``` r

# In a conda environment
# conda install -c conda-forge r-rfastlowess
```

## From Source (GitHub)

Requires [Rust](https://rustup.rs/) and Rtools (Windows).

``` r

devtools::install_github("thisisamirv/lowess-project", subdir = "bindings/r")
```

------------------------------------------------------------------------

## System Requirements

- **R**: 4.4.0 or later
- **Windows**: [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
  — required for source installation
- **Rust**: Required for source installation only
  ([rustup.rs](https://rustup.rs/))

On Windows with Rtools, add the MinGW GNU target:

``` sh
rustup target add x86_64-pc-windows-gnu
```

------------------------------------------------------------------------

## Verifying Installation

``` r

library(rfastlowess)

# Fit a simple example
x <- 1:10
y <- x + rnorm(10, sd = 0.5)
model <- Lowess()
result <- fit(model, x, y)
cat("Smoothed values:\n")
#> Smoothed values:
print(result$y)
#>  [1] 1.331578 2.133920 3.100833 4.157752 5.295655 6.275928 7.170805 8.009471
#>  [9] 8.765496 9.330796
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
#> [1] rfastlowess_4.0.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39     desc_1.4.3        R6_2.6.1          fastmap_1.2.0    
#>  [5] xfun_0.60         cachem_1.1.0      knitr_1.51        htmltools_0.5.9  
#>  [9] rmarkdown_2.32    lifecycle_1.0.5   cli_3.6.6         sass_0.4.10      
#> [13] pkgdown_2.2.1     textshaping_1.0.5 jquerylib_0.1.4   systemfonts_1.3.2
#> [17] compiler_4.6.1    tools_4.6.1       ragg_1.5.2        bslib_0.12.0     
#> [21] evaluate_1.0.5    yaml_2.3.12       otel_0.2.0        jsonlite_2.0.0   
#> [25] rlang_1.3.0       fs_2.1.0          htmlwidgets_1.6.4
```

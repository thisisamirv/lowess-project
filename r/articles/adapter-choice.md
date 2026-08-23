# Choosing an Execution Mode

`rfastlowess` provides three execution modes for different data sizes
and processing needs:

![Execution mode
comparison](../reference/figures/adapter_comparison.svg)

Execution mode comparison

| Mode | Class | Use Case | Memory | Key Features |
|----|----|----|----|----|
| **Batch** | `Lowess` | Full dataset | Entire dataset | CI, PI, CV, GPU |
| **Streaming** | `StreamingLowess` | \>100K points | Per chunk | With overlap |
| **Online** | `OnlineLowess` | Live / real-time | Sliding window | Per-point |

## Quick Decision Guide

- **Data fits in memory and you need intervals or CV** → [Batch
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/batch.md)
- **Data is too large for memory or arrives in file chunks** →
  [Streaming
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/streaming.md)
- **Data arrives point-by-point in real time** → [Online
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/online.md)

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
#> [1] BiocStyle_2.40.0
#> 
#> loaded via a namespace (and not attached):
#>  [1] digest_0.6.39       desc_1.4.3          R6_2.6.1           
#>  [4] bookdown_0.47       fastmap_1.2.0       xfun_0.60          
#>  [7] cachem_1.1.0        knitr_1.51          htmltools_0.5.9    
#> [10] rmarkdown_2.31      lifecycle_1.0.5     cli_3.6.6          
#> [13] sass_0.4.10         pkgdown_2.2.1       textshaping_1.0.5  
#> [16] jquerylib_0.1.4     systemfonts_1.3.2   compiler_4.6.1     
#> [19] tools_4.6.1         ragg_1.5.2          bslib_0.12.0       
#> [22] evaluate_1.0.5      yaml_2.3.12         BiocManager_1.30.27
#> [25] otel_0.2.0          jsonlite_2.0.0      rlang_1.3.0        
#> [28] fs_2.1.0            htmlwidgets_1.6.4
```

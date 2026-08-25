# Use Case: Genomic Data Smoothing

## Overview

LOWESS is well-suited for genomic data such as DNA methylation profiles,
ChIP-seq signals, and expression arrays. Sequencing depth variation, PCR
artifacts, and biological heterogeneity all create noise that LOWESS
handles gracefully without parametric assumptions.

------------------------------------------------------------------------

## Methylation Profile Smoothing

A small `fraction = 0.1` lets LOWESS follow fine-scale spatial structure
without smearing transitions between methylated and unmethylated
regions. `confidence_intervals = 0.95` produces uncertainty bands that
naturally widen at positions with sparser CpG coverage.

``` r

library(rfastlowess)

# Simulate methylation data along a chromosome
set.seed(42)
n <- 1000
positions <- sort(runif(n, 0, 1e6))

# True methylation pattern
true_meth <- 0.5 + 0.3 * sin(positions / 1e5)

# Observed with noise
observed <- true_meth + rnorm(n, sd = 0.15)
observed <- pmax(0, pmin(1, observed))

# Smooth
model <- Lowess(
    fraction = 0.1,
    iterations = 3,
    confidence_intervals = 0.95
)
result <- fit(model, positions, observed)

# Plot
plot(positions, observed, pch = ".", col = "gray",
    xlab = "Genomic Position (bp)", ylab = "Methylation Level",
    main = "Methylation Profile Smoothing")
lines(result$x, result$y, col = "blue", lwd = 2)
lines(result$x, result$confidence_lower, col = "blue", lty = 2)
lines(result$x, result$confidence_upper, col = "blue", lty = 2)
legend("topright", c("Observed", "Smoothed", "95% CI"),
        pch = c(1, NA, NA), lty = c(NA, 1, 2),
        col = c("gray", "blue", "blue"))
```

![](use-case-genomics_files/figure-html/use_case_genomics_1-1.png)

------------------------------------------------------------------------

## ChIP-seq Signal Smoothing

ChIP-seq experiments produce sparse, noisy coverage data. LOWESS can
help identify binding regions.

`fraction = 0.05` provides high spatial resolution — important for
resolving narrow binding peaks that would otherwise be smeared into the
background. The larger `iterations = 5` is deliberate:
Poisson-distributed read counts produce tall, isolated spikes, and extra
robustness iterations progressively down-weight them so the estimated
background level is not inflated by a handful of extreme counts.

``` r

library(rfastlowess)
set.seed(42)

# Simulated ChIP-seq enrichment
n <- 500
positions <- seq(1, 50000, length.out = n)
signal <- 10 + 5 * dnorm(positions, mean = 25000, sd = 5000) * 1000

# Add noise and spikes
noise <- rpois(n, lambda = 2)
signal_noisy <- signal + noise
signal_noisy[c(100, 200, 300)] <- signal_noisy[c(100, 200, 300)] * 5

# Smooth
model <- Lowess(
    fraction = 0.05,
    iterations = 5
)
threshold <- quantile(result$y, 0.75)

plot(positions, signal_noisy, pch = 16, cex = 0.4, col = "gray",
    xlab = "Genomic Position", ylab = "Read Count",
    main = "ChIP-seq Signal Smoothing")
lines(result$x, result$y, col = "blue", lwd = 2)
abline(h = threshold, col = "red", lty = 2)
```

![](use-case-genomics_files/figure-html/use_case_genomics_2-1.png)

------------------------------------------------------------------------

## Large Genome Coverage (Streaming)

For whole-genome datasets that do not fit in memory, use the streaming
adapter.

``` r

library(rfastlowess)

model <- StreamingLowess(
    fraction   = 0.05,
    chunk_size = 100000,
    overlap    = 10000,
    merge_strategy = "weighted_average"
)

# Process chromosome in chunks
# (in practice, read from a file or database)
set.seed(42)
for (chunk_i in 1:5) {
    pos_chunk <- seq((chunk_i - 1) * 1e5 + 1, chunk_i * 1e5, length.out = 1000)
    val_chunk <- sin(pos_chunk / 1e4) + rnorm(1000, sd = 0.2)
    result <- process_chunk(model, pos_chunk, val_chunk)
}
final <- finalize(model)
cat("First 6 smoothed values (streaming, weighted_average merge):\n")
#> First 6 smoothed values (streaming, weighted_average merge):
print(head(final$y))
#> [1] 0.3533869 0.3561734 0.3589968 0.3618561 0.3647504 0.3676785
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
#> [1] rfastlowess_3.1.0
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

# LOWESS Concepts

## What is LOWESS?

**LOWESS** (Locally Weighted Scatterplot Smoothing) is a nonparametric
regression method that fits smooth curves through scatter plots without
assuming a global functional form.

Unlike parametric methods (linear regression, polynomial fitting),
LOWESS adapts locally to the data structure, making it ideal for:

- **Exploratory data analysis** — Discover patterns without assumptions
- **Trend estimation** — Extract signals from noisy time series
- **Baseline correction** — Remove systematic effects in spectroscopy
- **Genomic smoothing** — Smooth methylation, ChIP-seq, or expression
  data

------------------------------------------------------------------------

## How It Works

![LOWESS smoothing concept](../reference/figures/lowess_concept.svg)

LOWESS smoothing concept

For each point in your data, LOWESS:

1.  **Selects neighbors** — Choose the nearest points (controlled by
    `fraction`)
2.  **Assigns weights** — Closer points get higher weights (using a
    kernel function)
3.  **Fits locally** — Perform weighted least squares regression
4.  **Extracts value** — Use the fitted value as the smoothed estimate
5.  **Iterates** (optional) — Reweight points based on residuals to
    reduce outlier influence

------------------------------------------------------------------------

## The Fraction Parameter

The `fraction` (also called bandwidth or span) is the most important
parameter. It controls what proportion of data is used for each local
fit.

![Effect of the fraction
parameter](../reference/figures/fraction_comparison.svg)

Effect of the fraction parameter

| Fraction    | Effect                            | When to Use                  |
|-------------|-----------------------------------|------------------------------|
| **0.1–0.3** | Fine detail, follows data closely | Rapidly changing signals     |
| **0.3–0.5** | Balanced smoothing                | Most applications            |
| **0.5–0.7** | Heavy smoothing                   | Noisy data, trend extraction |
| **0.7–1.0** | Very smooth                       | Strong noise, global trends  |

> **Tip:** Start with `fraction=0.67` (the default) and adjust based on
> visual inspection. Use cross-validation for automated selection.

------------------------------------------------------------------------

## Robustness Iterations

Standard LOWESS is sensitive to outliers. **Robustness iterations**
downweight points with large residuals:

![Effect of robustness
iterations](../reference/figures/robust_iter_comparison.svg)

Effect of robustness iterations

| Iterations | Effect                  | When to Use                |
|------------|-------------------------|----------------------------|
| **0**      | No robustness (fastest) | Clean data, speed-critical |
| **1–3**    | Moderate robustness     | Most applications          |
| **4–6**    | Strong robustness       | Data with outliers         |
| **7+**     | Very strong             | Heavy contamination        |

------------------------------------------------------------------------

## Confidence vs Prediction Intervals

![Confidence and prediction
intervals](../reference/figures/intervals_comparison.svg)

Confidence and prediction intervals

| Interval Type  | What It Represents                 | Width  |
|----------------|------------------------------------|--------|
| **Confidence** | Uncertainty in the *mean curve*    | Narrow |
| **Prediction** | Uncertainty for *new observations* | Wide   |

- Use **confidence intervals** to show where the true trend likely lies
- Use **prediction intervals** to show where new data points might fall

------------------------------------------------------------------------

## Execution Modes

Choose the right mode based on your use case:

| Mode          | Use Case          | Memory         | Features              |
|---------------|-------------------|----------------|-----------------------|
| **Batch**     | Complete datasets | Entire dataset | All features          |
| **Streaming** | \>100K points     | One chunk      | Residuals, robustness |
| **Online**    | Real-time data    | Fixed window   | Incremental updates   |

Decision guide:

- Data fits in memory and no real-time requirement → **Batch**
- Data does not fit in memory → **Streaming**
- Real-time / point-by-point processing needed → **Online**

------------------------------------------------------------------------

## Key Advantages

| Feature                    | LOWESS | Polynomial Regression | Moving Average |
|----------------------------|--------|-----------------------|----------------|
| No parametric assumptions  | ✓      | ✗                     | ✓              |
| Adapts to local structure  | ✓      | ✗                     | Partial        |
| Robust to outliers         | ✓      | ✗                     | ✗              |
| Uncertainty estimates      | ✓      | ✓                     | ✗              |
| Handles irregular sampling | ✓      | ✓                     | ✗              |

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

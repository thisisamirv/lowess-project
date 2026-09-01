---
title: "API"
weight: 30
---

Batch `Lowess` reference. Best suited when the dataset fits in memory and you need intervals, cross-validation, or diagnostics.

> **StreamingLowess** and **OnlineLowess** are documented separately: [api-streaming.md](api-streaming.md), [api-online.md](api-online.md)

## `Options.builder() Options.Builder`

Returns a fluent builder pre-populated with recommended defaults. Start from this and override only the settings you need:

```java
Options options = Options.builder()
        .fraction(0.3)
        .returnDiagnostics(true)
        .build();
```

## `Options` settings

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `fraction` | `double` | `0.67` | Smoothing fraction, in (0, 1]. |
| `iterations` | `int` | `3` | Robustness iterations, in [0, 1000]. |
| `delta` | `double` | `0.0` (auto) | Interpolation distance threshold, as a fraction of the x range. |
| `weightFunction` | `String` | `"tricube"` | Kernel: `tricube`, `gaussian`, `uniform`, `cosine`, `epanechnikov`, `biweight`, `triangle`. |
| `robustnessMethod` | `String` | `"bisquare"` | Outlier downweighting: `bisquare`, `huber`, `talwar`. |
| `scalingMethod` | `String` | `"mad"` | Residual scale estimator: `mad`, `mar`, `mean`. |
| `boundaryPolicy` | `String` | `"extend"` | Boundary handling: `extend`, `reflect`, `zero`, `noboundary`. |
| `zeroWeightFallback` | `String` | `"use_local_mean"` | Fallback when all robustness weights hit zero: `use_local_mean`, `return_original`, `return_none`. |
| `confidenceIntervals` | `double` | `NaN` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `predictionIntervals` | `double` | `NaN` (disabled) | Confidence level in (0, 1), e.g. `0.95`. |
| `autoConverge` | `double` | `NaN` (disabled) | Convergence tolerance for early stopping. |
| `returnDiagnostics` | `boolean` | `false` | Populate `Result.diagnostics()`. |
| `returnResiduals` | `boolean` | `false` | Populate `Result.residuals()`. |
| `returnRobustnessWeights` | `boolean` | `false` | Populate `Result.robustnessWeights()`. |
| `returnSe` | `boolean` | `false` | Populate `Result.standardErrors()` (hat-matrix statistics). |
| `cvFractions` | `double[]` | `null` (disabled) | Candidate fractions for cross-validation. |
| `cvMethod` | `String` | `"kfold"` | `kfold` or `loocv`. |
| `cvK` | `int` | `5` | Number of folds for k-fold CV. |
| `cvSeed` | `long` (boxed) | `null` (random) | RNG seed for reproducible k-fold splits. |
| `parallel` | `boolean` | `true` | Enable parallel processing. |
| `backend` | `String` | `"cpu"` | `cpu` or `gpu` (requires a `gpu`-feature build of the native library). |

## `new Lowess(Options options)`

Creates a new batch model. Throws `RuntimeException` if any option is out of range (some validation is eager at construction, e.g. `iterations`; some is deferred to `fit`, e.g. `fraction`).

## `Lowess.fit(double[] x, double[] y) Result`

## `Lowess.fit(double[] x, double[] y, double[] customWeights) Result`

Smooths `y` as a function of `x`. `x` and `y` must be non-empty and the same length. An optional `customWeights` array (same length) applies per-observation case weights; pass `null` (or use the two-argument overload) for uniform weights.

## `Lowess.close()`

Releases native resources. Safe to call multiple times. Implements `AutoCloseable`, so prefer try-with-resources over relying on the garbage collector.

## `Result` accessors

| Accessor | Type | Populated when |
| --- | --- | --- |
| `x()`, `y()` | `double[]` | Always. |
| `standardErrors()` | `Optional<double[]>` | `returnSe` |
| `confidenceLower()`, `confidenceUpper()` | `Optional<double[]>` | `confidenceIntervals` set |
| `predictionLower()`, `predictionUpper()` | `Optional<double[]>` | `predictionIntervals` set |
| `residuals()` | `Optional<double[]>` | `returnResiduals` |
| `robustnessWeights()` | `Optional<double[]>` | `returnRobustnessWeights` |
| `cvScores()` | `Optional<double[]>` | `cvFractions` set |
| `fractionUsed()` | `double` | Always. |
| `iterationsUsed()` | `OptionalInt` | Always (empty if not available). |
| `diagnostics()` | `Optional<Diagnostics>` | `returnDiagnostics` |

`Diagnostics` holds `rmse()`, `mae()`, `rSquared()`, `aic()`, `aicc()`, `effectiveDf()`, `residualSd()` (the last three are `Optional<Double>`).

## Custom weights

```java
double[] weights = new double[x.length];
java.util.Arrays.fill(weights, 1.0);
weights[0] = 5.0; // trust the first observation more

try (Lowess model = new Lowess(Options.builder().build())) {
    Result result = model.fit(x, y, weights);
}
```

## Cross-validation

```java
Options options = Options.builder()
        .cvFractions(new double[] { 0.1, 0.2, 0.3, 0.5 })
        .cvMethod("kfold")
        .cvK(5)
        .cvSeed(42L)
        .build();

try (Lowess model = new Lowess(options)) {
    Result result = model.fit(x, y);
    System.out.println(java.util.Arrays.toString(result.cvScores().orElseThrow()));
    System.out.println(result.fractionUsed()); // the CV-selected fraction
}
```

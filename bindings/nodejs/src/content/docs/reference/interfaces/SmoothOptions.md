[**fastlowess**](../index.md)

***

[fastlowess](../index.md) / SmoothOptions

# Interface: SmoothOptions

Defined in: [index.d.ts:107](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L107)

Configuration options for LOWESS smoothing.

## Properties

### auto\_converge?

> `optional` **auto\_converge?**: `number`

Defined in: [index.d.ts:128](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L128)

Auto-convergence tolerance. Default: None.

***

### backend?

> `optional` **backend?**: `string`

Defined in: [index.d.ts:156](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L156)

Execution backend: "cpu" (default) or "gpu" (requires the package to be
built with the `gpu` Cargo feature and a Vulkan/Metal/DX12-capable GPU).
Batch (`Lowess`) only; ignored by `StreamingLowess`/`OnlineLowess`.

***

### boundary\_policy?

> `optional` **boundary\_policy?**: `string`

Defined in: [index.d.ts:124](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L124)

Boundary handling ("extend", "reflect", "zero", "noboundary"). Default: "extend".

***

### confidence\_intervals?

> `optional` **confidence\_intervals?**: `number`

Defined in: [index.d.ts:136](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L136)

Calculate confidence intervals (e.g., 0.95). Default: None.

***

### cv\_fractions?

> `optional` **cv\_fractions?**: `number`[]

Defined in: [index.d.ts:140](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L140)

Fractions to use for cross-validation.

***

### cv\_k?

> `optional` **cv\_k?**: `number`

Defined in: [index.d.ts:144](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L144)

Number of folds for K-Fold CV. Default: 5.

***

### cv\_method?

> `optional` **cv\_method?**: `string`

Defined in: [index.d.ts:142](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L142)

CV method ("loocv", "kfold"). Default: "kfold".

***

### cv\_seed?

> `optional` **cv\_seed?**: `number`

Defined in: [index.d.ts:146](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L146)

Random seed for reproducible K-Fold cross-validation. Default: None.

***

### delta?

> `optional` **delta?**: `number`

Defined in: [index.d.ts:116](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L116)

Delta for interpolation speedup. Default: NaN (auto).
Set to 0.0 to disable interpolation.

***

### fraction?

> `optional` **fraction?**: `number`

Defined in: [index.d.ts:109](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L109)

Smoothing fraction (0 < fraction <= 1). Default: 0.67.

***

### iterations?

> `optional` **iterations?**: `number`

Defined in: [index.d.ts:111](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L111)

Number of robustness iterations. Default: 3.

***

### parallel?

> `optional` **parallel?**: `boolean`

Defined in: [index.d.ts:150](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L150)

Enable parallel execution. Default: true.

***

### prediction\_intervals?

> `optional` **prediction\_intervals?**: `number`

Defined in: [index.d.ts:138](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L138)

Calculate prediction intervals. Default: None.

***

### return\_diagnostics?

> `optional` **return\_diagnostics?**: `boolean`

Defined in: [index.d.ts:134](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L134)

Return diagnostics (RMSE, etc.). Default: false.

***

### return\_residuals?

> `optional` **return\_residuals?**: `boolean`

Defined in: [index.d.ts:130](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L130)

Return residuals in result. Default: false.

***

### return\_robustness\_weights?

> `optional` **return\_robustness\_weights?**: `boolean`

Defined in: [index.d.ts:132](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L132)

Return robustness weights in result. Default: false.

***

### return\_se?

> `optional` **return\_se?**: `boolean`

Defined in: [index.d.ts:148](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L148)

Compute standard errors. Default: false.

***

### robustness\_method?

> `optional` **robustness\_method?**: `string`

Defined in: [index.d.ts:120](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L120)

Robustness method ("bisquare", "huber", "talwar"). Default: "bisquare".

***

### scaling\_method?

> `optional` **scaling\_method?**: `string`

Defined in: [index.d.ts:126](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L126)

Scaling method ("mad", "mar", "mean"). Default: "mad".

***

### weight\_function?

> `optional` **weight\_function?**: `string`

Defined in: [index.d.ts:118](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L118)

Weight function ("tricube", "epanechnikov", "gaussian", "uniform", "biweight", "triangle", "cosine"). Default: "tricube".

***

### zero\_weight\_fallback?

> `optional` **zero\_weight\_fallback?**: `string`

Defined in: [index.d.ts:122](https://github.com/thisisamirv/lowess-project/blob/a0a9afcfad40aebb60e25a38abb07b76a65a3861/bindings/nodejs/index.d.ts#L122)

Fallback strategy when weights are zero ("use_local_mean", "return_original", "return_none"). Default: "use_local_mean".

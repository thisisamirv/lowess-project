# Node.js API

API reference for the `fastlowess` native Node.js package.

---

## Classes

### Lowess (Batch)

Stateful class for batch smoothing.

```javascript
const { Lowess } = require('fastlowess');

const model = new Lowess({
    fraction: 0.5,
    iterations: 3,
    delta: 0.01,
    weightFunction: "tricube",
    robustnessMethod: "bisquare",
    scalingMethod: "mad",
    zeroWeightFallback: "useLocalMean",
    boundaryPolicy: "extend",
    autoConverge: 1e-4,
    returnResiduals: false,
    returnDiagnostics: false,
    returnRobustnessWeights: false,
    confidenceIntervals: 0.95,
    predictionIntervals: 0.95,
    cvFractions: [0.3, 0.5, 0.7],
    cvMethod: "kfold",
    cvK: 5,
    parallel: true
});
```

**Methods:**

#### fit

Fit the model to data.

```javascript
const result = model.fit(x, y);
```

**Parameters:**

| Parameter | Type           | Description          |
|-----------|----------------|----------------------|
| `x`       | `Float64Array` | Independent variable |
| `y`       | `Float64Array` | Dependent variable   |

**Returns:** Object with fields (see Result Structure below).

**Example:**

```javascript
const fl = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2.1, 3.9, 6.2, 8.0, 10.1]);

const model = new fl.Lowess({ fraction: 0.3, iterations: 3 });
const result = model.fit(x, y);
console.log(result.y);
```

---

### StreamingLowess

Streaming mode for large datasets.

```javascript
const { StreamingLowess } = require('fastlowess');

const stream = new StreamingLowess({
    fraction: 0.67,
    // ...other Lowess options
}, {
    chunkSize: 5000,
    overlap: 500,
    mergeStrategy: "average"
});
```

**Methods:**

#### processChunk

Process a chunk of data.

```javascript
const result = stream.processChunk(xChunk, yChunk);
```

#### finalize

Finalize the stream and process remaining buffered data.

```javascript
const finalResult = stream.finalize();
```

**Example:**

```javascript
const stream = new StreamingLowess({ fraction: 0.05 }, { chunkSize: 10000 });
// ... process chunks ...
const r1 = stream.processChunk(x1, y1);
const r2 = stream.processChunk(x2, y2);
const rFinal = stream.finalize();
```

---

### OnlineLowess

Online mode for real-time data.

```javascript
const { OnlineLowess } = require('fastlowess');

const online = new OnlineLowess({
    fraction: 0.2,
    // ...other Lowess options
}, {
    windowCapacity: 100,
    minPoints: 2,
    updateMode: "incremental"
});
```

**Methods:**

#### addPoints

Add new points to the online processor.

```javascript
const result = online.addPoints(x, y);
```

**Example:**

```javascript
const online = new OnlineLowess({ fraction: 0.3 }, { windowCapacity: 25 });

for (const point of stream) {
    const x = new Float64Array([point.time]);
    const y = new Float64Array([point.value]);
    const res = online.addPoints(x, y);
    if (res.y.length > 0) {
        console.log("Smoothed:", res.y[0]);
    }
}
```

---

## Result Structure

Returns an object with properties:

| Property             | Type           | Description                         |
|----------------------|----------------|-------------------------------------|
| `x`                  | `Float64Array` | Input x values                      |
| `y`                  | `Float64Array` | Smoothed y values                   |
| `fractionUsed`       | `number`       | Actual fraction used                |
| `residuals`          | `Float64Array` | If `returnResiduals=true`           |
| `confidenceLower`    | `Float64Array` | If `confidenceIntervals` set        |
| `confidenceUpper`    | `Float64Array` | If `confidenceIntervals` set        |
| `predictionLower`    | `Float64Array` | If `predictionIntervals` set        |
| `predictionUpper`    | `Float64Array` | If `predictionIntervals` set        |
| `robustnessWeights`  | `Float64Array` | If `returnRobustnessWeights=true`   |
| `diagnostics`        | `object`       | If `returnDiagnostics=true`         |
| `cvScores`           | `number[]`     | If cross-validation used            |

---

## Options Reference

### SmoothOptions (Lowess/Streaming/Online)

| Field                      | Type      | Default            | Description                       |
|----------------------------|-----------|--------------------|-----------------------------------|
| `fraction`                 | `number`  | `0.67`             | Smoothing span (0, 1]             |
| `iterations`               | `number`  | `3`                | Robustness iterations             |
| `delta`                    | `number`  | `0.0`              | Interpolation threshold (0=auto)  |
| `weightFunction`           | `string`  | `"tricube"`        | Kernel function                   |
| `robustnessMethod`         | `string`  | `"bisquare"`       | Outlier handling method           |
| `scalingMethod`            | `string`  | `"mad"`            | Scale estimation method           |
| `zeroWeightFallback`       | `string`  | `"useLocalMean"`   | Zero weight handling              |
| `boundaryPolicy`           | `string`  | `"extend"`         | Boundary handling                 |
| `autoConverge`             | `number`  | `null`             | Auto-convergence tolerance        |
| `returnResiduals`          | `boolean` | `false`            | Return residuals                  |
| `returnDiagnostics`        | `boolean` | `false`            | Return fit diagnostics            |
| `returnRobustnessWeights`  | `boolean` | `false`            | Return robustness weights         |
| `confidenceIntervals`      | `number`  | `null`             | Confidence level (e.g., 0.95)     |
| `predictionIntervals`      | `number`  | `null`             | Prediction interval level         |
| `cvMethod`                 | `string`  | `"kfold"`          | Cross-validation method           |
| `cvK`                      | `number`  | `5`                | Number of CV folds                |
| `cvFractions`              | `number[]`| `null`             | Fractions for CV                  |
| `parallel`                 | `boolean` | `false`            | Enable parallelism                |

### StreamingOptions

| Field            | Type     | Default     | Description                        |
|------------------|----------|-------------|------------------------------------|
| `chunkSize`      | `number` | `5000`      | Points per chunk                   |
| `overlap`        | `number` | `500`       | Overlap between chunks             |
| `mergeStrategy`  | `string` | `"average"` | How to merge overlaps              |

### OnlineOptions

| Field             | Type     | Default         | Description          |
|-------------------|----------|-----------------|----------------------|
| `windowCapacity`  | `number` | `100`           | Max points in window |
| `minPoints`       | `number` | `2`             | Points before output |
| `updateMode`      | `string` | `"incremental"` | Update strategy      |

---

## String Options

### weightFunction

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"biweight"`
- `"cosine"`
- `"triangle"`
- `"uniform"`

### robustnessMethod

- `"bisquare"` (default)
- `"huber"`
- `"talwar"`

### boundaryPolicy

- `"extend"` (default)
- `"reflect"`
- `"zero"`
- `"noBoundary"`

### updateMode

- `"incremental"` (default)
- `"full"`

---

## Diagnostics

When `returnDiagnostics=true`, the result includes:

```javascript
result.diagnostics = {
    rmse: number,        // Root Mean Square Error
    mae: number,         // Mean Absolute Error
    rSquared: number,    // R² coefficient
    residualSd: number,  // Residual standard deviation
    effectiveDf: number  // Effective degrees of freedom
    aic: number,         // AIC (optional)
    aicc: number         // AICc (optional)
}
```

---

# WebAssembly API

API reference for the `fastlowess-wasm` WebAssembly bindings.

---

## Functions

### smooth

Main function for batch smoothing.

```javascript
import init, { smooth } from 'fastlowess-wasm';

await init(); // Initialize WASM module first

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2, 4, 6, 8, 10]);

const result = smooth(x, y, {
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
    cvK: 5
});
```

**Parameters:**

| Parameter | Type           | Default  | Description          |
|-----------|----------------|----------|----------------------|
| `x`       | `Float64Array` | required | Independent variable |
| `y`       | `Float64Array` | required | Dependent variable   |
| `options` | `object`       | `{}`     | Configuration options|

**Options Fields:**

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

**Returns:** Object with properties:

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

**Example:**

```javascript
import init, { smooth } from 'fastlowess-wasm';

await init();

const x = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const y = new Float64Array([2.1, 3.9, 6.2, 8.0, 10.1, 12.3, 13.8, 16.2, 17.9, 20.1]);

const result = smooth(x, y, { fraction: 0.3, iterations: 3 });
console.log(result.y);
```

---

### smooth_streaming

Streaming mode for large datasets.

```javascript
import { StreamingLowessWasm } from 'fastlowess-wasm';

const processor = new StreamingLowessWasm({
    fraction: 0.1,
    iterations: 3
}, {
    chunkSize: 5000,
    overlap: 500,
    mergeStrategy: "average"
});

const result = processor.processChunk(x, y);
const finalResult = processor.finalize();
```

**Additional Options:**

| Field            | Type     | Default     | Description                        |
|------------------|----------|-------------|------------------------------------|
| `chunkSize`      | `number` | `5000`      | Points per chunk                   |
| `overlap`        | `number` | `500`       | Overlap between chunks             |
| `mergeStrategy`  | `string` | `"average"` | How to merge overlaps              |

**Example:**

```javascript
// Process 100,000 points
const x = new Float64Array(100000);
const y = new Float64Array(100000);
// ... fill arrays ...

const processor = new StreamingLowessWasm({ fraction: 0.05 }, { chunkSize: 10000 });
const result = processor.processChunk(x, y);
```

---

### smooth_online

Online mode for real-time data.

```javascript
import { OnlineLowessWasm } from 'fastlowess-wasm';

const online = new OnlineLowessWasm({
    fraction: 0.2,
    iterations: 3
}, {
    windowCapacity: 100,
    minPoints: 5,
    updateMode: "incremental"
});

const value = online.update(x, y);
```

**Additional Options:**

| Field             | Type     | Default         | Description          |
|-------------------|----------|-----------------|----------------------|
| `windowCapacity`  | `number` | `100`           | Max points in window |
| `minPoints`       | `number` | `2`             | Points before output |
| `updateMode`      | `string` | `"incremental"` | Update strategy      |

**Example:**

```javascript
// Real-time sensor data
const online = new OnlineLowessWasm({ fraction: 0.3 }, { windowCapacity: 25 });

for (let i = 0; i < sensorData.length; i++) {
    const smoothed = online.update(sensorData[i].time, sensorData[i].value);
    if (smoothed !== null) {
        console.log(`Smoothed: ${smoothed}`);
    }
}
```

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

### mergeStrategy

- `"average"` (default)
- `"left"`
- `"right"`
- `"weighted"`

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
}
```

---

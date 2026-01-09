# Node.js API

API reference for the `fastlowess` native Node.js package.

## Installation

Install via npm:

```bash
npm install fastlowess
```

---

## Configuration Options

The `fastlowess` package uses a common configuration object for its smoothing functions.

### smoothingOptions

| Parameter                 | Type    | Default        | Description                      |
| :------------------------ | :------ | :------------- | :------------------------------- |
| `fraction`                | number  | 0.67           | Smoothing span (0, 1]            |
| `iterations`              | number  | 3              | Robustness iterations            |
| `delta`                   | number  | auto           | Interpolation threshold          |
| `weightFunction`          | string  | "tricube"      | Kernel function                  |
| `robustnessMethod`        | string  | "bisquare"     | Outlier handling                 |
| `scalingMethod`           | string  | "mad"          | Robust scale estimation          |
| `zeroWeightFallback`      | string  | "useLocalMean" | Fallback policy                  |
| `boundaryPolicy`          | string  | "extend"       | Boundary handling                |
| `autoConverge`            | number  | null           | Convergence threshold            |
| `returnResiduals`         | boolean | false          | Include residuals in output      |
| `returnDiagnostics`       | boolean | false          | Include quality metrics          |
| `returnRobustnessWeights` | boolean | false          | Include final weights            |
| `confidenceIntervals`     | number  | null           | 1-alpha for confidence intervals |
| `predictionIntervals`     | number  | null           | 1-alpha for prediction intervals |

---

## Core API

### smooth()

Main function for batch smoothing. Optimized for high-performance processing of static datasets.

```javascript
const fl = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2, 4, 6, 8, 10]);

const result = fl.smooth(x, y, {
  fraction: 0.3,
  returnDiagnostics: true
});
```

**Parameters:**

- `x`: `Float64Array` (required)
- `y`: `Float64Array` (required)
- `options`: `smoothingOptions` (optional)

**Returns:** Object with:

- `x`: `Float64Array`
- `y`: `Float64Array`
- `residuals`: `Float64Array` (optional)
- `confidenceLower`/`Upper`: `Float64Array` (optional)
- `predictionLower`/`Upper`: `Float64Array` (optional)
- `diagnostics`: Object (optional)

---

## Adapters

### StreamingLowess

Class for processing large datasets in chunks. Ideal for files or streams that don't fit in memory.

```javascript
const { StreamingLowess } = require('fastlowess');

const processor = new StreamingLowess(smoothingOptions, {
  chunkSize: 5000,
  overlap: 500
});

// Process chunks
const result1 = processor.processChunk(xChunk1, yChunk1);
const finalResult = processor.finalize();
```

---

### OnlineLowess

Class for real-time incremental smoothing. Provides immediate output as new data points arrive.

```javascript
const { OnlineLowess } = require('fastlowess');

const online = new OnlineLowess(smoothingOptions, {
  windowCapacity: 100,
  minPoints: 2,
  updateMode: "incremental"
});

const smoothedValue = online.update(x, y); // Returns number or null
```

---

## Option Values

### weightFunction

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"uniform"`
- `"biweight"`
- `"triangle"`
- `"cosine"`

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

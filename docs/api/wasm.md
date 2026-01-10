# WebAssembly API

API reference for the `fastlowess-wasm` WebAssembly bindings.

## Installation

Install via npm:

```bash
npm install fastlowess-wasm
```

---

## Initialization

The WebAssembly module must be initialized before use.

### Using with Bundlers (Vite, Webpack, etc.)

Most modern bundlers handle WASM loading automatically when you import the package.

```javascript
import init, { smooth } from 'fastlowess-wasm';

async function run() {
  await init(); // Initialize the WASM module

  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = smooth(x, y);
  console.log(result.y);
}
```

### Using in Browser (No Bundler)

You must provide the path to the `.wasm` file to the `init` function.

```html
<script type="module">
  import init, { smooth } from './pkg/fastlowess_wasm.js';
  
  async function run() {
    await init('./pkg/fastlowess_wasm_bg.wasm');
    const result = smooth(x, y);
  }
  run();
</script>
```

---

## Configuration Options

WebAssembly smoothing functions use the following configuration object.

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
| `cvFractions`             | number[]| null           | List of fractions for CV         |
| `cvMethod`                | string  | "kfold"        | CV method (kfold, loocv)         |
| `cvK`                     | number  | 5              | Number of folds for K-Fold CV    |
| `parallel`                | boolean | false          | Enable parallel execution        |

---

## Core API

### smooth()

Main function for batch smoothing.

```javascript
import { smooth } from 'fastlowess-wasm';

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2, 4, 6, 8, 10]);

const result = smooth(x, y, { fraction: 0.3 });
```

**Parameters:**

- `x`: `Float64Array` (required)
- `y`: `Float64Array` (required)
- `options`: `smoothingOptions` (optional)

**Returns:** Object containing smoothed `y` values and any requested optional output (residuals, diagnostics, etc.).

---

## Adapters

### StreamingLowessWasm

WASM-optimized class for processing large datasets in chunks.

```javascript
import { StreamingLowessWasm } from 'fastlowess-wasm';

const processor = new StreamingLowessWasm(smoothingOptions, {
  chunkSize: 5000,
  overlap: 500
});

const result = processor.processChunk(x, y);
const finalResult = processor.finalize();
```

---

### OnlineLowessWasm

WASM-optimized class for real-time incremental smoothing.

```javascript
import { OnlineLowessWasm } from 'fastlowess-wasm';

const online = new OnlineLowessWasm(smoothingOptions, {
  windowCapacity: 100,
  minPoints: 2,
  updateMode: "incremental"
});

const value = online.update(x, y);
```

---

## Performance & Memory Management

- **TypedArrays**: Always use `Float64Array` to avoid expensive copies and serialization.
- **WASM Memory**: For extremely large datasets, be mindful of the WASM linear memory limits (defaults to 2GB or 4GB).
- **Native vs WASM**: Use the native Node.js package (`fastlowess`) if running on the server for best performance and multi-threading support.

---

## Option Values

### weightFunction

- `"tricube"`, `"epanechnikov"`, `"gaussian"`, `"uniform"`, `"biweight"`, `"triangle"`, `"cosine"`

### robustnessMethod

- `"bisquare"`, `"huber"`, `"talwar"`

### boundaryPolicy

- `"extend"`, `"reflect"`, `"zero"`, `"noBoundary"`

### updateMode

- `"incremental"`, `"full"`

### cvMethod

- `"kfold"`, `"loocv"`

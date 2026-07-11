# Node.js Examples

Complete Node.js examples demonstrating fastlowess with native N-API bindings.

## Batch Smoothing

Process complete datasets with confidence intervals and diagnostics.

```javascript
--8<-- "examples/nodejs/batch_smoothing.js"
```

[:material-download: Download batch_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/nodejs/batch_smoothing.js)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks.

```javascript
--8<-- "examples/nodejs/streaming_smoothing.js"
```

[:material-download: Download streaming_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/nodejs/streaming_smoothing.js)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data.

```javascript
--8<-- "examples/nodejs/online_smoothing.js"
```

[:material-download: Download online_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/nodejs/online_smoothing.js)

---

## Running the Examples

```bash
# Install the package
cd bindings/nodejs
npm install
npm run build

# Run examples
node ../../examples/nodejs/batch_smoothing.js
node ../../examples/nodejs/streaming_smoothing.js
node ../../examples/nodejs/online_smoothing.js
```

## Quick Start

```javascript
const { Lowess } = require('fastlowess');

// Generate sample data
const x = Float64Array.from({ length: 100 }, (_, i) => i * 0.1);
const y = Float64Array.from(x, xi => Math.sin(xi) + Math.random() * 0.2);

// Basic smoothing
const model = new Lowess({ fraction: 0.3 });
const result = model.fit(x, y);
console.log('Smoothed values:', result.y);

// With options
const resultWithOptions = new Lowess({
    fraction: 0.3,
    iterations: 3,
    confidence_intervals: 0.95,
    return_diagnostics: true
}).fit(x, y);

console.log('R²:', resultWithOptions.diagnostics.r_squared);
```

## TypeScript Support

The package includes TypeScript definitions:

```typescript
const { Lowess, LowessResult } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const options = {
    fraction: 0.3,
    iterations: 3,
    confidence_intervals: 0.95
};

const model = new Lowess(options);
const result: LowessResult = model.fit(x, y);
```

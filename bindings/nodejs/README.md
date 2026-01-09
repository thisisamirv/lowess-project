# fastlowess-node

Node.js native bindings for **fastLowess**, the high-performance LOWESS implementation.

## Installation

```bash
npm install fastlowess
```

## Features

- **Blazing Fast**: Native Rust implementation with Node.js N-API.
- **Parallel Execution**: Automatically utilizes all CPU cores for batch and streaming tasks.
- **Feature Complete**: Supports robustness iterations, confidence intervals, interpolation optimization (delta), and more.
- **Multiple Modes**:
  - `smooth()`: Batch processing.
  - `StreamingLowess`: Chunked processing for large-scale data.
  - `OnlineLowess`: Real-time incremental smoothing.

## Documentation

Full documentation is available at [lowess.readthedocs.io](https://lowess.readthedocs.io/en/latest/api/javascript/).

## Usage

```javascript
const fastlowess = require('fastlowess');

const x = new Float64Array([1, 2, 3, 4, 5]);
const y = new Float64Array([2, 4, 6, 8, 10]);

const result = fastlowess.smooth(x, y, {
  fraction: 0.3,
  returnDiagnostics: true
});

console.log(result.y);
```

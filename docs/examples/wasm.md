# WebAssembly Examples

Complete WebAssembly examples demonstrating fastlowess-wasm for browser-based smoothing.

## Batch Smoothing

Process complete datasets in the browser.

```javascript
--8<-- "../../examples/wasm/batch_smoothing.js"
```

[:material-download: Download batch_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/batch_smoothing.js)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks in the browser.

```javascript
--8<-- "../../examples/wasm/streaming_smoothing.js"
```

[:material-download: Download streaming_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/streaming_smoothing.js)

---

## Online Smoothing

Real-time smoothing with sliding window for browser applications.

```javascript
--8<-- "../../examples/wasm/online_smoothing.js"
```

[:material-download: Download online_smoothing.js](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/online_smoothing.js)

---

## Installation

### NPM

```bash
npm install fastlowess-wasm
```

### CDN

```html
<script type="module">
  import init, { smooth } from 'https://unpkg.com/fastlowess-wasm@latest';
  
  await init();
  // Ready to use
</script>
```

## Quick Start

### Browser (ES Modules)

```javascript
import init, { smooth, smoothStreaming, smoothOnline } from 'fastlowess-wasm';

async function main() {
    // Initialize WASM module
    await init();

    // Generate sample data
    const x = Float64Array.from({ length: 100 }, (_, i) => i * 0.1);
    const y = Float64Array.from(x, xi => Math.sin(xi) + Math.random() * 0.2);

    // Basic smoothing
    const result = smooth(x, y, { fraction: 0.3 });
    console.log('Smoothed values:', result.y);

    // With options
    const resultWithOptions = smooth(x, y, {
        fraction: 0.3,
        iterations: 3,
        confidenceIntervals: 0.95,
        returnDiagnostics: true
    });

    console.log('R²:', resultWithOptions.diagnostics.rSquared);
}

main();
```

### Node.js

```javascript
const { smooth } = require('fastlowess-wasm');

// Same API as browser
const result = smooth(x, y, { fraction: 0.3 });
```

## Features

The WebAssembly bindings provide:

- **Zero dependencies** - Pure WASM, no runtime requirements
- **TypedArray support** - Works with `Float64Array` for efficiency
- **Same API as Node.js** - Consistent interface across platforms
- **Small bundle size** - Optimized with `wasm-opt`

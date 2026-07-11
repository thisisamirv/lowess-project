# WebAssembly Examples

Complete WebAssembly examples demonstrating fastlowess-wasm for browser-based smoothing.

## Batch Smoothing

Process complete datasets in the browser.

```html
--8<-- "examples/wasm/batch_smoothing.html"
```

[:material-download: Download batch_smoothing.html](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/batch_smoothing.html)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks in the browser.

```html
--8<-- "examples/wasm/streaming_smoothing.html"
```

[:material-download: Download streaming_smoothing.html](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/streaming_smoothing.html)

---

## Online Smoothing

Real-time smoothing with sliding window for browser applications.

```html
--8<-- "examples/wasm/online_smoothing.html"
```

[:material-download: Download online_smoothing.html](https://github.com/thisisamirv/lowess-project/blob/main/examples/wasm/online_smoothing.html)

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
const { Lowess } = require('./fastlowess_wasm.js');

async function main() {
    // Initialize WASM module

    // Generate sample data
    const x = Float64Array.from({ length: 100 }, (_, i) => i * 0.1);
    const y = Float64Array.from(x, xi => Math.sin(xi) + ((xi * 7 % 1) - 0.5) * 0.2);

    // Basic smoothing
    const model = new Lowess({ fraction: 0.3 });
    const result = model.fit(x, y);
    console.log('Smoothed values:', result.y);

    // With options
    const modelWithOptions = new Lowess({
        fraction: 0.3,
        iterations: 3,
        confidence_intervals: 0.95,
        return_diagnostics: true
    });
    const resultWithOptions = modelWithOptions.fit(x, y);

    console.log('R²:', resultWithOptions.diagnostics.r_squared);
}

main();
```

### Node.js

```javascript
const { Lowess } = require('fastlowess');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const model = new Lowess({ fraction: 0.3 });
const result = model.fit(x, y);
```

## Features

The WebAssembly bindings provide:

- **Zero dependencies** - Pure WASM, no runtime requirements
- **TypedArray support** - Works with `Float64Array` for efficiency
- **Same API as Node.js** - Consistent interface across platforms
- **Small bundle size** - Optimized with `wasm-opt`

# fastlowess-wasm

WebAssembly bindings for **fastLowess**, the high-performance LOWESS implementation.

## Installation

```bash
npm install fastlowess-wasm
```

## Usage

```javascript
import init, { smooth } from 'fastlowess-wasm';

async function run() {
  await init();
  
  const x = new Float64Array([1, 2, 3, 4, 5]);
  const y = new Float64Array([2, 4, 6, 8, 10]);
  
  const result = smooth(x, y, { fraction: 0.3 });
  console.log(result.y);
}

run();
```

## Documentation

Full documentation is available at [lowess.readthedocs.io](https://lowess.readthedocs.io/en/latest/api/javascript/).

---
title: Installation
---
<!-- markdownlint-disable MD024 MD046 -->
Install the LOWESS library for your preferred language.

## From NPM (recommended)

```bash
npm install fastlowess-wasm
```

## From CDN

```html
<script type="module">
  import init, { Lowess } from "https://cdn.jsdelivr.net/npm/fastlowess-wasm@0.99/fastlowess_wasm.js";
  await init();
</script>
```

## From Source

```bash
# Install Rust first: https://rustup.rs/
# Install wasm-pack: https://rustwasm.github.io/wasm-pack/installer/
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/wasm
# For bundlers (Webpack, Vite, etc.)
wasm-pack build --target bundler
# For Node.js
wasm-pack build --target nodejs
# For browser (no bundler)
wasm-pack build --target web
```

---

## Verify Installation

```javascript
import init, { Lowess } from 'fastlowess-wasm';

async function verify() {
await init();
const x = new Float64Array([1.0, 2.0, 3.0]);
const y = new Float64Array([2.0, 4.0, 6.0]);
const result = new Lowess({}).fit(x, y);
console.log("Installed successfully!");
}
verify();
```

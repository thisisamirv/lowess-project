<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

=== "From NPM (recommended)"

```bash
npm install fastlowess
```

=== "From Source"

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/nodejs
npm install
npm run build
```

---

## Verify Installation

```javascript
const fl = require('fastlowess');

const x = new Float64Array([1.0, 2.0, 3.0]);
const y = new Float64Array([2.0, 4.0, 6.0]);

const model = new fl.Lowess({});
const result = model.fit(x, y);
console.log("Installed successfully!");
```

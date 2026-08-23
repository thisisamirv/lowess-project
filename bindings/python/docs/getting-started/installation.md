<!-- markdownlint-disable MD024 MD046 -->
# Installation

Install the LOWESS library for your preferred language.

## From PyPI (recommended)

```bash
pip install fastlowess
```

## From conda-forge

```bash
conda install -c conda-forge fastlowess
```

## From Source

```bash
# Install Rust first: https://rustup.rs/
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/python
pip install maturin
maturin develop --release
```

---

## Verify Installation

```python
import fastlowess as fl
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

model = fl.Lowess()
result = model.fit(x, y)
print("Installed successfully!")
```

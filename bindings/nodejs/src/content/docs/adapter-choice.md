---
title: Execution Modes
---
<!-- markdownlint-disable MD024 MD033 MD046 -->
Choose the right adapter for your use case.

## Overview

Choose the first row below whose condition applies:

| Condition | Adapter |
| --- | --- |
| Data too large to fit in memory | `Streaming` |
| Fits in memory, need real-time/incremental updates | `Online` |
| Fits in memory, no real-time requirement | `Batch` |

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** | Complete datasets | Full | All features |
| **Streaming** | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](../assets/diagrams/adapter_comparison.svg)

---

## Feature Comparison

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Cross-validation | ✓ | ✗ | ✗ |
| Diagnostics | ✓ | ✓ | ✗ |
| Residuals | ✓ | ✓ | ✓ |
| Robustness weights | ✓ | ✓ | ✓ |
| Parallel execution | ✓ | ✓ | ✗ |

---

## Next Steps

- [API Reference](api.md) — All configuration options
- [Streaming API](api-streaming.md) · [Online API](api-online.md)

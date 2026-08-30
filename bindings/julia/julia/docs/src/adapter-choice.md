# Execution Modes

Choose the right adapter for your use case.

## Overview

```mermaid
graph LR
    A[Data] --> B{Size?}
    B -->|Fits in memory| C{Real-time?}
    B -->|Too large| D[Streaming]
    C -->|No| E[Batch]
    C -->|Yes| F[Online]
```

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** | Complete datasets | Full | All features |
| **Streaming** | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](assets/adapter_comparison.svg)

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
- [Tutorials](use-case-real-time.md) — Real-time processing guide

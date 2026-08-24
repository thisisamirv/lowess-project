---
title: Execution Modes
---
<!-- markdownlint-disable MD024 MD033 MD046 -->
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

![Adapter Comparison](../assets/diagrams/adapter_comparison.svg)

---

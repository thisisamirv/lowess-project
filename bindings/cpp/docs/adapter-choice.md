# Execution Modes

Choose the right adapter for your use case.

## Overview

```text
Data
  └─ Fits in memory?
       ├─ No  ─────────────► Streaming
       └─ Yes ─ Real-time? ─┬─ No  ─► Batch
                            └─ Yes ─► Online
```

| Mode | Use Case | Memory | Features |
| --- | --- | --- | --- |
| **Batch** | Complete datasets | Full | All features |
| **Streaming** | Large files (>100K) | Chunked | Residuals, robustness |
| **Online** | Real-time sensors | Fixed window | Incremental updates |

![Adapter Comparison](adapter_comparison.svg)

---

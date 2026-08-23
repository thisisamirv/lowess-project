# Choosing an Execution Mode

`rfastlowess` provides three execution modes for different data sizes
and processing needs:

![Execution mode
comparison](../reference/figures/adapter_comparison.svg)

Execution mode comparison

| Mode | Class | Use Case | Memory | Key Features |
|----|----|----|----|----|
| **Batch** | `Lowess` | Complete datasets in memory | Entire dataset | All features: CI, PI, CV, GPU |
| **Streaming** | `StreamingLowess` | Large files (\>100K points) | One chunk at a time | Chunked processing with overlap |
| **Online** | `OnlineLowess` | Real-time / live data | Fixed sliding window | Point-by-point incremental updates |

## Quick Decision Guide

- **Data fits in memory and you need intervals or CV** → [Batch
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/batch.md)
- **Data is too large for memory or arrives in file chunks** →
  [Streaming
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/streaming.md)
- **Data arrives point-by-point in real time** → [Online
  Mode](https://thisisamirv.github.io/lowess-project/r/articles/online.md)

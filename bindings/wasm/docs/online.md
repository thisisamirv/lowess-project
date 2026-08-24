# Online Adapter

Incremental updates with a sliding window for real-time data.

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](../assets/diagrams/online_comparison.svg)

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `window_capacity` | 1000 | Max points in window |
| `min_points` | 2 | Points before output starts |
| `update_mode` | `"incremental"` | Update strategy |

## Update Modes

| Mode | Behavior | Speed |
| --- | --- | --- |
| `"incremental"` | Update only affected fits | Faster |
| `"full"` | Recompute entire window | More accurate |

## Example

```javascript
const { OnlineLowess } = require('fastlowess-wasm');

const n = 100;
const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

const sensorStream = Array.from({ length: n }, (_, i) => [x[i], y[i]]);

const processor = new OnlineLowess(
    { fraction: 0.2, iterations: 1 },
    { window_capacity: 100, min_points: 5, update_mode: "incremental" }
);

// Add points; print first 5 results
let printed = 0;
for (const [xi, yi] of sensorStream) {
    const output = processor.add_point(xi, yi);
    if (output !== undefined && output !== null) {
        if (printed < 5) console.log(`Smoothed: ${output.y.toFixed(2)}`);
        printed++;
    }
}
console.log(`... (${printed - 5} more)`);
```

```output
Smoothed: 0.45
Smoothed: 0.15
Smoothed: 0.46
Smoothed: 0.17
Smoothed: 0.47
... (91 more)
```

---

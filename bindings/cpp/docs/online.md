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

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }


    fastlowess::OnlineOptions opts;
    opts.fraction = 0.2;
    opts.iterations = 1;
    opts.window_capacity = 100;
    opts.min_points = 5;
    opts.update_mode = "incremental";

    fastlowess::OnlineLowess model(opts);
    for (size_t i = 0; i < x.size(); ++i) {
        auto out = model.add_point(x[i], y[i]).value();
        if (out.has_value())
            std::cout << out.y() << std::endl;
    }

    return 0;
}
```

---

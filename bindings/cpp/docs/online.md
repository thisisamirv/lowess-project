# Online Adapter

Incremental updates with a sliding window for real-time data.

## When to Use

- Data arrives incrementally (sensors, streams)
- Need real-time smoothed values
- Fixed memory budget

![Online Adapter](online_comparison.svg)

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

```output
0.351148
0.412033
0.471662
0.529795
0.586197
0.640641
0.692908
0.742788
0.790079
0.834592
0.876146
0.91495
0.950118
0.981863
1.01006
1.03459
1.05606
1.073
1.08602
1.09506
1.10011
1.1022
1.09919
1.09216
1.08113
1.06615
1.04868
1.02598
0.999544
0.969488
0.935932
0.900589
0.860394
0.817138
0.770994
0.722148
0.672331
0.618585
0.562751
0.505055
0.445727
0.386231
0.324224
0.261316
0.197757
0.133805
0.0703588
0.00621752
-0.0575461
-0.120675
-0.182916
-0.244202
-0.304118
-0.362406
-0.418833
-0.47317
-0.526392
-0.57611
-0.623106
-0.66719
-0.708185
-0.748209
-0.782735
-0.813706
-0.840999
-0.864502
-0.887463
-0.903273
-0.915043
-0.922726
-0.926291
-0.929961
-0.925355
-0.91662
-0.903791
-0.886921
-0.870927
-0.846204
-0.817671
-0.785443
-0.749649
-0.715507
-0.672947
-0.627274
-0.578673
-0.52734
-0.478311
-0.421962
-0.36351
-0.303193
-0.241251
-0.182033
-0.117324
-0.0517404
0.0144546
0.080994
```

---

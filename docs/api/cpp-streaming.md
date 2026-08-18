# StreamingLowess — C++ API Reference

See also: [fastLowess C++ API Reference](cpp.md)

## Class

### `fastlowess::StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    fastlowess::StreamingOptions opts;
    opts.chunk_size = 5;
    fastlowess::StreamingLowess model(opts);

    return 0;
}
```

* `options`: A `StreamingOptions` struct (inherits from `LowessOptions`) with additional `chunk_size`, `overlap`, and `merge_strategy` parameters.

**Methods:**

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

    fastlowess::StreamingOptions opts;
    opts.chunk_size = 10;
    opts.overlap = 0;
    fastlowess::StreamingLowess model(opts);
    (void)model.process_chunk(x, y);

    return 0;
}
```

* Processes a chunk of data. Returns partial results.

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

    fastlowess::StreamingOptions opts;
    opts.chunk_size = 10;
    opts.overlap = 0;
    fastlowess::StreamingLowess model(opts);
    model.process_chunk(x, y);
    auto result = model.finalize().value();

    return 0;
}
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | 5000 | Data chunk size |
| `overlap` | `int` | 500 | Overlap between chunks |
| `merge_strategy` | `std::string` | "weighted_average" | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)

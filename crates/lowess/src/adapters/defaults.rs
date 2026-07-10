// Default values for adapter configuration (streaming, online, batch).

use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;

// Default number of data points per chunk.
pub const DEFAULT_STREAMING_CHUNK_SIZE: usize = 5_000;

// Default overlap between consecutive chunks.
pub const DEFAULT_STREAMING_OVERLAP: usize = 500;

// Default merge strategy for overlapping chunk regions.
pub const DEFAULT_STREAMING_MERGE_STRATEGY_ENUM: MergeStrategy = MergeStrategy::WeightedAverage;
#[cfg(feature = "dev")]
pub const DEFAULT_STREAMING_MERGE_STRATEGY: &str = "weighted_average";

// Default sliding-window capacity.
pub const DEFAULT_ONLINE_WINDOW_CAPACITY: usize = 1_000;

// Default minimum number of points required before output is produced.
pub const DEFAULT_ONLINE_MIN_POINTS: usize = 2;

// Default update mode for the **Online** adapter.
pub const DEFAULT_ONLINE_UPDATE_MODE_ENUM: UpdateMode = UpdateMode::Incremental;
#[cfg(feature = "dev")]
pub const DEFAULT_ONLINE_UPDATE_MODE: &str = "incremental";

// Default smoothing fraction. Approximately Cleveland's original recommendation of 2/3.
pub const DEFAULT_FRACTION: f64 = 0.67;

// Default interpolation-optimisation threshold for **Streaming** and **Online** adapters.
// `0.0` disables the optimisation (every point is fit individually).
pub const DEFAULT_DELTA: f64 = 0.0;

// Default interpolation-optimisation threshold for the **Batch** adapter.
// `None` means auto-compute as 1% of the x-range.
pub const fn default_batch_delta<T>() -> Option<T> {
    None
}

// Default CV seed: `None` means non-reproducible fold splitting.
pub const DEFAULT_CV_SEED: Option<u64> = None;

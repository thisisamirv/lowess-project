// Default values for adapter configuration (streaming, online, batch).

// Internal dependencies
use crate::primitives::policies::{MergeStrategy, UpdateMode};

pub const DEFAULT_STREAMING_CHUNK_SIZE: usize = 5_000;
pub fn default_overlap(chunk_size: usize) -> usize {
    let default = chunk_size / 10;
    default.min(chunk_size.saturating_sub(10)).max(1)
}
pub const DEFAULT_STREAMING_MERGE_STRATEGY_ENUM: MergeStrategy = MergeStrategy::WeightedAverage;
#[cfg(feature = "dev")]
pub const DEFAULT_STREAMING_MERGE_STRATEGY: &str = "weighted_average";
pub const DEFAULT_ONLINE_WINDOW_CAPACITY: usize = 1_000;
pub const DEFAULT_ONLINE_MIN_POINTS: usize = 2;
pub const DEFAULT_ONLINE_UPDATE_MODE_ENUM: UpdateMode = UpdateMode::Incremental;
#[cfg(feature = "dev")]
pub const DEFAULT_ONLINE_UPDATE_MODE: &str = "incremental";
pub const DEFAULT_ONLINE_ITERATIONS: usize = 0;
pub const fn default_batch_delta<T>() -> Option<T> {
    None
}

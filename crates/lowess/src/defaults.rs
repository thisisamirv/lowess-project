//! Canonical default values for all LOWESS parameters.
//!
//! This module is the single source of truth for every default parameter value
//! used across the `lowess` crate and all downstream consumers (`fastLowess`
//! and the language bindings for Python, R, Julia, Node.js, WASM, and C++).
//!
//! Adapter-specific constants are prefixed with the adapter name
//! (`STREAMING_`, `ONLINE_`). Shared constants (used by every
//! adapter) have no prefix.

use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;
use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;

// ── Streaming adapter ─────────────────────────────────────────────────────────

// Default number of data points per chunk.
pub const DEFAULT_STREAMING_CHUNK_SIZE: usize = 5_000;

// Default overlap between consecutive chunks.
pub const DEFAULT_STREAMING_OVERLAP: usize = 500;

// Default merge strategy for overlapping chunk regions.
pub const DEFAULT_STREAMING_MERGE_STRATEGY_ENUM: MergeStrategy = MergeStrategy::WeightedAverage;
pub const DEFAULT_STREAMING_MERGE_STRATEGY: &str = "weighted_average";

// ── Online adapter ────────────────────────────────────────────────────────────

// Default sliding-window capacity.
pub const DEFAULT_ONLINE_WINDOW_CAPACITY: usize = 1_000;

// Default minimum number of points required before output is produced.
pub const DEFAULT_ONLINE_MIN_POINTS: usize = 2;

// Default update mode for the **Online** adapter.
pub const DEFAULT_ONLINE_UPDATE_MODE_ENUM: UpdateMode = UpdateMode::Incremental;
pub const DEFAULT_ONLINE_UPDATE_MODE: &str = "incremental";

// ── Shared (all adapters) ─────────────────────────────────────────────────────

// Default smoothing fraction. Approximately Cleveland's original recommendation of 2/3.
pub const DEFAULT_FRACTION: f64 = 0.67;

// Default number of robustness iterations.
pub const DEFAULT_ITERATIONS: usize = 3;

// Default interpolation-optimisation threshold for **Streaming** and **Online** adapters.
// `0.0` disables the optimisation (every point is fit individually).
pub const DEFAULT_DELTA: f64 = 0.0;

// Default interpolation-optimisation threshold for the **Batch** adapter.
// `None` means auto-compute as 1 % of the x-range.
pub const fn default_batch_delta<T>() -> Option<T> {
    None
}

// Default auto-convergence tolerance: `None` disables early stopping.
pub const fn default_auto_converge<T>() -> Option<T> {
    None
}

// Default confidence-interval level: `None` means no confidence intervals.
pub const fn default_confidence_level<T>() -> Option<T> {
    None
}

// Default prediction-interval level: `None` means no prediction intervals.
pub const fn default_prediction_level<T>() -> Option<T> {
    None
}

// Default kernel weight function.
pub const DEFAULT_WEIGHT_FUNCTION_ENUM: WeightFunction = WeightFunction::Tricube;
pub const DEFAULT_WEIGHT_FUNCTION: &str = "tricube";

// Default robustness weighting method.
pub const DEFAULT_ROBUSTNESS_METHOD_ENUM: RobustnessMethod = RobustnessMethod::Bisquare;
pub const DEFAULT_ROBUSTNESS_METHOD: &str = "bisquare";

// Default robust scale estimator.
pub const DEFAULT_SCALING_METHOD_ENUM: ScalingMethod = ScalingMethod::MAD;
pub const DEFAULT_SCALING_METHOD: &str = "mad";

// Default boundary padding policy.
pub const DEFAULT_BOUNDARY_POLICY_ENUM: BoundaryPolicy = BoundaryPolicy::Extend;
pub const DEFAULT_BOUNDARY_POLICY: &str = "extend";

// Default zero-weight neighbourhood fallback.
pub const DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM: ZeroWeightFallback = ZeroWeightFallback::UseLocalMean;
pub const DEFAULT_ZERO_WEIGHT_FALLBACK: &str = "use_local_mean";

// Default: do not compute or return diagnostic statistics.
pub const DEFAULT_RETURN_DIAGNOSTICS: bool = false;

// Default: do not compute or return residuals.
pub const DEFAULT_RETURN_RESIDUALS: bool = false;

// Default: do not compute or return robustness weights.
pub const DEFAULT_RETURN_ROBUSTNESS_WEIGHTS: bool = false;

// ── Cross-validation ──────────────────────────────────────────────────────────

// Default CV seed: `None` means non-reproducible fold splitting.
pub const DEFAULT_CV_SEED: Option<u64> = None;

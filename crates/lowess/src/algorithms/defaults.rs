// Default values for algorithms module types (regression, robustness).

use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;

// Default number of robustness iterations.
pub const DEFAULT_ITERATIONS: usize = 3;

// Default robustness weighting method.
pub const DEFAULT_ROBUSTNESS_METHOD_ENUM: RobustnessMethod = RobustnessMethod::Bisquare;
#[cfg(feature = "dev")]
pub const DEFAULT_ROBUSTNESS_METHOD: &str = "bisquare";

// Default zero-weight neighbourhood fallback.
pub const DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM: ZeroWeightFallback = ZeroWeightFallback::UseLocalMean;
#[cfg(feature = "dev")]
pub const DEFAULT_ZERO_WEIGHT_FALLBACK: &str = "use_local_mean";

// Default auto-convergence tolerance: `None` disables early stopping.
pub const fn default_auto_converge<T>() -> Option<T> {
    None
}

// False means do not compute or return diagnostic statistics.
pub const DEFAULT_RETURN_DIAGNOSTICS: bool = false;

// False means do not compute or return residuals.
pub const DEFAULT_RETURN_RESIDUALS: bool = false;

// False means do not compute or return robustness weights.
pub const DEFAULT_RETURN_ROBUSTNESS_WEIGHTS: bool = false;

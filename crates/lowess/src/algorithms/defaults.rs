// Default values for algorithms module types (regression, robustness).

// Internal dependencies
use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::primitives::policies::MissingPolicy;

pub const DEFAULT_ITERATIONS: usize = 3;
pub const DEFAULT_ROBUSTNESS_METHOD_ENUM: RobustnessMethod = RobustnessMethod::Bisquare;
#[cfg(feature = "dev")]
pub const DEFAULT_ROBUSTNESS_METHOD: &str = "bisquare";
pub const DEFAULT_ZERO_WEIGHT_FALLBACK_ENUM: ZeroWeightFallback = ZeroWeightFallback::UseLocalMean;
#[cfg(feature = "dev")]
pub const DEFAULT_ZERO_WEIGHT_FALLBACK: &str = "use_local_mean";
pub const DEFAULT_MISSING_POLICY_ENUM: MissingPolicy = MissingPolicy::Error;
#[cfg(feature = "dev")]
pub const DEFAULT_MISSING_POLICY: &str = "error";
pub const fn default_auto_converge<T>() -> Option<T> {
    None
}
pub const DEFAULT_RETURN_DIAGNOSTICS: bool = false;
pub const DEFAULT_RETURN_RESIDUALS: bool = false;
pub const DEFAULT_RETURN_ROBUSTNESS_WEIGHTS: bool = false;
pub const DEFAULT_RETURN_DERIVATIVE: bool = false;
pub const DEFAULT_RETURN_SORTED: bool = false;
pub const DEFAULT_RETAIN_MODEL: bool = false;

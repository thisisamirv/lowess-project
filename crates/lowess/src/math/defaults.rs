// Default values for math module types (kernel, scaling, boundary).

// Internal dependencies
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;

pub const DEFAULT_WEIGHT_FUNCTION_ENUM: WeightFunction = WeightFunction::Tricube;
#[cfg(feature = "dev")]
pub const DEFAULT_WEIGHT_FUNCTION: &str = "tricube";
pub const DEFAULT_SCALING_METHOD_ENUM: ScalingMethod = ScalingMethod::MAD;
#[cfg(feature = "dev")]
pub const DEFAULT_SCALING_METHOD: &str = "mad";
pub const DEFAULT_BOUNDARY_POLICY_ENUM: BoundaryPolicy = BoundaryPolicy::Extend;
#[cfg(feature = "dev")]
pub const DEFAULT_BOUNDARY_POLICY: &str = "extend";

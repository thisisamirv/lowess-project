// String input compatibility for builder methods.
//
// Defines [`IntoEnum`], a sealed conversion trait that allows builder methods
// to accept either a typed enum value or a string literal/`String`, mirroring
// the case-insensitive string parsing used by all language bindings.
//
// Invalid strings do not panic; errors accumulate in the builder and are
// returned together as [`LowessError::ParseErrors`] by `build()`.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

// Internal dependencies
use crate::adapters::online::UpdateMode;
use crate::adapters::streaming::MergeStrategy;
use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::math::boundary::BoundaryPolicy;
use crate::math::kernel::WeightFunction;
use crate::math::scaling::ScalingMethod;
use crate::primitives::errors::LowessError;

// Converts a value into a typed enum, either infallibly (enum variant) or
// via case-insensitive string parsing (string literal / `String`).
pub trait IntoEnum<E> {
    fn into_enum(self) -> Result<E, LowessError>;
}

// Generate IntoEnum impls for a concrete (non-generic) enum type.
macro_rules! impl_into_enum_for {
    ($ty:ty) => {
        impl IntoEnum<$ty> for $ty {
            #[inline]
            fn into_enum(self) -> Result<$ty, LowessError> {
                Ok(self)
            }
        }

        impl IntoEnum<$ty> for &str {
            #[inline]
            fn into_enum(self) -> Result<$ty, LowessError> {
                self.parse()
            }
        }

        impl IntoEnum<$ty> for String {
            #[inline]
            fn into_enum(self) -> Result<$ty, LowessError> {
                self.as_str().parse()
            }
        }
    };
}

impl_into_enum_for!(BoundaryPolicy);
impl_into_enum_for!(MergeStrategy);
impl_into_enum_for!(RobustnessMethod);
impl_into_enum_for!(ScalingMethod);
impl_into_enum_for!(UpdateMode);
impl_into_enum_for!(WeightFunction);
impl_into_enum_for!(ZeroWeightFallback);

// String input compatibility for fastLowess parallel builder methods.
//
// Mirrors the `parse` module in lowess. Defines [`IntoEnum`] so that
// parallel builder methods can accept either typed enum values or strings,
// exactly as the underlying LowessBuilder from lowess does.

// External dependencies
use std::string::String;

// Enum types re-exported from the lowess crate
use lowess::internals::adapters::online::UpdateMode;
use lowess::internals::adapters::streaming::MergeStrategy;
use lowess::internals::algorithms::regression::ZeroWeightFallback;
use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::math::boundary::BoundaryPolicy;
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::primitives::errors::LowessError;

// Converts a value into a typed enum, either infallibly (enum variant) or
// via case-insensitive string parsing (string literal / `String`).
pub(crate) trait IntoEnum<E> {
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
impl_into_enum_for!(UpdateMode);
impl_into_enum_for!(WeightFunction);
impl_into_enum_for!(ZeroWeightFallback);

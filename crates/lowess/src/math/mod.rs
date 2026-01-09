//! Layer 2: Math
//!
//! This layer provides pure mathematical functions used throughout LOWESS:
//! - Kernel functions for distance-based weighting
//! - Robust statistics (MAD/MAR)

// Kernel (weight) functions for distance-based weighting.
pub mod kernel;

// Robust scale estimation (MAR/MAD).
pub mod scaling;

// Boundary padding utilities.
pub mod boundary;

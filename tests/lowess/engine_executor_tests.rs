#![cfg(feature = "dev")]
//! Tests for LOWESS execution engine.
//!
//! These tests verify the core execution engine components:
//! - LowessExecutor construction and builder methods
//! - LowessConfig default values
//! - ExecutorOutput structure
//! - Basic smoothing operations
//!
//! ## Test Organization
//!
//! 1. **Constructor Tests** - Default values and builder pattern
//! 2. **Config Tests** - LowessConfig defaults and construction
//! 3. **Output Tests** - ExecutorOutput structure
//! 4. **Builder Pattern** - Fluent API methods
//!
//! Note: The executor is primarily tested through integration tests
//! in api_tests.rs and adapter tests. These unit tests focus on the
//! executor's public interface and configuration.

use approx::assert_relative_eq;
use num_traits::Float;

use lowess::internals::algorithms::robustness::RobustnessMethod;
use lowess::internals::api::BoundaryPolicy;
use lowess::internals::engine::executor::{
    ExecutorOutput, LowessBuffer, LowessConfig, LowessExecutor,
};
use lowess::internals::math::kernel::WeightFunction;
use lowess::internals::math::scaling::ScalingMethod;

// ============================================================================
// Constructor Tests
// ============================================================================

/// Test LowessExecutor default constructor.
///
/// Verifies that default values are set correctly.
#[test]
fn test_executor_new_defaults() {
    let executor = LowessExecutor::<f64>::new();

    assert_relative_eq!(executor.fraction, 0.67, epsilon = 1e-6);
    assert_eq!(executor.iterations, 3, "Default iterations should be 3");
    assert_relative_eq!(executor.delta, 0.0, epsilon = 1e-12);
    assert_eq!(
        executor.weight_function,
        WeightFunction::Tricube,
        "Default weight function should be Tricube"
    );
    assert_eq!(
        executor.zero_weight_fallback, 0,
        "Default zero weight fallback should be 0 (UseLocalMean)"
    );
    assert_eq!(
        executor.robustness_method,
        RobustnessMethod::Bisquare,
        "Default robustness method should be Bisquare"
    );
}

/// Test LowessExecutor default trait.
///
/// Verifies that Default trait produces same result as new().
#[test]
fn test_executor_default_trait() {
    let executor1 = LowessExecutor::<f64>::new();
    let executor2 = LowessExecutor::<f64>::default();

    assert_relative_eq!(executor1.fraction, executor2.fraction, epsilon = 1e-12);
    assert_eq!(executor1.iterations, executor2.iterations);
    assert_relative_eq!(executor1.delta, executor2.delta, epsilon = 1e-12);
    assert_eq!(executor1.weight_function, executor2.weight_function);
    assert_eq!(
        executor1.zero_weight_fallback,
        executor2.zero_weight_fallback
    );
    assert_eq!(executor1.robustness_method, executor2.robustness_method);
}

// ============================================================================
// Config Tests
// ============================================================================

/// Test LowessConfig default constructor.
///
/// Verifies that default configuration values are set correctly.
#[test]
fn test_config_defaults() {
    let config = LowessConfig::<f64>::default();

    assert!(config.fraction.is_none(), "Default fraction should be None");
    assert_eq!(config.iterations, 3, "Default iterations should be 3");
    assert_relative_eq!(config.delta, 0.0, epsilon = 1e-12);
    assert_eq!(
        config.weight_function,
        WeightFunction::Tricube,
        "Default weight function should be Tricube"
    );
    assert_eq!(
        config.zero_weight_fallback, 0,
        "Default zero weight fallback should be 0"
    );
    assert_eq!(
        config.robustness_method,
        RobustnessMethod::Bisquare,
        "Default robustness method should be Bisquare"
    );
    assert!(
        config.cv_fractions.is_none(),
        "Default CV fractions should be None"
    );
    assert!(config.cv_kind.is_none(), "Default CV kind should be None");
    assert!(
        config.auto_convergence.is_none(),
        "Default auto-convergence should be None"
    );
    assert!(
        config.return_variance.is_none(),
        "Default return_variance should be None"
    );
}

/// Test LowessConfig with custom values.
///
/// Verifies that config can be constructed with custom values.
#[test]
fn test_config_custom() {
    let config = LowessConfig {
        fraction: Some(0.5),
        iterations: 5,
        delta: 0.01,
        weight_function: WeightFunction::Epanechnikov,
        zero_weight_fallback: 1,
        robustness_method: RobustnessMethod::Huber,
        cv_fractions: Some(vec![0.3, 0.5, 0.7]),
        cv_kind: None,
        auto_convergence: Some(1e-6),
        return_variance: None,
        boundary_policy: BoundaryPolicy::default(),
        scaling_method: ScalingMethod::default(),
        custom_smooth_pass: None,
        custom_cv_pass: None,
        custom_interval_pass: None,
        custom_fit_pass: None,
        parallel: false,
        backend: None,
        cv_seed: None,
        delegate_boundary_handling: false,
    };

    assert_eq!(config.fraction, Some(0.5));
    assert_eq!(config.iterations, 5);
    assert_relative_eq!(config.delta, 0.01, epsilon = 1e-12);
    assert_eq!(config.weight_function, WeightFunction::Epanechnikov);
    assert_eq!(config.zero_weight_fallback, 1);
    assert_eq!(config.robustness_method, RobustnessMethod::Huber);
    assert_eq!(config.cv_fractions, Some(vec![0.3, 0.5, 0.7]));
    assert_eq!(config.auto_convergence, Some(1e-6));
}

// ============================================================================
// Output Tests
// ============================================================================

/// Test ExecutorOutput structure.
///
/// Verifies that output can be constructed and accessed.
#[test]
fn test_executor_output_basic() {
    let output = ExecutorOutput {
        smoothed: vec![1.0, 2.0, 3.0],
        std_errors: None,
        iterations: None,
        used_fraction: 0.5,
        cv_scores: None,
        robustness_weights: vec![1.0, 1.0, 1.0],
        residuals: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
    };

    assert_eq!(output.smoothed, vec![1.0, 2.0, 3.0]);
    assert!(output.std_errors.is_none());
    assert!(output.iterations.is_none());
    assert_relative_eq!(output.used_fraction, 0.5, epsilon = 1e-12);
    assert!(output.cv_scores.is_none());
}

/// Test ExecutorOutput with all optional fields.
///
/// Verifies that output can contain all optional data.
#[test]
fn test_executor_output_complete() {
    let output = ExecutorOutput {
        smoothed: vec![1.0, 2.0, 3.0],
        std_errors: Some(vec![0.1, 0.2, 0.3]),
        iterations: Some(5),
        used_fraction: 0.5,
        cv_scores: Some(vec![0.1, 0.2, 0.3]),
        robustness_weights: vec![1.0, 1.0, 1.0],
        residuals: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
    };

    assert_eq!(output.smoothed.len(), 3);
    assert!(output.std_errors.is_some());
    assert_eq!(output.std_errors.unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(output.iterations, Some(5));
    assert!(output.cv_scores.is_some());
}

/// Test ExecutorOutput with empty smoothed values.
///
/// Verifies that empty output is valid.
#[test]
fn test_executor_output_empty() {
    let output = ExecutorOutput {
        smoothed: vec![],
        std_errors: None,
        iterations: None,
        used_fraction: 0.5,
        cv_scores: None,
        robustness_weights: vec![],
        residuals: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
    };

    assert!(output.smoothed.is_empty());
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

/// Test executor builder pattern for fraction.
///
/// Verifies that fraction can be set via builder.
#[test]
fn test_executor_builder_fraction() {
    let executor = LowessExecutor::<f64>::new().fraction(0.5);

    assert_relative_eq!(executor.fraction, 0.5, epsilon = 1e-12);
}

/// Test executor builder pattern for iterations.
///
/// Verifies that iterations can be set via builder.
#[test]
fn test_executor_builder_iterations() {
    let executor = LowessExecutor::<f64>::new().iterations(5);

    assert_eq!(executor.iterations, 5);
}

/// Test executor builder pattern for delta.
///
/// Verifies that delta can be set via builder.
#[test]
fn test_executor_builder_delta() {
    let executor = LowessExecutor::<f64>::new().delta(0.01);

    assert_relative_eq!(executor.delta, 0.01, epsilon = 1e-12);
}

/// Test executor builder pattern for weight function.
///
/// Verifies that weight function can be set via builder.
#[test]
fn test_executor_builder_weight_function() {
    let executor = LowessExecutor::<f64>::new().weight_function(WeightFunction::Epanechnikov);

    assert_eq!(executor.weight_function, WeightFunction::Epanechnikov);
}

/// Test executor builder pattern for zero weight fallback.
///
/// Verifies that zero weight fallback can be set via builder.
#[test]
fn test_executor_builder_zero_weight_fallback() {
    let executor = LowessExecutor::<f64>::new().zero_weight_fallback(1);

    assert_eq!(executor.zero_weight_fallback, 1);
}

/// Test executor builder pattern for robustness method.
///
/// Verifies that robustness method can be set via builder.
#[test]
fn test_executor_builder_robustness_method() {
    let executor = LowessExecutor::<f64>::new().robustness_method(RobustnessMethod::Huber);

    assert_eq!(executor.robustness_method, RobustnessMethod::Huber);
}

/// Test executor builder pattern chaining.
///
/// Verifies that multiple builder methods can be chained.
#[test]
fn test_executor_builder_chaining() {
    let executor = LowessExecutor::<f64>::new()
        .fraction(0.5)
        .iterations(5)
        .delta(0.01)
        .weight_function(WeightFunction::Gaussian)
        .zero_weight_fallback(2)
        .robustness_method(RobustnessMethod::Talwar);

    assert_relative_eq!(executor.fraction, 0.5, epsilon = 1e-12);
    assert_eq!(executor.iterations, 5);
    assert_relative_eq!(executor.delta, 0.01, epsilon = 1e-12);
    assert_eq!(executor.weight_function, WeightFunction::Gaussian);
    assert_eq!(executor.zero_weight_fallback, 2);
    assert_eq!(executor.robustness_method, RobustnessMethod::Talwar);
}

/// Test executor with f32 type.
///
/// Verifies that executor works with f32.
#[test]
fn test_executor_f32() {
    let executor = LowessExecutor::<f32>::new().fraction(0.5f32);

    assert_relative_eq!(executor.fraction, 0.5f32, epsilon = 1e-6);
    assert_eq!(executor.iterations, 3);
}

/// Test config with f32 type.
///
/// Verifies that config works with f32.
#[test]
fn test_config_f32() {
    let config = LowessConfig::<f32> {
        fraction: Some(0.5f32),
        iterations: 3,
        delta: 0.0f32,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: 0,
        robustness_method: RobustnessMethod::Bisquare,
        cv_fractions: None,
        cv_kind: None,
        auto_convergence: None,
        return_variance: None,
        boundary_policy: BoundaryPolicy::default(),
        scaling_method: ScalingMethod::default(),
        custom_smooth_pass: None,
        custom_cv_pass: None,
        custom_interval_pass: None,
        custom_fit_pass: None,
        parallel: false,
        backend: None,
        cv_seed: None,
        delegate_boundary_handling: false,
    };

    assert_eq!(config.fraction, Some(0.5f32));
}

/// Test output with f32 type.
///
/// Verifies that output works with f32.
#[test]
fn test_output_f32() {
    let output = ExecutorOutput {
        smoothed: vec![1.0f32, 2.0f32, 3.0f32],
        std_errors: None,
        iterations: None,
        used_fraction: 0.5f32,
        cv_scores: None,
        robustness_weights: vec![1.0f32, 1.0f32, 1.0f32],
        residuals: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
    };

    assert_eq!(output.smoothed.len(), 3);
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test auto-convergence with zero tolerance.
#[test]
fn test_executor_convergence_zero_tolerance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.5, 6.0, 7.5, 10.0]; // Non-linear data

    // With tolerance = 0.0, convergence should never be reached
    // (unless smoothed values are exactly identical, which is unlikely)
    let config = LowessConfig {
        fraction: Some(0.5),
        iterations: 10,
        delta: 0.0,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: 0,
        robustness_method: RobustnessMethod::Bisquare,
        cv_fractions: None,
        cv_kind: None,
        auto_convergence: Some(0.0), // Zero tolerance
        return_variance: None,
        boundary_policy: BoundaryPolicy::Extend,
        scaling_method: ScalingMethod::default(),
        custom_smooth_pass: None,
        custom_cv_pass: None,
        custom_interval_pass: None,
        custom_fit_pass: None,
        parallel: false,
        backend: None,
        cv_seed: None,
        delegate_boundary_handling: false,
    };

    let output = LowessExecutor::run_with_config(&x, &y, config).unwrap();

    // Should run all iterations since tolerance=0 is impossible to meet
    // (or very few if it happens to converge exactly)
    assert!(output.iterations.is_some());
    assert!(output.iterations.unwrap() >= 1);
}

/// Test executor with delta equal to or greater than data range.
#[test]
fn test_executor_delta_equals_range() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];

    // Data range is 4.0 (max - min)
    let config = LowessConfig {
        fraction: Some(0.5),
        iterations: 0,
        delta: 4.0,
        weight_function: WeightFunction::Tricube,
        zero_weight_fallback: 0,
        robustness_method: RobustnessMethod::Bisquare,
        cv_fractions: None,
        cv_kind: None,
        auto_convergence: None,
        return_variance: None,
        boundary_policy: BoundaryPolicy::Extend,
        scaling_method: ScalingMethod::default(),
        custom_smooth_pass: None,
        custom_cv_pass: None,
        custom_interval_pass: None,
        custom_fit_pass: None,
        parallel: false,
        backend: None,
        cv_seed: None,
        delegate_boundary_handling: false,
    };

    let output = LowessExecutor::run_with_config(&x, &y, config).unwrap();

    // Should still produce valid output
    assert_eq!(output.smoothed.len(), 5);
    assert!(output.smoothed.iter().all(|v| v.is_finite()));
}

/// Test LowessBuffer allocation without convergence tracking.
#[test]
fn test_buffer_allocation_no_convergence() {
    let mut buffers = LowessBuffer::<f64>::with_capacity(5);
    buffers.prepare(5, false);

    assert_eq!(buffers.y_smooth.len(), 5);
    assert!(buffers.y_prev.is_empty());
    assert_eq!(buffers.robustness_weights.len(), 5);
    assert_eq!(buffers.residuals.len(), 5);
    assert_eq!(buffers.weights.len(), 5);
}

/// Test LowessBuffer allocation with convergence tracking.
#[test]
fn test_buffer_allocation_with_convergence() {
    let mut buffers = LowessBuffer::<f64>::with_capacity(5);
    buffers.prepare(5, true);

    assert_eq!(buffers.y_smooth.len(), 5);
    assert_eq!(buffers.y_prev.len(), 5);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test executor with zero iterations.
///
/// Verifies that zero iterations is valid.
#[test]
fn test_executor_zero_iterations() {
    let executor = LowessExecutor::<f64>::new().iterations(0);

    assert_eq!(executor.iterations, 0);
}

/// Test executor with large iterations.
///
/// Verifies that large iteration count is valid.
#[test]
fn test_executor_large_iterations() {
    let executor = LowessExecutor::<f64>::new().iterations(1000);

    assert_eq!(executor.iterations, 1000);
}

/// Test executor with fraction at boundaries.
///
/// Verifies that boundary fraction values are valid.
#[test]
fn test_executor_fraction_boundaries() {
    let executor1 = LowessExecutor::<f64>::new().fraction(0.01);
    assert_relative_eq!(executor1.fraction, 0.01, epsilon = 1e-12);

    let executor2 = LowessExecutor::<f64>::new().fraction(1.0);
    assert_relative_eq!(executor2.fraction, 1.0, epsilon = 1e-12);
}

/// Test config clone trait.
///
/// Verifies that config can be cloned.
#[test]
fn test_config_clone() {
    let config1 = LowessConfig::<f64>::default();
    let config2 = config1.clone();

    assert_eq!(config1.iterations, config2.iterations);
    assert_relative_eq!(config1.delta, config2.delta, epsilon = 1e-12);
}

/// Test executor clone trait.
///
/// Verifies that executor can be cloned.
#[test]
fn test_executor_clone() {
    let executor1 = LowessExecutor::<f64>::new().fraction(0.5);
    let executor2 = executor1.clone();

    assert_relative_eq!(executor1.fraction, executor2.fraction, epsilon = 1e-12);
    assert_eq!(executor1.iterations, executor2.iterations);
}

/// Test output clone trait.
///
/// Verifies that output can be cloned.
#[test]
fn test_output_clone() {
    let output1 = ExecutorOutput {
        smoothed: vec![1.0, 2.0, 3.0],
        std_errors: Some(vec![0.1, 0.2, 0.3]),
        iterations: Some(5),
        used_fraction: 0.5,
        cv_scores: Some(vec![0.1, 0.2]),
        robustness_weights: vec![1.0, 1.0, 1.0],
        residuals: None,
        confidence_lower: None,
        confidence_upper: None,
        prediction_lower: None,
        prediction_upper: None,
    };

    let output2 = output1.clone();

    assert_eq!(output1.smoothed, output2.smoothed);
    assert_eq!(output1.std_errors, output2.std_errors);
    assert_eq!(output1.iterations, output2.iterations);
}

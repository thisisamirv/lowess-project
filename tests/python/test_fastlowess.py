"""Tests for fastlowess Python bindings.

Comprehensive test suite covering:
- Basic smoothing functionality
- Full smooth() with all options
- Cross-validation
- Streaming and online adapters
- Error handling
- Edge cases
"""

import numpy as np
import pytest
import fastlowess


class TestSmooth:
    """Tests for the smooth() function."""

    def test_basic_smooth(self):
        """Test basic smooth with default parameters."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

        result = fastlowess.smooth(x, y, fraction=0.5)

        assert isinstance(result, fastlowess.LowessResult)
        assert len(result.y) == len(x)
        assert len(result.x) == len(x)
        assert result.fraction_used == pytest.approx(0.5)

    def test_basic_smooth_serial(self):
        """Test basic smooth with parallel=False."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

        result = fastlowess.smooth(x, y, fraction=0.5, parallel=False)

        assert isinstance(result, fastlowess.LowessResult)
        assert len(result.y) == len(x)

    def test_lowess_with_diagnostics(self):
        """Test lowess with diagnostics enabled."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

        result = fastlowess.smooth(x, y, fraction=0.5, return_diagnostics=True)

        assert result.diagnostics is not None
        diag = result.diagnostics
        assert isinstance(diag, fastlowess.Diagnostics)
        assert diag.rmse >= 0
        assert diag.mae >= 0
        assert 0 <= diag.r_squared <= 1
        assert diag.residual_sd >= 0

    def test_lowess_with_residuals(self):
        """Test lowess with residuals enabled."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 5.9, 8.2, 9.8])

        result = fastlowess.smooth(x, y, fraction=0.5, return_residuals=True)

        assert result.residuals is not None
        assert len(result.residuals) == len(x)

    def test_lowess_with_robustness_weights(self):
        """Test lowess with robustness weights enabled."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.1, 100.0, 8.2, 9.8])  # Outlier

        result = fastlowess.smooth(
            x, y, fraction=0.7, iterations=3, return_robustness_weights=True
        )

        assert result.robustness_weights is not None
        assert len(result.robustness_weights) == len(x)
        # All weights should be in [0, 1]
        assert np.all(result.robustness_weights >= 0)
        assert np.all(result.robustness_weights <= 1)

    def test_lowess_with_confidence_intervals(self):
        """Test lowess with confidence intervals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + np.random.normal(0, 1, 20)

        result = fastlowess.smooth(x, y, fraction=0.5, confidence_intervals=0.95)

        assert result.confidence_lower is not None
        assert result.confidence_upper is not None
        assert len(result.confidence_lower) == len(x)
        assert len(result.confidence_upper) == len(x)
        # Lower bound should be <= upper bound
        for i in range(len(x)):
            assert result.confidence_lower[i] <= result.confidence_upper[i]

    def test_lowess_with_prediction_intervals(self):
        """Test lowess with prediction intervals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2 * x + np.random.normal(0, 1, 20)

        result = fastlowess.smooth(x, y, fraction=0.5, prediction_intervals=0.95)

        assert result.prediction_lower is not None
        assert result.prediction_upper is not None
        assert len(result.prediction_lower) == len(x)
        assert len(result.prediction_upper) == len(x)

    def test_lowess_different_weight_functions(self):
        """Test lowess with different weight functions."""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)

        kernels = [
            "tricube",
            "epanechnikov",
            "gaussian",
            "uniform",
            "biweight",
            "triangle",
        ]

        for kernel in kernels:
            result = fastlowess.smooth(x, y, fraction=0.5, weight_function=kernel)
            assert len(result.y) == len(x)

    def test_lowess_different_robustness_methods(self):
        """Test lowess with different robustness methods."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 100.0, 8.0, 10.0])  # Outlier

        methods = ["bisquare", "huber", "talwar"]

        for method in methods:
            result = fastlowess.smooth(
                x, y, fraction=0.7, iterations=3, robustness_method=method
            )
            assert len(result.y) == len(x)

    def test_lowess_with_delta(self):
        """Test lowess with delta optimization."""
        x = np.linspace(0, 100, 200)
        y = np.sin(x / 10)

        result = fastlowess.smooth(x, y, fraction=0.1, delta=0.1)
        assert len(result.y) == len(x)

    def test_lowess_iterations(self):
        """Test lowess with different iteration counts."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 100.0, 8.0, 10.0])

        for iterations in [0, 1, 3, 5]:
            result = fastlowess.smooth(x, y, fraction=0.7, iterations=iterations)
            assert len(result.y) == len(x)


class TestSmoothStreaming:
    """Tests for the smooth_streaming() function."""

    def test_streaming_returns_all_points(self):
        """Test that streaming returns smoothed values for all input points."""
        # Create data where we can verify all points are returned
        x = np.linspace(0, 100, 100)
        y = 2 * x + 1  # Linear data

        result = fastlowess.smooth_streaming(x, y, fraction=0.3, chunk_size=5000)

        assert isinstance(result, fastlowess.LowessResult)
        # Critical: verify all points are returned
        assert len(result.y) == len(x), f"Expected {len(x)} points, got {len(result.y)}"
        assert len(result.x) == len(x), (
            f"Expected {len(x)} x-values, got {len(result.x)}"
        )

    def test_streaming_basic(self):
        """Test basic streaming smoothing."""
        # Need large enough data and chunk_size > default overlap (500)
        x = np.linspace(0, 1000, 2000)
        y = np.sin(x / 100)

        result = fastlowess.smooth_streaming(x, y, fraction=0.1, chunk_size=1000)

        assert isinstance(result, fastlowess.LowessResult)
        # Verify we get all points back
        assert len(result.y) == len(x), f"Expected {len(x)} points, got {len(result.y)}"

    def test_streaming_larger_data(self):
        """Test streaming with larger dataset."""
        np.random.seed(42)
        # Use a dataset where chunking makes sense
        x = np.linspace(0, 1000, 5000)
        y = np.sin(x / 100) + np.random.normal(0, 0.1, 5000)

        result = fastlowess.smooth_streaming(x, y, fraction=0.05, chunk_size=1500)

        assert isinstance(result, fastlowess.LowessResult)
        # Verify we get all points back
        assert len(result.y) == len(x), f"Expected {len(x)} points, got {len(result.y)}"

    def test_streaming_accuracy(self):
        """Test that streaming produces accurate results for linear data."""
        x = np.linspace(0, 100, 200)
        y = 2 * x + 1  # Perfect linear

        result = fastlowess.smooth_streaming(x, y, fraction=0.5, chunk_size=1000)

        # Compare streaming with batch (both will have same padding bias)
        batch_res = fastlowess.smooth(x, y, fraction=0.5)
        np.testing.assert_allclose(result.y, batch_res.y, rtol=1e-10)

        # Also verify it's generally accurate for the middle part (less padding bias)
        # Skip the edges (25% on each side)
        start, end = 50, 150
        np.testing.assert_allclose(result.y[start:end], y[start:end], rtol=0.1)

    def test_streaming_residuals(self):
        """Test streaming with return_residuals=True."""
        x = np.linspace(0, 100, 200)
        y = np.sin(x / 10)

        result = fastlowess.smooth_streaming(
            x, y, fraction=0.1, chunk_size=50, return_residuals=True
        )

        assert isinstance(result, fastlowess.LowessResult)
        assert result.residuals is not None
        assert len(result.residuals) == len(x)
        assert len(result.y) == len(x)

    def test_streaming_zero_weight_fallback(self):
        """Test streaming with zero_weight_fallback parameter."""
        x = np.linspace(0, 100, 200)
        y = np.sin(x)
        # Just verifying the parameter is accepted and runs
        result = fastlowess.smooth_streaming(
            x, y, fraction=0.1, chunk_size=50, zero_weight_fallback="return_original"
        )
        assert len(result.y) == len(x)


class TestSmoothOnline:
    """Tests for the smooth_online() function."""

    def test_online_zero_weight_fallback(self):
        """Test online with zero_weight_fallback parameter."""
        x = np.arange(20, dtype=float)
        y = x.copy()

        # Verify parameter is accepted
        result = fastlowess.smooth_online(
            x, y, fraction=0.5, window_capacity=10, zero_weight_fallback="return_none"
        )
        assert len(result.y) == len(x)

    def test_online_basic(self):
        """Test basic online smoothing."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

        result = fastlowess.smooth_online(
            x, y, fraction=0.5, window_capacity=10, min_points=3
        )

        assert len(result.y) == len(x)
        assert isinstance(result, fastlowess.LowessResult)

    def test_online_with_noise(self):
        """Test online smoothing with noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 20, 50)
        y = 2 * x + np.random.normal(0, 1, 50)

        result = fastlowess.smooth_online(
            x, y, fraction=0.3, window_capacity=20, min_points=5
        )

        assert len(result.y) == len(x)


class TestSmoothResult:
    """Tests for the LowessResult class."""

    def test_result_repr(self):
        """Test result string representation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = fastlowess.smooth(x, y, fraction=0.5)

        repr_str = repr(result)
        assert "LowessResult" in repr_str
        assert "n=5" in repr_str

    def test_result_optional_fields_none(self):
        """Test that optional fields are None when not requested."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = fastlowess.smooth(x, y, fraction=0.5)

        # These should be None when not requested
        assert result.diagnostics is None
        assert result.residuals is None
        assert result.robustness_weights is None
        assert result.confidence_lower is None
        assert result.confidence_upper is None
        assert result.prediction_lower is None
        assert result.prediction_upper is None


class TestDiagnostics:
    """Tests for the Diagnostics class."""

    def test_diagnostics_repr(self):
        """Test diagnostics string representation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = fastlowess.smooth(x, y, fraction=0.5, return_diagnostics=True)

        repr_str = repr(result.diagnostics)
        assert "Diagnostics" in repr_str
        assert "rmse" in repr_str
        assert "mae" in repr_str
        assert "r_squared" in repr_str

    def test_diagnostics_values(self):
        """Test diagnostic values are reasonable."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect linear

        result = fastlowess.smooth(x, y, fraction=0.5, return_diagnostics=True)

        diag = result.diagnostics
        # Perfect linear data should have very low error
        assert diag.rmse < 0.1
        assert diag.mae < 0.1
        assert diag.r_squared > 0.99


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_fraction_high(self):
        """Test error on fraction > 1."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, fraction=1.5)

    def test_invalid_fraction_low(self):
        """Test error on fraction <= 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, fraction=0.0)

    def test_mismatched_lengths(self):
        """Test error on mismatched x and y lengths."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, fraction=0.5)

    def test_invalid_weight_function(self):
        """Test error on invalid weight function."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, fraction=0.5, weight_function="invalid")

    def test_invalid_robustness_method(self):
        """Test error on invalid robustness method."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, fraction=0.5, robustness_method="invalid")

    def test_invalid_cv_method(self):
        """Test error on invalid CV method."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        with pytest.raises(ValueError):
            fastlowess.smooth(x, y, cv_fractions=[0.5], cv_method="invalid")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_points(self):
        """Test with minimum number of points."""
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])

        result = fastlowess.smooth(x, y, fraction=1.0)
        assert len(result.y) == 2

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        n = 1000
        x = np.linspace(0, 100, n)
        y = np.sin(x / 10) + np.random.normal(0, 0.1, n)

        result = fastlowess.smooth(x, y, fraction=0.1)
        assert len(result.y) == n

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        np.random.seed(42)
        x = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        y = np.array([6.0, 2.0, 10.0, 4.0, 8.0])

        result = fastlowess.smooth(x, y, fraction=0.7)
        assert len(result.y) == 5

    def test_duplicate_x_values(self):
        """Test with duplicate x values."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        y = np.array([2.0, 2.1, 4.0, 3.9, 6.0])

        result = fastlowess.smooth(x, y, fraction=0.7)
        assert len(result.y) == 5

    def test_all_same_y(self):
        """Test with constant y values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = fastlowess.smooth(x, y, fraction=0.5)
        np.testing.assert_allclose(result.y, y, rtol=1e-10)


class TestCrossValidation:
    """Tests for cross-validation via smooth()."""

    def test_cv_basic(self):
        """Test basic cross-validation selects a fraction."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.sin(x)

        result = fastlowess.smooth(x, y, cv_fractions=[0.2, 0.3, 0.5, 0.7])

        assert result.fraction_used in [0.2, 0.3, 0.5, 0.7]
        assert result.cv_scores is not None
        assert len(result.cv_scores) == 4
        assert len(result.y) == len(x)

    def test_cv_kfold(self):
        """Test k-fold cross-validation."""
        x = np.linspace(0, 10, 30)
        y = x**2

        result = fastlowess.smooth(
            x, y, cv_fractions=[0.3, 0.5], cv_method="kfold", cv_k=5
        )

        assert result.fraction_used in [0.3, 0.5]
        assert result.cv_scores is not None

    def test_cv_loocv(self):
        """Test leave-one-out cross-validation."""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)

        result = fastlowess.smooth(x, y, cv_fractions=[0.4, 0.6], cv_method="loocv")

        assert result.fraction_used in [0.4, 0.6]
        assert result.cv_scores is not None

    def test_cv_with_other_params(self):
        """Test CV works with other parameters."""
        x = np.linspace(0, 10, 40)
        y = 2 * x + 0.5 * np.sin(x)

        result = fastlowess.smooth(
            x,
            y,
            cv_fractions=[0.3, 0.5, 0.7],
            iterations=2,
            return_diagnostics=True,
            return_residuals=True,
        )

        assert result.fraction_used in [0.3, 0.5, 0.7]
        assert result.diagnostics is not None
        assert result.residuals is not None

    def test_cv_single_fraction(self):
        """Test CV with single fraction still works."""
        x = np.linspace(0, 10, 25)
        y = x + np.random.normal(0, 0.1, 25)

        result = fastlowess.smooth(x, y, cv_fractions=[0.5])

        assert result.fraction_used == 0.5
        assert result.cv_scores is not None
        assert len(result.cv_scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

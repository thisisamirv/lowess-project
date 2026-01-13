import numpy as np
import pytest
from fastlowess import OnlineLowess, StreamingLowess, LowessResult


def test_online_lowess_kwargs():
    """Test that OnlineLowess can be initialized with keyword arguments."""
    ol = OnlineLowess(
        fraction=0.3,
        window_capacity=50,
        iterations=2,
        weight_function="gaussian",
        update_mode="incremental",
    )

    # Test update
    val = ol.update(1.0, 2.0)
    # With only 1 point and min_points=2 (default), should return None or original depending on implementation
    # Here we expect None if min_points=2
    assert val is None or val == 2.0

    val2 = ol.update(2.0, 4.0)
    assert val2 is not None


def test_online_lowess_add_points():
    """Test adding multiple points to OnlineLowess."""
    ol = OnlineLowess(fraction=0.5, window_capacity=10)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = ol.add_points(x, y)
    assert isinstance(result, LowessResult)
    assert len(result.y) == 5
    assert np.allclose(result.y, y, rtol=1e-5)


def test_streaming_lowess_kwargs():
    """Test that StreamingLowess can be initialized with keyword arguments."""
    sl = StreamingLowess(
        fraction=0.1, chunk_size=100, overlap=10, iterations=3, return_residuals=True
    )

    x = np.linspace(0, 10, 50)
    y = 2 * x + 1

    res1 = sl.process_chunk(x, y)
    assert isinstance(res1, LowessResult)

    res2 = sl.finalize()
    assert isinstance(res2, LowessResult)


if __name__ == "__main__":
    pytest.main([__file__])

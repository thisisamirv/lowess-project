import math

import fastlowess
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def _boundary_cases(draw):
    n = draw(st.integers(min_value=3, max_value=32))
    gaps = draw(
        st.lists(
            st.floats(
                min_value=0.05,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
    x = np.concatenate(([0.0], np.cumsum(gaps)))
    x /= x[-1]
    shape = draw(st.sampled_from(["random", "constant", "linear", "quadratic", "step"]))
    if shape == "random":
        y = np.asarray(
            draw(
                st.lists(
                    st.floats(
                        min_value=-5.0,
                        max_value=5.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=n,
                    max_size=n,
                )
            )
        )
    elif shape == "constant":
        y = np.full(n, 2.5)
    elif shape == "linear":
        y = 1.5 * x - 0.75
    elif shape == "quadratic":
        y = x**2 - 0.5 * x
    else:
        y = np.where(x < 0.5, -1.0, 1.0)

    fraction = draw(
        st.floats(
            min_value=0.15,
            max_value=0.9,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return x, y, fraction


def _pad_inputs(x: np.ndarray, y: np.ndarray, q: int, policy: str):
    pad_len = min(q // 2, len(x) - 1)
    if pad_len == 0:
        return x.copy(), y.copy(), 0

    if policy in ("extend", "zero"):
        left_x = x[0] - np.arange(pad_len, 0, -1) * (x[1] - x[0])
        right_x = x[-1] + np.arange(1, pad_len + 1) * (x[-1] - x[-2])
        if policy == "extend":
            left_y = np.full(pad_len, y[0])
            right_y = np.full(pad_len, y[-1])
        else:
            left_y = np.zeros(pad_len)
            right_y = np.zeros(pad_len)
    elif policy == "reflect":
        left_indices = np.arange(pad_len, 0, -1)
        right_indices = np.arange(len(x) - 2, len(x) - pad_len - 2, -1)
        left_x = 2 * x[0] - x[left_indices]
        right_x = 2 * x[-1] - x[right_indices]
        left_y = y[left_indices]
        right_y = y[right_indices]
    else:
        raise ValueError(f"unsupported boundary policy: {policy}")

    return (
        np.concatenate((left_x, x, right_x)),
        np.concatenate((left_y, y, right_y)),
        pad_len,
    )


@pytest.mark.parametrize("policy", ["extend", "reflect", "zero"])
@settings(max_examples=40, derandomize=True, database=None, deadline=None)
@given(case=_boundary_cases())
def test_boundary_policy_matches_explicit_padded_python_binding(policy: str, case):
    x, y, fraction = case
    q = min(len(x), max(2, math.floor(len(x) * fraction + 1e-7)))
    x_padded, y_padded, pad_len = _pad_inputs(x, y, q, policy)
    padded_fraction = q / len(x_padded)
    padded_q = math.floor(len(x_padded) * padded_fraction + 1e-7)
    assert padded_q == q

    options = {
        "iterations": 0,
        "delta": 0.0,
        "weight_function": "tricube",
        "parallel": False,
        "backend": "cpu",
    }
    actual = fastlowess.Lowess(
        fraction=fraction, boundary_policy=policy, **options
    ).fit(x, y)
    explicitly_padded = fastlowess.Lowess(
        fraction=padded_fraction, boundary_policy="noboundary", **options
    ).fit(x_padded, y_padded)
    expected = explicitly_padded.y[pad_len : pad_len + len(x)]

    np.testing.assert_allclose(actual.y, expected, rtol=1e-12, atol=1e-12)

Quick Start
===========

This guide covers basic usage of the ``fastlowess`` package for batch smoothing.

.. note::
   Ensure you have installed the package first (see :doc:`installation`).

Basic Smoothing
---------------

To smooth a dataset, provide the ``x`` and ``y`` arrays to the ``smooth()`` function. By default, it uses a smoothing fraction of 0.67 and 3 robustness iterations.

.. code-block:: python

   import numpy as np
   import fastlowess

   # Generate linear data with some noise
   x = np.linspace(0, 10, 100)
   y = 2 * x + np.random.normal(0, 1, 100)

   # Simple smoothing
   result = fastlowess.smooth(x, y, fraction=0.3)

   print(f"Smoothed values: {result.y[:5]}...")

Understanding the Result
------------------------

The ``smooth()`` function returns a ``LowessResult`` object containing:

*   ``x``: The potentially sorted independent variable values.
*   ``y``: The smoothed dependent variable values.
*   ``fraction_used``: The smoothing fraction applied.
*   ``iterations_used``: Number of robustness iterations performed.
*   Optional fields (if requested): ``standard_errors``, ``confidence_lower``, ``residuals``, etc.

Customizing Parameters
----------------------

You can control the smoothing behavior using core parameters:

*   **fraction**: Controls the span of the smoothing window (span). Values: (0.0, 1.0].
*   **iterations**: Number of robustness iterations to downweight outliers.

.. image:: _static/images/fraction_effect_comparison.svg
   :alt: Fraction Effect Comparison
   :align: center
   :width: 100%
.. code-block:: python

   # More smoothing, higher robustness
   result = fastlowess.smooth(x, y, fraction=0.7, iterations=5)

Uncertainty Quantification
--------------------------

Request confidence or prediction intervals by providing the desired confidence level (e.g., 0.95 for 95% intervals).

.. code-block:: python

   result = fastlowess.smooth(
       x, y,
       fraction=0.3,
       confidence_intervals=0.95,
       prediction_intervals=0.95
   )

   # Access results
   lower = result.confidence_lower
   upper = result.confidence_upper

*   **Range**: ``(0, 1]``
*   **0.1 - 0.3**: Captures fine details. Best for data with high-frequency variation.
*   **0.3 - 0.5**: Moderate smoothing (recommended for most cases).
*   **0.5 - 0.7**: Heavy smoothing, emphasizes long-term trends (default is 0.67).
*   **> 0.8**: Extremely smooth, may over-smooth local features.

**Iterations (Robustness)**

Number of robust re-weighting iterations to handle outliers.

*   **0**: No robustness (fastest). Use for clean data.
*   **1-3**: Light to moderate robustness (recommended). Default is 3.
*   **4-6**: Strong robustness for heavily contaminated data.

Getting Diagnostics
-------------------

You can request fit statistics like R-squared ($R^2$), RMSE, and AIC:

.. code-block:: python

    result = fastlowess.smooth(x, y, fraction=0.3, return_diagnostics=True)

    if result.diagnostics:
        print(f"R-squared: {result.diagnostics.r_squared:.4f}")
        print(f"RMSE: {result.diagnostics.rmse:.4f}")

    # Output:
    # R-squared: 0.9012
    # RMSE: 0.2164

Next Steps
----------

*   Learn about handling outliers and uncertainty in :doc:`advanced_usage`.
*   Working with massive datasets? Check out :doc:`execution_modes`.

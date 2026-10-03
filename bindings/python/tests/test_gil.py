"""Regression test that verifies heavy LOWESS fitting releases the Python GIL."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastlowess import Lowess, OnlineLowess


def heavy_computation():
    """Run a deliberately heavy LOWESS fit in a worker thread."""
    # Create large random dataset to ensure computation takes time
    n_points = 50_000  # Enough to take a few hundred ms
    x = np.linspace(0, 100, n_points)
    y = np.sin(x) + np.random.normal(0, 0.5, n_points)

    # Configure with expensive settings to prolong runtime
    lowess = Lowess(
        fraction=0.3, iterations=3, parallel=False
    )  # parallel=False to stress single thread logic if needed, but here we test GIL
    print("Starting fit...")
    lowess.fit(x, y)
    print("Fit finished.")


def heartbeat():
    """Emit periodic ticks while the worker thread is fitting."""
    start = time.time()
    ticks = 0
    while time.time() - start < 2.0:  # Run for 2 seconds
        time.sleep(0.1)
        ticks += 1
        print(".", end="", flush=True)
    return ticks


def test_gil_release():
    """Check that the main thread stays responsive during model fitting."""
    print("Verifying GIL release...")

    # Thread for heavy computation
    t = threading.Thread(target=heavy_computation)

    start_time = time.time()
    t.start()

    # Run heartbeat on main thread
    ticks = heartbeat()

    t.join()
    duration = time.time() - start_time

    print(f"\nTotal duration: {duration:.2f}s")
    print(f"Heartbeat ticks: {ticks}")

    # If GIL was NOT released, the main thread would be blocked and ticks would be 0 or very low
    # until the heavy computation finished.
    # We expect ticks to be roughly duration / 0.1

    # If GIL was NOT released, the main thread would be blocked effectively completely.
    # We lower the threshold to 0.25 (25% of theoretical max) to account for CI noise/scheduling.
    expected_ticks = (duration / 0.1) * 0.25
    if ticks < expected_ticks and duration > 0.5:
        print(
            f"FAIL: Main thread was blocked! Got {ticks} ticks, expected > {expected_ticks:.2f}"
        )
        sys.exit(1)
    else:
        print("PASS: Main thread remained responsive.")


def test_online_add_point_releases_gil():
    """A full-window online update should not block unrelated Python threads."""
    online = OnlineLowess(
        fraction=0.5,
        window_capacity=2000,
        min_points=2000,
        iterations=5,
        update_mode="full",
        intervals={"confidence": 0.95, "bootstrap": 50},
    )
    for index in range(1999):
        online.add_point(float(index), float(index))

    started = threading.Event()

    def update():
        started.set()
        online.add_point(1999.0, 1999.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(update)
        assert started.wait(timeout=2.0)

        ticks = 0
        while not future.done():
            ticks += 1
            time.sleep(0.01)

        future.result()
    assert ticks > 0


if __name__ == "__main__":
    test_gil_release()

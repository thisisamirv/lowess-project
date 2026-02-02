import threading
import time
import numpy as np
from fastlowess import Lowess


def heavy_computation():
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
    start = time.time()
    ticks = 0
    while time.time() - start < 2.0:  # Run for 2 seconds
        time.sleep(0.1)
        ticks += 1
        print(".", end="", flush=True)
    return ticks


def test_gil_release():
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

    expected_ticks = (duration / 0.1) * 0.5  # Allow 50% margin
    if ticks < expected_ticks and duration > 0.5:
        print(
            f"FAIL: Main thread was blocked! Got {ticks} ticks, expected > {expected_ticks}"
        )
        exit(1)
    else:
        print("PASS: Main thread remained responsive.")


if __name__ == "__main__":
    test_gil_release()

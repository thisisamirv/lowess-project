"""
benchmarks/gpu_transfer.py — GPU data-transfer overhead benchmark.

Isolates host↔device buffer transfer time from kernel execution time by
reproducing the exact buffer footprint of one fastLowess GPU fit:
  - Upload:   2 × n f32 arrays  (x, y)
  - Download: 3 × n f32 arrays  (y_smooth, robustness_weights, residuals)

The script uses wgpu-py (same wgpu version as fastLowess) to run the transfers
without any compute, then cross-references against an end-to-end GPU benchmark
JSON (if present) to compute the transfer fraction of total GPU time.

Install:
    pip install wgpu numpy
Run:
    python benchmarks/gpu_transfer.py
    python benchmarks/gpu_transfer.py --sizes 1000 10000 100000 --iters 30
    python benchmarks/gpu_transfer.py --output benchmarks/output/gpu_transfer.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────────────────

# Sizes that span the interesting crossover region (where transfer dominates
# at small n and kernel dominates at large n).
DEFAULT_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
F32_BYTES = 4

# fastLowess GPU buffer layout (without optional SE / interval buffers):
#   upload arrays  : x, y                        → 2 × n f32
#   download arrays: y_smooth, weights, residuals → 3 × n f32
UPLOAD_ARRAYS = 2
DOWNLOAD_ARRAYS = 3


# ── wgpu helpers ──────────────────────────────────────────────────────────────


def _require_wgpu():
    try:
        import wgpu

        return wgpu
    except ImportError:
        print(
            "ERROR: wgpu not installed.\n"
            "       Run:  pip install wgpu\n"
            "       Docs: https://wgpu-py.readthedocs.io/",
            file=sys.stderr,
        )
        sys.exit(1)


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
        sys.exit(1)


def _init_device(wgpu):
    """Request a high-performance adapter and return (adapter, device)."""
    # wgpu-py exposes the GPU singleton at wgpu.gpu, not on the module directly.
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        print("ERROR: No GPU adapter found.", file=sys.stderr)
        sys.exit(1)
    device = adapter.request_device_sync()
    return adapter, device


# ── core measurement ──────────────────────────────────────────────────────────


def _sync_read(device, wgpu, sync_buf: object) -> None:
    """Force CPU/GPU synchronisation by reading a tiny 4-byte buffer."""
    device.queue.read_buffer(sync_buf, 0, size=4)


def bench_transfer(wgpu, np, device, sync_buf, n: int, iters: int, warmup: int) -> dict:
    """
    Measure upload and download latency for one fastLowess-sized dataset of n
    points.  Returns a dict with per-phase and round-trip timing statistics.
    """
    nbytes_upload = UPLOAD_ARRAYS * n * F32_BYTES  # x + y
    nbytes_download = DOWNLOAD_ARRAYS * n * F32_BYTES  # y_smooth + weights + residuals

    # Synthetic data — same deterministic pattern used in the R/Python benchmarks.
    t = np.linspace(0.0, 2.0 * math.pi, n, dtype=np.float32)
    upload_data = np.concatenate([np.sin(t) + 0.1, t]).astype(np.float32)  # x ∥ y

    upload_buf = device.create_buffer(
        size=nbytes_upload,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
    )
    download_buf = device.create_buffer(
        size=nbytes_download,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.STORAGE,
    )

    upload_ms: list[float] = []
    download_ms: list[float] = []
    roundtrip_ms: list[float] = []

    for i in range(warmup + iters):
        # ── upload (host → device) ────────────────────────────────────────────
        t0 = time.perf_counter()
        device.queue.write_buffer(upload_buf, 0, upload_data.tobytes())
        device.queue.submit([])
        _sync_read(device, wgpu, sync_buf)  # wait for GPU to acknowledge the write
        t_up = (time.perf_counter() - t0) * 1_000.0

        # ── download (device → host) ─────────────────────────────────────────
        t1 = time.perf_counter()
        device.queue.read_buffer(
            download_buf
        )  # blocking; creates staging buf internally
        t_dn = (time.perf_counter() - t1) * 1_000.0

        if i >= warmup:
            upload_ms.append(t_up)
            download_ms.append(t_dn)
            roundtrip_ms.append(t_up + t_dn)

    def _stats(times_ms: list[float], nbytes: int) -> dict:
        mean_ms = statistics.mean(times_ms)
        return {
            "mean_ms": round(mean_ms, 4),
            "median_ms": round(statistics.median(times_ms), 4),
            "stdev_ms": round(
                statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0, 4
            ),
            "min_ms": round(min(times_ms), 4),
            "max_ms": round(max(times_ms), 4),
            # Effective bandwidth: bytes transferred / wall time (includes PCIe latency)
            "bandwidth_gbps": round((nbytes / 1e9) / (mean_ms / 1_000.0), 4),
        }

    return {
        "n": n,
        "upload_nbytes": nbytes_upload,
        "download_nbytes": nbytes_download,
        "upload": _stats(upload_ms, nbytes_upload),
        "download": _stats(download_ms, nbytes_download),
        "round_trip": _stats(roundtrip_ms, nbytes_upload + nbytes_download),
    }


# ── cross-reference against end-to-end GPU benchmark ─────────────────────────


def _load_e2e(path: str) -> dict[int, float] | None:
    """
    Parse an existing GPU benchmark JSON (rfastlowess_parallel.json or
    rust_benchmark_gpu.json) and return a {size: mean_time_ms} mapping.
    Handles both the flat-list and the {"scalability": [...]} envelope formats.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None

    entries = (
        raw
        if isinstance(raw, list)
        else raw.get("scalability", raw.get("benchmarks", []))
    )
    if not isinstance(entries, list):
        return None

    result: dict[int, float] = {}
    for entry in entries:
        n = entry.get("size") or entry.get("n")
        t = entry.get("mean_time_ms") or entry.get("mean_ms")
        if isinstance(n, int) and isinstance(t, (int, float)):
            result[n] = float(t)
    return result or None


def _transfer_fraction(
    results: list[dict], e2e_by_size: dict[int, float]
) -> list[dict]:
    rows = []
    for r in results:
        n = r["n"]
        rt_ms = r["round_trip"]["mean_ms"]
        e2e_ms = e2e_by_size.get(n)
        rows.append(
            {
                "n": n,
                "transfer_ms": rt_ms,
                "e2e_ms": e2e_ms,
                "transfer_fraction": (round(rt_ms / e2e_ms, 4) if e2e_ms else None),
                "kernel_ms": (round(e2e_ms - rt_ms, 4) if e2e_ms else None),
            }
        )
    return rows


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="output/gpu_transfer.json",
        help="Path to write results JSON (default: output/gpu_transfer.json)",
    )
    parser.add_argument(
        "--e2e",
        default="output/rfastlowess_parallel.json",
        help="End-to-end GPU benchmark JSON for transfer-fraction comparison",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        metavar="N",
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help=f"Measurement iterations per size (default: {DEFAULT_ITERS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Warmup iterations per size (default: {DEFAULT_WARMUP})",
    )
    args = parser.parse_args()

    wgpu = _require_wgpu()
    np = _require_numpy()

    print("Initialising GPU device…")
    adapter, device = _init_device(wgpu)
    # adapter.summary format varies by wgpu-py version: "Device on Backend" or "Device via Backend"
    summary = adapter.summary
    adapter_name, backend = summary, "?"
    for sep in (" via ", " on "):
        if sep in summary:
            adapter_name, backend = summary.rsplit(sep, 1)
            break
    print(f"  Adapter : {adapter_name}")
    print(f"  Backend : {backend}")
    print()

    # Tiny 4-byte buffer used as a synchronisation barrier after write_buffer calls.
    sync_buf = device.create_buffer(
        size=4,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.STORAGE,
    )

    results: list[dict] = []
    for n in args.sizes:
        up_kb = UPLOAD_ARRAYS * n * F32_BYTES / 1024
        dn_kb = DOWNLOAD_ARRAYS * n * F32_BYTES / 1024
        print(
            f"  n = {n:>9,}   upload {up_kb:>8.1f} KB   download {dn_kb:>8.1f} KB …",
            flush=True,
        )
        r = bench_transfer(wgpu, np, device, sync_buf, n, args.iters, args.warmup)
        results.append(r)
        up = r["upload"]
        dn = r["download"]
        rt = r["round_trip"]
        print(
            f"             upload   {up['mean_ms']:8.3f} ms  ({up['bandwidth_gbps']:.2f} GB/s)"
        )
        print(
            f"             download {dn['mean_ms']:8.3f} ms  ({dn['bandwidth_gbps']:.2f} GB/s)"
        )
        print(f"             round-trip {rt['mean_ms']:6.3f} ms")

    e2e_by_size = _load_e2e(args.e2e)
    fractions: list[dict] = []
    if e2e_by_size:
        fractions = _transfer_fraction(results, e2e_by_size)
        print("\nTransfer fraction of end-to-end GPU time")
        print(f"  (e2e reference: {args.e2e})")
        print(
            f"  {'n':>10}  {'transfer':>10}  {'e2e':>10}  {'fraction':>10}  {'kernel':>10}"
        )
        print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for row in fractions:
            if row["transfer_fraction"] is not None:
                print(
                    f"  {row['n']:>10,}  "
                    f"{row['transfer_ms']:>9.3f}ms  "
                    f"{row['e2e_ms']:>9.3f}ms  "
                    f"{row['transfer_fraction']:>9.1%}  "
                    f"{row['kernel_ms']:>9.3f}ms"
                )
    else:
        print(
            f"\n(No end-to-end GPU data found at {args.e2e} — skipping transfer-fraction table.)"
        )

    output = {
        "benchmark": "gpu_transfer",
        "description": (
            "Host↔device buffer transfer overhead isolated from kernel execution. "
            f"Upload models {UPLOAD_ARRAYS} f32 arrays (x, y); "
            f"download models {DOWNLOAD_ARRAYS} f32 arrays (y_smooth, weights, residuals)."
        ),
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "methodology": {
            "upload": "queue.write_buffer → submit([]) → sync read of 4-byte sentinel buffer",
            "download": "queue.read_buffer (blocking; creates staging buffer + map internally)",
            "round_trip": "upload_ms + download_ms per iteration",
        },
        "results": results,
        "transfer_vs_e2e": fractions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

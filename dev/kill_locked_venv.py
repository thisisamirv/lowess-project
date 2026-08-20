"""Kill any process holding a lock on files inside the given virtualenv directories.

On Windows, a lingering ``python.exe`` from a venv (e.g. a VS Code Python
extension language server still pointed at that interpreter) can hold an open
handle on ``Scripts/python.exe``. When `make clean` then tries to remove the
venv via `git clean`/`rm`, the delete fails and Git's Windows compat layer
blocks waiting for an interactive "Unlink of file '...' failed. Should I try
again? (y/n)" answer, hanging any non-interactive `make clean` run.

Run this before removing a venv to proactively free that lock. No-op on
non-Windows platforms, where this failure mode doesn't occur.

Usage: python dev/kill_locked_venv.py .venv docs-venv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(venv_dirs: list[str]) -> None:
    if sys.platform != "win32":
        return

    targets = [str(Path(d).resolve()) for d in venv_dirs if Path(d).exists()]
    if not targets:
        return

    conditions = " -or ".join(f"$_.Path -like '{t}\\*'" for t in targets)
    script = (
        "Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.Path -and ({conditions}) }} | "
        "Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

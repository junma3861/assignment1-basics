"""Windows-friendly shim for the Unix-only resource module."""

from __future__ import annotations

from typing import Tuple

# Keep the interface minimal: only what tests require.
RLIMIT_AS = 9


def getrlimit(resource: int) -> Tuple[int, int]:
    # Windows does not support resource limits; return "unlimited".
    return (-1, -1)


def setrlimit(resource: int, limits: Tuple[int, int]) -> None:
    # No-op on Windows.
    return None

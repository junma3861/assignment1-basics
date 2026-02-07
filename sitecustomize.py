"""Configure UTF-8 defaults on Windows for tests that omit encodings."""

from __future__ import annotations

import locale
import os
import sys

if sys.platform.startswith("win"):
    # Best-effort: make the preferred encoding UTF-8 for open().
    os.environ.setdefault("PYTHONUTF8", "1")
    for loc in ("C.UTF-8", "en_US.UTF-8"):
        try:
            locale.setlocale(locale.LC_CTYPE, loc)
            break
        except locale.Error:
            continue

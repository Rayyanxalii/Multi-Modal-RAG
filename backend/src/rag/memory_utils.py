"""Memory telemetry and MuPDF cache controls for constrained cloud containers.

Render/HF Spaces commonly run the app with 512 MB RAM.  These helpers make
memory pressure visible in the logs and release PyMuPDF's internal caches
between pages so peak RSS stays below the container ceiling.
"""

from __future__ import annotations

import gc
import os
import sys


def current_rss_mb() -> float | None:
    """Return the current resident set size in MB when the platform exposes it."""
    # Linux (Render, Docker, HF Spaces): /proc is the cheapest and most accurate source.
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as statm:
            rss_pages = int(statm.read().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        pass

    # Fallback for local Windows/macOS development when psutil happens to be installed.
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def peak_rss_mb() -> float | None:
    """Return the process's peak RSS in MB when supported by the platform."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return value / 1024 if sys.platform != "darwin" else value / (1024 * 1024)
    except Exception:
        return None


def log_memory(label: str) -> float | None:
    """Log current and peak RSS so deployment logs pinpoint memory spikes."""
    current = current_rss_mb()
    peak = peak_rss_mb()
    if current is None and peak is None:
        return None

    parts = []
    if current is not None:
        parts.append(f"current {current:.0f} MB")
    if peak is not None:
        parts.append(f"peak {peak:.0f} MB")
    print(f"[Memory] {label}: {' / '.join(parts)}", flush=True)
    return current


def configure_mupdf_low_memory() -> None:
    """Disable MuPDF per-device display-list caching to reduce peak memory."""
    try:
        import pymupdf

        pymupdf.TOOLS.set_low_memory(True)
    except Exception:
        pass


def release_mupdf_caches() -> None:
    """Flush MuPDF's decoded-object store and run a Python GC cycle.

    Safe to call between pages: any still-referenced object is simply reloaded
    on demand.  This prevents decoded images, fonts and display lists from
    accumulating across a multi-page parse.
    """
    try:
        import pymupdf

        pymupdf.TOOLS.store_shrink(100)
    except Exception:
        pass
    gc.collect()

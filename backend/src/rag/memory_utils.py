"""Memory telemetry and MuPDF cache helpers for constrained containers."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_MB = 1024.0 * 1024.0
# cgroup v1 reports a near-2^63 sentinel when no memory limit is set.
_UNLIMITED_CGROUP_BYTES = 1 << 60

_CLOUD_ENV_VARS = (
    "RENDER",
    "RENDER_SERVICE_ID",
    "RENDER_INSTANCE_ID",
    "SPACE_ID",
    "K_SERVICE",
    "FLY_APP_NAME",
    "DYNO",
    "AWS_EXECUTION_ENV",
)


@dataclass(frozen=True)
class MemorySnapshot:
    """Raw memory measurements from the host and cgroup, in MB."""

    total_mb: float | None = None
    available_mb: float | None = None
    limit_mb: float | None = None
    limit_available_mb: float | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class ResourceProfile:
    """Resolved runtime profile used by the parser, renderer, and upload API."""

    low_memory: bool
    use_docling: bool
    total_mb: float | None
    available_mb: float | None
    limit_mb: float | None
    threshold_mb: float
    headroom_mb: float
    cloud: bool
    reason: str

    def summary(self) -> str:
        """Return a short text summary of this profile."""
        def _fmt(value: float | None) -> str:
            """Format a value in MB or 'unknown'."""
            return f"{value:.0f} MB" if value is not None else "unknown"

        parser = "docling" if self.use_docling else "pymupdf-lite"
        return (
            f"parser={parser}, low_memory={self.low_memory}, "
            f"total={_fmt(self.total_mb)}, available={_fmt(self.available_mb)}, "
            f"cgroup_limit={_fmt(self.limit_mb)}, threshold={_fmt(self.threshold_mb)}, "
            f"cloud={self.cloud}, reason={self.reason}"
        )


def _env_flag(name: str) -> bool | None:
    """Return a tri-state boolean from an environment variable."""
    raw = os.getenv(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def _env_float(name: str, default: float) -> float:
    """Read a positive float from an env var, or return the default."""
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _read_cgroup_memory() -> tuple[float | None, float | None]:
    """Read the container memory limit and free space from cgroup v2 or v1."""
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
        except Exception:
            continue
        if not raw_limit or raw_limit == "max":
            continue
        try:
            limit_bytes = int(raw_limit)
        except ValueError:
            continue
        if limit_bytes <= 0 or limit_bytes >= _UNLIMITED_CGROUP_BYTES:
            continue

        limit_mb = limit_bytes / _MB
        available_mb = None
        try:
            usage_bytes = int(usage_path.read_text(encoding="ascii").strip())
            available_mb = max(0.0, (limit_bytes - usage_bytes) / _MB)
        except Exception:
            pass
        return limit_mb, available_mb
    return None, None


def _read_host_memory() -> tuple[float | None, float | None]:
    """Read total/available physical memory using only the standard library."""
    # Linux: MemAvailable includes reclaimable page cache.
    try:
        values: dict[str, str] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo:
            for line in meminfo:
                key, _, rest = line.partition(":")
                values[key] = rest.strip()
        total_kb = float(values["MemTotal"].split()[0])
        available_kb = float(values.get("MemAvailable", values.get("MemFree", "0")).split()[0])
        return total_kb / 1024.0, available_kb / 1024.0
    except Exception:
        pass

    # Windows: GlobalMemoryStatusEx returns both total and currently available RAM.
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / _MB, status.ullAvailPhys / _MB
        except Exception:
            pass

    # macOS / generic POSIX.
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_mb = (os.sysconf("SC_PHYS_PAGES") * page_size) / _MB
        available_mb = None
        try:
            available_mb = (os.sysconf("SC_AVPHYS_PAGES") * page_size) / _MB
        except (AttributeError, ValueError, OSError):
            pass
        return total_mb, available_mb
    except (AttributeError, ValueError, OSError):
        pass

    # Optional psutil fallback (not a required dependency).
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / _MB, vm.available / _MB
    except Exception:
        return None, None


def _memory_snapshot() -> MemorySnapshot:
    """Collect host and cgroup memory data into a single snapshot."""
    total_mb, available_mb = _read_host_memory()
    limit_mb, limit_available_mb = _read_cgroup_memory()
    sources = []
    if total_mb is not None:
        sources.append("host")
    if limit_mb is not None:
        sources.append("cgroup")
    return MemorySnapshot(
        total_mb=total_mb,
        available_mb=available_mb,
        limit_mb=limit_mb,
        limit_available_mb=limit_available_mb,
        source="+".join(sources) or "unknown",
    )


def get_resource_profile() -> ResourceProfile:
    """Detect the usable memory budget and choose Docling or PyMuPDF Lite."""
    snapshot = _memory_snapshot()
    threshold_mb = _env_float("DOCLING_MIN_RAM_MB", 3500.0)
    headroom_mb = _env_float("DOCLING_MIN_AVAILABLE_MB", 1800.0)
    docling_flag = _env_flag("USE_DOCLING")
    low_flag = _env_flag("LOW_MEMORY_PARSER")
    cloud = any(os.getenv(name) for name in _CLOUD_ENV_VARS)

    effective_total = snapshot.total_mb
    if snapshot.limit_mb is not None:
        effective_total = (
            snapshot.limit_mb
            if effective_total is None
            else min(effective_total, snapshot.limit_mb)
        )

    effective_available = snapshot.available_mb
    if snapshot.limit_available_mb is not None:
        effective_available = (
            snapshot.limit_available_mb
            if effective_available is None
            else min(effective_available, snapshot.limit_available_mb)
        )
    elif snapshot.limit_mb is not None:
        # Usage is unavailable; cap host "available" at the container limit so
        # a large host reading cannot mask a small cgroup budget.
        effective_available = (
            snapshot.limit_mb
            if effective_available is None
            else min(effective_available, snapshot.limit_mb)
        )

    if docling_flag is True:
        low_memory = False
        reason = "USE_DOCLING=1 forces full mode"
    elif low_flag is True:
        low_memory = True
        reason = "LOW_MEMORY_PARSER=true forces low-memory mode"
    elif low_flag is False:
        low_memory = False
        reason = "LOW_MEMORY_PARSER=false forces full mode"
    elif effective_total is None:
        low_memory = True
        reason = "memory budget could not be detected; using safe low-memory mode"
    elif cloud and snapshot.limit_mb is None:
        low_memory = True
        reason = "cloud platform without a readable cgroup limit; using safe low-memory mode"
    elif effective_total < threshold_mb:
        low_memory = True
        reason = (
            f"detected {effective_total:.0f} MB usable RAM "
            f"< {threshold_mb:.0f} MB required for Docling"
        )
    elif effective_available is not None and effective_available < headroom_mb:
        low_memory = True
        reason = (
            f"only {effective_available:.0f} MB currently available "
            f"< {headroom_mb:.0f} MB headroom required for Docling"
        )
    else:
        low_memory = False
        available_text = (
            f"{effective_available:.0f} MB available"
            if effective_available is not None
            else "available RAM unknown"
        )
        reason = (
            f"detected {effective_total:.0f} MB usable RAM "
            f">= {threshold_mb:.0f} MB threshold ({available_text})"
        )

    if docling_flag is True:
        use_docling = True
    elif docling_flag is False:
        use_docling = False
    else:
        use_docling = not low_memory

    return ResourceProfile(
        low_memory=low_memory,
        use_docling=use_docling,
        total_mb=effective_total,
        available_mb=effective_available,
        limit_mb=snapshot.limit_mb,
        threshold_mb=threshold_mb,
        headroom_mb=headroom_mb,
        cloud=cloud,
        reason=reason,
    )


def log_resource_profile() -> ResourceProfile:
    """Log the resolved parser/memory profile for startup diagnostics."""
    profile = get_resource_profile()
    print(f"[Memory] Resource profile: {profile.summary()}", flush=True)
    return profile


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
    """Flush MuPDF caches and run garbage collection to free memory."""
    try:
        import pymupdf

        pymupdf.TOOLS.store_shrink(100)
    except Exception:
        pass
    gc.collect()

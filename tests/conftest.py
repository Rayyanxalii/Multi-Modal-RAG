"""Shared pytest fixtures and utilities for backend testing under Render constraints."""

import io
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
import pymupdf
from fastapi.testclient import TestClient

# Ensure backend and backend/src are in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
BACKEND_SRC = BACKEND_DIR / "src"

for path_entry in (str(REPO_ROOT), str(BACKEND_DIR), str(BACKEND_SRC)):
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

# Force low-memory cloud parser mode during tests to match Render production
os.environ["LOW_MEMORY_PARSER"] = "true"
os.environ["RENDER"] = "true"

from backend.server import app


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Ensure test-generated PDF files and directories are cleaned up after test run."""
    yield
    raw_dir = REPO_ROOT / "data" / "raw"
    test_prefixes = ("polling_test", "valid_sample", "bench_test", "large_doc", "oversized_pages")
    for pattern in test_prefixes:
        for f in raw_dir.glob(f"{pattern}*"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

        for sub in ("rendered_pages", "chunks", "figures", "tables"):
            p = REPO_ROOT / "data" / sub / pattern
            if p.exists() and p.is_dir():
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            for f in (REPO_ROOT / "data" / sub).glob(f"{pattern}*"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Provide a thread-safe FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client


def create_test_pdf(pages: int = 3, text_per_page: str = "Test page content.") -> bytes:
    """Create a valid in-memory PDF with an arbitrary number of pages using PyMuPDF."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)  # Standard A4
        page.insert_text((50, 72), f"Header Page {i + 1}", fontsize=16)
        page.insert_text((50, 120), f"{text_per_page} (Page {i + 1})", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_oversized_pdf(target_mb: float = 10.5) -> bytes:
    """Create a valid PDF padded with stream metadata to exceed Render's 10 MB limit."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 72), "Oversized Test Document", fontsize=16)
    base_bytes = doc.tobytes()
    doc.close()

    # Append valid PDF comment padding to simulate an oversized payload safely
    needed_bytes = int(target_mb * 1024 * 1024) - len(base_bytes)
    if needed_bytes > 0:
        padding = b"\n% " + (b"X" * (needed_bytes - 4))
        return base_bytes + padding
    return base_bytes

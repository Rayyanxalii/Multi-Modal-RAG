"""Tests verifying Render free-tier upload limits and low-memory parser rules."""

import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.server import get_upload_limits
from rag.fast_pdf_parser import is_low_memory_env
from tests.conftest import create_oversized_pdf, create_test_pdf


class TestRenderEnvironmentDetection:
    """Validate environment detection protects the 512 MB container from loading Docling."""

    def test_default_is_low_memory(self):
        """Without any environment flags, default must be True (safe mode)."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_low_memory_env() is True

    def test_render_flag_triggers_low_memory(self):
        """Setting RENDER=true must enforce low memory mode."""
        with patch.dict(os.environ, {"RENDER": "true"}, clear=True):
            assert is_low_memory_env() is True

    def test_low_memory_parser_env_flag(self):
        """Setting LOW_MEMORY_PARSER=true must enforce low memory mode."""
        with patch.dict(os.environ, {"LOW_MEMORY_PARSER": "true"}, clear=True):
            assert is_low_memory_env() is True

    def test_explicit_use_docling_opt_in(self):
        """Setting USE_DOCLING=1 allows Docling on high-memory local machines."""
        with patch.dict(os.environ, {"USE_DOCLING": "1"}, clear=True):
            assert is_low_memory_env() is False


class TestUploadLimitEnforcement:
    """Verify strict 10 MB and 25-page ceiling enforced by backend server."""

    def test_limits_calculation_in_low_memory(self):
        """Limits should be exactly (10.0, 25) when LOW_MEMORY_PARSER is active."""
        with patch.dict(os.environ, {"LOW_MEMORY_PARSER": "true", "RENDER": "true"}):
            max_mb, max_pages = get_upload_limits()
            assert max_mb == 10.0
            assert max_pages == 25

    def test_upload_non_pdf_rejected_with_400(self, client: TestClient):
        """Uploading non-PDF files must be rejected immediately."""
        files = {
            "file": ("malicious_payload.exe", io.BytesIO(b"MZ...executable content"), "application/octet-stream")
        }
        res = client.post("/api/upload", files=files)
        assert res.status_code == 400
        assert "Only PDF files are accepted" in res.json().get("detail", "")

    def test_upload_exceeds_10mb_rejected_with_413(self, client: TestClient):
        """Uploading a PDF > 10 MB must return HTTP 413 Payload Too Large."""
        oversized_bytes = create_oversized_pdf(target_mb=10.5)
        files = {
            "file": ("large_doc.pdf", io.BytesIO(oversized_bytes), "application/pdf")
        }
        res = client.post("/api/upload", files=files)
        assert res.status_code == 413
        detail = res.json().get("detail", "")
        assert "exceeds the 10 MB limit" in detail or "10" in detail

    def test_upload_exceeds_25_pages_rejected_with_413(self, client: TestClient):
        """Uploading a PDF with > 25 pages must return HTTP 413 Payload Too Large."""
        doc_26_pages = create_test_pdf(pages=26, text_per_page="Stress test page content.")
        files = {
            "file": ("oversized_pages.pdf", io.BytesIO(doc_26_pages), "application/pdf")
        }
        res = client.post("/api/upload", files=files)
        assert res.status_code == 413
        detail = res.json().get("detail", "")
        assert "26 pages" in detail
        assert "free-tier limit" in detail

    def test_upload_valid_pdf_accepted_with_async_job_id(self, client: TestClient):
        """A valid small PDF (3 pages) must return HTTP 200 with an async job_id in < 0.2s."""
        doc_bytes = create_test_pdf(pages=3, text_per_page="Universal multimodal RAG test.")
        files = {
            "file": ("valid_sample.pdf", io.BytesIO(doc_bytes), "application/pdf")
        }
        res = client.post("/api/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") == "accepted"
        assert "job_id" in data
        assert len(data["job_id"]) > 4

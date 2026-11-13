"""Integration tests for all core FastAPI backend endpoints under Render constraints."""

import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_test_pdf


class TestSystemEndpoints:
    """Verify system health, static routing, and favicon handling."""

    def test_health_check(self, client: TestClient):
        """Uptime monitoring probe must return 200 OK."""
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json().get("status") == "ok"

    def test_favicon_returns_204(self, client: TestClient):
        """Browser automatic favicon requests must return 204 No Content."""
        res = client.get("/favicon.ico")
        assert res.status_code == 204

    def test_root_serves_html(self, client: TestClient):
        """Root endpoint serves the interactive working sheet HTML."""
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")


class TestAuthStatusEndpoint:
    """Verify BYOK (Bring Your Own Key) status reporting and upload limits propagation."""

    def test_auth_status_without_keys(self, client: TestClient):
        """Without any key configured, endpoint must report 'needs_key'."""
        with patch.dict(os.environ, {"GROQ_API": ""}, clear=False):
            res = client.get("/api/auth/status")
            assert res.status_code == 200
            data = res.json()
            assert "max_upload_mb" in data
            assert "max_upload_pages" in data
            assert data["max_upload_mb"] == 10.0
            assert data["max_upload_pages"] == 25

    def test_auth_status_with_client_header_key(self, client: TestClient):
        """Supplying X-Groq-Api-Key header must report 'ready' with has_user_key=True."""
        with patch.dict(os.environ, {"GROQ_API": ""}, clear=False):
            res = client.get("/api/auth/status", headers={"X-Groq-Api-Key": "gsk_test_client_key_12345"})
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "ready"
            assert data["has_user_key"] is True


class TestDocumentArchiveEndpoints:
    """Verify document archive listing, PDF serving, and rendered pages."""

    def test_list_documents(self, client: TestClient):
        """Listing documents must return an array containing at least the default document."""
        res = client.get("/api/documents")
        assert res.status_code == 200
        docs = res.json()
        assert isinstance(docs, list)
        assert len(docs) >= 1
        doc_ids = [d.get("doc_id") for d in docs]
        assert "attention" in doc_ids

    def test_get_attention_pdf(self, client: TestClient):
        """Raw PDF retrieval must serve inline application/pdf."""
        res = client.get("/api/documents/attention/pdf")
        assert res.status_code == 200
        assert "application/pdf" in res.headers.get("content-type", "")

    def test_get_attention_pages(self, client: TestClient):
        """Rendered pages listing must return page images and total count."""
        res = client.get("/api/documents/attention/pages")
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") == "success"
        assert data.get("total_pages", 0) > 0
        assert len(data.get("pages", [])) > 0


class TestAsyncUploadJobQueue:
    """Verify background ingestion job creation and polling."""

    def test_poll_unknown_job_returns_404(self, client: TestClient):
        """Polling a nonexistent job_id must return 404 Not Found."""
        res = client.get("/api/upload/status/nonexistent_job_123")
        assert res.status_code == 404
        assert "not found" in res.json().get("detail", "").lower()

    def test_poll_active_job_progress(self, client: TestClient):
        """Polling an active upload job must return stage, progress, and done flag."""
        doc_bytes = create_test_pdf(pages=2, text_per_page="Polling lifecycle test.")
        files = {
            "file": ("polling_test.pdf", io.BytesIO(doc_bytes), "application/pdf")
        }
        upload_res = client.post("/api/upload", files=files)
        assert upload_res.status_code == 200
        job_id = upload_res.json()["job_id"]

        status_res = client.get(f"/api/upload/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert "stage" in status_data
        assert "progress" in status_data
        assert "done" in status_data
        assert isinstance(status_data["progress"], (int, float))


class TestChatAuthValidation:
    """Verify chat endpoints enforce user-supplied Groq key requirement."""

    def test_chat_empty_question_rejected_with_400(self, client: TestClient):
        """Submitting an empty question must return 400 Bad Request."""
        res = client.post("/api/chat", json={"question": "   ", "doc_id": "attention"})
        assert res.status_code == 400

    def test_chat_missing_groq_key_rejected_with_401(self, client: TestClient):
        """Submitting a question without any server or client Groq key must return 401."""
        with patch.dict(os.environ, {"GROQ_API": ""}, clear=False):
            res = client.post(
                "/api/chat",
                json={"question": "What is the Transformer architecture?", "doc_id": "attention"},
            )
            assert res.status_code == 401
            assert "Vision LLM API key required" in res.json().get("detail", "")

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

    def test_get_specific_page_image_on_demand(self, client: TestClient):
        """Dedicated page image endpoint must serve image/png on demand."""
        res = client.get("/api/documents/attention/pages/1")
        assert res.status_code == 200
        assert "image/png" in res.headers.get("content-type", "")
        assert len(res.content) > 1000


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

    def test_upload_duplicate_name_deduplication_and_auto_increment(self, client: TestClient):
        """Uploading identical content reuses doc_id; uploading different content auto-increments doc_id."""
        from rag.config import RAW_DATA_DIR, CHUNKS_DIR
        for p in RAW_DATA_DIR.glob("same_name_doc*.*"):
            p.unlink(missing_ok=True)
        for p in CHUNKS_DIR.glob("same_name_doc*.*"):
            p.unlink(missing_ok=True)

        pdf_bytes_v1 = create_test_pdf(pages=1, text_per_page="Original Document Version 1")
        pdf_bytes_v2 = create_test_pdf(pages=1, text_per_page="Different Content Version 2")
        pdf_bytes_v3 = create_test_pdf(pages=1, text_per_page="Yet Another Content Version 3")

        try:
            with patch("backend.server.threading.Thread"):
                # 1. First upload -> doc_id="same_name_doc"
                files1 = {"file": ("same_name_doc.pdf", io.BytesIO(pdf_bytes_v1), "application/pdf")}
                res1 = client.post("/api/upload", files=files1)
                assert res1.status_code == 200
                data1 = res1.json()
                assert data1["doc_id"] == "same_name_doc"

                # 2. Identical re-upload -> reuses "same_name_doc"
                files_identical = {"file": ("same_name_doc.pdf", io.BytesIO(pdf_bytes_v1), "application/pdf")}
                res_same = client.post("/api/upload", files=files_identical)
                assert res_same.status_code == 200
                data_same = res_same.json()
                assert data_same["doc_id"] == "same_name_doc"

                # 3. Different content with same filename -> auto-increments to "same_name_doc_1"
                files2 = {"file": ("same_name_doc.pdf", io.BytesIO(pdf_bytes_v2), "application/pdf")}
                res2 = client.post("/api/upload", files=files2)
                assert res2.status_code == 200
                data2 = res2.json()
                assert data2["doc_id"] == "same_name_doc_1"

                # 4. Third different content -> auto-increments to "same_name_doc_2"
                files3 = {"file": ("same_name_doc.pdf", io.BytesIO(pdf_bytes_v3), "application/pdf")}
                res3 = client.post("/api/upload", files=files3)
                assert res3.status_code == 200
                data3 = res3.json()
                assert data3["doc_id"] == "same_name_doc_2"
        finally:
            for p in RAW_DATA_DIR.glob("same_name_doc*.*"):
                p.unlink(missing_ok=True)
            for p in CHUNKS_DIR.glob("same_name_doc*.*"):
                p.unlink(missing_ok=True)


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


class TestDocumentIsolation:
    """Verify strict document isolation so multi-tenant documents never leak across queries."""

    def test_dense_retriever_node_enforces_active_doc_id(self):
        """dense_retriever_node must derive doc_id and filter out any foreign document chunks."""
        from rag.graph.nodes import dense_retriever_node

        mock_chunks = [
            {"chunk_id": "dell_1", "document_id": "dell", "text": "Dell revenue"},
            {"chunk_id": "nestle_1", "document_id": "nestle", "text": "Nestle profit"},
        ]

        with patch("rag.graph.nodes.search_chunks", return_value=mock_chunks) as mock_search:
            state = {
                "question": "What is the profit?",
                "pdf_path": "data/raw/dell.pdf",
                "doc_name": "dell",
            }
            res = dense_retriever_node(state)

            # Assert search_chunks was called with doc_id="dell"
            mock_search.assert_called_once_with("What is the profit?", top_k=5, doc_id="dell")

            # Assert defense-in-depth discarded the foreign nestle chunk
            matches = res["dense_matches"]
            assert len(matches) == 1
            assert matches[0]["document_id"] == "dell"

    def test_text_fusion_node_purges_cross_document_leakage(self):
        """text_fusion_node must discard any chunks from another document before RRF fusion."""
        from rag.graph.nodes import text_fusion_node

        state = {
            "question": "Financial results",
            "pdf_path": "data/raw/dell.pdf",
            "doc_name": "dell",
            "dense_matches": [
                {"chunk_id": "dell_1", "document_id": "dell", "text": "Dell chunk"},
                {"chunk_id": "nestle_9", "document_id": "nestle", "text": "Nestle chunk"},
            ],
            "bm25_matches": [
                {"chunk_id": "dell_2", "document_id": "dell", "text": "Dell BM25"},
            ],
        }

        res = text_fusion_node(state)
        text_results = res["text_results"]
        # All items must belong to dell
        for r in text_results:
            if r.get("document_id"):
                assert r["document_id"] == "dell"

    def test_search_chunks_constructs_doc_id_filter(self):
        """search_chunks must attach a FieldCondition for document_id in Qdrant query filter."""
        from rag.text_retrieval.search_chunks import search_chunks

        with patch("rag.text_retrieval.search_chunks.get_client") as mock_get_client, \
             patch("rag.services.embedding_service.embed_query", return_value=[0.1] * 768):

            mock_client = mock_get_client.return_value
            mock_client.query_points.return_value.points = []

            search_chunks("test query", top_k=3, doc_id="dell_financials")

            mock_client.query_points.assert_called_once()
            _, kwargs = mock_client.query_points.call_args
            query_filter = kwargs.get("query_filter")
            assert query_filter is not None
            assert query_filter.must[0].key == "document_id"
            assert query_filter.must[0].match.value == "dell_financials"


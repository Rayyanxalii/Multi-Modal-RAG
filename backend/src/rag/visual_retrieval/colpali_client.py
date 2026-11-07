"""Cloud client for Hugging Face ZeroGPU ColPali Visual Search Microservice."""

import os
from pathlib import Path
from typing import Any
from functools import lru_cache

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None


class ColPaliCloudClient:
    """Client for querying and indexing documents on the remote ColPali HF Space."""

    def __init__(self, space_name: str | None = None, hf_token: str | None = None):
        target = space_name or os.getenv("COLPALI_SPACE", "https://rayyan0-colpali.hf.space")
        if "/" in target and not target.startswith("http"):
            user, repo = target.split("/", 1)
            self.space_name = f"https://{user}-{repo}.hf.space"
        else:
            self.space_name = target
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if Client is None:
                raise ImportError(
                    "gradio_client is required for cloud ColPali. Run 'pip install gradio-client'."
                )
            print(f"[ColPali Cloud] Connecting to Hugging Face Space: {self.space_name}...")
            self._client = Client(self.space_name, token=self.hf_token)
            print(f"[ColPali Cloud] Connected successfully.")
        return self._client

    def index_document(
        self,
        doc_id: str,
        image_paths: list[Path | str],
    ) -> dict[str, Any]:
        """Upload and compute ColPali visual embeddings for page images on the HF Space GPU."""
        doc_id = (doc_id or "").strip().lower()
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if not image_paths:
            raise ValueError(f"No page images provided for document '{doc_id}'")

        print(f"[ColPali Cloud] Uploading {len(image_paths)} pages for '{doc_id}' to ZeroGPU...")
        files = [handle_file(str(p)) for p in sorted(image_paths)]

        result = self.client.predict(
            new_doc_id=doc_id,
            upload_pages=files,
            api_name="/index",
        )
        print(f"[ColPali Cloud] Indexing complete: {result}")
        return result

    def search(
        self,
        query: str,
        doc_id: str = "attention",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the remote ColPali model via late interaction MaxSim."""
        query = (query or "").strip()
        if not query:
            return []

        doc_id = (doc_id or "attention").strip().lower()

        raw_results = None
        for attempt in range(1, 4):
            try:
                raw_results = self.client.predict(
                    doc_id=doc_id,
                    query=query,
                    top_k=int(top_k),
                    api_name="/search",
                )
                break
            except Exception as exc:
                print(f"[ColPali Cloud] Search attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    import time
                    time.sleep(1.5)

        if not isinstance(raw_results, list):
            return []

        # Check for error payload from server
        if raw_results and isinstance(raw_results[0], dict) and "error" in raw_results[0]:
            print(f"[ColPali Cloud] Space returned error: {raw_results[0]['error']}")
            return []

        normalized = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            page_no = item.get("page", 1)
            score = item.get("score", 0.0)
            page_name = item.get("page_name", f"page_{page_no}.png")
            normalized.append({
                "rank": item.get("rank", len(normalized) + 1),
                "page": page_name,
                "page_number": int(page_no),
                "score": float(score),
            })

        return normalized

    def list_documents(self) -> list[str]:
        """List documents available on the Space."""
        try:
            return self.client.predict(api_name="/documents") or []
        except Exception as exc:
            print(f"[ColPali Cloud] Failed to list documents: {exc}")
            return []


@lru_cache(maxsize=1)
def get_colpali_cloud_client() -> ColPaliCloudClient:
    return ColPaliCloudClient()

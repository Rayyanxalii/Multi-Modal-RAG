"""Universal Embedding Service supporting Google Gemini Embeddings with Ollama fallback."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from rag.config import (
    EMBEDDING_MODEL,
    VECTOR_SIZE,
    get_ollama_client,
)

GEMINI_MODEL = "models/gemini-embedding-001"
MAX_BATCH_SIZE = 50


def get_gemini_api_key() -> str | None:
    """Retrieve Gemini API key from environment."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip().strip("'\"") if key else None


def _call_gemini_api(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Call Google Gemini REST API with automatic 429 rate limit backoff."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{endpoint}?key={api_key}"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, 6):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            if exc.code == 429 and attempt < 5:
                wait_sec = 15.0 * attempt
                try:
                    err_json = json.loads(err_body)
                    for detail in err_json.get("error", {}).get("details", []):
                        if "retryDelay" in detail:
                            delay_str = str(detail["retryDelay"]).rstrip("s")
                            wait_sec = max(float(delay_str) + 2.0, wait_sec)
                except Exception:
                    pass
                print(f"[Gemini Embedding] 429 Quota Pause: Waiting {wait_sec:.1f}s before retry (attempt {attempt}/5)...")
                time.sleep(wait_sec)
                continue

            if attempt == 5 or exc.code in (400, 401, 403):
                raise RuntimeError(f"Gemini API Error ({exc.code}): {err_body}") from exc
            time.sleep(2.0 * attempt)
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError(f"Gemini API Connection Error: {exc}") from exc
            time.sleep(2.0 * attempt)

    raise RuntimeError("Gemini API call failed after retries.")


def embed_query(query: str) -> list[float]:
    """Embed a search query string into a 768-dimensional vector."""
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty.")

    api_key = get_gemini_api_key()
    if api_key:
        payload = {
            "model": GEMINI_MODEL,
            "content": {"parts": [{"text": clean_query}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": VECTOR_SIZE,
        }
        res = _call_gemini_api(f"{GEMINI_MODEL}:embedContent", payload, api_key)
        values = res.get("embedding", {}).get("values", [])
        if len(values) != VECTOR_SIZE:
            raise ValueError(f"Expected {VECTOR_SIZE} dimensions, got {len(values)}")
        return values

    # Fallback to Ollama if no Gemini API key configured
    ollama_client = get_ollama_client()
    res = ollama_client.embed(model="nomic-embed-text", input=clean_query)
    return res["embeddings"][0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document passages into 768-dimensional vectors."""
    if not texts:
        return []

    api_key = get_gemini_api_key()
    if api_key:
        all_embeddings: list[list[float]] = []

        for start_idx in range(0, len(texts), MAX_BATCH_SIZE):
            chunk_batch = texts[start_idx : start_idx + MAX_BATCH_SIZE]
            requests = [
                {
                    "model": GEMINI_MODEL,
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "outputDimensionality": VECTOR_SIZE,
                }
                for t in chunk_batch
            ]
            payload = {"requests": requests}
            res = _call_gemini_api(f"{GEMINI_MODEL}:batchEmbedContents", payload, api_key)
            raw_embs = res.get("embeddings", [])
            for item in raw_embs:
                vals = item.get("values", [])
                if len(vals) != VECTOR_SIZE:
                    raise ValueError(f"Expected {VECTOR_SIZE} dimensions, got {len(vals)}")
                all_embeddings.append(vals)

            if start_idx + MAX_BATCH_SIZE < len(texts):
                time.sleep(1.0)  # Gentle pacing between batches

        return all_embeddings

    # Fallback to Ollama
    ollama_client = get_ollama_client()
    res = ollama_client.embed(model="nomic-embed-text", input=texts)
    return res["embeddings"]

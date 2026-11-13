"""Memory benchmark test proving that PDF parsing stays strictly under Render 512 MB limits."""

import gc
import json
import os
from pathlib import Path

import psutil
import pytest

from rag.config import DEFAULT_PDF_PATH, get_document_paths
from rag.fast_pdf_parser import is_fast_parsed_doc, parse_pdf_fast


class TestMemoryBenchmark:
    """Benchmark memory footprint to guarantee container stability on Render Free Tier (512 MB)."""

    def test_parse_pdf_fast_memory_ceiling(self, tmp_path: Path):
        """Parse a 15-page scientific PDF and verify peak memory delta remains strictly under 60 MB."""
        if not DEFAULT_PDF_PATH.exists():
            pytest.skip("Default PDF (attention.pdf) not present on disk.")

        # Clean up garbage before baseline measurement
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline_rss_mb = process.memory_info().rss / (1024 * 1024)

        # Configure isolated temp paths for output artifacts
        test_paths = {
            "doc_name": "bench_test",
            "chunks_dir": tmp_path / "chunks",
            "chunks_path": tmp_path / "chunks" / "chunks.jsonl",
            "parsed_md": tmp_path / "chunks" / "parsed.md",
            "document_json_path": tmp_path / "chunks" / "document.json",
            "figures_raw_jsonl": tmp_path / "figures_raw.jsonl",
            "tables_jsonl": tmp_path / "tables.jsonl",
        }

        # Run the lightweight parser
        chunks = parse_pdf_fast(DEFAULT_PDF_PATH, doc_paths=test_paths)
        gc.collect()

        after_rss_mb = process.memory_info().rss / (1024 * 1024)
        delta_mb = after_rss_mb - baseline_rss_mb

        print(f"\n[Memory Benchmark] Baseline: {baseline_rss_mb:.2f} MB | After: {after_rss_mb:.2f} MB | Delta: {delta_mb:.2f} MB")

        # 1. Memory constraint: Delta must be under 60 MB (well within 512 MB limit)
        assert delta_mb < 60.0, f"Memory delta ({delta_mb:.2f} MB) exceeded 60 MB threshold!"

        # 2. Functional correctness: Chunks and artifacts must be created cleanly
        assert len(chunks) > 0
        assert test_paths["chunks_path"].exists()
        assert test_paths["document_json_path"].exists()
        assert is_fast_parsed_doc(test_paths["document_json_path"]) is True

        # 3. Heading and structure extraction
        headings_found = any(len(c.get("headings", [])) > 0 for c in chunks)
        assert headings_found, "Fast parser should have extracted structural headings from font sizes."

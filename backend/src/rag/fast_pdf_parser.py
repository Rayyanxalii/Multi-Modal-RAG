"""PDF parser using PyMuPDF, tuned for low-memory containers."""

from __future__ import annotations

import gc
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from rag.config import DEFAULT_PDF_PATH, get_document_paths
from rag.memory_utils import (
    configure_mupdf_low_memory,
    current_rss_mb,
    get_resource_profile,
    log_memory,
    release_mupdf_caches,
)


def is_low_memory_env() -> bool:
    """Return True when the detected memory budget cannot safely run Docling.

    Detection uses the container cgroup limit when present, otherwise host RAM.
    Set ``LOW_MEMORY_PARSER=1/0`` to force a mode, or ``USE_DOCLING=1/0`` to
    force only the parser.
    """
    return get_resource_profile().low_memory


def should_use_docling() -> bool:
    """Return True when Docling should be used for the current memory budget."""
    return get_resource_profile().use_docling


def is_fast_parsed_doc(document_json_path: Path) -> bool:
    """Return True if the JSON metadata marker belongs to the lightweight PyMuPDF parser."""
    if not document_json_path.exists():
        return False
    try:
        with document_json_path.open("r", encoding="utf-8") as f:
            head = f.read(256)
        return '"pymupdf_lite"' in head
    except Exception:
        return False


def should_extract_tables(low_mem: bool | None = None) -> bool:
    """Decide whether PyMuPDF table detection may run."""
    flag = os.getenv("ENABLE_TABLE_DETECTION", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    if low_mem is None:
        low_mem = is_low_memory_env()
    return not low_mem


def _detect_body_font_size(pdf: pymupdf.Document, sample_limit: int = 8) -> float:
    """Estimate primary body font size by weighting character frequency across sample pages."""
    size_counts: Counter[float] = Counter()
    sample_pages = min(len(pdf), max(1, sample_limit))
    for idx in range(sample_pages):
        page_dict = pdf[idx].get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        size = round(float(span.get("size", 10.0)), 1)
                        size_counts[size] += len(txt)
        del page_dict
    if not size_counts:
        return 10.0
    return size_counts.most_common(1)[0][0]


def _clean_text(text: str) -> str:
    """Collapse whitespace and trim a text string."""
    return " ".join((text or "").split()).strip()


def _format_table_markdown(rows: list[list[Any]]) -> str:
    """Convert raw extracted 2D table cells into clean GFM Markdown."""
    if not rows:
        return ""
    cleaned_rows = []
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols == 0:
        return ""
    for row in rows:
        cells = [_clean_text(str(cell if cell is not None else "")).replace("|", "\\|") for cell in row]
        while len(cells) < max_cols:
            cells.append("")
        cleaned_rows.append(cells)

    lines = ["| " + " | ".join(cleaned_rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    for row in cleaned_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file + rename so interrupted parsers never look complete."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL via a temp file + rename so interrupted parsers never look complete."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def parse_pdf_fast(
    pdf_path: Path = DEFAULT_PDF_PATH,
    doc_paths: dict[str, Any] | None = None,
    max_words_per_chunk: int = 380,
) -> list[dict[str, Any]]:
    """Parse PDF text chunks, headings, figures and (optionally) tables with a page-bounded memory footprint."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

    if doc_paths is None:
        doc_paths = get_document_paths(pdf_path)

    doc_name = doc_paths["doc_name"]
    chunks_dir = doc_paths["chunks_dir"]
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks_path: Path = doc_paths["chunks_path"]
    markdown_path: Path = doc_paths["parsed_md"]
    document_json_path: Path = doc_paths["document_json_path"]
    figures_path: Path = doc_paths["figures_raw_jsonl"]
    tables_jsonl_path: Path = doc_paths["tables_jsonl"]

    print(f"[FastPDF] Parsing '{pdf_path.name}' with PyMuPDF engine...")
    low_mem = is_low_memory_env()
    if low_mem:
        configure_mupdf_low_memory()

    extract_tables = should_extract_tables(low_mem)
    if extract_tables:
        rss = current_rss_mb()
        # The 320 MB guard only applies to the 512 MB cloud profile. On a
        # high-RAM host, a larger baseline RSS is not a reason to skip tables.
        if low_mem and rss is not None and rss > 320:
            print(
                f"[FastPDF] RSS already {rss:.0f} MB — skipping table detection to protect the container "
                "(set ENABLE_TABLE_DETECTION=1 on an instance with more RAM to re-enable it)."
            )
            extract_tables = False

    if not extract_tables:
        table_mode = "off"
    elif low_mem:
        table_mode = "on (text strategy)"
    else:
        table_mode = "on (lines strategy)"
    print(f"[FastPDF] Mode: low_memory={low_mem}, table_detection={table_mode}")
    log_memory("parse:start")

    chunks: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    md_sections: list[str] = []

    current_headings: list[str] = []
    acc_texts: list[str] = []
    acc_pages: set[int] = set()
    acc_bboxes: list[dict[str, Any]] = []
    acc_words = 0

    def flush_chunk() -> None:
        """Flush accumulated text into a new chunk."""
        nonlocal acc_texts, acc_pages, acc_bboxes, acc_words
        combined = "\n\n".join(t for t in acc_texts if t.strip()).strip()
        if not combined:
            acc_texts, acc_pages, acc_bboxes, acc_words = [], set(), [], 0
            return

        pages_sorted = sorted(acc_pages) if acc_pages else [1]
        idx = len(chunks)
        heading_prefix = " > ".join(current_headings)
        contextualized = f"{heading_prefix}\n\n{combined}" if heading_prefix else combined

        chunks.append({
            "chunk_id": f"{pdf_path.stem}_{idx:04d}",
            "document_id": doc_name,
            "file_name": pdf_path.name,
            "chunk_index": idx,
            "page_number": pages_sorted[0],
            "page_numbers": pages_sorted,
            "headings": list(current_headings),
            "captions": [],
            "item_types": ["TextItem"],
            "bounding_boxes": acc_bboxes[:20],
            "source_file": pdf_path.name,
            "source_mimetype": "application/pdf",
            "text": combined,
            "contextualized_text": contextualized,
        })
        acc_texts, acc_pages, acc_bboxes, acc_words = [], set(), [], 0

    with pymupdf.open(pdf_path) as pdf:
        total_pages = len(pdf)
        body_size = _detect_body_font_size(pdf, sample_limit=3 if low_mem else 8)

        for page_idx, page in enumerate(pdf):
            page_no = page_idx + 1
            page_dict = page.get_text("dict")

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox", [0, 0, 0, 0])
                block_lines = []
                max_span_size = 0.0
                is_bold = False

                for line in block.get("lines", []):
                    line_txt = "".join(s.get("text", "") for s in line.get("spans", []))
                    for s in line.get("spans", []):
                        if s.get("text", "").strip():
                            sz = float(s.get("size", body_size))
                            if sz > max_span_size:
                                max_span_size = sz
                            if s.get("flags", 0) & 16:  # Bold flag
                                is_bold = True
                    if line_txt.strip():
                        block_lines.append(line_txt.strip())

                block_text = _clean_text(" ".join(block_lines))
                if not block_text:
                    continue

                is_heading = (
                    (max_span_size >= body_size * 1.15 or (is_bold and max_span_size >= body_size * 1.05))
                    and len(block_text.split()) <= 16
                    and len(block_text) >= 2
                )

                if is_heading:
                    flush_chunk()
                    current_headings = [block_text]
                    md_sections.append(f"\n## {block_text}\n")
                    continue

                words = len(block_text.split())
                if acc_words + words > max_words_per_chunk and acc_texts:
                    flush_chunk()

                acc_texts.append(block_text)
                acc_pages.add(page_no)
                if len(acc_bboxes) < 20:
                    acc_bboxes.append({
                        "page_number": page_no,
                        "left": round(float(bbox[0]), 2),
                        "top": round(float(bbox[1]), 2),
                        "right": round(float(bbox[2]), 2),
                        "bottom": round(float(bbox[3]), 2),
                        "coordinate_origin": "TOPLEFT",
                    })
                acc_words += words
                md_sections.append(block_text)

            # Disabled by default on low-memory containers: get_drawings()+rawdict
            # can allocate hundreds of MB on vector-heavy financial pages.
            try:
                tab_finder = None
                if extract_tables:
                    find_kwargs: dict[str, Any] = {"use_layout": False}
                    if low_mem:
                        # The text strategy avoids page.get_drawings() entirely.
                        find_kwargs["strategy"] = "text"
                    tab_finder = page.find_tables(**find_kwargs)
                for tbl in (tab_finder.tables if tab_finder else []):
                    rows = tbl.extract()
                    md_tbl = _format_table_markdown(rows)
                    if not md_tbl:
                        continue
                    tbbox = tbl.bbox
                    t_idx = len(tables) + 1
                    t_id = f"{doc_name}_table_{t_idx:03d}"
                    tables.append({
                        "table_id": t_id,
                        "document_id": doc_name,
                        "file_name": pdf_path.name,
                        "page_number": page_no,
                        "page_numbers": [page_no],
                        "bounding_boxes": [{
                            "page_number": page_no,
                            "left": round(float(tbbox[0]), 2),
                            "top": round(float(tbbox[1]), 2),
                            "right": round(float(tbbox[2]), 2),
                            "bottom": round(float(tbbox[3]), 2),
                            "coordinate_origin": "TOPLEFT",
                        }],
                        "caption": None,
                        "table_markdown": md_tbl,
                        "image_path": None,
                        "image_width": None,
                        "image_height": None,
                        "item_type": "TableItem",
                    })
                    md_sections.append(f"\n{md_tbl}\n")
                    # Also ensure table content is indexed cleanly in text chunks
                    chunks.append({
                        "chunk_id": f"{pdf_path.stem}_{len(chunks):04d}",
                        "document_id": doc_name,
                        "file_name": pdf_path.name,
                        "chunk_index": len(chunks),
                        "page_number": page_no,
                        "page_numbers": [page_no],
                        "headings": list(current_headings),
                        "captions": [],
                        "item_types": ["TableItem"],
                        "bounding_boxes": [],
                        "source_file": pdf_path.name,
                        "source_mimetype": "application/pdf",
                        "text": md_tbl,
                        "contextualized_text": (" > ".join(current_headings) + "\n\n" if current_headings else "") + md_tbl,
                    })
            except Exception as exc:
                print(f"[FastPDF] Table scan warning on page {page_no}: {exc}")

            try:
                seen_boxes: list[tuple[float, float, float, float]] = []
                for info in page.get_image_info(xrefs=True):
                    ib = info.get("bbox")
                    if not ib:
                        continue
                    w = float(ib[2]) - float(ib[0])
                    h = float(ib[3]) - float(ib[1])
                    # Filter out tiny icons, divider lines, or empty spans
                    if w < 60 or h < 60:
                        continue
                    # Skip full-page decorative backgrounds (>95% page width & height)
                    if w > page.rect.width * 0.95 and h > page.rect.height * 0.95:
                        continue
                    box_tuple = (round(float(ib[0]), 1), round(float(ib[1]), 1), round(float(ib[2]), 1), round(float(ib[3]), 1))
                    if box_tuple in seen_boxes:
                        continue
                    seen_boxes.append(box_tuple)

                    f_idx = len(figures) + 1
                    f_id = f"{doc_name}_fig_{f_idx:03d}"
                    figures.append({
                        "figure_id": f_id,
                        "document_id": doc_name,
                        "file_name": pdf_path.name,
                        "page_number": page_no,
                        "page_numbers": [page_no],
                        "bounding_boxes": [{
                            "page_number": page_no,
                            "left": box_tuple[0],
                            "top": box_tuple[1],
                            "right": box_tuple[2],
                            "bottom": box_tuple[3],
                            "coordinate_origin": "TOPLEFT",
                        }],
                        "caption": None,
                        "image_path": None,
                        "image_width": None,
                        "image_height": None,
                        "item_type": "PictureItem",
                    })
            except Exception as exc:
                print(f"[FastPDF] Figure scan warning on page {page_no}: {exc}")

            # Release per-page extraction structures and MuPDF decoded-object caches.
            del page_dict
            tab_finder = None
            if low_mem:
                release_mupdf_caches()
            elif page_no % 5 == 0:
                gc.collect()

        flush_chunk()

    # Handle scanned/image-only PDF edge case where no text chunks were found
    if not chunks:
        chunks.append({
            "chunk_id": f"{pdf_path.stem}_0000",
            "document_id": doc_name,
            "file_name": pdf_path.name,
            "chunk_index": 0,
            "page_number": 1,
            "page_numbers": [1],
            "headings": [pdf_path.stem.replace("_", " ").title()],
            "captions": [],
            "item_types": ["TextItem"],
            "bounding_boxes": [],
            "source_file": pdf_path.name,
            "source_mimetype": "application/pdf",
            "text": f"Visual document: {pdf_path.name} ({total_pages} pages). Query visual contents directly via ColPali vision retrieval.",
            "contextualized_text": f"Visual document: {pdf_path.name} ({total_pages} pages).",
        })

    _atomic_write_text(markdown_path, "\n\n".join(md_sections))
    _atomic_write_jsonl(chunks_path, chunks)
    _atomic_write_jsonl(figures_path, figures)
    _atomic_write_jsonl(tables_jsonl_path, tables)
    _atomic_write_text(
        document_json_path,
        json.dumps(
            {
                "parser": "pymupdf_lite",
                "doc_id": doc_name,
                "file_name": pdf_path.name,
                "pages": total_pages,
            }
        ),
    )

    log_memory("parse:end")
    print(
        f"[FastPDF] Complete: {len(chunks)} chunks, {len(figures)} figures, {len(tables)} tables "
        f"({total_pages} pages) saved."
    )
    return chunks

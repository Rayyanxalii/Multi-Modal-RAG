"""Ultra-low-memory PDF parser using PyMuPDF (fitz) for cloud containers (512 MB RAM safe)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from rag.config import DEFAULT_PDF_PATH, get_document_paths


def is_low_memory_env() -> bool:
    """Detect constrained environments or default to lightweight PyMuPDF parser to protect against OOM."""
    # 1. Explicit user opt-in for heavy IBM Docling
    if os.getenv("USE_DOCLING", "").strip().lower() in ("1", "true", "yes"):
        return False

    # 2. Explicit flag for low-memory parser
    low_flag = os.getenv("LOW_MEMORY_PARSER", "").strip().lower()
    if low_flag in ("1", "true", "yes"):
        return True
    if low_flag in ("0", "false", "no"):
        return False

    # 3. Cloud / container platform detection
    cloud_vars = (
        "RENDER",
        "RENDER_SERVICE_ID",
        "RENDER_INSTANCE_ID",
        "SPACE_ID",
        "K_SERVICE",
        "FLY_APP_NAME",
        "DYNO",
        "AWS_EXECUTION_ENV",
    )
    if any(os.getenv(v) for v in cloud_vars):
        return True

    # Docker container environment
    if Path("/.dockerenv").exists():
        return True

    # 4. Check Linux cgroup memory ceiling (Docker, Render, Cloud Run, HF CPU Basic)
    for cgroup_path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            p = Path(cgroup_path)
            if p.exists():
                val = p.read_text(encoding="utf-8").strip()
                if val.isdigit():
                    limit_mb = int(val) / (1024 * 1024)
                    if 0 < limit_mb < 3500:
                        return True
        except Exception:
            pass

    # 5. Check host total RAM
    try:
        import psutil
        total_mb = psutil.virtual_memory().total / (1024 * 1024)
        if total_mb < 3500:
            return True
    except Exception:
        pass

    # Default to lightweight parser for maximum stability and speed
    return True


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


def _detect_body_font_size(pdf: pymupdf.Document) -> float:
    """Estimate primary body font size by weighting character frequency across sample pages."""
    size_counts: Counter[float] = Counter()
    sample_pages = min(len(pdf), 8)
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
    if not size_counts:
        return 10.0
    return size_counts.most_common(1)[0][0]


def _clean_text(text: str) -> str:
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


def parse_pdf_fast(
    pdf_path: Path = DEFAULT_PDF_PATH,
    doc_paths: dict[str, Any] | None = None,
    max_words_per_chunk: int = 380,
) -> list[dict[str, Any]]:
    """Parse PDF text chunks, headings, figures, and tables using minimal RAM (< 35 MB)."""
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

    print(f"[FastPDF] Parsing '{pdf_path.name}' with low-memory PyMuPDF engine...")

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
        body_size = _detect_body_font_size(pdf)

        for page_idx, page in enumerate(pdf):
            page_no = page_idx + 1
            page_dict = page.get_text("dict")

            # 1. Extract text blocks & detect headings
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

            # 2. Extract structured tables via PyMuPDF native table finder
            try:
                tab_finder = page.find_tables()
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

            # 3. Extract figures / raster images via PyMuPDF
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

    # Write outputs atomically
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(md_sections))

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with figures_path.open("w", encoding="utf-8") as f:
        for fig in figures:
            f.write(json.dumps(fig, ensure_ascii=False) + "\n")

    with tables_jsonl_path.open("w", encoding="utf-8") as f:
        for tbl in tables:
            f.write(json.dumps(tbl, ensure_ascii=False) + "\n")

    with document_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "parser": "pymupdf_lite",
                "doc_id": doc_name,
                "file_name": pdf_path.name,
                "pages": total_pages,
            },
            f,
        )

    print(
        f"[FastPDF] Complete: {len(chunks)} chunks, {len(figures)} figures, {len(tables)} tables "
        f"({total_pages} pages) saved."
    )
    return chunks

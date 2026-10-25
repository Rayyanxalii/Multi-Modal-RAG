import json
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHUNKS_DIR = PROJECT_ROOT / "src" / "pdf" / "chunks"

INPUT_PATH = CHUNKS_DIR / "attention_figures.jsonl"
CLEANED_PATH = CHUNKS_DIR / "attention_figures_cleaned.jsonl"
REPORT_PATH = CHUNKS_DIR / "attention_figure_review.jsonl"

MIN_WIDTH = 25
MIN_HEIGHT = 25
MIN_AREA = 1500


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def figure_id(record):
    return record.get("figure_id", record.get("id", "unknown"))


def page_number(record):
    value = record.get("page_number")

    if value is None:
        pages = record.get("page_numbers", [])
        value = pages[0] if pages else None

    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def caption_text(record):
    value = record.get("caption")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def bbox(record):
    boxes = record.get("bounding_boxes") or []

    if not boxes or not isinstance(boxes[0], dict):
        return None

    box = boxes[0]

    try:
        return {
            "left": float(box["left"]),
            "right": float(box["right"]),
            "top": float(box["top"]),
            "bottom": float(box["bottom"]),
        }
    except (KeyError, ValueError, TypeError):
        return None


def dimensions(record):
    box = bbox(record)

    if box is None:
        return None

    width = abs(box["right"] - box["left"])
    height = abs(box["top"] - box["bottom"])

    return width, height, width * height


def suspicious_reasons(record):
    reasons = []
    size = dimensions(record)

    if size is None:
        return ["missing_or_invalid_bbox"]

    width, height, area = size

    if width < MIN_WIDTH:
        reasons.append("width_below_threshold")

    if height < MIN_HEIGHT:
        reasons.append("height_below_threshold")

    if area < MIN_AREA:
        reasons.append("area_below_threshold")

    if not record.get("image_path"):
        reasons.append("missing_image_path")

    return reasons


def has_left_right_caption(text):
    return bool(
        re.search(r"\bleft\b", text, re.I)
        and re.search(r"\bright\b", text, re.I)
    )


def parent_figure(text):
    match = re.search(r"\bFigure\s+(\d+[a-z]?)\b", text, re.I)
    return f"Figure {match.group(1)}" if match else None


def panel_caption(text, panel):
    pattern = (
        r"\(\s*" + re.escape(panel) + r"\s*\)\s*(.*?)"
        r"(?=\(\s*(?:left|right)\s*\)|$)"
    )

    match = re.search(pattern, text, re.I | re.S)

    if not match:
        return ""

    return re.sub(r"\s+", " ", match.group(1)).strip()


def horizontal_order(first, second):
    a, b = bbox(first), bbox(second)

    if a is None or b is None:
        return None

    center_a = (a["left"] + a["right"]) / 2
    center_b = (b["left"] + b["right"]) / 2

    if abs(center_a - center_b) < 1:
        return None

    return (first, second) if center_a < center_b else (second, first)


def main():
    original = read_jsonl(INPUT_PATH)

    # Work on independent copies. Never modify the raw input file.
    cleaned = [dict(record) for record in original]
    report = []
    by_page = defaultdict(list)

    for record in cleaned:
        page = page_number(record)
        if page is not None:
            by_page[page].append(record)

        reasons = suspicious_reasons(record)
        if reasons:
            report.append({
                "status": "review_suspicious_detection",
                "figure_id": figure_id(record),
                "page_number": page,
                "reasons": reasons,
                "image_path": record.get("image_path"),
            })

    # Associate shared captions using same-page geometry.
    # Do not require the panel boxes to be within a fixed distance.
    for page, records in by_page.items():
        for first, second in combinations(records, 2):
            first_caption = caption_text(first)
            second_caption = caption_text(second)

            caption = (
                first_caption
                if has_left_right_caption(first_caption)
                else second_caption
                if has_left_right_caption(second_caption)
                else ""
            )

            if not caption:
                continue

            ordered = horizontal_order(first, second)
            if ordered is None:
                continue

            left_record, right_record = ordered

            # Reject a pair if the caption's left/right descriptions
            # cannot be extracted reliably.
            left_text = panel_caption(caption, "left")
            right_text = panel_caption(caption, "right")

            if not left_text or not right_text:
                continue

            # A caption record can be associated with another image
            # only when its own record is one of the two detections.
            caption_record = (
                first if first_caption == caption else second
            )

            # Both detections must be on the caption's page.
            if page_number(caption_record) != page:
                continue

            assignments = [
                (left_record, "left", left_text),
                (right_record, "right", right_text),
            ]

            for target, panel, text in assignments:
                target["shared_caption"] = caption
                target["parent_figure"] = parent_figure(caption)
                target["panel"] = panel
                target["panel_caption"] = text

                # Fill missing captions, but preserve existing ones.
                if not caption_text(target):
                    target["caption"] = caption

            report.append({
                "status": "review_caption_association",
                "page_number": page,
                "parent_figure": parent_figure(caption),
                "figure_ids": [
                    figure_id(left_record),
                    figure_id(right_record),
                ],
                "shared_caption": caption,
                "panels": {
                    figure_id(left_record): {
                        "panel": "left",
                        "panel_caption": left_text,
                    },
                    figure_id(right_record): {
                        "panel": "right",
                        "panel_caption": right_text,
                    },
                },
                "action": "keep_images_separate_add_shared_caption",
            })

    write_jsonl(CLEANED_PATH, cleaned)
    write_jsonl(REPORT_PATH, report)

    print(f"Input records: {len(original)}")
    print(f"Output records: {len(cleaned)}")
    print(f"Review records: {len(report)}")
    print(f"Saved: {CLEANED_PATH}")
    print(f"Saved: {REPORT_PATH}")

    for item in report:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()

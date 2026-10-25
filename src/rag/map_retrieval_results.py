
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_CHUNKS_PATH = PROJECT_ROOT / "src/pdf/chunks/attention_chunks.jsonl"
FIGURES_PATH = PROJECT_ROOT / "src/pdf/chunks/attention_figures_cleaned.jsonl"
TABLES_PATH = PROJECT_ROOT / "src/pdf/chunks/attention_tables.jsonl"




def load_jsonl_index(path: Path, id_field: str) -> dict[str, dict]:
    """Load JSONL records into a dictionary indexed by their unique IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    index = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            chunk_id = record.get(id_field)

            if not chunk_id:
                raise ValueError(
                    f"Missing '{id_field}' in {path.name}, "
                    f"line {line_number}"
                )

            chunk_id = str(chunk_id)

            if chunk_id in index:
                raise ValueError(
                    f"Duplicate ID '{chunk_id}' in {path.name}"
                )

            index[chunk_id] = record

    return index







def map_results(
    results: list[dict[str, Any]],
    source_index: dict[str, dict[str, Any]],
    id_field: str,
    source_name: str,
) -> list[dict[str, Any]]:
    """
    Map retrieval results to complete source records.

    Args:
        results: Retrieved results from the orchestrator.
        source_index: Source records indexed by their unique identifier.
        id_field: Identifier field, e.g. 'chunk_id'.
        source_name: Name used in error messages.

    Returns:
        Complete source records with retrieval metadata attached.
    """
    mapped_results = []

    for rank, result in enumerate(results, start=1):
        record_id = result.get(id_field)

        # Also support identifiers stored in a Qdrant payload.
        if record_id is None:
            payload = result.get("payload") or {}
            record_id = payload.get(id_field)

        if record_id is None:
            raise ValueError(
                f"Missing '{id_field}' in {source_name} "
                f"retrieval result at rank {rank}."
            )

        record_id = str(record_id)
        source_record = source_index.get(record_id)

        if source_record is None:
            raise KeyError(
                f"Could not find {source_name} record "
                f"with {id_field}='{record_id}' in the source index."
            )

        # Preserve the full source record and attach retrieval details.
        mapped_results.append({
            **source_record,
            "_retrieval": {
                key: value
                for key, value in result.items()
                if key != "payload"
            },
        })

    return mapped_results




def get_result_id(result: dict, id_field: str) -> str | None:
    """Extract an ID from a retrieval result or its Qdrant payload."""
    value = result.get(id_field)

    if value is None:
        payload = result.get("payload") or {}
        value = payload.get(id_field)

    return str(value) if value is not None else None






def map_figure_table_results(
    results: list[dict[str, Any]],
    figure_index: dict[str, dict[str, Any]],
    table_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Map visual retrieval results to their complete source records.

    Figures are looked up using figure_id.
    Tables are looked up using table_id.
    """
    mapped_results = []

    for result in results:
        record_type = result.get("record_type")
        record_id = result.get("record_id")

        # Fall back to metadata if needed.
        metadata = result.get("metadata") or {}

        if record_type is None:
            record_type = metadata.get("record_type")

        if record_id is None:
            if record_type == "figure":
                record_id = metadata.get("figure_id")
            elif record_type == "table":
                record_id = metadata.get("table_id")

        if not record_type or not record_id:
            raise ValueError(
                f"Visual result is missing its type or ID: {result}"
            )

        record_type = str(record_type).lower()
        record_id = str(record_id)

        if record_type == "figure":
            source_index = figure_index
        elif record_type == "table":
            source_index = table_index
        else:
            raise ValueError(
                f"Unknown visual record type '{record_type}' "
                f"for record '{record_id}'."
            )

        source_record = source_index.get(record_id)

        if source_record is None:
            raise KeyError(
                f"Could not find {record_type} '{record_id}' "
                f"in its source JSONL index."
            )

        mapped_results.append({
            **source_record,
            "_retrieval": {
                key: value
                for key, value in result.items()
                if key != "metadata"
            },
        })

    return mapped_results






def map_orchestrator_results(
    orchestrator_result: dict[str, Any],
) -> dict[str, list[dict]]:
    """
    Map universal orchestrator output to complete text and
    figure/table chunk records.

    Expected input keys:
        text_results
        figure_table_results
    """
    text_results = orchestrator_result.get("text_results", [])
    figure_table_results = orchestrator_result.get(
        "figure_table_results", []
    )

    text_index = load_jsonl_index(
        TEXT_CHUNKS_PATH,
        id_field="chunk_id",
    )

    figure_index = load_jsonl_index(
        FIGURES_PATH,
        id_field="figure_id",
    )

    table_index = load_jsonl_index(
        TABLES_PATH,
        id_field="table_id",
    )

    text_chunks = map_results(
        text_results,
        text_index,
        id_field="chunk_id",
        source_name="text",
    )

    figure_table_chunks = map_figure_table_results(
        figure_table_results,
        figure_index,
        table_index,
    )

    return {
        "text_chunks": text_chunks,
        "figure_table_chunks": figure_table_chunks,
    }

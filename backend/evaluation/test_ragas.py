"""test_ragas.py


Evaluates RAG pipeline performance across all 8 official Ragas RAG metrics:
  1. Context Precision         (context_precision)
  2. Context Recall            (context_recall)
  3. Context Entities Recall   (context_entity_recall)
  4. Noise Sensitivity         (NoiseSensitivity)
  5. Response Relevancy        (answer_relevancy)
  6. Faithfulness              (faithfulness)
  7. Multimodal Faithfulness   (multimodal_faithness)
  8. Multimodal Relevance      (multimodal_relevance)

Target Document: data/raw/nestle_financial_report.pdf (or configured PDF)
Dataset: 69 curated benchmark samples from backend/evaluation/create_dataset.py


"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend and backend/src to sys.path
EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
BACKEND_SRC = BACKEND_DIR / "src"

for p in [str(PROJECT_ROOT), str(BACKEND_DIR), str(BACKEND_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

import ollama
from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

# Official Ragas evaluation imports
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# All 8 Ragas RAG metrics
from ragas.metrics import (
    context_precision,
    context_recall,
    context_entity_recall,
    NoiseSensitivity,
    answer_relevancy,
    faithfulness,
    multimodal_faithness,
    multimodal_relevance,
)

# Benchmark samples and RAG pipeline
from backend.evaluation.create_dataset import SAMPLES
from rag.graph import run_agentic_rag


# ---------------------------------------------------------------------------
# Default Constants & Paths
# ---------------------------------------------------------------------------
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "nestle_financial_report.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"
DEFAULT_EVAL_MODEL = "openai/gpt-oss-120b"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Independent Embeddings Adapter (SentenceTransformers)
# ---------------------------------------------------------------------------
class IndependentSentenceTransformerEmbeddings(Embeddings):
    """Independent embedding model (all-MiniLM-L6-v2) for unbiased Ragas evaluation."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.client = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [str(t) if str(t).strip() else " " for t in texts]
        embeddings = self.client.encode(clean_texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        clean_text = str(text) if str(text).strip() else " "
        embedding = self.client.encode(clean_text, convert_to_numpy=True)
        return embedding.tolist()


# ---------------------------------------------------------------------------
# Resilient LLM Adapter (Groq with Automatic Rate-Limit Fallback)
# ---------------------------------------------------------------------------
class ResilientChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that automatically falls back to gpt-oss-20b if rate limited."""
    fallback_model: str = "openai/gpt-oss-20b"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                orig = self.model_name
                try:
                    self.model_name = self.fallback_model
                    return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                finally:
                    self.model_name = orig
            raise


# ---------------------------------------------------------------------------
# Evaluator Setup (LLM + Embeddings + 8 Metrics)
# ---------------------------------------------------------------------------
def setup_evaluator(
    eval_model: str = DEFAULT_EVAL_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper, list[Any]]:
    """Configure evaluator LLM, embeddings, and instantiate all 8 Ragas metrics."""
    groq_api_key = os.getenv("GROQ_API")
    if not groq_api_key:
        raise ValueError("GROQ_API key is missing in environment or .env file.")

    # 1. Evaluator LLM (Groq with resilient fallback)
    eval_llm = LangchainLLMWrapper(
        ResilientChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            model=eval_model,
            temperature=0.0,
            max_retries=2,
            request_timeout=60,
        ),
        is_finished_parser=lambda _: True,
    )

    # 2. Independent Evaluator Embeddings (all-MiniLM-L6-v2, independent from nomic-embed-text)
    eval_emb = LangchainEmbeddingsWrapper(
        IndependentSentenceTransformerEmbeddings(model_name=embedding_model)
    )

    # 3. Instantiate NoiseSensitivity
    noise_sensitivity = NoiseSensitivity(mode="relevant")

    # 4. Consolidated list of all 8 evaluation metrics
    metrics = [
        context_precision,
        context_recall,
        context_entity_recall,
        noise_sensitivity,
        answer_relevancy,
        faithfulness,
        multimodal_faithness,
        multimodal_relevance,
    ]

    return eval_llm, eval_emb, metrics


# ---------------------------------------------------------------------------
# Context Extraction Helper
# ---------------------------------------------------------------------------
def extract_contexts_from_rag_result(result: dict[str, Any]) -> list[str]:
    """Extract grounded context passages (text, tables, figures) from RAG result."""
    contexts: list[str] = []

    for src in result.get("sources", []):
        s_type = src.get("type", "text")
        s_id = src.get("id", "Unknown")
        pages = ", ".join(str(p) for p in src.get("pages", []))

        if s_type == "text":
            txt = src.get("text", "")
            if txt:
                contexts.append(f"[Text Chunk {s_id} | Page: {pages}]\n{txt}")
        elif s_type == "table":
            cap = src.get("caption", "")
            t_md = src.get("table_markdown", "")
            block = f"[Table {s_id} | Page: {pages}]\nCaption: {cap}"
            if t_md:
                block += f"\nTable Data:\n{t_md}"
            contexts.append(block)
        elif s_type == "figure":
            cap = src.get("caption") or src.get("panel_caption") or ""
            if cap:
                contexts.append(f"[Figure {s_id} | Page: {pages}]\nCaption: {cap}")

    # Fallback or supplement with context_str blocks if contexts is empty or lean
    if result.get("context_str") and (not contexts or not any(len(c) > 80 for c in contexts)):
        raw_blocks = [b.strip() for b in result["context_str"].split("\n\n") if b.strip()]
        if raw_blocks:
            contexts = raw_blocks[:5]

    if not contexts:
        contexts = ["No relevant document context found."]

    return contexts


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------
def run_evaluation(
    pdf_path: str | Path = DEFAULT_PDF,
    limit: int | None = None,
    offset: int = 0,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tag: str = "nestle_eval_8metrics",
    eval_model: str = DEFAULT_EVAL_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_workers: int = 2,
) -> dict[str, Any]:
    """Execute complete Ragas test evaluation across all 8 metrics.

    Workflow:
      1. Slices evaluation test dataset (from SAMPLES).
      2. Executes Generalized Multimodal Agentic RAG for each question.
      3. Captures query, generated answer, retrieved contexts, and expected answer.
      4. Converts to Hugging Face Dataset format.
      5. Runs official ragas.evaluate() across all 8 metrics.
      6. Saves results to JSON & CSV and prints summary scorecard.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select samples
    selected_samples = SAMPLES[offset:]
    if limit is not None and limit > 0:
        selected_samples = selected_samples[:limit]

    total = len(selected_samples)
    print("\n" + "=" * 75)
    print(f"RAGAS EVALUATION TEST SUITE (8 METRICS)")
    print(f"Target Document      : {pdf_path.name}")
    print(f"Number of Samples    : {total}")
    print(f"Evaluator LLM        : {eval_model} (Groq)")
    print(f"Evaluator Embeddings : {embedding_model} (Local Ollama)")
    print("=" * 75 + "\n")

    # 2. Phase 1: Run RAG pipeline (with persistent caching)
    print("--- [Phase 1/2] Executing Multimodal Agentic RAG Pipeline ---")
    cache_file = PROJECT_ROOT / "data" / "experiments" / f".cache_rag_{tag}_off{offset}_lim{total}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    rag_records = []

    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) == total:
                rag_records = cached
                print(f"Loaded {len(rag_records)} cached RAG responses from {cache_file.name}")
        except Exception as e:
            print(f"Could not load cache: {e}. Re-running pipeline.")
            rag_records = []

    if not rag_records:
        start_time = time.time()
        for idx, (query, expected, complexity, q_type, modality, page) in enumerate(selected_samples, start=offset + 1):
            s_id = f"nestle_{idx:03d}"
            print(f"[{idx}/{offset + total}] ({q_type} | {modality}) {query[:75]}...")
            t0 = time.time()

            try:
                thread_id = f"eval_{s_id}_{int(time.time()*1000)}"
                result = run_agentic_rag(question=query, pdf_path=pdf_path, thread_id=thread_id)
                latency = time.time() - t0
                answer = result.get("answer", "").strip()
                contexts = extract_contexts_from_rag_result(result)

                rag_records.append({
                    "id": s_id,
                    "user_input": query,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": expected,
                    "latency_sec": round(latency, 2),
                    "metadata": {
                        "complexity": complexity,
                        "question_type": q_type,
                        "modality": modality,
                        "source_page": str(page) if page is not None else "",
                    },
                    "sources_count": len(result.get("sources", [])),
                    "route": result.get("route", "both"),
                })
                print(f"    -> Done in {latency:.2f}s | Sources: {len(result.get('sources', []))} | Route: {result.get('route')}")

            except Exception as exc:
                print(f"    -> ERROR running query: {exc}")
                rag_records.append({
                    "id": s_id,
                    "user_input": query,
                    "response": f"Error: {exc}",
                    "retrieved_contexts": ["Pipeline execution error."],
                    "reference": expected,
                    "latency_sec": 0.0,
                    "metadata": {
                        "complexity": complexity,
                        "question_type": q_type,
                        "modality": modality,
                        "source_page": str(page) if page is not None else "",
                    },
                    "sources_count": 0,
                    "route": "error",
                })

        # Cache completed RAG records
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(rag_records, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(rag_records)} RAG records to cache.")
        except Exception as e:
            print(f"Warning: could not write cache file: {e}")

        pipeline_duration = time.time() - start_time
        print(f"\nPhase 1 Completed: {len(rag_records)} samples generated in {pipeline_duration:.2f}s.\n")
    else:
        print(f"Phase 1 Skipped: using {len(rag_records)} cached sample responses.\n")

    # 3. Phase 2: Build HF Dataset & Run Ragas 8-Metric Evaluation
    print("--- [Phase 2/2] Running Ragas Evaluation across 8 Metrics ---")
    hf_dataset = HFDataset.from_dict({
        "user_input": [r["user_input"] for r in rag_records],
        "response": [r["response"] for r in rag_records],
        "retrieved_contexts": [r["retrieved_contexts"] for r in rag_records],
        "reference": [r["reference"] for r in rag_records],
    })

    eval_llm, eval_emb, metrics = setup_evaluator(
        eval_model=eval_model,
        embedding_model=embedding_model,
    )

    run_cfg = RunConfig(
        max_workers=max_workers if max_workers > 1 else 4,
        timeout=120,
        max_retries=2,
    )

    eval_results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
        run_config=run_cfg,
    )

    df_results = eval_results.to_pandas()
    try:
        raw_scores = dict(getattr(eval_results, "_repr_dict", {}))
    except Exception:
        raw_scores = {}

    import numpy as np

    # Normalize metric column lookups
    def get_column_value(df, row_idx: int, metric_name: str) -> float | None:
        if df is None:
            return None
        matched_col = None
        if metric_name in df.columns:
            matched_col = metric_name
        else:
            aliases = {
                "noise_sensitivity": ["noise_sensitivity(mode=relevant)", "noise_sensitivity"],
                "multimodal_faithness": ["faithful_rate", "multimodal_faithness"],
                "multimodal_relevance": ["relevance_rate", "multimodal_relevance"],
            }.get(metric_name, [metric_name])
            for alias in aliases:
                for col in df.columns:
                    if alias.lower() in col.lower():
                        matched_col = col
                        break
                if matched_col:
                    break

        if matched_col and matched_col in df.columns:
            val = df.at[row_idx, matched_col]
            try:
                f_val = float(val)
                return None if np.isnan(f_val) else round(f_val, 4)
            except (ValueError, TypeError):
                return None
        return None

    # Attach metric scores to each individual record
    metric_keys = [
        "context_precision",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
        "answer_relevancy",
        "faithfulness",
        "multimodal_faithness",
        "multimodal_relevance",
    ]

    for i, r in enumerate(rag_records):
        r["scores"] = {}
        for m in metric_keys:
            r["scores"][m] = get_column_value(df_results, i, m)

    # Build clean overall scorecard dictionary
    clean_overall_scores = {}
    for m in metric_keys:
        vals = [r["scores"][m] for r in rag_records if r["scores"].get(m) is not None]
        clean_overall_scores[m] = round(float(np.mean(vals)), 4) if vals else 0.0

    # 4. Save results directly to results.json (sample 4 to sample 10)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{tag}_{timestamp}"
    json_path = output_dir / f"{run_id}.json"
    csv_path = output_dir / f"{run_id}.csv"

    backend_results_json = EVAL_DIR / "results.json"

    export_data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document_name": pdf_path.name,
        "num_samples": total,
        "sample_range": f"Sample {offset + 1} to Sample {offset + total}",
        "overall_scores": clean_overall_scores,
        "raw_scores": {k: round(float(v), 4) for k, v in raw_scores.items()},
        "eval_config": {
            "eval_llm": eval_model,
            "eval_embeddings": embedding_model,
            "max_workers": max_workers,
        },
        "records": rag_records,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    with backend_results_json.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question", "expected_answer", "model_answer",
            *metric_keys,
            "complexity", "question_type", "modality", "source_page", "latency_sec",
        ])
        for r in rag_records:
            meta = r.get("metadata", {})
            sc = r.get("scores", {})
            writer.writerow([
                r["id"],
                r["user_input"],
                r["reference"],
                r["response"],
                *[sc.get(m) for m in metric_keys],
                meta.get("complexity", ""),
                meta.get("question_type", ""),
                meta.get("modality", ""),
                meta.get("source_page", ""),
                r.get("latency_sec", 0.0),
            ])

    # 5. Display Scorecard Summary
    print("\n" + "=" * 75)
    print("                      RAGAS 8-METRIC EVALUATION SCORECARD")
    print(f"               Evaluated Samples: {offset + 1} to {offset + total} (Total: {total})")
    print("=" * 75)
    for m in metric_keys:
        val = clean_overall_scores.get(m, 0.0)
        label = m.replace("_", " ").title()
        print(f"  * {label:<27}: {val:.4f}")
    print("-" * 75)
    print(f"Saved Results JSON: {backend_results_json}")
    print(f"Saved Run JSON    : {json_path}")
    print(f"Saved Run CSV     : {csv_path}")
    print("=" * 75 + "\n")

    return export_data


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ragas 8-Metric Evaluation Test.")
    parser.add_argument("--pdf", type=str, default=str(DEFAULT_PDF), help="Target PDF document path.")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate (e.g. 5 for smoke test).")
    parser.add_argument("--offset", type=int, default=0, help="Starting index in dataset.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for results.")
    parser.add_argument("--tag", type=str, default="nestle_eval_8metrics", help="Experiment identifier tag.")
    parser.add_argument("--eval-llm", type=str, default=DEFAULT_EVAL_MODEL, help="Evaluator LLM model on Groq.")
    parser.add_argument("--eval-embeddings", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Evaluator embedding model.")
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrency for Ragas evaluation.")

    args = parser.parse_args()

    run_evaluation(
        pdf_path=args.pdf,
        limit=args.limit,
        offset=args.offset,
        output_dir=args.output_dir,
        tag=args.tag,
        eval_model=args.eval_llm,
        embedding_model=args.eval_embeddings,
        max_workers=args.max_workers,
    )

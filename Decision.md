# Architectural & Engineering Decisions Record (ADR)
## Project: Multi-Modal Hybrid RAG System

This document captures every major architectural, technical, algorithmic, and engineering decision taken during the conception, design, implementation, and optimization of this Multi-Modal Hybrid Retrieval-Augmented Generation (RAG) system (demonstrated on the foundational paper [*Attention Is All You Need*](src/pdf/attention.pdf)).

---

## Table of Contents
1. [Core Philosophy & Target Architecture](#1-core-philosophy--target-architecture)
2. [Project Tooling & Environment Decisions](#2-project-tooling--environment-decisions)
3. [Document Ingestion, Layout Parsing & Provenance](#3-document-ingestion-layout-parsing--provenance)
4. [Text Retrieval Architecture (Tri-Modal Search & Fusion)](#4-text-retrieval-architecture-tri-modal-search--fusion)
5. [Dedicated Figure & Table Processing & Retrieval](#5-dedicated-figure--table-processing--retrieval)
6. [Retrieval Orchestration, Lean Logging & ID Hydration](#6-retrieval-orchestration-lean-logging--id-hydration)
7. [Grounded Answer Generation & Citation Synthesis](#7-grounded-answer-generation--citation-synthesis)
8. [Empirical Evaluation & Benchmark Design](#8-empirical-evaluation--benchmark-design)
9. [Summary Table of Architectural Decisions](#9-summary-table-of-architectural-decisions)
10. [ColPali Re-alignment & Visual Architecture Refactoring (Option A)](#10-colpali-re-alignment--visual-architecture-refactoring-option-a)
11. [Project Reorganization & Unified Ingestion Pipeline](#11-project-reorganization--unified-ingestion-pipeline)
12. [Generalization to Arbitrary Documents & LangGraph Agentic RAG](#12-generalization-to-arbitrary-documents--langgraph-agentic-rag)

---

## 1. Core Philosophy & Target Architecture

### Decision 1.1: Multi-Modal Hybrid Architecture Over Naive Text RAG
- **Context & Problem**: Standard RAG pipelines extract raw PDF text into flat strings, chunk by fixed token counts, embed into a single vector database, and retrieve top-$k$ text passages. In complex scientific literature, this fails catastrophically:
  1. Formulas and math notations are mangled or omitted.
  2. Tables lose structural row/column alignment.
  3. Architecture diagrams (e.g., Transformer encoder/decoder structures) and attention heatmaps are completely invisible to text embeddings.
  4. Exact keyword searches for unique variables or abbreviations (e.g., $d_k, d_v, \sqrt{d_k}$, BLEU) suffer from semantic vector drift.
- **Decision**: Architect a **tri-modal hybrid retrieval pipeline** combining:
  1. **Dense Semantic Search** (Ollama `nomic-embed-text` in Qdrant) for conceptual semantic matching.
  2. **Sparse Lexical Search** (Heading-aware Okapi BM25) for precision technical terms and section names.
  3. **Visual Late-Interaction Search** (ColPali v1.3 multi-vector page embeddings) for charts, layout, diagrams, and math formulas.
  4. **Reciprocal Rank Fusion (RRF)** to combine distinct rank distributions without arbitrary score normalization.
- **Outcome**: A robust system capable of answering questions that rely on text, formulas, diagrams, tables, or structural document hierarchy.

---

## 2. Project Tooling & Environment Decisions

### Decision 2.1: Package Management with `uv`
- **Context & Problem**: Traditional Python tooling (`pip`, `pip-tools`, `poetry`) struggles with dependency resolution speed and deterministic platform locking when mixing heavy machine-learning packages (`torch`, `sentence-transformers`, `docling`, `pymupdf`, `llama-index`).
- **Decision**: Adopt Astral's **`uv`** as the project manager, pinning dependencies in `pyproject.toml` and locking them with `uv.lock`.
- **Rationale**: Sub-second virtual environment creation, ultra-fast lockfile resolution, and seamless multi-dependency build orchestration.

### Decision 2.2: Modern Python Runtime (`>=3.14` / Modern Standard Library)
- **Decision**: Target modern Python using native type annotations (`list[str]`, `dict[str, Any]`, `str | None`) and modern standard library capabilities (`pathlib.Path`, `uuid.uuid5`, `dataclasses`).
- **Outcome**: Clean, robust, type-annotated code across all modules with zero external typing dependencies.

### Decision 2.3: Modular Monorepo Architecture (`src/` Layout)
- **Decision**: Structure the codebase into explicit, isolated sub-packages:
  - `src/rag/Text_Retrieval/`: Text indexing, BM25, dense search, ColPali chunk retriever, and hybrid fusion.
  - `src/rag/Figure_Table_Retrieval/`: Dedicated sub-pipelines for visual extraction, coordinate cropping, caption disambiguation, and figure/table hybrid search.
  - `src/rag/retreival_orchestrator.py`: High-level multi-branch retrieval orchestrator.
  - `src/rag/map_retrieval_results.py`: Hydration and LLM context extraction layer.
  - `src/rag/answer_generation.py`: End-to-end question-answering pipeline with citation grounding.
  - `src/services/llm_service.py`: LLM provider abstraction (Groq via LlamaIndex).
  - `src/evaluation/`: Benchmark harness and golden test sets.

---

## 3. Document Ingestion, Layout Parsing & Provenance

### Decision 3.1: IBM Docling Over Simple PDF Parsers (PyPDF, pdfplumber, pypdf)
- **Context & Problem**: Basic PDF parsers discard reading order, two-column layouts, table markdown, and document object hierarchies.
- **Decision**: Adopt **IBM Docling (`docling`)** as the primary document converter:
  - Parses PDF into an object-oriented structural document (`DoclingDocument`).
  - Serializes full structural JSON (`attention_document.json`) and clean hierarchical markdown (`attention_parsed.md`).
  - Distinguishes document item types (`DocItem`, `TableItem`, `PictureItem`).
- **Outcome**: High-fidelity layout extraction preserving document hierarchy and native table markdown.

### Decision 3.2: Structure-Aware Chunking via Docling `HybridChunker`
- **Context & Problem**: Fixed-token chunking cuts across sentences, paragraphs, and section boundaries, isolating technical terms from the section titles that define them.
- **Decision**: Use Docling's **`HybridChunker`**:
  - Respects document layout boundaries (paragraphs, section breaks, lists).
  - Computes `contextualized_text` via `chunker.contextualize(chunk)`, prepending parent headings and section breadcrumbs to each chunk.
- **Outcome**: Dense embeddings capture the full conceptual context of where a paragraph belongs (e.g., `3.2 Attention > 3.2.1 Scaled Dot-Product Attention`).

### Decision 3.3: Zero-LLM Metadata & Provenance Extraction
- **Context & Problem**: Many modern RAG pipelines use an LLM during ingestion to summarize chunks or extract metadata, which is expensive, slow, non-deterministic, and prone to hallucinations.
- **Decision**: Extract all metadata deterministically from Docling's native provenance objects (`item.prov`):
  - Exact page numbers (`page_numbers`).
  - Bounding box coordinates (`left`, `top`, `right`, `bottom`, `coord_origin`).
  - Document item types (`DocItem`, `PictureItem`, `TableItem`).
  - Hierarchical headings and captions.
- **Outcome**: Instant, free, 100% accurate structural metadata for every chunk without a single LLM API call.

### Decision 3.4: High-DPI Page Rendering via PyMuPDF (Dual-Rendering Pipeline)
- **Context & Problem**: Both visual late-interaction models (ColPali) and accurate sub-figure cropping require high-resolution raster images of the original PDF pages.
- **Decision**: Implement `doc_loader.py` using **PyMuPDF (`pymupdf`)** with a $2\times$ scale matrix (`pymupdf.Matrix(2, 2)`):
  - Renders crisp 300 DPI PNG images (`page_0001.png` - `page_0015.png`).
  - Saves images to `src/pdf/rendered_pages/` as shared assets for ColPali indexing and figure cropping.

---

## 4. Text Retrieval Architecture (Tri-Modal Search & Fusion)

### Decision 4.1: Dense Semantic Search via Local Ollama + Cloud Qdrant
- **Embedding Model**: Chose Ollama running **`nomic-embed-text`** (768-dimensional embeddings).
  - *Rationale*: Free local execution, privacy, no rate limits, large context window (8192 tokens), and high performance on retrieval benchmarks.
- **Vector Database**: **Qdrant** (`qdrant-client`).
  - *Rationale*: Production-ready vector database supporting rich JSON payload filtering, cosine similarity, payload indexing, and scalable cloud/local deployments.
- **Deterministic Point IDs**:
  - Created UUID5 identifiers: `uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_id}")`.
  - *Rationale*: Guarantees deterministic, idempotent indexing. Running the indexer multiple times updates existing points instead of creating duplicate vectors.
- **Index Target**: Vectorized `contextualized_text` rather than raw body text, ensuring embeddings reflect section context.

### Decision 4.2: Heading-Aware Lexical Search (Rank-BM25)
- **Context & Problem**: Dense vectors can miss exact alphanumeric queries (e.g., "$d_k$", "WMT 2014", "tensor2tensor", "LayerNorm"). Furthermore, queries asking about broad concepts like *"What is attention?"* should strongly match chunks under the heading `3.2 Attention`.
- **Decision**:
  - **Technical Tokenizer**: Built a regex tokenizer `[a-zA-Z0-9]+(?:[-_./][a-zA-Z0-9]+)*` preserving hyphens, underscores, slashes, and periods (keeping terms like `multi-head`, `Q/K/V`, `dk`, `O(1)` intact).
  - **Heading Prioritization**: Built dual BM25 indices (body text and headings). Implemented `search_headings` to retrieve chunks where query terms match section titles with non-zero BM25 score.

### Decision 4.3: Visual Late-Interaction Retrieval (ColPali v1.3)
- **Context & Problem**: Architecture diagrams (Figure 1), multi-head attention diagrams (Figure 2), and sentence dependency visualizations (Figures 3, 4, 5) contain critical conceptual information that no text OCR can fully represent.
- **Decision**: Integrated **ColPali v1.3** (`vidore/colpali-v1.3-hf`) via `sentence-transformers`:
  - Embeds full rendered page images into multi-vector representations.
  - Queries are encoded into multi-vectors; late interaction (MaxSim operator) scores visual patches against query tokens.
  - Precomputed embeddings saved to `src/data/colpali/attention_colpali.pt` (enabling instant local search or Colab pre-computation).
  - **Page-to-Chunk Mapping (`colpali_chunk_retriever.py`)**: Devised a ranking mapper that translates visual page-level rankings down to the specific text chunks situated on those pages, integrating visual relevance into the text retrieval pipeline.

### Decision 4.4: Reciprocal Rank Fusion (RRF) with $k = 60$
- **Context & Problem**: The system produces three separate score distributions:
  - Dense search: Cosine similarity scores $\in [0, 1]$.
  - BM25: Unbounded positive keyword matching scores $\in [0, \infty)$.
  - ColPali: Late-interaction multi-vector MaxSim scores.
  Normalizing and linearly weighting these scores is brittle and query-dependent.
- **Decision**: Combine ranked candidate lists using **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}, \text{colpali}\}} \frac{1}{k + r_m(d)} \quad (k = 60)$$
- **Rationale**: $k = 60$ (the standard Cormack et al. constant) balances top-ranked items while preventing single-retriever bias. Items retrieved by multiple modalities receive significant score boosts.

---

## 5. Dedicated Figure & Table Processing & Retrieval

### Decision 5.1: Decoupling Visual Assets from Text Chunks
- **Context & Problem**: Naively embedding tables and figures inside large text chunks dilutes their specificity. Users often ask questions requiring visual inspection of a figure or numerical lookup in a table (e.g., *"What is the complexity per layer for self-attention in Table 1?"*).
- **Decision**: Create a dedicated sub-package `src/rag/Figure_Table_Retrieval/` with its own pre-processing, cropping, and hybrid search pipeline.

### Decision 5.2: Coordinate Space Transformation for Image Cropping
- **Context & Problem**: Docling outputs bounding box coordinates in PDF point space with a bottom-left coordinate origin (`CoordOrigin.BOTTOMLEFT`). Image libraries (PIL) and rendered page images operate in top-left pixel space.
- **Decision**: In `images_to_figures.py` and `image_to_table.py`, implemented mathematical coordinate transformation:
  $$x_0 = \text{left} \times \text{scale}_x, \quad y_0 = (\text{page\_height} - \text{top}) \times \text{scale}_y$$
  $$x_1 = \text{right} \times \text{scale}_x, \quad y_1 = (\text{page\_height} - \text{bottom}) \times \text{scale}_y$$
  where $\text{scale}_x$ and $\text{scale}_y$ are the ratios of rendered image pixels to PDF point dimensions.
- **Outcome**: Clean, perfectly bounded cropped PNG images for all figures (`attention_fig_001.png` - `attention_fig_007.png`) and tables (`attention_table_001.png`).

### Decision 5.3: Disentangling Multi-Panel Figures & Shared Captions
- **Context & Problem**: In *Attention Is All You Need*, **Figure 2** consists of two distinct sub-figures side-by-side:
  - Left panel: Scaled Dot-Product Attention.
  - Right panel: Multi-Head Attention.
  Docling detects these as separate `PictureItem` bounding boxes, but only assigns one composite caption to one of them, leaving the other uncaptioned or misassigned.
- **Decision**: Developed `Caption_Merge_or_Incorrect.py`:
  1. **Sub-panel Detection**: Detects shared `"left"` / `"right"` patterns in captions using regex.
  2. **Geometric Ordering (`horizontal_order`)**: Compares horizontal center coordinates ($x$-centers) of the bounding boxes on the same page to assign the left box to `panel="left"` and the right box to `panel="right"`.
  3. **Disentangled Metadata**: Preserves individual cropped images while associating both with `parent_figure="Figure 2"`, `shared_caption`, and specific `panel_caption`.
  4. **Quality Filtering**: Flags suspicious bounding boxes (width/height $< 25$ px or area $< 1500$ px²).
  5. **Data Immutability & Audit Trail**: Keeps raw inputs untouched; writes cleaned records to `attention_figures_cleaned.jsonl` and detailed decisions to `attention_figure_review.jsonl`.

### Decision 5.4: Dedicated Figure/Table Hybrid Qdrant Collection
- **Decision**: Created a separate Qdrant collection `research_figures_tables`:
  - Indexes structured Markdown tables and figure captions using dense embeddings (`nomic-embed-text`) and BM25.
  - Searchable by `record_type` (`"figure"` or `"table"`).
  - Merged via figure/table RRF fusion (`rrf.py`).

---

## 6. Retrieval Orchestration, Lean Logging & ID Hydration

### Decision 6.1: Dual-Branch Independent Retrieval Orchestration
- **Context & Problem**: Mixing text chunks and figures/tables into a single top-$k$ pool causes text chunks to crowd out visual assets, or vice versa.
- **Decision**: In `retreival_orchestrator.py`, execute text hybrid retrieval (`text_top_k=10`) and figure/table hybrid retrieval (`figure_table_top_k=5`) as **two parallel, independent branches**.
- **Outcome**: Guarantees that the LLM receives the most relevant text context *and* the most relevant tables/diagrams.

### Decision 6.2: Elimination of Run Log Bloat (Lean Evaluation Logging)
- **Context & Problem**: Initially, `run_hybrid_retrieval` dumped the full retrieved payload into JSON files (`data/retrieval_runs/retrieval_<timestamp>.json`). Each run was ~35 KB to ~50 KB because it saved:
  - All bounding box coordinate dictionaries.
  - Raw image dimensions and internal point IDs.
  - Multiple duplicate text fields (`text`, `contextualized_text`, and typo duplicates like `cpntextualized_text`).
  Saving full text repeatedly duplicated static data across dozens of files.
- **Decision**: Implemented `format_eval_run` with `lean_save=True`:
  - Retains only: `question`, `created_at`, `retrieval_config`, `counts`, and clean ranked candidate lists with `rank`, `chunk_id`/`record_id`, `page`, scores, and source ranking metadata.
- **Outcome**: **Reduced run log size by 84% to 95%** (down to ~1–5 KB per run) while preserving 100% of the diagnostic data required for offline IR evaluation.

### Decision 6.3: The ID-Pointer Hydration Pattern (`map_retrieval_results.py`)
- **Context & Problem**: Downstream consumers (evaluators vs. LLM generators) have different data needs:
  - Evaluation requires only ranked IDs and scores.
  - The LLM requires clean text, table markdown, and page numbers.
- **Decision**: Implement the **Hydration Pattern**:
  - The orchestrator passes around lightweight IDs and rankings.
  - `map_orchestrator_results()` looks up full source records on demand from canonical index files:
    - `attention_chunks.jsonl` (for text).
    - `attention_figures_cleaned.jsonl` (for figures).
    - `attention_tables.jsonl` (for tables).

### Decision 6.4: LLM Context Sanitization (`format_llm_context`)
- **Context & Problem**: Passing full JSON payloads (with bounding boxes, OCR provenance, and image pixel dimensions) into an LLM context window wastes tokens, increases latency, and distracts the reasoning model.
- **Decision**: In `map_retrieval_results.py`, implemented `format_llm_context()` and `build_grounded_context_str()`:
  - Formats clean, minimal context items:
    - **Text**: `[Source N] (Text ID: attention_0011, Page: 3)\n{contextualized_text}`
    - **Tables**: `[Source N] (Table ID: attention_table_001, Page: 6)\n{table_markdown}`
    - **Figures**: `[Source N] (Figure ID: attention_fig_002, Page: 4)\n{caption}`
  - Completely strips bounding boxes and internal metadata before prompt assembly.

---

## 7. Grounded Answer Generation & Citation Synthesis

### Decision 7.1: Ultra-Fast LLM Inference via Groq (`openai/gpt-oss-20b`)
- **Context & Problem**: Complex technical RAG queries require multi-paragraph reasoning. Standard cloud LLM endpoints often have high time-to-first-token (TTFT) and high inference latency.
- **Decision**: Use **Groq** accelerated inference via `llama-index-llms-groq` (`openai/gpt-oss-20b`):
  - Sub-second generation speed.
  - High reasoning capability on technical architecture concepts.

### Decision 7.2: Anti-Hallucination & Explicit Citation Guardrails
- **Decision**: Enforce strict system prompt instructions:
  1. *Answer exclusively using the supplied document context.*
  2. *Cite factual claims immediately using `[Source N, Page P]`.*
  3. *Never invent citations or extrapolate beyond provided context.*
  4. *If context is insufficient, explicitly state what is missing.*
  5. *Acknowledge conflicting statements across sources if present.*
- **Outcome**: Verifiable, trustworthy answers with traceable provenance down to the specific page number in the original PDF.

---

## 8. Empirical Evaluation & Benchmark Design

### Decision 8.1: Human-Curated Golden Evaluation Benchmark
- **Context & Problem**: Evaluating retrieval quality using synthetic LLM-generated QA pairs often yields trivial keyword matches.
- **Decision**: Created `src/evaluation/golden_set.json` consisting of 23 human-curated questions targeting:
  - Core definitions (e.g., attention mapping function).
  - Exact mathematical formulas (Scaled Dot-Product formula, scaling factor motivation).
  - Architecture details (encoder layer structure, residual connections, layer norm).
  - Hyperparameter choices ($h = 8, d_k = d_v = 64$).
  - Multi-head attention application modes (encoder self-attention, masked decoder attention, encoder-decoder cross-attention).

### Decision 8.2: Comprehensive IR Evaluation Metrics at Top-$K$
- **Decision**: Built `src/evaluation/evaluate.py` to benchmark retrievers against standard Information Retrieval metrics:
  - **Precision@K**: Fraction of retrieved chunks that are relevant.
  - **Recall@K**: Fraction of relevant chunks retrieved in top-$K$.
  - **MRR@K (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant chunk.
  - **Hit Rate@K**: Proportion of queries with at least one relevant chunk in top-$K$.
- **Empirical Validation (at $K = 5$)**:
  - **Dense Search (Qdrant + nomic-embed-text)**: Precision@5: **0.3478**, Recall@5: **0.7616**, MRR@5: **0.8514**, Hit Rate@5: **91.30%**.
  - **Sparse Search (Heading BM25)**: Precision@5: 0.3217, Recall@5: 0.7036, MRR@5: 0.7101, Hit Rate@5: 82.61%.
  - **Hybrid RRF**: Precision@5: **0.3478**, Recall@5: 0.7558, MRR@5: 0.7971, Hit Rate@5: 86.96%.
- **Outcome**: Verified that dense retrieval provides the strongest individual baseline, while BM25 and visual search provide essential fallback for specific technical terms and multi-panel diagram queries.

---

## 9. Summary Table of Architectural Decisions

| Area | Decision Taken | Rejected Alternative | Key Rationale |
| :--- | :--- | :--- | :--- |
| **Package Manager** | `uv` | `pip`, `poetry`, `conda` | 10–100x faster dependency resolution, deterministic `uv.lock`. |
| **PDF Parsing** | IBM Docling (`docling`) | `pypdf`, `pdfplumber`, `PyPDF2` | Object-oriented layout hierarchy, native tables, and bounding box coordinates. |
| **Page Rendering** | PyMuPDF ($2\times$ scale) | `pdf2image`, `pypdfium2` | High-DPI 300+ DPI page images needed for visual late-interaction and cropping. |
| **Chunking** | Docling `HybridChunker` | Fixed token count (e.g., 512 tokens) | Preserves section headings and semantic boundaries via `contextualized_text`. |
| **Metadata Extraction** | Pure Docling Provenance | LLM metadata extraction | Instant, deterministic, zero API cost, exact bounding boxes. |
| **Dense Embeddings** | `nomic-embed-text` via Ollama | OpenAI / Cohere Cloud APIs | Free local execution, privacy, 8192 token context, zero rate limits. |
| **Vector DB** | Qdrant | FAISS, Chroma, Pinecone | Production-grade payload filtering, cosine distance, deterministic UUID5 upserts. |
| **Sparse Retrieval** | Heading-Aware Okapi BM25 | Pure unigram BM25 / TF-IDF | Prioritizes structural section headers for technical topics. |
| **Visual Retrieval** | ColPali v1.3 (`sentence-transformers`) | OCR text extraction on images | Captures diagrams, formulas, and visual layout via multi-vector late interaction. |
| **Rank Fusion** | Reciprocal Rank Fusion ($k = 60$) | Linear weighted score combination | Robust rank-based fusion; eliminates uncalibrated score scale mismatches. |
| **Figure/Table Crop** | Bottom-left to top-left coordinate mapping | Whole page embedding only | Produces standalone, inspectable cropped PNG assets for each figure and table. |
| **Multi-Panel Figures** | Geometric horizontal ordering & regex | Ignoring panel differences | Disentangles sub-figures (e.g., Figure 2 left/right) and assigns correct sub-captions. |
| **Figure/Table Index** | Dedicated Qdrant collection | Merging into main text collection | Prevents visual assets from being diluted by dense prose chunks. |
| **Orchestration** | Dual-branch independent retrieval | Single monolithic retrieval step | Guarantees top-$k$ text and top-$k$ visual assets simultaneously. |
| **Run Persistence** | Lean Evaluation JSON (~1–5 KB) | Dumping raw merged payloads (~35 KB) | Eliminates 85–95% file bloat while retaining all metrics for IR evaluation. |
| **Data Presentation** | ID-Pointer Hydration Pattern | Passing heavy payloads through pipeline | Clean separation of concerns between retrieval logging and LLM prompt context. |
| **LLM Context** | Minimal grounded text blocks | Raw JSON with bounding boxes | Drastically saves tokens, eliminates layout noise, prevents LLM distraction. |
| **LLM Inference** | Groq (`openai/gpt-oss-20b`) | Local 7B model / OpenAI standard | Ultra-low latency generation with strong technical reasoning. |
| **Citations** | Explicit `[Source N, Page P]` citations | Uncited answers / URL links | Verifiable provenance linking directly to PDF page numbers. |
| **Evaluation** | Human-curated 23-query Golden Set | LLM-generated synthetic queries | Realistic technical questions testing formulas, motivations, and architecture. |
| **ColPali Deployment** | ColPali strictly in Visual Retrieval (Option A) | ColPali in Text Retrieval / Caption Dense & BM25 | Native visual search for diagrams & tables; eliminates chunk granularity mismatch & redundant latency. |

---

## 10. ColPali Re-alignment & Visual Architecture Refactoring (Option A)

### Decision 10.1: Removing ColPali from Text Retrieval
- **Context & Problem**: Initially, ColPali was invoked inside `text_retrieval/hybrid_retriever.py` to retrieve full pages, which were then mapped to all text chunks located on those pages. This created three engineering issues:
  1. **Granularity Mismatch**: Giving every paragraph chunk on Page 3 the exact same visual score polluted sentence-level chunk ranking.
  2. **Latency Bottleneck**: Multi-vector Late Interaction (MaxSim) on CPU took ~1–2 seconds per text query, slowing down otherwise ultra-fast (<20 ms) Qdrant/BM25 text searches.
  3. **Redundancy**: Text chunks already have dedicated dense semantic vectors and BM25 lexical indexes that outperform whole-page vision models on prose text.
- **Decision**: Streamline `text_retrieval` to **pure Hybrid Sparse + Dense (Dense vector search + BM25 heading search fused via RRF)**. ColPali was completely removed from the text retrieval path.

### Decision 10.2: Dedicated Visual Retrieval via ColPali Page-to-Asset Mapping (Option A)
- **Context & Problem**: Previously, `visual_retrieval` attempted to search figures and tables using text embeddings and BM25 on captions. This was fundamentally flawed: figures with terse or uninformative captions (e.g., *"Figure 1: The Transformer - model architecture"*) were invisible to text searches querying internal diagram components (*"Feed Forward"*, *"Positional Encoding"*, *"LayerNorm"*).
- **Decision (Option A)**:
  1. Use **ColPali** as the sole visual retriever for figures and tables.
  2. ColPali runs query-to-page multi-vector MaxSim against precomputed rendered page embeddings (`data/embeddings/pages_embeddings.pt`).
  3. The top-ranked pages are mapped directly to the extracted figures (`attention_figures_cleaned.jsonl`) and tables (`attention_tables.jsonl`) residing on those pages.
  4. For tables, both the visual cropped image and the full structured `table_markdown` are returned to downstream LLM/VLM generation.
- **Outcome & Benefits**:
  - Eliminated the `figures_tables` Qdrant collection, removing duplicate indexing and embedding maintenance.
  - Figures and complex diagrams are now retrieved using actual visual image features rather than text captions alone.
  - Verified empirical test: A query for *"The Transformer model architecture"* ranks Figure 1 at **Rank 1 with a ColPali score of 15.2930**.

### Decision 10.3: Elimination of Secondary Qdrant Collection (Single-Collection Qdrant)
- **Context & Problem**: Previously, two Qdrant collections were maintained: `research_papers` (text chunks) and `research_figures_tables` (caption text & table markdown embeddings).
- **Decision**: Completely decommission the `research_figures_tables` Qdrant collection.
- **Rationale**: Visual retrieval is now handled exclusively by ColPali's multi-vector late-interaction over rendered page images, rendering text-based vector search over visual captions redundant. Qdrant is dedicated solely to storing and searching `text_chunks`.

---

## 11. Project Reorganization & Unified Ingestion Pipeline

### Decision 11.1: Separation of Data Assets and Codebase (`data/` & `notebooks/`)
- **Context & Problem**: Raw PDFs, rendered page images, cropped figures, cropped tables, and PyTorch embedding files were previously scattered inside `src/` (`src/pdf/`, `src/data/`, `src/notebook/`). This violated standard Python packaging practices and polluted source control.
- **Decision**: Restructure the project root into clean directories:
  - `data/raw/`: Original documents (e.g., `attention.pdf`).
  - `data/rendered_pages/`: High-resolution PNG page images.
  - `data/chunks/`: JSONL chunks, parsed Docling documents, and cleaned figure/table metadata.
  - `data/figures/`: Cropped figure PNG images.
  - `data/tables/`: Cropped table PNG images.
  - `data/embeddings/`: ColPali PyTorch multi-vector embeddings (`pages_embeddings.pt`).
  - `data/retrieval_runs/`: Trace logs and evaluation run outputs.
  - `notebooks/`: Exploratory Jupyter notebooks (`colpali_visual_retrieval.ipynb`).
  - `src/rag/`: Pure Python source code.

### Decision 11.2: Centralized Configuration (`src/rag/config.py`)
- **Context & Problem**: File paths, collection names, model identifiers, and hyperparameters were hardcoded across multiple files with conflicting relative path assumptions.
- **Decision**: Create `src/rag/config.py` as the single source of truth for all paths (`PROJECT_ROOT`, `DATA_DIR`), collection names (`TEXT_COLLECTION_NAME`), embedding models, and Qdrant environment credential loaders.

### Decision 11.3: Unified Ingestion Pipeline (`src/rag/ingest.py`)
- **Context & Problem**: Preparing a new document previously required manually executing 7 to 8 separate scripts across different subdirectories in strict sequential order. Running them out of order resulted in `FileNotFoundError`.
- **Decision**: Implement a unified command-line orchestrator in `src/rag/ingest.py`:
  - **Stage 1**: Page rendering via PyMuPDF (`data/rendered_pages/`).
  - **Stage 2**: Document hierarchy layout parsing & contextual chunking via Docling (`data/chunks/`).
  - **Stage 3**: Figure & table bounding box metadata extraction via Docling.
  - **Stage 4**: Visual element cropping & caption normalization (`data/figures/`, `data/tables/`).
  - **Stage 5**: Text chunk vector indexing into Qdrant (`research_papers`).
- **Outcome**: A single CLI command (`python -m rag.ingest --pdf <path>`) runs the entire multi-stage pipeline with modular skip flags (`--skip-rendering`, `--skip-parsing`, `--skip-cropping`, `--only-indexing`).

### Decision 11.4: Purging Dead Code & Module Naming Standardization
- **Context & Problem**: Legacy commented-out functions, dead scoring functions in BM25, and mixed-case directory names (`Text_Retrieval`, `Figure_Table_Retrieval`, `pre_process`, `query_figures_tables`, `images_to_figures.py`) reduced maintainability.
- **Decision**:
  - Removed all dead and commented-out code blocks (e.g. `format_llm_context`, `build_grounded_context_str`, `normalize_scores`, `search_heading_aware`).
  - Removed the empty `src/evaluation/` directory.
  - Standardized all directories and files into standard Python snake_case (`text_retrieval`, `visual_retrieval/preprocess`, `visual_retrieval/search`, `retrieval_orchestrator.py`).

---

## 12. Generalization to Arbitrary Documents & LangGraph Agentic RAG

### Decision 12.1: Document-Agnostic Generalization
- **Context & Problem**: The prototype initially hardcoded assumptions tailored solely to *Attention Is All You Need* (hardcoded `"attention"` prefixes in chunk IDs, figure IDs, table IDs, and paper-specific system prompts).
- **Decision**: Generalize the system to handle any arbitrary technical document containing text, formulas, diagrams, system architectures, charts, and tables:
  - Document IDs and filenames are dynamically derived from input paths (`pdf_path.stem`).
  - Generalized system prompts for grounded synthesis, query optimization, and document grading.
  - Dynamic path resolution via `get_document_paths()` in `src/rag/config.py`.

### Decision 12.2: LangGraph StateGraph Architecture
- **Context & Problem**: A purely static linear pipeline lacks adaptive capabilities: it executes every retriever regardless of query intent, cannot recover from vocabulary mismatches, and cannot self-correct.
- **Decision**: Implement a stateful agentic workflow using **LangGraph** (`src/rag/graph/`):
  - Typed `RAGState` maintaining query, route, text/visual candidates, grounded context, retry counter, and verification flags.
  - Compiled `StateGraph` with conditional edges enabling adaptive branching and corrective loops.

### Decision 12.3: Query Routing (Selective Multi-Modal Retrieval)
- **Context & Problem**: ColPali multi-vector Late-Interaction on CPU takes ~1–2 seconds per query. Running it for pure conceptual text questions (*"What is the motivation for self-attention?"*) is computationally wasteful.
- **Decision**: Implement `route_query_node` to classify query intent:
  - `"text"`: Routes exclusively to Dense + BM25 (<20 ms).
  - `"visual"`: Routes exclusively to ColPali page-to-asset mapping.
  - `"both"`: Executes parallel multimodal retrieval for comprehensive queries.

### Decision 12.4: Corrective RAG (CRAG) Without Web Search
- **Context & Problem**: Standard CRAG literature falls back to external web search (Google / Tavily) when internal documents are graded irrelevant. In technical research papers or proprietary enterprise PDFs, web search causes hallucinations, breaks page-level citation provenance, and risks data leakage.
- **Decision**: Implement an internal **Corrective RAG (CRAG)** loop:
  - `grade_context_node` verifies whether retrieved context contains relevant information.
  - If graded irrelevant and `retry_count < 1`, `rewrite_query_node` uses the LLM to reformulate conversational language into formal technical documentation vocabulary, looping back to retrieval.
  - Strict retry guardrail (`retry_count < 1`) prevents infinite loops while guaranteeing predictable response times.

### Decision 12.5: Graph Visualization Artifacts
- **Context & Problem**: Inspecting and debugging complex cyclic state graphs requires visual confirmation of nodes, conditional edges, and fallback paths.
- **Decision**: Implement [`src/rag/graph/visualize.py`](src/rag/graph/visualize.py) using LangGraph's native `draw_mermaid_png()`:
  - `graph.png` / `data/graph.png`: Native Mermaid PNG rendered via `graph_view.get_graph().draw_mermaid_png()`.
  - `data/graph.mmd`: Raw Mermaid syntax exported directly from LangGraph's compiled graph via `draw_mermaid()`.
  - CLI integration: Accessible via `python -m rag.main --visualize` or `python -m rag.graph.visualize`.

### Decision 12.6: Unified Ingestion Verification & Granular Multi-Modal State Graph
- **Context & Problem**: Previously, retrieval was bundled inside a single monolithic `retrieve_node`, and document ingestion was completely decoupled from the runtime state graph. This hid the inner mechanics of dense vector search, BM25 keyword matching, Reciprocal Rank Fusion (RRF), ColPali late-interaction, and visual asset hydration from visual graph inspections. Furthermore, querying a new document required manual out-of-band ingestion commands.
- **Decision**:
  1. **Granular Retrieval Nodes**: Decomposed retrieval into explicit, modular LangGraph nodes:
     - `dense_retriever_node`: Qdrant dense vector search using `nomic-embed-text`.
     - `bm25_retriever_node`: Okapi BM25 keyword matching over document chunk headings (with in-memory index caching).
     - `text_fusion_node`: Reciprocal Rank Fusion (RRF, $k=60$) combining dense and sparse rankings.
     - `colpali_retriever_node`: Vision-language late-interaction MaxSim search over high-res page embeddings.
     - `visual_hydration_node`: Automatic hydration mapping top-scoring pages to cropped figure PNGs and structured Markdown tables.
  2. **Automated Document Ingestion Verification**: Integrated document pre-flight checking and ingestion directly into the StateGraph:
     - `check_doc_status_node`: Performs a non-blocking (<2 ms) verification of rendered pages, JSONL chunks, cropped visuals, and Qdrant points.
     - **Fast Path**: If the document is already ingested, the graph immediately transitions to `route_query_node`, preserving sub-second response times.
     - **Ingestion Path**: If any artifact is missing or a new PDF is provided, the graph automatically triggers `render_pages_node` $\rightarrow$ `parse_document_node` $\rightarrow$ `extract_and_crop_node` $\rightarrow$ `index_vectors_node` before continuing to query execution.
  3. **Visual Representation**: The compiled graph (`graph.png`) now displays all 15 nodes, conditional branches, parallel retrieval tracks, and self-correction loops.

### Decision 12.7: Graceful Degradation & Single-Retriever RRF Tolerance
- **Context & Problem**: When executing Reciprocal Rank Fusion (RRF), if a user queries with a dense-only or BM25-only route, or if one retrieval engine returns 0 results or encounters a network/indexing timeout, standard RRF implementations can raise `TypeError` or drop all results.
- **Decision**: Harden both `reciprocal_rank_fusion` and `text_fusion_node`:
  - `dense_results = dense_results or []` and `heading_results = heading_results or []` ensure `None` or missing state keys never raise exceptions.
  - If only dense results exist, RRF ranks and returns the dense candidates with calculated reciprocal rank scores (`1 / (k + rank)`).
  - If only BM25 results exist, RRF ranks and returns the BM25 candidates accordingly.
  - If both return empty lists, `text_fusion_node` gracefully yields `{"text_results": []}` without crashing.
  - Exception fallback: In the unexpected event of a fusion calculation error, the node falls back directly to `dense_matches or bm25_matches or []`.

### Decision 12.8: LLM-Based Structured Intent Routing
- **Context & Problem**: Static keyword matching in query routing is brittle: it fails on implicit visual intent (e.g., *"What does the encoder look like?"* contains no visual keyword like "figure" or "diagram"), misclassifies nuanced conceptual questions, and struggles with broad hybrid queries.
- **Decision**: Upgrade `route_query_node` to use an LLM-based structured classifier:
  - Defined `QueryIntent` using Pydantic schema: `route` (`"text" | "visual" | "both"`), `confidence` (`float`), and `reasoning` (`str`).
  - The LLM assesses semantic intent, metaphors, and implicit references to select the optimal retrieval branch.
  - **Explainability & Transparency**: The routing decision and `routing_reasoning` are recorded in `RAGState` and displayed in CLI run headers.
  - **Complete Elimination of Keyword Matching**: Removed all brittle static keyword lists (`VISUAL_KEYWORDS`, `TEXT_KEYWORDS`). Routing is 100% LLM-driven via semantic intent. In the rare event of an API communication error, the router defaults safely to comprehensive `"both"` retrieval.






### Decision 12.9: Deprecation & Purge of Pre-LangGraph Legacy Modules
- **Context & Problem**: Early prototypes relied on a static, linear pipeline (`answer_generation.py` -> `retrieval_orchestrator.py` -> `map_retrieval_results.py` and `hybrid_retrieval.py`). After migrating to the LangGraph 15-node StateGraph, these files became dead code, creating confusion and polluting repository structure.
- **Decision**:
  - Moved `hybrid_retrieve()` into `src/rag/visual_retrieval/search/colpali_retriever.py` to preserve backwards compatibility.
  - Permanently deleted the 4 dead modules:
    1. `src/rag/answer_generation.py` (superseded by `generate_answer_node`)
    2. `src/rag/retrieval_orchestrator.py` (superseded by LangGraph retrieval nodes)
    3. `src/rag/map_retrieval_results.py` (superseded by native node hydration)
    4. `src/rag/visual_retrieval/search/hybrid_retrieval.py` (redundant pass-through wrapper)
  - Eliminates ~556 lines of dead code.

### Decision 12.10: Dedicated Multimodal Vision Model (`llm_vision`) for Answer Generation
- **Context & Problem**: In earlier iterations, answer generation used a purely textual LLM (`openai/gpt-oss-20b`), feeding figure captions and markdown tables as text. While table markdown captured numbers, diagrams, flowcharts, architectures, and visual graphics could not be inspected directly by the model. Furthermore, passing multiple high-resolution images together can quickly exceed API input token quotas (e.g. Groq on-demand 7,000 ITPM limit).
- **Decision**:
  1. **Separation of LLM Roles**:
     - Fast Text LLM (`llm`, `openai/gpt-oss-20b`): Handles intent routing (`route_query_node`), query rewriting (`rewrite_query_node`), and context relevance grading (`grade_context_node`).
     - Multimodal Vision LLM (`llm_vision`, `qwen/qwen3.8-27b`): Dedicated exclusively to answer generation in `generate_answer_node`.
  2. **Direct Unified Multimodal Message Construction**:
     - `generate_answer_node` bundles retrieved text chunks, structured table markdown, figure captions, and actual cropped PNG images (`ImageBlock`) into a single multimodal `ChatMessage(role=MessageRole.USER, blocks=...)` along with system instructions.
     - Zero unnecessary intermediate processing: images and text are dispatched directly to the vision model.
  3. **Robust Asset Resolution & Token Budget Control**:
     - Implemented `_resolve_image_path` to map legacy chunk paths to canonical visual asset locations (`data/figures/` and `data/tables/`).
     - Caps attached visual image blocks to the top 2 highest-scoring assets (`max_images = 2`) and text chunks to the top 4 candidates. Subsequent visual assets retain their full captions and markdown table grids in text, ensuring rich context while remaining well under the 7,000 ITPM limit.
     - Retains rigorous provenance and citation constraints (`[Source N, Page P]`).

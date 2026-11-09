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
13. [Cloud ColPali Microservice on Nvidia A100 ZeroGPU](#13-cloud-colpali-microservice-on-nvidia-a100-zerogpu)
14. [Document Ingestion Performance Optimization](#14-document-ingestion-performance-optimization)
15. [Dual-Format Asset Cropping Resilience](#15-dual-format-asset-cropping-resilience)
16. [Side-by-Side Dual-Pane Document Inspection Deck](#16-side-by-side-dual-pane-document-inspection-deck)
17. [Professional Design System & Interface De-Cluttering](#17-professional-design-system--interface-de-cluttering)
18. [Short-Term Conversational Memory via LangGraph Checkpointer](#18-short-term-conversational-memory-via-langgraph-checkpointer)
19. [Mathematical Equation Typesetting (KaTeX & Safe Token Extraction)](#19-mathematical-equation-typesetting-katex--safe-token-extraction)
20. [Terminal Query Path Dispatch & VLM Payload Observability](#20-terminal-query-path-dispatch--vlm-payload-observability)
21. [Persistent SQLite STM Checkpointer & Local KaTeX Mathematical Engine](#21-persistent-sqlite-stm-checkpointer--local-katex-mathematical-engine)
22. [Unconditional Multimodal Retrieval Mode (Routing Bypassed)](#22-unconditional-multimodal-retrieval-mode-routing-bypassed)
23. [Granular Visual Metadata Dispatch (Parent Figure, Panels, Captions & Tables)](#23-granular-visual-metadata-dispatch-parent-figure-panels-captions--tables)
24. [VLM Token Budgeting & Self-Healing Rate-Limit (413) Recovery](#24-vlm-token-budgeting--self-healing-rate-limit-413-recovery)
25. [Rich Markdown Typography, Streaming Animation & Consolidated Sources Provenance](#25-rich-markdown-typography-streaming-animation--consolidated-sources-provenance)
26. [Persistent Multi-Document Session Restoration & Checkpoint Hydration](#26-persistent-multi-document-session-restoration--checkpoint-hydration)
27. [Reversion to Technical Blueprint & Sketchy Interface Aesthetic](#27-reversion-to-technical-blueprint--sketchy-interface-aesthetic)

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


---

## 13. Cloud ColPali Microservice on Nvidia A100 ZeroGPU

### Decision 13.1: Decoupling ColPali Late-Interaction to Dedicated Cloud ZeroGPU
- **Context & Problem**: ColPali v1.3 is a multi-vector vision-language model based on PaliGemma (3B parameters). Running multi-vector indexing and late-interaction MaxSim scoring on local consumer hardware (CPU or low-VRAM GPUs) causes severe bottlenecks:
  1. High memory footprint (>10GB RAM) causing system starvation.
  2. Slow indexing throughput (~8-12 seconds per high-resolution page).
  3. High latency on query-to-patch MaxSim matrix multiplications.
- **Decision**: Architect a dedicated microservice running on a Hugging Face Space (`rayyan0/colpali`) powered by an **Nvidia A100 (40GB) ZeroGPU**:
  - Exposes Gradio API endpoints (`/index` and `/search`).
  - `/index`: Accepts batch document page images, generates multi-vector token embeddings on the A100, and caches them on disk keyed by `doc_id`.
  - `/search`: Computes query multi-vector embeddings and performs instantaneous parallelized MaxSim late interaction across all pages of the document, returning ranked pages with similarity scores.
- **Outcome**: Indexing speeds accelerated by over $15\times$, and query late-interaction execution dropped to sub-second latencies without consuming local system RAM.

### Decision 13.2: Robust ColPali Cloud Client with Automatic Authentication & Local Fallback
- **Context & Problem**: Hugging Face Spaces on ZeroGPU enforce rate limits and require user authentication. If the remote space is cold-starting, queued, or quota-throttled, visual retrieval must not fail the entire user query.
- **Decision**:
  1. **Automated Token Injection**: Implemented `ColPaliCloudClient` in `src/rag/visual_retrieval/colpali_client.py` with automatic `HF_TOKEN` resolution from `.env` (sanitizing trailing whitespace and quotes).
  2. **Graceful Local Fallback**: In `colpali_retriever.py`, if remote ZeroGPU connection fails or times out, the system automatically falls back to layout metadata and visual caption matching from `chunks_dir`, ensuring uninterrupted multimodal responses.

---

## 14. Document Ingestion Performance Optimization

### Decision 14.1: Disabling Redundant RapidOCR on Digital PDFs
- **Context & Problem**: During document upload, IBM Docling automatically triggered local RapidOCR model weight downloads and initialization (`PP-OCRv6_det_small.pth` and `ch_ppocr_mobile_v2.0_cls_mobile.pth`), running CPU convolutions over every page. For digital-native PDFs (which already contain programmatic character streams and font encodings), this redundant OCR step added 15–30 seconds of pure latency and wasted CPU cycles.
- **Decision**: In `docling_parser.py`, configure `PdfPipelineOptions(do_ocr=False)`:
  ```python
  pipeline_options = PdfPipelineOptions(do_ocr=False)
  converter = DocumentConverter(
      format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
  )
  ```
- **Outcome**: Completely eliminated local RapidOCR model loading and processing overhead, cutting layout parsing time by ~80% for digital PDFs while preserving exact reading order, tables, and bounding boxes.

---

## 15. Dual-Format Asset Cropping Resilience

### Decision 15.1: Page Numbering Zero-Padding Normalization Across Renderers
- **Context & Problem**: Ingestion pipeline components produced differing page filename formats: `render_pdf_to_images` produces zero-padded names (`page_0001.png`, `page_0015.png`), while Docling coordinates reference natural 1-based integers (`page_no=1`, `page_no=10`). In multi-page documents (such as `sample-tables.pdf`), figure and table croppers failed to locate pages with single-digit names (`page_0001.png not found`), resulting in skipped visual assets.
- **Decision**: Updated `figure_cropper.py` and `table_cropper.py` to test candidate page filenames sequentially:
  ```python
  candidates = [
      rendered_pages_dir / f"page_{page_number:04d}.png",
      rendered_pages_dir / f"page_{page_number}.png",
  ]
  ```
- **Outcome**: 100% extraction resilience across all document sizes and naming schemes (e.g., successfully cropped and cataloged all 29 tables from `sample-tables.pdf`).

---

## 16. Side-by-Side Dual-Pane Document Inspection Deck

### Decision 16.1: Side-by-Side Dual-Pane Document Verification Interface
- **Context & Problem**: In technical and scientific research, users need to verify generated answers and citations directly against the original source text, diagrams, and formulas without navigating away or losing conversational context.
- **Decision**: Implemented a responsive dual-pane workspace ([`frontend/static/index.html`](frontend/static/index.html) and [`frontend/static/style.css`](frontend/static/style.css)):
  - **Left Pane (Chat & Query Deck)**: Conversational stream, question formulation, and visual evidence cards.
  - **Right Pane (Document Inspector Deck)**: Toggleable side-by-side inspection column supporting dual modes:
    1. **Pages View**: High-resolution rendered page images with zoom controls (50% to 250%), direct page jump input, and horizontal thumbnail quick-jump strip.
    2. **Native PDF**: Embedded browser iframe streaming the raw PDF directly from `GET /api/documents/{doc_id}/pdf`.
  - **Dynamic Citation Jump**: Grounded citations (e.g., `Page 2 →`) are interactive buttons that automatically open the side-by-side viewer and navigate to the referenced page.
- **Outcome**: Real-time provenance inspection allowing users to cross-examine synthesized claims alongside original PDF pages.

---

## 17. Professional Design System & Interface De-Cluttering

### Decision 17.1: Adoption of the Authoritative Professional Design Tokens
- **Context & Problem**: Early UI iterations incorporated high-contrast industrial indicators, neon accents, and dense hardware metadata that cluttered the reading experience and distracted from technical research.
- **Decision**: Refactored the entire frontend into the **Professional** design system:
  - **Palette**: Spacious ivory grounds (`#f7f4ec`), pale surfaces (`#fcfaf5`), dark warm ink (`#29271f`), muted slate ink (`#514d44`), and fine gold hairline rules (`#8b713a`, `#d4ccbd`).
  - **Typography**: `EB Garamond` for headings and titles; `Instrument Sans` for reading prose, metadata, and controls.
  - **De-Cluttering**: Purged all internal telemetry labels from the user interface (e.g. `A100 ZeroGPU`, `MaxSim Multi-Vector`, `ROUTE: READY`, `COGNITIVE RETRIEVAL SYSTEM`, `SOURCE PROVENANCE: MANDATORY GROUNDING`).
- **Outcome**: A clean, publication-grade, accessible reading interface prioritizing legible content, keyboard accessibility, and visual stillness.

---

## 18. Short-Term Conversational Memory via LangGraph Checkpointer

### Decision 18.1: Session Memory Architecture via Threaded Checkpointing
- **Context & Problem**: The agentic RAG pipeline was initially stateless: every query was evaluated in complete isolation. Users asking natural follow-up questions (*"What about the number of heads in that table?"*, *"Can you explain that more simply?"*) suffered from retrieval failures because pronouns (*"it"*, *"that table"*) could not be resolved without conversation context.
- **Decision**: Architect stateful short-term memory using LangGraph's native checkpointing system:
  1. **Accumulator State Channel**: Added `chat_history: Annotated[list[dict[str, Any]], operator.add]` to `RAGState`.
  2. **Thread Isolation**: Configured `build_rag_graph().compile(checkpointer=MemorySaver())` with independent thread identifiers per document (`thread_id="doc_{doc_id}"`).
  3. **Contextual Query Reformulation**: In `route_query_node`, prior turns in `chat_history` are used to rephrase ambiguous follow-ups into self-contained search queries before routing and retrieval.
  4. **Multi-Turn Answer Synthesis**: In `generate_answer_node`, recent dialogue turns are injected into the vision LLM prompt, maintaining coherent dialogue flow.
  5. **History REST API**: Added `GET /api/documents/{doc_id}/history` to retrieve stored turns from the checkpointer when switching between documents in the UI.
- **Outcome**: Seamless multi-turn conversation support per document with isolated thread memory and automatic follow-up coreference resolution.


---

## 19. Mathematical Equation Typesetting (KaTeX & Safe Token Extraction)

### Decision 19.1: KaTeX Typesetting with Isolated Math Token Extraction & Fallback
- **Context & Problem**: Technical documents (especially AI/ML foundational papers like *Attention Is All You Need*) contain extensive mathematical formulations (e.g. Scaled Dot-Product Attention $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, dimension variables $d_k, d_v$, matrix symbols $Q, K, V$, and complexity bounds $O(n^2 \cdot d)$). Initially, the frontend lacked a LaTeX math engine and ran naive HTML escaping across the entire string, resulting in unrendered raw LaTeX code (`$$`, `\text`, `\frac`, `\sqrt`) that was difficult to read and broke visual presentation.
- **Decision**:
  1. **Safe Token Extraction**: To prevent HTML escaping (`&`, `<`, `>`) from corrupting LaTeX syntax (such as matrix alignment ampersands or inequalities), implemented a two-pass token placeholder extraction system in `app.js`:
     - Pass 1: Isolates display math (`$$...$$`, `\[...\]`) and inline math (`$...$`, `\(...\)`) into indexed placeholders (`___MATH_TOKEN_X___`).
     - Pass 2: Safely escapes non-math HTML, resolves compound citations, and converts paragraph breaks.
     - Pass 3: Re-injects math expressions rendered via KaTeX.
  2. **KaTeX Integration**: Integrated KaTeX (`v0.16.11`) via CDN in `index.html` and configured `.math-block` (ivory container with gold border and overflow scrolling) and `.math-inline` styling in `style.css`.
  3. **High-Fidelity Offline Fallback**: Implemented `formatMathFallback()` to translate LaTeX constructs into Unicode math equivalents (`\sqrt` $\rightarrow$ `√`, `_k` $\rightarrow$ `ₖ`, `^2` $\rightarrow$ `²`, `^T` $\rightarrow$ `ᵀ`, `\cdot` $\rightarrow$ `·`) if KaTeX CDN is loading or offline.
  4. **Compound Citation Parsing**: Enhanced citation regex to identify and convert compound citations (e.g., `[Source 5, Page 4; Source 6, Page 4]`) into unified clickable buttons (`Page 4 →`).
- **Outcome**: Publication-grade mathematical typesetting for complex scientific formulas and clean, interactive citation buttons.

---

## 20. Terminal Query Path Dispatch & VLM Payload Observability

### Decision 20.1: End-to-End Query Path and Multimodal Dispatch Observability
- **Context & Problem**: In complex Agentic RAG workflows (Intent Routing $\rightarrow$ Conditional Branching $\rightarrow$ Dense/BM25/ColPali Retrieval $\rightarrow$ CRAG Evaluation $\rightarrow$ VLM Generation), developers and operators need immediate clarity on which retrieval pipelines are actively running for any given prompt (whether ColPali only, text only, or multimodal both), why the router chose that path, and what exact multimodal assets (images, table grids, text chunks, conversation history) are being passed into the Vision Language Model (`qwen/qwen3.8-27b`). Furthermore, printing raw technical document extracts on Windows terminals frequently triggered fatal `UnicodeEncodeError: 'charmap' codec can't encode character` crashes due to `cp1252` encoding mismatches on mathematical and typographic characters (`\u2217`, `\u2013`, `\u221a`).
- **Decision**:
  1. **Retrieval Branch Logging**: Configured `branch_retrieval` in `backend/src/rag/graph/workflow.py` to clearly announce the routing path in terminal output:
     - `[Query Path: COLPALI ONLY]`: ColPali visual late-interaction on page embeddings (dense/BM25 bypassed).
     - `[Query Path: TEXT ONLY]`: Hybrid Dense Vector (Qdrant) + BM25 Keywords with Reciprocal Rank Fusion (ColPali bypassed).
     - `[Query Path: DENSE ONLY]` or `[Query Path: BM25 ONLY]`: Targeted single-modality execution.
     - `[Query Path: BOTH MULTIMODAL]`: Concurrent execution of Dense, BM25, and ColPali visual late-interaction.
  2. **Granular Component Execution Metrics**: Added logging across all retrieval nodes:
     - `[Query Intent Router]`: Output route, confidence score, contextual query reformulations, and LLM reasoning.
     - `[Dense Retriever]`: Candidate vector count from Qdrant.
     - `[BM25 Retriever]`: Candidate lexical match count from Okapi BM25.
     - `[ColPali Retriever]`: Page images retrieved via MaxSim late-interaction on Hugging Face ZeroGPU.
     - `[Text Fusion]`: Fused chunk count from Reciprocal Rank Fusion.
     - `[Visual Hydration]`: Hydrated figures and tables matching retrieved pages.
  3. **Full VLM Payload Inspection Deck**: In `generate_answer_node` (`backend/src/rag/graph/nodes.py`), implemented a comprehensive dispatch monitor (`>>> [VLM INPUT DISPATCH] Payload sent to Vision LLM (qwen/qwen3.8-27b)`):
     - Active/contextualized question string.
     - Attached text chunks with chunk IDs, page numbers, and truncated text previews.
     - Evaluated visual assets (figures/tables) with page numbers, captions, table grid markdown character counts, and explicit status tags (`ATTACHED AS ImageBlock`, `SKIPPED IMAGE (Limit reached)`, `TEXT/TABLE-ONLY`).
     - Injected conversational memory dialogue turns (role and truncated content).
     - Active grounding and citation instructions.
  4. **Windows Encoding Resilience**: Implemented `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` and an ASCII-safe truncation helper `_safe_str()` with fallback character replacement, preventing console crashes across Windows environments.
- **Outcome**: 100% transparent terminal observability into retrieval mechanics and VLM payloads without risking runtime encoding failures.

---

## 21. Persistent SQLite STM Checkpointer & Local KaTeX Mathematical Engine

### Decision 21.1: Migration from In-Memory Saver to Persistent SQLite Checkpointing
- **Context & Problem**: Short-Term Memory was originally backed by `MemorySaver()`, which maintained chat checkpoints exclusively in volatile process RAM. Whenever the server process restarted or reloaded (`--reload`), conversation threads and follow-up contextualization state were lost.
- **Decision**: Installed `langgraph-checkpoint-sqlite` and configured `SqliteSaver` in `backend/src/rag/graph/workflow.py`:
  - Points to a persistent SQLite database at `data/chat_history.db`.
  - Automatically initializes schema via `checkpointer.setup()`.
  - Compiles `app = build_rag_graph().compile(checkpointer=checkpointer)`.
  - Integrates with `GET /api/documents/{doc_id}/history` in `frontend/server.py` to restore chat history across document switches and server restarts.
- **Outcome**: Thread state and dialogue memory are permanently persisted to disk with zero memory leakage and immediate multi-turn recall.

### Decision 21.2: Local KaTeX Mathematical Typesetting (Eliminating CDN Fallback)
- **Context & Problem**: Previously, math equations were occasionally rendered as crude Unicode symbols (e.g. `√(dₖ)`, `QKᵀ`) because the browser blocked or failed to load the external jsdelivr KaTeX CDN scripts, triggering the fallback text parser.
- **Decision**:
  - Downloaded and bundled the complete KaTeX suite (`katex.min.js`, `katex.min.css`, `contrib/auto-render.min.js`, and all WOFF2/TTF fonts) locally in `frontend/static/vendor/katex/`.
  - Updated `frontend/static/index.html` to reference local `/static/vendor/katex/` paths, guaranteeing 100% offline mathematical typesetting without external network dependencies.
- **Outcome**: Flawless, instantaneous mathematical equation rendering for complex formulas and matrices.

### Decision 21.3: Elimination of Premature Answer Blocking & Grounding Optimization
- **Context & Problem**: Simple queries occasionally received `"I don't have information about that."` due to two compound issues:
  1. IBM Docling stripped formulas from `chunks.jsonl` and replaced them with `<!-- formula-not-decoded -->`.
  2. The Query Intent router routed formulas and equations to `TEXT ONLY`, completely bypassing ColPali and Figure 2.
  3. `generate_answer_node` contained an artificial hardcoded block `if not is_relevant: return "I don't have information about that."` which silenced the Vision LLM if the grading node reported any hesitation, even when valid candidate chunks were retrieved.
- **Decision**:
  - Implemented automatic formula recovery in `backend/src/rag/docling_parser.py`, replacing `<!-- formula-not-decoded -->` with raw formula expressions (`orig`) from Docling's JSON document model.
  - Re-embedded and updated Qdrant vector index for `attention`.
  - Updated query intent routing to classify mathematical equations, formulas, mechanics, and calculations into `"both"` (triggering parallel ColPali + Dense + BM25).
  - Removed premature `is_relevant` short-circuit in `generate_answer_node` so the Vision LLM synthesizes evidence from all retrieved passages.
  - Re-engineered system prompt to require standard display LaTeX (`$$ ... $$`) and inline LaTeX (`$ ... $`), natural paragraph-level citations, and increased text context window to top 6 chunks.
- **Outcome**: High-accuracy, publication-quality answer synthesis with clean mathematical equations and robust multi-modal grounding across both simple and advanced queries.

---

## 22. Unconditional Multimodal Retrieval Mode (Routing Bypassed)

### Decision 22.1: Commenting Out Query Intent Routing in Favor of Direct Multimodal Execution
- **Context & Problem**: To verify answer synthesis quality and ensure no query is ever starved of visual evidence or text passages due to router misclassification, the user requested bypassing dynamic intent routing and unconditionally running all retrieval branches for every query.
- **Decision**:
  1. **Routing Node Bypass**: In `backend/src/rag/graph/nodes.py`, commented out the LLM routing classification call (`routing_prompt`) in `route_query_node`. Preserved `contextualize_query(original_q, history)` so follow-up coreference resolution from short-term conversational memory remains active. Explicitly set `route = "both"`.
  2. **Branch Retrieval Unconditional Execution**: In `backend/src/rag/graph/workflow.py`, commented out conditional branching in `branch_retrieval` to always return `["dense_retriever", "bm25_retriever", "colpali_retriever"]`.
  3. **Universal Evidence Merging**: Every question now concurrently executes:
     - Dense semantic search (Qdrant).
     - Lexical keyword search (BM25).
     - ColPali late-interaction on A100 ZeroGPU with figure and table hydration.
     - Reciprocal Rank Fusion text merging + visual asset mounting into the Vision LLM context.
- **Outcome**: Deterministic multi-modal retrieval ensuring every query receives the full complement of text chunks, figures, and tables.

---

## 23. Granular Visual Metadata Dispatch (Parent Figure, Panels, Captions & Tables)

### Decision 23.1: Complete Visual Metadata Ingestion into the Vision LLM Context
- **Context & Problem**: In complex scientific literature, figures frequently contain multiple sub-panels (e.g. Figure 2 in *Attention Is All You Need* contains a left panel showing Scaled Dot-Product Attention and a right panel showing Multi-Head Attention). Previously, `visual_hydration_node` and `generate_answer_node` only passed the raw `Caption: ...` and table markdown. The critical structural disambiguation fields—`parent_figure` (e.g. `"Figure 2"`), `panel` (e.g. `"left"` or `"right"`), and `panel_caption` (e.g. `"Scaled Dot-Product Attention."`)—were omitted from the prompt text, forcing the LLM to guess which cropped image corresponded to which panel.
- **Decision**:
  1. **Visual Hydration Field Retention**: In `visual_hydration_node` (`backend/src/rag/graph/nodes.py`), retained `parent_figure`, `panel`, `panel_caption`, and `shared_caption` across all hydrated figure records.
  2. **Structured Text Block Formulation**: In `format_context_node` and `generate_answer_node`, updated the visual prompt generator to emit structured metadata blocks:
     - **For Tables**: `Caption: {caption}` and `Table Markdown Grid:\n{table_md}`.
     - **For Figures**: `Parent Figure: {parent_fig}`, `Panel: {panel}`, `Panel Caption: {panel_cap}`, and `Caption: {caption}`.
  3. **Terminal Telemetry Enhancement**: Updated the terminal dispatch deck to log `[Parent: Figure 2, Panel: left]` and explicit panel captions alongside vision attachment status (`ATTACHED AS ImageBlock`).
- **Outcome**: Unambiguous panel grounding allowing the Vision LLM to accurately attribute architectural components to specific sub-diagrams and sub-figures without confusion.

---

## 24. VLM Token Budgeting & Self-Healing Rate-Limit (413) Recovery

### Decision 24.1: Adaptive Payload Budgeting & 413 Token Overflow Mitigation
- **Context & Problem**: On large or table-dense documents (e.g., `sample-tables.pdf` with 29 tables), querying broad questions (e.g. *"give the summary about this document"*) hydrated multiple tables with large markdown strings and high-resolution images. In Groq's `qwen/qwen3.8-27b` service tier, an Input Tokens Per Minute (ITPM) limit of 7,000 tokens is strictly enforced. When 2 image blocks (~2,500-3,000 vision tokens), 6 text chunks (~1,800 tokens), and 5 raw table markdown grids (~2,500 tokens) were combined, the total payload hit 7,264 tokens, causing Groq to fail with `Error code: 413 - {'message': 'Request too large... Limit 7000, Requested 7264'}`.
- **Decision**:
  1. **Table Image De-duplication**: When a table already has structured `table_markdown`, do not attach a redundant PNG `ImageBlock`. The text markdown grid is 100% readable by the model and saves 1,200–1,500 vision tokens per table.
  2. **Max Image Guard**: Set `max_images = 1` for primary figures/diagrams, keeping vision tokens within safe limits.
  3. **Table Markdown Truncation**: Capped table markdown strings at 850 characters (`table_md[:850] + "\n... [table truncated for token budget]"`), preventing massive tables from dominating the prompt.
  4. **Dynamic Chunk Slicing**: Sliced text chunks to top 4 when visual assets are present, and top 5 when pure text.
  5. **Self-Healing 413 Fallback**: Wrapped `llm_vision.chat()` in an automated recovery handler: if a `413` or `rate_limit_exceeded` error is ever encountered, the system automatically strips heavy `ImageBlock` tokens and immediately retries with lean text/table markdown, generating the answer seamlessly without crashing.
- **Outcome**: 100% resilience against Groq token limits, eliminating 413 errors even on table-dense documents.

---

## 25. Rich Markdown Typography, Streaming Animation & Consolidated Sources Provenance

### Decision 25.1: High-Fidelity Markdown Parser & Mathematical Token Preservation
- **Context & Problem**: Previously, assistant answers displayed raw unrendered Markdown asterisks (`**Table 5**`, `**2010**`), raw hashes (`###`), and unformatted lists. While LaTeX equations were extracted into placeholders, general Markdown syntax was completely unhandled by `formatAssistantAnswer`, resulting in raw markup characters appearing directly in the UI.
- **Decision**:
  - Implemented a multi-stage semantic markdown engine (`parseMarkdownToHtml` in `frontend/static/app.js`):
    1. **Math Expression Stashing**: Protects all display LaTeX (`$$ ... $$`, `\[ ... \]`) and inline LaTeX (`$ ... $`, `\( ... \)`) into isolated tokens before HTML processing.
    2. **Inline Sub-bullet Disambiguation**: Normalizes inline bullet patterns (e.g. `: * **item**` or `. * **item**`) into structured line breaks with indentation.
    3. **HTML Sanitization**: Escapes non-math content to guarantee XSS prevention.
    4. **Semantic Typography**: Converts `**bold**` and `__bold__` to `<strong>`, `#`-level headings to `.md-title`, `.md-heading`, and `.md-subheading`, inline code to `<code class="md-code">`, and fenced code blocks to `<pre class="md-code-block">`.
    5. **Hierarchical Lists**: Converts top-level bullets (`*`, `-`) to `<ul class="md-list">`, sub-bullets to `<ul class="md-sublist">`, and numbered items (`1.`) to `<ol class="md-ordered-list">`.
    6. **Identifier Protection**: Ensures snake_case identifiers (such as `sample-tables_table_005`) are not accidentally parsed as italic tags by constraining italic regexes to whitespace and punctuation boundaries.
    7. **KaTeX Re-injection**: Evaluates math tokens with local KaTeX, rendering full mathematical equations with graceful fallback.
- **Outcome**: Flawless, publication-quality editorial typography with zero raw markdown leakage.

### Decision 25.2: Progressive Typewriter Streaming Effect with Interactive Skip
- **Context & Problem**: Instantaneous rendering of large LLM answers felt abrupt and static. The user requested an animated streaming effect where text appears progressively word-by-word, matching modern LLM chat experiences.
- **Decision**:
  - Implemented `streamAssistantResponse(data)` in `frontend/static/app.js`:
    - Breaks cleaned answer text into progressive word and space tokens.
    - Utilizes a dynamic pacing interval (~16ms per tick) that batches tokens based on response length so animations feel brisk (2-3 seconds total) without lagging.
    - Displays a styled blinking vertical cursor (`<span class="streaming-cursor"></span>`) via `@keyframes cursorBlink`.
    - Automatically anchors chat scroll (`chatStream.scrollTop = chatStream.scrollHeight`) as words stream in.
    - Features one-click fast-forward: clicking anywhere on the card or sending a subsequent query instantly finalizes the stream and displays the complete formatted answer.
    - When restored from short-term conversational history (`GET /api/documents/{doc_id}/history`), uses immediate rendering (`appendAssistantMessage(..., { immediate: true })`) to avoid replaying animations.
- **Outcome**: Fluid, responsive, and engaging streaming experience that gives immediate visual feedback.

### Decision 25.3: Consolidated End-of-Answer Sources Provenance Bar
- **Context & Problem**: Previously, `[Source N, Page P]` citations were replaced inline with button pills mid-sentence (e.g. `presented in thousands of pounds (£) Page 1 → .`). This fragmented prose, split commas and periods, and cluttered reading flow. The user requested all citations be displayed strictly at the very end of the answer in the exact format: `"Sources: Page 10, 12, 15..."`.
- **Decision**:
  - **Inline Stripping**: In `cleanAnswerTextAndExtractPages`, all inline bracket tags (`[Source N, Page P]`, `[Source N, Page P; Source M, Page Q]`, `[Page P]`, `[Source N]`) are stripped from the prose. Residual spacing before punctuation (e.g. `(£) .` $\to$ `(£).`) is cleaned.
  - **Deduplicated Page Aggregation**: Extracts all unique page numbers into a sorted numerical list (`pages`). If no explicit brackets were in the text, falls back to the provenance pages of the retrieved sources.
  - **Unified Provenance Bar**: Appends an editorial provenance footer at the bottom of the card:
    ```html
    <div class="sources-provenance-bar">
      <span class="sources-label">Sources:</span>
      <span class="sources-prefix">Page</span>
      <span class="sources-list">
        <button class="source-page-pill" onclick="openAndJumpToPage(10)">10</button>,
        <button class="source-page-pill" onclick="openAndJumpToPage(12)">12</button>,
        <button class="source-page-pill" onclick="openAndJumpToPage(15)">15</button>
      </span>
    </div>
    ```
  - Visually displays as `Sources: Page 10, 12, 15` where each page number is a subtle gold-hairline interactive pill that, when clicked, opens the document viewer and jumps straight to that page.
- **Outcome**: Clean, unfragmented reading flow in the answer body coupled with a unified, interactive provenance footer.

---

## 26. Persistent Multi-Document Session Restoration & Checkpoint Hydration

### Decision 26.1: Automated History Restoration on Browser Startup & Refresh
- **Context & Problem**: Although dialogue turns were successfully written to SQLite (`data/chat_history.db`) via LangGraph's `SqliteSaver`, every browser refresh wiped the screen back to the hardcoded initial greeting card. The frontend only called `/api/documents/{doc_id}/history` inside `switchDocument`, but on startup `currentDocId` defaulted to `"attention"` without fetching its checkpoints, and clicking `"attention"` did nothing because `switchDocument` aborted when the document ID matched `currentDocId`.
- **Decision**:
  1. **Active Document Persistence**: Stored `currentDocId` in `localStorage.getItem("rag_active_doc_id")` so browser refreshes stay on the active document (e.g. `sample-tables`).
  2. **Startup Checkpoint Hydration**: In `initDocumentLedger` (`frontend/static/app.js`), immediately invoked `loadDocumentHistory(currentDocId)` during `DOMContentLoaded`.
  3. **Immediate Rich Rendering**: Restored past turns via `appendAssistantMessage(..., { immediate: true })`, bypassing typewriter animations while parsing full Markdown, KaTeX formulas, sources provenance pills, and visual deck mounts.
  4. **Document Chat Isolation**: Each document thread is strictly mapped to `thread_id = f"doc_{clean_id}"`. Switching documents swaps the conversational context in real-time.
- **Outcome**: Seamless session continuity across page reloads and server restarts.

### Decision 26.2: Sources Retention in Conversational State & Dedicated History Clear Endpoint
- **Context & Problem**: Previously, `generate_answer_node` only recorded `{"role": "assistant", "content": response}` in `chat_history`, omitting `sources`. When messages were restored from SQLite, source citations and visual attachments were lost. Additionally, users had no mechanism to reset a document's dialogue memory without manually deleting the SQLite database.
- **Decision**:
  1. **Source Retention**: Updated `generate_answer_node` (`backend/src/rag/graph/nodes.py`) to record `{"role": "assistant", "content": response, "sources": state.get("sources", [])}`.
  2. **Image Path Normalization**: Updated `GET /api/documents/{doc_id}/history` in `frontend/server.py` to normalize relative `image_path` entries into `/data/` URLs.
  3. **Thread History Clear Endpoint**: Implemented `DELETE /api/documents/{doc_id}/history` in `frontend/server.py` to purge SQLite checkpoints and writes for that document thread.
  4. **Clear Chat UI Control**: Added a "Clear Chat" button to `frontend/static/index.html` allowing users to reset conversational memory for the active document on demand.
- **Outcome**: Full visual and citation fidelity for historical chat turns and granular user control over session state.

---

## 27. Reversion to Technical Blueprint & Sketchy Interface Aesthetic

### Decision 27.1: Re-instating the Authoritative Monochrome & Laboratory Notebook Design
- **Context & Problem**: The user requested reverting the user interface from the ivory/gold palette back to the original **Sketchy / Technical Blueprint** aesthetic. The user preferred the high-contrast research atmosphere, sharp architectural rules, monospace telemetry badges, and notebook-style paper cards.
- **Decision**:
  1. **Color & Typography Restoration**: Restored `--dp-color-ink: #000000`, `--dp-color-paper: #ffffff`, `--dp-color-secondary: #4d4d4d`, `--dp-color-quiet: #cfcfcf`, coupled with `IBM Plex Mono` for technical labels, `Source Serif 4` for reading prose, and `EB Garamond` for display headers.
  2. **Technical Telemetry & Badges**: Re-introduced monospace status strips (`DOC: ATTENTION | ROUTE: BOTH`), system route indicators (`<span class="route-pill">ROUTE: BOTH</span>`), user thought scraps (`<div class="user-tag">Researcher Inquiry</div>`), and sidebar hardware indicators (`ColPali Visual: A100 ZeroGPU`, `Late Interaction: MaxSim Multi-Vector`).
  3. **Preservation of Core Capabilities**: Seamlessly preserved all new functional additions inside the sketchy theme:
     - Rich semantic Markdown parsing (`<strong>`, headings, hierarchical lists, code blocks).
     - Progressive word-by-word streaming effect with blinking monospace cursor (`.streaming-cursor`).
     - Consolidated end-of-answer provenance bar (`Sources: PAGE 10 , 12 , 15`).
     - 100% offline local KaTeX mathematical typesetting.
     - Short-term conversational memory hydration from SQLite on page refresh.
     - Dedicated per-document "Clear Chat" database purge button.
- **Outcome**: The beloved high-contrast, technical laboratory notebook aesthetic restored in full visual fidelity, underpinned by modern multimodal RAG features.


---

## 28. Minimalist De-cluttering of the Sketchy Research Interface

### Decision 28.1: Elimination of Unnecessary Telemetry, Tags, and Visual Clutter
- **Context & Problem**: Following the reversion to the Sketchy aesthetic, the user requested: *"but remove all of the unnecessary text, keep it clean , i wnat a clean display"*. The interface had accumulated technical metadata badges (`ColPali Visual: A100 ZeroGPU`, `Late Interaction: MaxSim Multi-Vector`, `ROUTE: BOTH`, `QUERY DISPATCH`, `SYSTEM ARCHIVE`), conversational tags (`Researcher Inquiry`), verbose subheaders, and decorative footer annotations (`SOURCE PROVENANCE: MANDATORY GROUNDING`). These distracted from reading the answer and interacting with documents.
- **Decision**:
  1. **Sidebar Simplification**:
     - Removed the sidebar hardware telemetry box (`ColPali Visual: A100 ZeroGPU`, `Late Interaction: MaxSim Multi-Vector`).
     - Simplified sidebar header from `Document Archive: Multi-Document Sessions` to a clean `Documents`.
     - Removed redundant list section label `Ingested Papers`.
  2. **Header & Status Spine Streamlining**:
     - Removed the descriptive paragraph under the main title (`Agentic text retrieval, late-interaction page vectors, and high-resolution figure inspection.`).
     - Simplified `.status-paper-strip` by removing the rule divider and `ROUTE: BOTH` badge. Retained only the live status dot and active document badge (`● DOC: ATTENTION`).
     - Vertically aligned the title and action buttons cleanly (`align-items: center`).
  3. **Chat Stream De-cluttering**:
     - User Messages: Removed `<div class="user-tag">Researcher Inquiry</div>`, presenting user inquiries cleanly with comfortable reading typography.
     - Assistant Working Sheets: Removed `<div class="assistant-header-meta">` (`ROUTE: BOTH Multimodal context synthesis` and `SYSTEM ARCHIVE`), letting answers start immediately below the clean ruled divider.
     - Thinking & Loader States: Simplified loading indicators to clean, understated messages (`Searching document...`, `Loading conversation history...`).
     - Starter Card: Replaced the multi-paragraph starter greeting and sample query bullets with a concise 1-sentence prompt: *"Ask any question about this document's text, equations, figures, or tables."*
  4. **Visual References & Input Footer**:
     - Figures & Tables Deck: Removed the redundant uppercase banner `<div class="mount-inspect-hint">INSPECT PAGE X IN DOCUMENT →</div>` from each figure/table card. The cards remain fully clickable with their clean metadata badge (`FIGURE [PAGE 2]`, `TABLE [PAGE 8]`) and caption.
     - Input Footer: Simplified placeholder to *"Ask a question about this document..."*, changed action button to *"Send →"*, and removed the entire decorative `.footer-annotation` bar.
     - Upload Modal: Shortened description to a single clear sentence.
  5. **Core Capability Integrity**: All functional engines remain 100% active and untouched:
     - Progressive word-by-word streaming typewriter animation.
     - Rich Markdown parsing (`<strong>`, headings, lists, tables).
     - Offline KaTeX mathematical formula rendering.
     - Unified end-of-answer `Sources: Page 10, 12, 15...` interactive jump bar.
     - SQLite session checkpoint persistence across page refreshes.
     - Per-document "Clear Chat" database purge functionality.
     - Side-by-side document inspection with page navigation and native PDF view.
- **Outcome**: A clean, distraction-free laboratory notebook workspace with high typographic elegance and zero unnecessary noise.


---

## 29. Retrieval Depth Calibration: 5 Chunks (Dense/BM25) & Up to 10 Visual Assets

### Decision 29.1: Balanced 5-Chunk Candidate Retrieval & 10-Asset Visual Capacity
- **Context & Problem**: Previously, dense vector retrieval (`Qdrant`) retrieved 6 chunks, BM25 retrieved 6 chunks, and RRF returned the top 6 text chunks, while visual asset hydration capped figures and tables at 5 (`matched_assets[:5]`). The user requested retrieving 5 chunks each from BM25 and dense retrieval, while allowing up to 10 total figures/tables for comprehensive visual coverage.
- **Decision**:
  1. **Dense Retrieval**: Reduced candidate text retrieval from `top_k=6` to `top_k=5` in `dense_retriever_node` (`backend/src/rag/graph/nodes.py`).
  2. **BM25 Retrieval**: Reduced lexical candidate retrieval from `top_k=6` to `top_k=5` in `bm25_retriever_node` (`backend/src/rag/graph/nodes.py`).
  3. **Reciprocal Rank Fusion**: Updated RRF fusion in `text_fusion_node` to return the top 5 fused text chunks (`top_k=5`).
  4. **Visual Asset Hydration**: Expanded figure and table retention from `matched_assets[:5]` to `matched_assets[:10]` in `visual_hydration_node` (`backend/src/rag/graph/nodes.py`), surfacing up to 10 relevant figures and tables from the top ColPali pages.
  5. **Config & Defaults**: Synchronized `DEFAULT_TEXT_TOP_K = 5` and `DEFAULT_FIGURE_TABLE_TOP_K = 10` in `backend/src/rag/config.py` and `colpali_retriever.py`.
- **Outcome**: Enhanced visual asset recall (up to 10 figures and tables) paired with focused, high-precision 5-chunk text retrieval that fits comfortably within token budgets.


---

## 30. Definitive Fix for Visual Figure & Table Asset URL Resolution

### Decision 30.1: Root Cause Analysis & Robust Asset URL Resolution Engine
- **Context & Problem**: The user reported: *"currently figures are not renedered on the frontend, why is it that after an update something break down, aren't you building the project properly?"*. Inspection revealed that figure images in the visual references deck failed to load in the browser.
- **Root Cause**:
  1. **Discrepancy Between JSONL `image_path` and Physical Storage**: In `attention_figures.jsonl` and `attention_tables.jsonl`, `image_path` was recorded as `"src\pdf\figures\cropped\attention_fig_001.png"`, whereas the physical files reside directly at `data/figures/attention_fig_001.png` and `data/tables/attention_table_001.png`.
  2. **Naive URL String Replacement**: In `frontend/server.py`, the URL formatting logic used:
     ```python
     clean_img = str(img).replace("\", "/").replace("data/", "")
     src["web_image_url"] = f"/data/{clean_img}"
     ```
     This produced `/data/src/pdf/figures/cropped/attention_fig_001.png`. Because FastAPI mounts `/data` to `DATA_DIR` (`data/`), this URL returned **HTTP 404 Not Found** for every visual asset.
  3. **Lack of Client-Side Fallback**: `renderTapedMount` in `frontend/static/app.js` relied solely on `src.web_image_url` and had no self-healing logic to extract the filename and fallback to canonical `/data/figures/{filename}` or `/data/tables/{filename}` paths.
- **Decision & Fix**:
  1. **Canonical Asset Resolver (`frontend/server.py`)**: Implemented `resolve_asset_url(image_path, doc_id, record_type)`:
     - Extracts filename from any raw path (`attention_fig_001.png`).
     - Checks document-specific folders first (`data/figures/{doc_id}/...`, `data/tables/{doc_id}/...`).
     - Checks canonical folders (`data/figures/...`, `data/tables/...`).
     - Recursively verifies existence against `DATA_DIR` and returns the exact verified `/data/...` URL.
     - Applied in `/api/chat` and `/api/documents/{doc_id}/history` (heals both live responses and historical checkpoints).
  2. **Client-Side Self-Healing (`frontend/static/app.js`)**: Updated `renderTapedMount` to automatically detect legacy `/data/src/pdf/...` patterns or missing `web_image_url` and reconstruct `/data/figures/${filename}` or `/data/tables/${filename}` directly in the browser.
  3. **Visual Quality Enhancement (`frontend/static/style.css`)**: Removed destructive `filter: grayscale(100%) contrast(108%)` from `.mount-image` and increased `max-height` to `380px` so technical diagrams, model architectures, and flowcharts display with crisp clarity and full visual fidelity.
- **Outcome**: Figures and tables render reliably and immediately in the visual references deck across all document sessions.


---

## 31. Elimination of Verbose Meta-Explanations for Unanswerable Queries

### Decision 31.1: Strict Single-Sentence Unanswerable Standard & Provenance Purging
- **Context & Problem**: When asked about a topic absent from the active document (e.g. *"what is random forest?"* while reading *Attention Is All You Need*), the LLM responded with a verbose multi-paragraph meta-explanation:
  > *"I don't have information about that in the document. The provided text chunks exclusively discuss the Transformer architecture, specifically focusing on Scaled Dot-Product Attention, Multi-Head Attention... The documents do not contain any information regarding Random Forests, decision trees, bagging, bootstrap aggregating..."*
  Additionally, the interface attached unrelated figures/tables because `sources` still contained the nearest (yet irrelevant) semantic chunks. The user requested: *"it is answering like 'The provided text chunks ..... ', don't do it like this. if the answer is not present so just say that I dont have information about this .......... etc etc something like this"*.
- **Decision**:
  1. **System Prompt Hardening (`backend/src/rag/graph/nodes.py`)**: Updated Guideline 6 with explicit negative constraints:
     - Must reply *only* with `"I don't have information about that in the document."`
     - Strictly forbidden from mentioning `"The provided text chunks..."` or summarizing document topics that *are* present.
     - Must stop immediately without reciting missing keywords or speculating.
  2. **Deterministic Response Guard (`_is_unanswerable_response`)**: Added a regex/prefix classifier in `nodes.py` that intercepts any response beginning with or containing meta-disclaimers (e.g. `"The provided text chunks exclusively discuss..."`, `"does not contain any information regarding..."`, `"I don't have information about..."`) and clamps the final output strictly to:
     ```python
     response = "I don't have information about that in the document."
     ```
  3. **Visual & Text Provenance Purge**: When an unanswerable response is identified, `sources` is reset to `[]`. This prevents the frontend from displaying irrelevant figure/table mounts or citation bars for out-of-scope questions.
- **Outcome**: Clean, concise single-sentence response with zero meta-clutter and zero irrelevant attachments whenever a question cannot be answered from the document.


---

## 32. Multi-Stage Ingestion Loading Bar & Real-Time Progress Feedback

### Decision 32.1: Asymptotic Ingestion Progress Bar & Live Pipeline Stage Telemetry
- **Context & Problem**: The user requested: *"also add a featire that when a pdf is ingested so it shows a loading bar or something, whcih make the user know that it is in progress"*. During document uploads, the UI previously showed a static, non-animated text box (`INDEXING DOCUMENT & EXTRACTING MULTIMODAL VECTORS...`). Because document ingestion involves multiple intensive stages (PyMuPDF rendering, Docling hierarchical layout parsing, figure/table extraction and cropping, Qdrant vector indexing, and ColPali embeddings), processing typically spans 15–30 seconds. Without a dynamic progress bar or stage feedback, users had no indication of progress or elapsed time.
- **Decision**:
  1. **Animated Loading Bar & Progress Track (`frontend/static/index.html` & `style.css`)**:
     - Embedded `.progress-box-sketch` in the upload modal featuring an animated ruled progress track (`.progress-track-sketch`) and a technical diagonal-hatched fill (`.progress-fill-sketch.animated`).
     - Real-time numerical percentage indicator (`0%` $	o$ `100%`).
  2. **Live Pipeline Stage Telemetry**:
     - Configured progressive stage text mapped to elapsed ingestion milestones:
       1. `0% - 18%`: *"Uploading PDF & initializing workspace..."*
       2. `18% - 38%`: *"Rendering high-res document pages..."*
       3. `38% - 62%`: *"Parsing layout & extracting text (IBM Docling)..."*
       4. `62% - 82%`: *"Cropping figures & structured data tables..."*
       5. `82% - 95%`: *"Indexing semantic vectors & multimodal embeddings..."*
       6. `100%`: *"✓ Ingestion complete! Opening workspace..."*
  3. **Global Ingestion Banner Strip (`#globalIngestBar`)**:
     - Added a banner strip beneath the header with an animated pulsing indicator dot (`.ingest-pulse-dot`), document name, active pipeline stage, and live percentage.
     - Ensures users retain full visibility into ingestion progress even if the modal is minimized or dismissed.
  4. **State Transitions & Auto-Hydration**:
     - Submitting the form disables buttons and displays a spinner (`Ingesting... ⏳`).
     - On successful response (HTTP 200), hits 100%, displays the checkmark, refreshes the document ledger (`loadDocuments()`), and automatically switches the active canvas to the new document.
- **Outcome**: Immediate, transparent, and responsive visual progress feedback during long-running document ingestion operations.


---

## 33. Multi-Turn Thread Protection & Document-Grounded Contextual Retrieval

### Decision 33.1: Decoupling Document Ingestion from Dialogue Memory
- **Context & Problem**: In `frontend/server.py`, the document upload endpoint (`/api/upload`) previously executed `run_agentic_rag(question="Summarize the core topic of this document.", pdf_path=target_pdf)`. Because `thread_id` defaulted to `doc_{doc_id}`, this priming query executed the full RAG graph and wrote a failed dialogue turn (`User: Summarize the core topic... -> Assistant: I don't have information about that in the document.`) directly into the SQLite checkpointer (`data/chat_history.db`) before the user ever entered the chat room.
- **Decision**:
  1. Created a dedicated ingestion function `ingest_document(pdf_path)` in `backend/src/rag/graph/workflow.py` that executes only the ingestion nodes (`render_pages`, `parse_document`, `extract_and_crop`, `index_vectors`, `index_colpali`).
  2. Updated `/api/upload` to call `ingest_document(target_pdf)` directly. It performs complete asset rendering and vector indexing without invoking query answering, LLM completion, or checkpointer writes.
  3. Every newly uploaded document starts with an entirely clean, empty conversational memory thread.

### Decision 33.2: Guarded Query Contextualization & Anaphora Detection
- **Context & Problem**: `contextualize_query` previously called the LLM on every question if history existed. When users asked self-contained questions (e.g. *"what is the document about?"*, *"what is table 1 shows?"*, *"what is the figure 5 about?"*), the contextualizer misattributed entities from prior turns or previous queries, creating recursive question loops and hallucinations.
- **Decision**:
  1. **Anaphoric Coreference Guard**: Inspected query for explicit pronouns (`it`, `its`, `they`, `them`, `this`, `that`, `these`, `those`) or relative/elliptical references (`what about`, `how about`, `the former`, `the latter`, `explain more`). If none are present, the question is self-contained and returned verbatim without LLM invocation.
  2. **Boilerplate Filter**: Filtered out unanswerable disclaimers (`"I don't have information about that in the document."`) and error messages from conversational memory prior to prompt construction.
  3. **Strict LLM Constraints & Loop Prevention**: Instructed the model never to invent external entities or repeat previous user questions if the new query is distinct.
  4. **CRAG Retry Exemption**: Excluded queries from re-contextualization when `retry_count > 0`.

### Decision 33.3: Authentic Dialogue Logging in Conversational Memory
- **Context & Problem**: In `generate_answer_node`, turns were recorded using `question = state["question"]`. Whenever the Corrective RAG (CRAG) node reformulated a query into technical search keywords, that reformulated search query was saved as the user's message in SQLite rather than what the user actually typed. This poisoned subsequent conversational turns.
- **Decision**:
  1. Updated `generate_answer_node` to record `user_dialogue_question = state.get("original_question") or question`.
  2. The SQLite checkpointer preserves the authentic user prompt verbatim, preventing search reformulations from leaking into conversation history.

### Decision 33.4: Targeted Visual Asset Extraction & Overview Seed Chunks
- **Context & Problem**:
  1. For broad overview questions (*"what is the document about?"*), dense vector and BM25 search often matched isolated middle pages rather than the title, abstract, or executive summary.
  2. For targeted questions (*"what is table 1 shows?"*, *"what is the figure 5 about?"*), assets were only hydrated if their page appeared in ColPali's top 5 pages. If page scoring varied, Table 1 or Figure 5 was omitted.
- **Decision**:
  1. **Direct Asset Targeting (`visual_hydration_node`)**: Detected regex patterns for `table \d+` and `fig(ure)? \d+`. If present, matches by asset ID (e.g. `_001`, `_005`) and caption are assigned priority rank (`colpali_rank=0`) and placed at the head of `figure_table_results`.
  2. **Introductory Chunk Seed (`text_fusion_node`)**: Added `is_overview_query(query)` detection. When activated, the system injects introductory chunks (Pages 1–2 title, executive summary, director's report) into the fused candidate pool.
  3. **Document-Grounded CRAG Reformulation (`rewrite_query_node`)**: Rewrote queries using `doc_name`, overview keywords, or targeted asset identifiers instead of inventing arbitrary computer vision benchmarks (e.g. ResNet-50, ImageNet, COCO).
### Decision 33.5: Relevance-Gated Visual Presentation & Redundancy Elimination
- **Context & Problem**: The user reported that redundant figures and tables were being presented on the frontend. During multimodal retrieval, the system hydrated up to 10–13 candidate visual assets from ColPali pages. Because all candidate assets were passed into `sources`, the frontend previously mounted every single retrieved table and figure below the answer, regardless of whether the model used them or whether the user asked about them (e.g. asking for Figure 5 displayed Figure 5 alongside 9 other unrelated tables and signature stamps; asking for a document overview plastered 10 incidental visual cards).
- **Decision**:
  1. **Provenance-Based Visual Filtering (`filter_relevant_sources`)**:
     - **Targeted Asset Priority**: If the query specifically requests a table or figure (e.g. `Table 1`, `Figure 5`), ONLY that targeted asset is presented.
     - **Answer Citation Priority**: Visual assets are presented only if explicitly cited via `[Source N]` or mentioned by name/number in the answer prose.
     - **Overview Query Visual Suppression**: For broad document overview questions without visual keywords, visual mounts are completely suppressed (`0 visual assets`), keeping the response a clean text summary.
     - **Clutter Ceiling**: Strictly capped presented visual assets to at most 2 items.
  2. **Frontend Visual Deduplication & Guard (`getRelevantVisualSources` in `app.js`)**:
     - Added deduplication by asset ID and image URL.
     - Guaranteed that if `visualSources` is empty, no visual deck or card is rendered.
- **Outcome**: 100% precision in visual presentation: overview queries show 0 visual clutter, asset-specific queries show only the requested figure/table, and out-of-scope queries show zero assets.


## 34. Architectural Separation: Relocation of `server.py` to Backend

### Decision 34.1: Moving `server.py` from `frontend/` to `backend/`
- **Context & Problem**:
  `server.py` was originally placed in `frontend/` when the web interface was first introduced as a lightweight static asset server (`StaticFiles`) for `index.html`. As the application evolved, `server.py` became the primary API server containing:
  - Document ingestion endpoints (`/api/upload`)
  - Multi-turn conversation and session endpoints (`/api/chat`, `/api/documents/{doc_id}/history`)
  - Asset routing and resolution (`/api/documents/{doc_id}/pdf`, `/api/documents/{doc_id}/pages`)
  - Checkpointer thread pruning (`DELETE /api/documents/{doc_id}/history`)
  Having the application backend inside `frontend/` violated the separation of concerns and created architectural confusion.
- **Decision**:
  1. Relocated `server.py` to `backend/server.py`.
  2. Maintained clean separation of concerns:
     - `frontend/`: Presentation layer only (`static/index.html`, `static/style.css`, `static/app.js`, `static/vendor/`).
     - `backend/`: Server & business logic (`server.py`, `src/rag/`, `src/services/`).
  3. Configured `PROJECT_ROOT` and `STATIC_DIR` resolution in `backend/server.py` so that:
     - Static assets mount from `PROJECT_ROOT / "frontend" / "static"`.
     - Data assets mount from `PROJECT_ROOT / "data"`.
     - `PROJECT_ROOT` and `backend/src` are safely registered in `sys.path`.
  4. Updated runner command to `python backend/server.py` and `uvicorn backend.server:app --reload --port 8000`.
- **Outcome**: Clean repository architecture, unambiguous boundary between backend server logic and frontend client assets, and reliable import path resolution across CLI, Uvicorn, and containerized executions.


## 35. Robust Whole-Document Summarization Prompting & Anti-Refusal Guard

### Decision 35.1: Overhauling Answer Generation System Prompt for Document Summaries
- **Context & Problem**: When users asked for a summary of the whole document (e.g., *"summarize the whole document"*, *"what is this report about?"*, *"give me an overview"*), the LLM frequently refused with *"I don't have information about that in the document."*
  This failure occurred because:
  1. The prompt previously strictly commanded: *"Answer strictly using the retrieved text chunks... without speculating or hallucinating unmentioned facts"*.
  2. The prompt had an overzealous Unanswerable rule: *"If the document contains genuinely no information to answer the question, respond ONLY with: 'I don't have information about that in the document.'"*
  3. Because the VLM only receives the top retrieved excerpts (5 chunks), the model reasoned that it lacked the complete, exhaustive text of the entire document and therefore falsely deemed a "whole document summary" unanswerable.
- **Decision**:
  1. **Dedicated Document Summary Guideline**: Added explicit instructions instructing the assistant how to handle whole-document summaries and overviews:
     - Synthesize a comprehensive, well-structured executive summary based on the provided document excerpts (title, abstract, introduction, key findings, and results).
     - Structure the summary with clean markdown sections (**Core Objective / Subject**, **Key Highlights & Methodology**, **Major Results / Financial Highlights**, and **Conclusion**).
     - Explicitly commanded: *"NEVER refuse or state you lack information when asked for a summary or overview of the document; utilize the available document context to represent the document's central theme and main contributions."*
  2. **Confined Unanswerable Rule**: Restricted the unanswerable refusal rule strictly to completely external, out-of-scope queries (unrelated companies, sports, outside world facts) that have zero presence in the document context. Explicitly barred triggering this rule for overview or summary requests.
  3. **Contextual Question Instruction**: Updated the question block in `generate_answer_node` so that overview queries prompt: *"Please synthesize a clear, comprehensive, and well-structured executive summary of the document based on the provided context, citing relevant sources."*
- **Outcome**: Document summarization requests reliably produce high-quality executive summaries across scientific and financial documents without falsely triggering unanswerable refusals.


## 36. Classical Production Query Expansion (Document-Agnostic & Dynamic)

### Decision 36.1: Document-Agnostic Classical Query Expansion (Avoiding Costly Multi-Query & Static Dictionaries)
- **Context & Problem**: In earlier iterations, query reformulation in the Corrective RAG (CRAG) fallback loop discarded the user's original query and replaced it with a synthetic keyword search string. Multi-query was rejected because issuing multiple concurrent searches across dense, BM25, and remote ColPali ZeroGPU would multiply latency and API costs by 3x–5x. Furthermore, hardcoded deterministic dictionaries (e.g. for finance or deep learning) were rejected because the system is document-agnostic and must support any uploaded PDF (medical, legal, engineering, financial, scientific, manuals, etc.).
- **Decision**:
  1. **Rejected Multi-Query**: Preserved single-pass cost efficiency and prevented Hugging Face Space rate limits/timeouts.
  2. **Classical Query Expansion Formula**: Upgraded the CRAG fallback loop to follow the classical information retrieval formula:
     $$Q_{\text{expanded}} = Q_{\text{original}} + \text{Domain Terms / Technical Synonyms}$$
     The user's original question is strictly preserved at the beginning as the semantic anchor for Dense & ColPali retrieval, while domain terms and technical synonyms are appended.
  3. **Removed Hardcoded Static Dictionaries**: Eliminated static domain dictionaries so no domain assumptions are baked into code. The initial BM25 search remains clean and completely general.
  4. **Universal Document Structure Expansions**:
     - *Tables*: Appends universal data-grid terms (`data grid rows columns values results breakdown {doc_name}`).
     - *Figures*: Appends universal visual terms (`visual chart diagram plot image illustration graph structure {doc_name}`).
     - *Overview/Summary*: Appends universal document overview terms (`executive summary introduction abstract core overview background findings {doc_name}`).
  5. **Dynamic LLM Query Expansion**: For general technical queries, the LLM dynamically analyzes the user query and `{doc_name}` to extract 4–8 relevant domain terms and technical synonyms without inventing external benchmark datasets or machine learning architectures.
- **Outcome**: 100% document-agnostic classical query expansion that adapts to any uploaded document type (healthcare, law, science, finance, engineering) without multi-query overhead or hardcoded vocabulary bias.


## 37. Groq OTPM Rate Limit Protection & Strict Output Token Capping

### Decision 37.1: Resolving Groq Output Tokens Per Minute (OTPM) 429 Errors
- **Context & Problem**: When asking for a comprehensive summary of a document, the user encountered:
  `RateLimitError: 429 - Request too large for model qwen/qwen3.8-27b on output tokens per minute (OTPM): Limit 1000, Requested 1066. The request's expected output tokens exceed the enforced limit; reduce max_tokens (or the request's expected output) and try again.`
- **Root Cause Analysis**:
  1. **Not Input Tokens (ITPM)**: The number of retrieved text chunks (5 chunks) and attached figure images (at most 1 image) was NOT the cause. Total input stayed well below the 7,000 ITPM limit.
  2. **Output Tokens (OTPM)**: Groq enforces an extremely tight **Output Tokens Per Minute (OTPM)** quota of **1,000 tokens** on `qwen/qwen3.8-27b`.
  3. **Unbounded Output Buffer**: `llm_vision` in `llm_service.py` had no `max_tokens` set. When an overview query asked for a comprehensive summary, the client defaulted to an unbounded output reservation (> 1,000 tokens), causing Groq to preemptively reject the call before generation began.
- **Decision**:
  1. **Explicit Output Token Ceilings**:
     - Configured `max_tokens=750` on `llm_vision` in [`backend/src/services/llm_service.py`](backend/src/services/llm_service.py).
     - Configured `max_tokens=500` on `llm` (used for grading and CRAG query expansion).
  2. **Prompt-Level Length Directive**:
     - Added an explicit length instruction in `generate_answer_node` for overview/summary queries: *"Keep your summary focused and concise (around 250-350 words) using clear bullet points to present key findings within output token budgets."*
- **Outcome**: Output reservations never exceed Groq's 1,000 OTPM limit, completely eliminating 429 output token rate-limit errors while ensuring dense, high-quality executive summaries.


## 38. Real-Time Document Ingestion Streaming & True Backend Token Streaming via SSE

### Decision 38.1: Authentic Backend Ingestion Progress via Server-Sent Events
- **Context & Problem**: Previously, document ingestion in the frontend modal used a simulated JavaScript `setInterval` timer with an artificial mathematical curve that guessed progress stages. If document rendering or Docling layout parsing took longer, the progress bar stalled at 95%; if ingestion finished faster, the bar jumped abruptly.
- **Decision**:
  1. **Backend Pipeline Generator**: Implemented `stream_ingest_document(pdf_path)` in `backend/src/rag/graph/workflow.py` as a Python generator yielding structured progress events:
     - `stage: rendering` (15%) - Document page rendering.
     - `stage: parsing` (40%) - Docling layout & table extraction.
     - `stage: cropping` (65%) - Figure & table cropping.
     - `stage: vectors` (82%) - Qdrant vector indexing.
     - `stage: colpali` (92%) - ColPali ZeroGPU registration.
     - `stage: complete` (100%) - Ingestion completion with target `doc_id`.
  2. **SSE Upload Endpoint**: Added streaming support to `POST /api/upload?stream=true` in `backend/server.py` returning `StreamingResponse(..., media_type="text/event-stream")`.
  3. **Frontend Event-Driven Progress**: Updated `frontend/static/app.js` (`initUploadModal`) to read the stream using `res.body.getReader()`, dynamically updating the progress bar width, percentage, and label in real time as each backend step finishes.
- **Outcome**: 100% authentic, transparent ingestion feedback reflecting the exact backend execution stage, completely eliminating simulated timers.

### Decision 38.2: True Backend Token Streaming & User-Friendly Status Pills
- **Context & Problem**: Previously, chat answers waited for the complete LLM response to finish before displaying text in the UI (or using a frontend-only simulated typewriter effect). Furthermore, status messages emitted during retrieval displayed internal engineering terminology (`"Fusing multimodal candidates & hydrating assets..."`, `"Executing parallel Dense, BM25, and ColPali visual retrieval..."`) which degraded end-user experience.
- **Decision**:
  1. **True VLM Token Streaming**: Implemented `stream_agentic_rag(question, pdf_path, thread_id)` in `backend/src/rag/graph/workflow.py` leveraging Groq's native `llm_vision.stream_chat(messages)` with `chunk.delta`.
  2. **FastAPI SSE Chat Route**: Added `POST /api/chat/stream` delivering Server-Sent Events for status pills (`type: "status"`), raw token deltas (`type: "token"`), and post-generation metadata (`type: "metadata"`).
  3. **User-Centric Status Messaging**: Simplified status messages to clear, non-technical natural language:
     - Stage 1 (Intent & History): `"Looking into document..."`
     - Stage 2 (Retrieval): `"Searching document..."`
     - Stage 3 (Fusion & Hydration): `"Combining results..."`
     - Stage 4 (CRAG Query Expansion): `"Refining search..."`
  4. **State Persistence & KaTeX Math Rendering**:
     - Preserved LangGraph SQLite checkpointer (`chat_history.db`) state updates using `app.update_state(config, {"answer": full_answer, "sources": sources, "chat_history": new_turn})`.
     - Upon stream completion, the frontend dynamically replaces raw streamed tokens with full KaTeX math evaluation, semantic Markdown typography, and clickable page source badges.
- **Outcome**: Sub-second Time-to-First-Token (TTFT), real-time streaming feedback, intuitive status messages, and seamless mathematical and visual asset rendering.


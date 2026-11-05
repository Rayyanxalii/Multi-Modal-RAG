from typing import Any

from rag.retrieval_orchestrator import run_hybrid_retrieval
from services.llm_service import llm


def generate_answer(question: str | None = None) -> dict[str, Any]:
    """Generate an answer using multimodal retrieval (text + visual ColPali)."""
    if question is None:
        question = input("Ask a question about the paper: ").strip()

    if not question:
        return {
            "question": "",
            "answer": "No question provided.",
            "sources": [],
        }

    # 1. Run independent text and visual retrieval
    retrieval_output = run_hybrid_retrieval(
        question=question,
        text_top_k=5,
        figure_table_top_k=3,
        save_results=True,
    )

    text_results = retrieval_output.get("text_results", [])
    fig_tab_results = retrieval_output.get("figure_table_results", [])

    if not text_results and not fig_tab_results:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the document.",
            "sources": [],
        }

    # 2. Build structured source blocks for prompt grounding
    sources = []
    context_blocks = []
    source_idx = 1

    # Add text chunk sources
    for item in text_results:
        chunk_id = item.get("chunk_id", "Unknown")
        pages = item.get("page_numbers") or []
        page_str = ", ".join(str(p) for p in pages) if pages else "Unknown"
        raw_text = item.get("text") or item.get("contextualized_text") or ""
        text = str(raw_text).strip()

        sources.append({
            "source": source_idx,
            "type": "text",
            "id": chunk_id,
            "pages": pages,
        })
        context_blocks.append(
            f"[Source {source_idx}] (Text Chunk: {chunk_id}, Pages: {page_str})\n{text}"
        )
        source_idx += 1

    # Add visual figure and table sources (from ColPali)
    for item in fig_tab_results:
        rec_type = item.get("record_type", "visual")
        rec_id = item.get("record_id", item.get("figure_id", item.get("table_id", "Unknown")))
        page = item.get("page", "Unknown")
        caption = str(item.get("caption") or item.get("panel_caption") or "").strip()

        content_parts = []
        if caption:
            content_parts.append(f"Caption: {caption}")
        table_md = str(item.get("table_markdown") or "").strip()
        if rec_type == "table" and table_md:
            content_parts.append(f"Table Data:\n{table_md}")

        content_str = "\n".join(content_parts) if content_parts else "(Visual element)"

        sources.append({
            "source": source_idx,
            "type": rec_type,
            "id": rec_id,
            "pages": [page] if isinstance(page, int) else [],
        })
        context_blocks.append(
            f"[Source {source_idx}] ({rec_type.capitalize()}: {rec_id}, Page: {page})\n{content_str}"
        )
        source_idx += 1

    grounded_context = "\n\n".join(context_blocks)

    system_prompt = """You are a research assistant answering questions about a scientific paper.

Rules:
1. Answer using only the supplied document context (text passages, figures, and tables).
2. Cite factual claims using the provided source IDs and page numbers (e.g., [Source 1, p. 3]).
3. Use only source IDs and page numbers explicitly supplied in the context.
4. Never invent citations or page numbers.
5. If the context does not support an answer, clearly state that.
6. Be clear, accurate, and concise."""

    user_prompt = f"""Question:
{question}

Document Context:
{grounded_context}

Write a comprehensive, grounded answer supported by the context. Cite relevant sources and pages next to the claims they support."""

    final_prompt = f"{system_prompt}\n\n{user_prompt}"

    # 3. Generate answer via LLM
    response = llm.complete(final_prompt)
    answer = getattr(response, "text", response)
    answer = str(answer).strip()

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    result = generate_answer()

    print("\n" + "=" * 50)
    print("QUESTION:")
    print("=" * 50)
    print(result["question"])

    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])

    print("\n" + "=" * 50)
    print("SOURCES:")
    print("=" * 50)
    for s in result["sources"]:
        pages_str = ", ".join(str(p) for p in s["pages"]) if s["pages"] else "Unknown"
        print(f"Source {s['source']} [{s['type'].capitalize()}] | ID: {s['id']} | Pages: {pages_str}")

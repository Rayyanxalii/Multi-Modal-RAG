
from rag.hybrid_retriever import main
from services.llm_service import llm


def generate_answer() -> dict:
    # 1. Retrieve the question, fused chunks, and context.
    question, fused_results = main()

    if not fused_results:
        return {
            "question": question,
            "answer": (
                "I couldn't find relevant information "
                "in the document."
            ),
            "sources": [],
        }

    # 2. Build source references from the actual fused chunks.
    sources = []

    for index, item in enumerate(fused_results, start=1):
        if isinstance(item, dict):
            text = item.get("text", item.get("content", ""))
            metadata = item.get("metadata") or {}

            # Your retrievers store page numbers directly
            # on the chunk, not necessarily in metadata.
            pages = item.get("page_numbers") or []

            if not pages:
                page = metadata.get(
                    "page",
                    metadata.get("page_number"),
                )

                if page is not None:
                    pages = [page]

            chunk_id = item.get("chunk_id", "Unknown")

        elif hasattr(item, "node"):
            text = item.node.get_content()
            metadata = item.node.metadata or {}
            pages = metadata.get("page_numbers") or []

            if not pages:
                page = metadata.get(
                    "page",
                    metadata.get("page_number"),
                )
                pages = [page] if page is not None else []

            chunk_id = getattr(item.node, "id_", "Unknown")

        elif hasattr(item, "get_content"):
            text = item.get_content()
            metadata = getattr(item, "metadata", {}) or {}
            pages = metadata.get("page_numbers") or []

            if not pages:
                page = metadata.get(
                    "page",
                    metadata.get("page_number"),
                )
                pages = [page] if page is not None else []

            chunk_id = getattr(item, "id_", "Unknown")

        else:
            text = str(item)
            pages = []
            chunk_id = "Unknown"

        sources.append({
            "source": index,
            "chunk_id": chunk_id,
            "pages": pages,
            "text": text,
        })

    # 3. Build a prompt that explicitly identifies source pages.
    # Use the existing context if hybrid_retriever provides it.
    context_parts = []

    for source in sources:
        page_label = (
            ", ".join(str(page) for page in source["pages"])
            if source["pages"]
            else "Unknown"
        )

        context_parts.append(
            f"[Source {source['source']}]\n"
            f"Chunk ID: {source['chunk_id']}\n"
            f"Page numbers: {page_label}\n"
            f"Content:\n{source['text']}"
        )

    grounded_context = "\n\n".join(context_parts)

    system_prompt = """
You are a research assistant answering questions about a document.

Rules:
1. Answer using only the supplied document context.
2. Cite factual claims using the provided source IDs and page numbers.
   Example: [Source 1, p. 3]
3. Use only source IDs and page numbers explicitly supplied.
4. Never invent citations or page numbers.
5. If the context does not support an answer, clearly say so.
6. If sources disagree, acknowledge the disagreement.
7. Be clear and concise.
"""

    user_prompt = f"""
Question:
{question}

Document context:
{grounded_context}

Write an answer supported by the context.
Cite relevant sources and pages next to the claims they support.
"""

    final_prompt = f"{system_prompt}\n\n{user_prompt}"

    # 4. Generate the answer with your existing LLM service.
    response = llm.complete(final_prompt)

    # Handle services that return an object rather than plain text.
    answer = getattr(response, "text", response)
    answer = str(answer).strip()

    # 5. Return the answer with traceable source information.
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": source["source"],
                "chunk_id": source["chunk_id"],
                "pages": source["pages"],
            }
            for source in sources
        ],
    }


if __name__ == "__main__":
    result = generate_answer()

    print("\n--- Question ---\n")
    print(result["question"])

    print("\n--- Answer ---\n")
    print(result["answer"])

    print("\n--- Sources ---\n")

    for source in result["sources"]:
        print(
            f"Source {source['source']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Pages: {source['pages'] or 'Unknown'}"
        )

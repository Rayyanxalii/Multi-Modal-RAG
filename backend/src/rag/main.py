"""Main entrypoint for the Generalized Multimodal Agentic RAG system."""

import sys
import argparse

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag.graph import run_agentic_rag


def run_single(question: str, pdf_path: str | None = None) -> None:
    """Run a question through the LangGraph agent and display grounded results."""
    result = run_agentic_rag(question, pdf_path=pdf_path)
    route = result.get("route", "both").capitalize()
    retries = result.get("retry_count", 0)

    print("\n" + "=" * 55)
    print(f"ROUTE: {route} | CRAG RETRIES: {retries}")
    if result.get("routing_reasoning"):
        print(f"REASONING: {result['routing_reasoning']}")
    if result.get("rewritten_query"):
        print(f"REFORMULATED QUERY: {result['rewritten_query']}")
    print("=" * 55)
    print("ANSWER:")
    print("=" * 55)
    print(result.get("answer", ""))
    print("\n--- Sources ---")
    for source in result.get("sources", []):
        pages_str = ", ".join(str(p) for p in source.get("pages", [])) or "Unknown"
        src_type = source.get("type", "Text").capitalize()
        src_id = source.get("id", source.get("chunk_id", "Unknown"))
        print(f"Source {source['source']} [{src_type}] | ID: {src_id} | Pages: {pages_str}")



def run_interactive(pdf_path: str | None = None) -> None:
    """Run an interactive session where models stay loaded in RAM across queries."""
    print("=" * 65)
    print("Generalized Multi-Modal Agentic RAG Assistant (LangGraph)")
    print(f"Active Document: {pdf_path or 'attention.pdf'}")
    print("Supports technical documents with text passages, diagrams & tables.")
    print("Type your question below, or type 'exit', 'quit', or 'q' to stop.")
    print("=" * 65)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break

        run_single(question, pdf_path=pdf_path)




def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Agentic RAG CLI")
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Optional single question to ask. If omitted, starts interactive session.",
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default=None,
        help="Optional path to custom PDF document (defaults to data/raw/attention.pdf).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save the LangGraph workflow image and Mermaid file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all currently ingested documents in the system.",
    )
    parser.add_argument(
        "--delete",
        type=str,
        default=None,
        help="Purge all assets, vectors, and cloud embeddings for a document ID.",
    )
    args = parser.parse_args()

    if args.list:
        from rag.document_manager import list_all_documents
        docs = list_all_documents()
        print("\n" + "=" * 65)
        print("INGESTED DOCUMENTS IN THE SYSTEM:")
        print("=" * 65)
        if not docs:
            print("No documents ingested yet.")
        for d in docs:
            print(f"- [{d['doc_id']}] {d['title']}: {d['pages']} pages | {d['chunks']} chunks | {d['figures']} figures | {d['tables']} tables")
        print("=" * 65)
        return

    if args.delete:
        from rag.document_manager import delete_document
        res = delete_document(args.delete, delete_raw_pdf=True)
        print(f"\nDeletion result for '{args.delete}': {res.get('status').upper()}")
        for item in res.get("deleted_items", []):
            print(f"  ✓ {item}")
        for err in res.get("errors", []):
            print(f"  ✗ {err}")
        print()
        return

    if args.visualize:
        from rag.graph.visualize import save_graph_artifacts
        png_p, mmd_p = save_graph_artifacts()
        print(f"Graph image saved to: {png_p}")
        print(f"Mermaid file saved to: {mmd_p}")
        return

    if args.question:
        run_single(args.question, pdf_path=args.pdf)
    else:
        run_interactive(pdf_path=args.pdf)



if __name__ == "__main__":
    main()

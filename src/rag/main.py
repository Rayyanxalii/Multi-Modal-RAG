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


def run_single(question: str) -> None:
    """Run a question through the LangGraph agent and display grounded results."""
    result = run_agentic_rag(question)
    route = result.get("route", "both").capitalize()
    retries = result.get("retry_count", 0)

    print("\n" + "=" * 55)
    print(f"ROUTE: {route} | SELF-RAG RETRIES: {retries}")
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



def run_interactive() -> None:
    """Run an interactive session where models stay loaded in RAM across queries."""
    print("=" * 65)
    print("Generalized Multi-Modal Agentic RAG Assistant (LangGraph)")
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

        run_single(question)




def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Agentic RAG CLI")
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Optional single question to ask. If omitted, starts interactive session.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save the LangGraph workflow image and Mermaid file.",
    )
    args = parser.parse_args()

    if args.visualize:
        from rag.graph.visualize import save_graph_artifacts
        png_p, mmd_p = save_graph_artifacts()
        print(f"Graph image saved to: {png_p}")
        print(f"Mermaid file saved to: {mmd_p}")
        return

    if args.question:
        run_single(args.question)
    else:
        run_interactive()



if __name__ == "__main__":
    main()

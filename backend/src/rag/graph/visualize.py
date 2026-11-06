"""Graph visualization generator for the LangGraph Multimodal Agentic RAG system."""

from pathlib import Path
from rag.config import DATA_DIR, PROJECT_ROOT
from rag.graph.workflow import app


def save_graph_artifacts(
    output_png: Path | str = DATA_DIR / "graph.png",
    output_mmd: Path | str = DATA_DIR / "graph.mmd",
    root_png: Path | str = PROJECT_ROOT / "graph.png",
) -> tuple[Path, Path]:
    """Generate and save the LangGraph workflow PNG using draw_mermaid_png()."""
    output_png = Path(output_png)
    output_mmd = Path(output_mmd)
    root_png = Path(root_png)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_mmd.parent.mkdir(parents=True, exist_ok=True)

    graph_view = app.get_graph()

    # 1. Export Mermaid markdown definition
    try:
        mermaid_syntax = graph_view.draw_mermaid()
        output_mmd.write_text(mermaid_syntax, encoding="utf-8")
        print(f"Mermaid syntax saved to: {output_mmd}")
    except Exception as exc:
        print(f"Could not export Mermaid syntax: {exc}")

    # 2. Generate Mermaid PNG via LangGraph draw_mermaid_png()
    png_data = graph_view.draw_mermaid_png()

    with open(output_png, "wb") as f:
        f.write(png_data)
    print(f"Graph image saved to: {output_png}")

    with open(root_png, "wb") as f:
        f.write(png_data)
    print(f"Graph image saved to: {root_png}")

    return output_png, output_mmd


if __name__ == "__main__":
    save_graph_artifacts()

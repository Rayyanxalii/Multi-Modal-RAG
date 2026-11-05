from pathlib import Path
from functools import lru_cache
import torch
from sentence_transformers import MultiVectorEncoder

from rag.config import COLPALI_EMBEDDINGS_PATH as EMBEDDINGS_PATH


class ColPaliRetriever:
    """Retrieve document pages using precomputed ColPali embeddings."""

    def __init__(
        self,
        embeddings_path: str | Path = EMBEDDINGS_PATH,
        model_name: str = "vidore/colpali-v1.3-hf",
    ):
        self.embeddings_path = Path(embeddings_path)

        if not self.embeddings_path.is_file():
            raise FileNotFoundError(
                f"ColPali embeddings not found: {self.embeddings_path}"
            )

        # Load the saved page names and multi-vector embeddings.
        saved_data = torch.load(
            self.embeddings_path,
            map_location="cpu",
            weights_only=True,
        )

        self.page_names = saved_data["page_names"]
        self.page_embeddings = saved_data["page_embeddings"]

        if len(self.page_names) != len(self.page_embeddings):
            raise ValueError(
                "Page names and embeddings have different lengths."
            )

        if not self.page_names:
            raise ValueError("No page embeddings found.")

        # Choose GPU when available; otherwise use CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the pretrained model for query encoding.
        self.model = MultiVectorEncoder(model_name, device=self.device)

        print(f"ColPali model loaded on: {self.device}")
        print(f"Pages indexed: {len(self.page_names)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search precomputed page embeddings for the query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Encode the query into a multi-vector embedding.
        query_embeddings = self.model.encode(
            [query],
            device=self.device,
        )

        with torch.no_grad():
            # Compare the query against the saved page embeddings.
            scores = self.model.similarity(
                query_embeddings,
                self.page_embeddings,
            )[0]

        ranked_indices = scores.argsort(descending=True)[:top_k].tolist()

        return [
            {
                "rank": rank,
                "page": self.page_names[index],
                "score": float(scores[index].item()),
                "chunk_id": f"page_{self.page_names[index]}",
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]


@lru_cache(maxsize=1)
def get_colpali_retriever() -> ColPaliRetriever:
    return ColPaliRetriever(
        embeddings_path=EMBEDDINGS_PATH
    )

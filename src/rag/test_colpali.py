from time import perf_counter
from functools import lru_cache
import torch
from rag.colpali_model import ColPaliRetriever


@lru_cache(maxsize=1)
def get_colpali_retriever() -> ColPaliRetriever:
    return ColPaliRetriever(
        embeddings_path="src/data/colpali/attention_colpali.pt"
    )


print(f"CUDA available: {torch.cuda.is_available()}")

for query in [
    "What is the attention mechanism?",
    "How does self-attention work?",
]:
    start = perf_counter()

    retriever = get_colpali_retriever()
    elapsed_init_and_lookup = perf_counter() - start

    start = perf_counter()
    results = retriever.search(query=query, top_k=5)
    query_time = perf_counter() - start

    print(f"\nInitialization/cache lookup: {elapsed_init_and_lookup:.2f}s")
    print(f"Query encoding and search: {query_time:.2f}s")
    print("Top result:", results[0])

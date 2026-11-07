"""ColPali visual retrieval interface powered by Hugging Face ZeroGPU Cloud Microservice."""

from functools import lru_cache
from rag.visual_retrieval.colpali_client import (
    ColPaliCloudClient,
    get_colpali_cloud_client,
)

# ColPaliRetriever is now an alias to the Cloud Client (no 6 GB local weights needed!)
ColPaliRetriever = ColPaliCloudClient


@lru_cache(maxsize=1)
def get_colpali_retriever(use_cloud: bool = True) -> ColPaliCloudClient:
    """Return cloud-backed ColPali retriever connected to Hugging Face Spaces."""
    return get_colpali_cloud_client()

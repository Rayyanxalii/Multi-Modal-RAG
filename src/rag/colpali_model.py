from sentence_transformers import MultiVectorEncoder
import torch

model = MultiVectorEncoder("vidore/colpali-v1.3-hf")

print("ColPali loaded successfully")
print("CUDA available:", torch.cuda.is_available())
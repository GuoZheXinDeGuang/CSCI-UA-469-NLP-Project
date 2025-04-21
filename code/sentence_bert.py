import numpy as np
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SentenceBERT:
    """
    A high‑quality, 384‑d sentence embedder using all‑MiniLM‑L6‑v2 model. 
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.dim   = self.model.get_sentence_embedding_dimension()

    @torch.inference_mode()
    def encode(self, texts, batch_size=128):
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE,
            show_progress_bar=False,
        )
        return embs.cpu().numpy()

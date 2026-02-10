from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, dim: int, model_path: str, normalize: bool = True):
        self.dim = dim
        self.model = SentenceTransformer(model_path)
        self.normalize = normalize

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

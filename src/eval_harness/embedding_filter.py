import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingFilter:
    """Cheap, local (no API cost) pre-filter: skips the paid LLM judge when two
    outputs are near-identical in embedding space. Per LIT-39 research finding #3,
    embedding similarity misses meaning nuance (e.g. reordered words) — it is
    ONLY used here as a cost-saving pre-filter, never as the source of truth for
    the semantic diff itself (that's the LLM judge, see judge.py)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def is_likely_unchanged(self, old_output: str, new_output: str, threshold: float = 0.97) -> bool:
        embeddings = self._model.encode([old_output, new_output])
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        return similarity >= threshold

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

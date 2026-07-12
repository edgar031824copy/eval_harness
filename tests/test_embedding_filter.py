from unittest.mock import MagicMock, patch
import numpy as np
from eval_harness.embedding_filter import EmbeddingFilter


def test_identical_text_is_likely_unchanged():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Same text -> same embedding -> cosine similarity 1.0
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("The sky is blue.", "The sky is blue.") is True


def test_very_different_text_is_not_likely_unchanged():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Orthogonal embeddings -> cosine similarity 0.0
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("The sky is blue.", "I like pizza.") is False


def test_threshold_is_configurable():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Cosine similarity ~0.95, below a strict 0.99 threshold
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [0.95, 0.312]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("a", "b", threshold=0.99) is False
        assert ef.is_likely_unchanged("a", "b", threshold=0.90) is True

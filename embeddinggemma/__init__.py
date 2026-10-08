"""
EmbeddingGemma 2 Model Implementation

A clean PyTorch implementation of the text encoder of Google's EmbeddingGemma 2.
"""

from .embeddinggemma2 import (
    ModelConfig,
    EmbeddingGemma2Model,
    EmbeddingGemmaTokenizer,
    load_pretrained_weights,
    load_text_weights,
)

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "EmbeddingGemma2Model",
    "EmbeddingGemmaTokenizer",
    "load_pretrained_weights",
    "load_text_weights",
]

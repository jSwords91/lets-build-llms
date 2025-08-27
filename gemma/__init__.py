"""
Gemma 3 Model Implementation

A clean PyTorch implementation of Google's Gemma 3 model with sliding window attention.
"""

from .gemma3 import (
    ModelConfig,
    Gemma3Model,
    GemmaTokenizer,
    generate_text_stream,
    load_pretrained_weights,
)

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "Gemma3Model", 
    "GemmaTokenizer",
    "generate_text_stream",
    "load_pretrained_weights",
]

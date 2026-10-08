# lets-build-llms
Various LLMs written from scratch, for pedagogical purposes only

I'll include code to load in the pre-trained weights from Huggingface for whatever we implement so we can ensure validity. 

I will endeavour to add a training loop along with a sample dataset too to showcase the training process.

# Implementations

- Gemma 270m Instruct
- EmbeddingGemma 2 (text encoder): `python -m embeddinggemma.embeddinggemma2 --query "What causes the northern lights?"`. A parity test against the Hugging Face reference is in `embeddinggemma/test_parity.py`.

# References

Sebastian Raschka's work and in particular his books are incredibly helpful. Please check out: https://github.com/rasbt

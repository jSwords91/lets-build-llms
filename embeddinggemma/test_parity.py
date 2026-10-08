"""
Parity test against the Hugging Face transformers reference implementation.

No download needed: builds the reference at full size, randomises every weight (norm scales and
layer scalars included, so a missed term shows up), saves it in the checkpoint format, loads that
into our model and compares outputs on a padded batch.

    pip install "transformers>=5.19" safetensors
    python -m embeddinggemma.test_parity
"""
import json
import os
import tempfile

import torch
import torch.nn.functional as F
from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model as ReferenceModel

from .embeddinggemma2 import ModelConfig, EmbeddingGemma2Model, load_pretrained_weights, load_text_weights


def check(sliding_window: int, seq_len: int, lengths: list[int]) -> bool:
    torch.manual_seed(0)
    reference = ReferenceModel(EmbeddingGemma2Config(text_config={"sliding_window": sliding_window})).eval()
    # Eager attention computes the same explicit softmax(QK^T)V as our model
    reference.language_model.config._attn_implementation = "eager"
    with torch.no_grad():
        for param in reference.parameters():
            if param.ndim == 1:
                param.copy_(1.0 + 0.3 * torch.randn_like(param))
            else:
                param.copy_(torch.randn_like(param) * param.shape[-1] ** -0.5)
        for layer in reference.language_model.layers:
            layer.layer_scalar.copy_(0.5 + torch.rand(1))

    with tempfile.TemporaryDirectory() as weights_dir:
        reference.save_pretrained(weights_dir)
        with open(os.path.join(weights_dir, "config.json")) as f:
            config = ModelConfig.from_hf_config(json.load(f), dtype=torch.float32)
        model = EmbeddingGemma2Model(config).eval()
        load_pretrained_weights(model, load_text_weights(weights_dir))

    input_ids = torch.randint(3, 250_000, (len(lengths), seq_len))
    attention_mask = torch.zeros_like(input_ids)
    for i, n in enumerate(lengths):
        attention_mask[i, :n] = 1
        input_ids[i, n:] = 0

    with torch.no_grad():
        ref_tokens = reference(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        our_tokens = model.encode_tokens(input_ids, attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        ref_emb = F.normalize((ref_tokens * mask).sum(1) / mask.sum(1), dim=-1)
        our_emb = model(input_ids, attention_mask)

    valid = attention_mask.bool()
    max_diff = (ref_tokens[valid] - our_tokens[valid]).abs().max().item()
    min_cos = F.cosine_similarity(ref_emb, our_emb).min().item()
    print(f"sliding_window={sliding_window:4d} seq_len={seq_len} lengths={lengths}: "
          f"max token diff {max_diff:.2e}, min embedding cosine {min_cos:.8f}")
    return max_diff < 1e-3 and min_cos > 0.99999


if __name__ == "__main__":
    ok = check(sliding_window=512, seq_len=48, lengths=[48, 30, 7])   # window wider than the input
    ok &= check(sliding_window=8, seq_len=64, lengths=[64, 41, 13])   # window narrower: exercises the local mask
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

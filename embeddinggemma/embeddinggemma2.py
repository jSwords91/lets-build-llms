import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

Tensor = torch.Tensor

@dataclass
class ModelConfig:
    """EmbeddingGemma 2 (text encoder) configuration"""
    vocab_size: int = 262_144
    context_length: int = 8_192
    emb_dim: int = 512
    n_heads: int = 4
    n_layers: int = 24
    hidden_dim: int = 2048
    # Sliding (local) layers
    head_dim: int = 256
    n_kv_groups: int = 2
    rope_local_base: float = 10_000.0
    sliding_window: int = 512             # Radius: a token sees every token with |i - j| <= sliding_window
    # Full (global) layers are wider, with fewer KV heads
    global_head_dim: int = 512
    global_n_kv_groups: int = 1
    rope_base: float = 1_000_000.0
    layer_types: list[str] = None
    # Per-layer embeddings (PLE) and the output head
    ple_dim: int = 512
    out_dim: int = 768
    pad_token_id: int = 0
    dtype: torch.dtype = torch.bfloat16   # bfloat16 or float32, never float16 (activations overflow)

    def __post_init__(self):
        if self.layer_types is None:
            # Default EmbeddingGemma 2 layer configuration: every 6th layer is global
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.n_layers)
            ]
        assert len(self.layer_types) == self.n_layers, f"layer_types length ({len(self.layer_types)}) must match n_layers ({self.n_layers})"

    @classmethod
    def from_hf_config(cls, hf_config: dict, **kwargs) -> "ModelConfig":
        """Build from the checkpoint's config.json"""
        text = hf_config.get("text_config", hf_config)
        # Global layers override head_dim and KV heads via per_layer_config, e.g. {"5": {"head_dim": 512, ...}}
        global_overrides = next(iter((text.get("per_layer_config") or {}).values()), {})
        rope = text.get("rope_parameters") or {}
        return cls(
            vocab_size=text["vocab_size"],
            emb_dim=text["hidden_size"],
            n_heads=text["num_attention_heads"],
            n_layers=text["num_hidden_layers"],
            hidden_dim=text["intermediate_size"],
            head_dim=text["head_dim"],
            n_kv_groups=text["num_key_value_heads"],
            rope_local_base=rope.get("sliding_attention", {}).get("rope_theta", 10_000.0),
            sliding_window=text["sliding_window"],
            global_head_dim=global_overrides.get("head_dim", 512),
            global_n_kv_groups=global_overrides.get("num_key_value_heads", 1),
            rope_base=rope.get("full_attention", {}).get("rope_theta", 1_000_000.0),
            layer_types=text["layer_types"],
            ple_dim=text["hidden_size_per_layer_input"],
            out_dim=text["embedding_dim"],
            pad_token_id=text["pad_token_id"],
            **kwargs,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Gemma 4 variant: plain weight, no (1 + weight) offset)"""
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        # Unlike Gemma 3, the stored weight is used as-is. The value norm in attention has no weight at all.
        self.scale = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: Tensor) -> Tensor:
        # Compute norm in float32 for stability
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.pow(var + self.eps, -0.5)
        if self.scale is not None:
            x_norm = x_norm * self.scale.float()
        return x_norm.to(input_dtype)


class RotaryPositionalEmbedding:
    """Rotary Position Embedding (RoPE) utilities"""
    @staticmethod
    def precompute_freqs_cis(head_dim: int, context_length: int, theta_base: float = 10_000.0) -> tuple[Tensor, Tensor]:
        """Precompute cosine and sine frequencies for RoPE"""
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        # Compute inverse frequencies
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # Generate position indices
        positions = torch.arange(context_length, dtype=torch.float32)
        # Compute angles: (seq_len, head_dim // 2)
        angles = positions[:, None] * inv_freq[None, :]
        # Expand to full head dimension
        angles = torch.cat([angles, angles], dim=1)  # (seq_len, head_dim)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary position embedding to input tensor"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        # Split into first and second half
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        # Adjust cos/sin shapes for broadcasting
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)  # (1, 1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        # Apply rotation: x_rotated = x * cos + rotate(x) * sin
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)
        return x_rotated.to(x.dtype)


class FeedForward(nn.Module):
    """GeGLU Feed Forward Network"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class PerLayerEmbedding(nn.Module):
    """Per-Layer Embedding (PLE) gate: the third residual sub-block of every layer.

    Each layer gets its own 512-d slice of a projection of the token embeddings ("per_layer_input"),
    and gates its hidden state against it through a small bottleneck.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.ple_dim, dtype=config.dtype, bias=False)
        self.out_proj = nn.Linear(config.ple_dim, config.emb_dim, dtype=config.dtype, bias=False)
        self.post_norm = RMSNorm(config.emb_dim, eps=1e-6)

    def forward(self, x: Tensor, per_layer_input: Tensor) -> Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.post_norm(self.out_proj(gate * per_layer_input))


class GroupedQueryAttention(nn.Module):
    """Bidirectional Grouped Query Attention with Q, K and V normalization"""
    def __init__(self, config: ModelConfig, attn_type: str):
        super().__init__()
        is_global = attn_type == "full_attention"
        self.num_heads = config.n_heads
        self.num_kv_groups = config.global_n_kv_groups if is_global else config.n_kv_groups
        assert self.num_heads % self.num_kv_groups == 0, "n_heads must be divisible by n_kv_groups"
        self.group_size = self.num_heads // self.num_kv_groups
        self.head_dim = config.global_head_dim if is_global else config.head_dim
        self.d_out = self.num_heads * self.head_dim

        # Projections
        self.q_proj = nn.Linear(config.emb_dim, self.d_out, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(config.emb_dim, self.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(config.emb_dim, self.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype)
        self.o_proj = nn.Linear(self.d_out, config.emb_dim, bias=False, dtype=config.dtype)

        # QK normalization, plus an unweighted V normalization (new in Gemma 4)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = RMSNorm(self.head_dim, eps=1e-6, with_scale=False)

        # Scaling factor: q and k are already RMS-normalized, so no 1/sqrt(head_dim)
        self.scaling = 1.0

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        # Apply projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # Q, K, V normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        # Apply RoPE
        q = RotaryPositionalEmbedding.apply_rotary_emb(q, cos, sin)
        k = RotaryPositionalEmbedding.apply_rotary_emb(k, cos, sin)
        # Expand K,V to match number of query heads (for grouped query attention)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        # Scaled dot-product attention (no causal mask: this is an encoder)
        q = q * self.scaling
        scores = q @ k.transpose(-2, -1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_out)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Single encoder block: attention, feed forward and PLE sub-blocks, each with pre/post norms"""
    def __init__(self, config: ModelConfig, attn_type: str):
        super().__init__()
        self.attn_type = attn_type
        self.attention = GroupedQueryAttention(config, attn_type)
        self.feed_forward = FeedForward(config)
        self.per_layer_embedding = PerLayerEmbedding(config)
        # Layer normalizations (pre and post norms, as in Gemma 3)
        self.input_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        # Learned per-layer output scale (stored as a buffer in the checkpoint)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=config.dtype))

    def forward(self, x: Tensor, per_layer_input: Tensor, mask_global: Tensor, mask_local: Tensor, cos_global: Tensor, sin_global: Tensor, cos_local: Tensor, sin_local: Tensor) -> Tensor:
        # Select appropriate mask and RoPE based on attention type
        if self.attn_type == "sliding_attention":
            mask, cos, sin = mask_local, cos_local, sin_local
        else:  # full_attention
            mask, cos, sin = mask_global, cos_global, sin_global

        # Attention block with residual connection
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, mask, cos, sin)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # Feed forward block with residual connection
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.feed_forward(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # Per-layer embedding block with residual connection
        x = x + self.per_layer_embedding(x, per_layer_input)
        return x * self.layer_scalar


class EmbeddingGemma2Model(nn.Module):
    """EmbeddingGemma 2 text encoder: bidirectional Gemma 4 style transformer + mean pooling"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        # PLE inputs are a projection of the token embeddings: one ple_dim slice per layer
        self.per_layer_projection = nn.Linear(config.emb_dim, config.n_layers * config.ple_dim, bias=False, dtype=config.dtype)
        self.per_layer_norm = RMSNorm(config.ple_dim, eps=1e-6)
        self.layers = nn.ModuleList([
            TransformerBlock(config, attn_type) for attn_type in config.layer_types
        ])
        self.final_norm = RMSNorm(config.emb_dim, eps=1e-6)
        # Projects every token to the output embedding size before pooling (in place of an lm_head)
        self.out_proj = nn.Linear(config.emb_dim, config.out_dim, bias=False, dtype=config.dtype)

        # Precompute RoPE frequencies for both local and global attention (different head dims)
        cos_local, sin_local = RotaryPositionalEmbedding.precompute_freqs_cis(config.head_dim, config.context_length, config.rope_local_base)
        cos_global, sin_global = RotaryPositionalEmbedding.precompute_freqs_cis(config.global_head_dim, config.context_length, config.rope_base)

        # Register as buffers (not parameters, but part of model state)
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_attention_masks(self, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Create bidirectional and sliding window attention masks (True = masked out)"""
        seq_len = attention_mask.shape[1]
        # Global mask: every token sees every non-padding token, in both directions
        mask_global = (attention_mask == 0)[:, None, None, :]  # (batch, 1, 1, seq_len)
        # Local mask: additionally mask tokens further than sliding_window away, in either direction
        positions = torch.arange(seq_len, device=attention_mask.device)
        too_far = (positions[:, None] - positions[None, :]).abs() > self.config.sliding_window
        mask_local = mask_global | too_far
        # Always let a token see itself. Real tokens already can; this stops a padding token with no
        # visible keys from producing NaN, which would otherwise leak into real tokens via 0 * NaN.
        self_attend = torch.eye(seq_len, dtype=torch.bool, device=attention_mask.device)
        return mask_global & ~self_attend, mask_local & ~self_attend

    def encode_tokens(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Per-token output embeddings (batch, seq_len, out_dim), before pooling"""
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # The embedding scale is rounded to the model dtype first (sqrt(512) -> 22.625 in bfloat16)
        x = self.token_embedding(input_ids) * torch.tensor(self.config.emb_dim ** 0.5, dtype=self.token_embedding.weight.dtype)

        # Per-layer embeddings: (batch, seq_len, n_layers, ple_dim)
        per_layer_inputs = self.per_layer_projection(x) * (self.config.emb_dim ** -0.5)
        per_layer_inputs = self.per_layer_norm(per_layer_inputs.view(batch_size, seq_len, self.config.n_layers, self.config.ple_dim))

        mask_global, mask_local = self._create_attention_masks(attention_mask)
        for i, layer in enumerate(self.layers):
            x = layer(x, per_layer_inputs[:, :, i], mask_global, mask_local, self.cos_global, self.sin_global, self.cos_local, self.sin_local)
        x = self.final_norm(x)
        return self.out_proj(x)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None, dim: int = None) -> Tensor:
        """Sentence embeddings (batch, out_dim): mean pooled over non-padding tokens and L2 normalized.

        dim truncates to a Matryoshka size (768, 512, 256 or 128) before re-normalizing.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        token_embeddings = self.encode_tokens(input_ids, attention_mask).float()
        mask = attention_mask.unsqueeze(-1).float()
        embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if dim is not None:
            embeddings = embeddings[:, :dim]
        return F.normalize(embeddings, dim=-1)

    def count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


class EmbeddingGemmaTokenizer:
    """Simple wrapper around HuggingFace tokenizer for EmbeddingGemma 2, with the task prompts it was trained with"""
    TASK_PROMPTS = {
        "search_query": "task: search result | query: ",
        "question_answering": "task: question answering | query: ",
        "fact_checking": "task: fact checking | query: ",
        "code_retrieval": "task: code retrieval | query: ",
        "classification": "task: classification | query: ",
        "clustering": "task: clustering | query: ",
        "sentence_similarity": "task: sentence similarity | query: ",
    }

    def __init__(self, tokenizer_path: str, pad_token_id: int = 0):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = pad_token_id

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids (special tokens such as <bos> are added by the tokenizer itself)"""
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str], max_length: int = 8_192) -> tuple[Tensor, Tensor]:
        """Encode and right-pad a batch, returning (input_ids, attention_mask)"""
        ids = [self.encode(text)[:max_length] for text in texts]
        seq_len = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids), seq_len), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x)
            attention_mask[i, :len(x)] = 1
        return input_ids, attention_mask

    @classmethod
    def apply_query_template(cls, query: str, task: str = "search_query") -> str:
        """Prefix for queries, or for every input in symmetric tasks (classification, clustering, similarity)"""
        return cls.TASK_PROMPTS[task] + query

    @staticmethod
    def apply_document_template(text: str, title: str = None) -> str:
        """Prefix for documents in retrieval tasks"""
        return f"title: {title or 'none'} | text: {text}"


def load_pretrained_weights(model: EmbeddingGemma2Model, weights_dict: dict[str, torch.Tensor]) -> None:
    """Load pretrained weights from HuggingFace format into our model (text encoder only)"""
    def assign_weight(module_param: nn.Parameter, weight_tensor: Tensor, name: str = "unknown") -> nn.Parameter:
        """Helper to assign weights with shape validation"""
        if module_param.shape != weight_tensor.shape:
            raise ValueError(f"Shape mismatch for {name}: expected {module_param.shape}, got {weight_tensor.shape}")
        return nn.Parameter(weight_tensor.clone().detach().to(module_param.dtype))

    # Text weights live under "language_model." (possibly with a "model." prefix); vision/audio weights are ignored
    embed_key = next(k for k in weights_dict if k.endswith("language_model.embed_tokens.weight"))
    root = embed_key[:-len("embed_tokens.weight")]

    # Token embeddings
    model.token_embedding.weight = assign_weight(model.token_embedding.weight, weights_dict[f"{root}embed_tokens.weight"], "token_embedding")

    # Per-layer embedding projection
    model.per_layer_projection.weight = assign_weight(
        model.per_layer_projection.weight,
        weights_dict[f"{root}ple.per_layer_model_projection.weight"],
        "per_layer_projection"
    )
    model.per_layer_norm.scale = assign_weight(
        model.per_layer_norm.scale,
        weights_dict[f"{root}ple.per_layer_projection_norm.weight"],
        "per_layer_norm"
    )

    # Load weights for each transformer layer
    for layer_idx in range(model.config.n_layers):
        layer = model.layers[layer_idx]
        prefix = f"{root}layers.{layer_idx}"

        # Attention weights
        layer.attention.q_proj.weight = assign_weight(
            layer.attention.q_proj.weight,
            weights_dict[f"{prefix}.self_attn.q_proj.weight"],
            f"layer_{layer_idx}_q_proj"
        )
        layer.attention.k_proj.weight = assign_weight(
            layer.attention.k_proj.weight,
            weights_dict[f"{prefix}.self_attn.k_proj.weight"],
            f"layer_{layer_idx}_k_proj"
        )
        layer.attention.v_proj.weight = assign_weight(
            layer.attention.v_proj.weight,
            weights_dict[f"{prefix}.self_attn.v_proj.weight"],
            f"layer_{layer_idx}_v_proj"
        )
        layer.attention.o_proj.weight = assign_weight(
            layer.attention.o_proj.weight,
            weights_dict[f"{prefix}.self_attn.o_proj.weight"],
            f"layer_{layer_idx}_o_proj"
        )

        # QK normalization (v_norm has no weights)
        layer.attention.q_norm.scale = assign_weight(
            layer.attention.q_norm.scale,
            weights_dict[f"{prefix}.self_attn.q_norm.weight"],
            f"layer_{layer_idx}_q_norm"
        )
        layer.attention.k_norm.scale = assign_weight(
            layer.attention.k_norm.scale,
            weights_dict[f"{prefix}.self_attn.k_norm.weight"],
            f"layer_{layer_idx}_k_norm"
        )

        # Feed forward weights
        layer.feed_forward.gate_proj.weight = assign_weight(
            layer.feed_forward.gate_proj.weight,
            weights_dict[f"{prefix}.mlp.gate_proj.weight"],
            f"layer_{layer_idx}_gate_proj"
        )
        layer.feed_forward.up_proj.weight = assign_weight(
            layer.feed_forward.up_proj.weight,
            weights_dict[f"{prefix}.mlp.up_proj.weight"],
            f"layer_{layer_idx}_up_proj"
        )
        layer.feed_forward.down_proj.weight = assign_weight(
            layer.feed_forward.down_proj.weight,
            weights_dict[f"{prefix}.mlp.down_proj.weight"],
            f"layer_{layer_idx}_down_proj"
        )

        # Per-layer embedding weights
        layer.per_layer_embedding.gate_proj.weight = assign_weight(
            layer.per_layer_embedding.gate_proj.weight,
            weights_dict[f"{prefix}.ple_block.per_layer_input_gate.weight"],
            f"layer_{layer_idx}_ple_gate_proj"
        )
        layer.per_layer_embedding.out_proj.weight = assign_weight(
            layer.per_layer_embedding.out_proj.weight,
            weights_dict[f"{prefix}.ple_block.per_layer_projection.weight"],
            f"layer_{layer_idx}_ple_out_proj"
        )
        layer.per_layer_embedding.post_norm.scale = assign_weight(
            layer.per_layer_embedding.post_norm.scale,
            weights_dict[f"{prefix}.ple_block.post_per_layer_input_norm.weight"],
            f"layer_{layer_idx}_ple_post_norm"
        )

        # Layer normalization weights
        layer.input_layernorm.scale = assign_weight(
            layer.input_layernorm.scale,
            weights_dict[f"{prefix}.input_layernorm.weight"],
            f"layer_{layer_idx}_input_layernorm"
        )
        layer.post_attention_layernorm.scale = assign_weight(
            layer.post_attention_layernorm.scale,
            weights_dict[f"{prefix}.post_attention_layernorm.weight"],
            f"layer_{layer_idx}_post_attention_layernorm"
        )
        layer.pre_feedforward_layernorm.scale = assign_weight(
            layer.pre_feedforward_layernorm.scale,
            weights_dict[f"{prefix}.pre_feedforward_layernorm.weight"],
            f"layer_{layer_idx}_pre_feedforward_layernorm"
        )
        layer.post_feedforward_layernorm.scale = assign_weight(
            layer.post_feedforward_layernorm.scale,
            weights_dict[f"{prefix}.post_feedforward_layernorm.weight"],
            f"layer_{layer_idx}_post_feedforward_layernorm"
        )

        # Layer output scale (a buffer, so copied rather than assigned as a Parameter)
        layer.layer_scalar.copy_(weights_dict[f"{prefix}.layer_scalar"])

    # Final layer norm
    model.final_norm.scale = assign_weight(model.final_norm.scale, weights_dict[f"{root}norm.weight"], "final_norm")

    # Output projection
    model.out_proj.weight = assign_weight(model.out_proj.weight, weights_dict[f"{root}embedding_projection.weight"], "out_proj")


def load_text_weights(weights_dir: str) -> dict[str, torch.Tensor]:
    """Read only the text encoder tensors from the checkpoint's safetensors files (skips ~470M vision/audio params)"""
    from safetensors import safe_open
    weights_dict = {}
    for file_name in sorted(os.listdir(weights_dir)):
        if not file_name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(weights_dir, file_name), framework="pt") as f:
            for key in f.keys():
                if "language_model." in key:
                    weights_dict[key] = f.get_tensor(key)
    return weights_dict


def main():
    """Example usage of the EmbeddingGemma 2 model"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="What causes the northern lights?")
    parser.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128])
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(device)
    # bfloat16 where supported, float32 elsewhere (including most CPUs). Never float16.
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    try:
        from huggingface_hub import snapshot_download

        repo_id = "google/embeddinggemma-2"
        local_dir = Path(repo_id).parts[-1]
        snapshot_download(repo_id=repo_id, allow_patterns=["config.json", "*.safetensors", "tokenizer.json"], local_dir=local_dir)

        with open(os.path.join(local_dir, "config.json")) as f:
            config = ModelConfig.from_hf_config(json.load(f), dtype=dtype)

        print("Initialising EmbeddingGemma 2 Model")
        model = EmbeddingGemma2Model(config); print(model)
        print(f"Total parameters: {model.count_parameters():,}")

        weights_dict = load_text_weights(local_dir)
        load_pretrained_weights(model, weights_dict)
        model.to(device).eval()
        del weights_dict

        tokenizer = EmbeddingGemmaTokenizer(os.path.join(local_dir, "tokenizer.json"), pad_token_id=config.pad_token_id)
        print("Model loaded successfully")

        documents = [
            "The northern lights are caused by charged particles from the sun hitting the atmosphere.",
            "Mars is often called the Red Planet because of the iron oxide on its surface.",
            "Pasta should be cooked in plenty of salted, boiling water.",
        ]
        query_ids, query_mask = tokenizer.encode_batch([tokenizer.apply_query_template(args.query)])
        doc_ids, doc_mask = tokenizer.encode_batch([tokenizer.apply_document_template(d) for d in documents])
        with torch.no_grad():
            query_emb = model(query_ids.to(device), query_mask.to(device), dim=args.dim)
            doc_emb = model(doc_ids.to(device), doc_mask.to(device), dim=args.dim)
        scores = (query_emb @ doc_emb.T)[0]

        print(f"\nQuery: {args.query}  (dim={args.dim})")
        for score, doc in sorted(zip(scores.tolist(), documents), reverse=True):
            print(f"  {score:.4f}  {doc}")
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    main()

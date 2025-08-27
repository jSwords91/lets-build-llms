import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

Tensor = torch.Tensor

@dataclass
class ModelConfig:
    """Gemma 3 model configuration"""
    vocab_size: int = 262_144
    context_length: int = 32_768
    emb_dim: int = 640
    n_heads: int = 4
    n_layers: int = 18
    hidden_dim: int = 2048
    head_dim: int = 256
    qk_norm: bool = True
    n_kv_groups: int = 1
    rope_local_base: float = 10_000.0
    rope_base: float = 1_000_000.0
    sliding_window: int = 512
    layer_types: list[str] = None
    dtype: torch.dtype = torch.bfloat16
    query_pre_attn_scalar: int = 256

    def __post_init__(self):
        if self.layer_types is None:
            # Default Gemma 3 270M layer configuration
            self.layer_types = [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ]
        assert len(self.layer_types) == self.n_layers, f"layer_types length ({len(self.layer_types)}) must match n_layers ({self.n_layers})"


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Gemma 3 variant with zero-centered weights)"""
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        # Gemma 3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        # Compute norm in float32 for stability, then scale by (1 + weight)
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(input_dtype)


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
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        # Apply rotation: x_rotated = x * cos + rotate(x) * sin
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)
        return x_rotated.to(x.dtype)


class FeedForward(nn.Module):
    """SwiGLU Feed Forward Network"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with optional QK normalization"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"
        self.num_heads = config.n_heads
        self.num_kv_groups = config.n_kv_groups
        self.group_size = config.n_heads // config.n_kv_groups
        self.head_dim = config.head_dim
        self.d_out = config.n_heads * config.head_dim

        # Projections
        self.q_proj = nn.Linear(config.emb_dim, self.d_out, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(config.emb_dim, config.n_kv_groups * config.head_dim, bias=False, dtype=config.dtype)
        self.o_proj = nn.Linear(self.d_out, config.emb_dim, bias=False, dtype=config.dtype)

        # Optional QK normalization (used in Gemma 3)
        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(config.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        # Scaling factor
        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar ** -0.5
        else:
            self.scaling = config.head_dim ** -0.5

    def forward(self, x: Tensor, mask: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        # Apply projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # Optional QK normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        # Apply RoPE
        q = RotaryPositionalEmbedding.apply_rotary_emb(q, cos, sin)
        k = RotaryPositionalEmbedding.apply_rotary_emb(k, cos, sin)
        # Expand K,V to match number of query heads (for grouped query attention)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        # Scaled dot-product attention
        q = q * self.scaling
        scores = q @ k.transpose(-2, -1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_out)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre/post layer norms as in Gemma 3"""
    def __init__(self, config: ModelConfig, attn_type: str):
        super().__init__()
        self.attn_type = attn_type
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        # Layer normalizations (Gemma 3 uses pre and post norms)
        self.input_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(config.emb_dim, eps=1e-6)

    def forward(self, x: Tensor, mask_global: Tensor, mask_local: Tensor, cos_global: Tensor, sin_global: Tensor, cos_local: Tensor, sin_local: Tensor) -> Tensor:
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
        return x


class Gemma3Model(nn.Module):
    """Gemma 3 Model with sliding window attention"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(config, attn_type) for attn_type in config.layer_types
        ])
        self.final_norm = RMSNorm(config.emb_dim, eps=1e-6)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

        # Precompute RoPE frequencies for both local and global attention
        cos_local, sin_local = RotaryPositionalEmbedding.precompute_freqs_cis(config.head_dim, config.context_length, config.rope_local_base)
        cos_global, sin_global = RotaryPositionalEmbedding.precompute_freqs_cis(config.head_dim, config.context_length, config.rope_base)

        # Register as buffers (not parameters, but part of model state)
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_attention_masks(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Create causal and sliding window attention masks"""
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        # Global mask: standard causal mask (mask future tokens)
        mask_global = torch.triu(ones, diagonal=1)
        # Local mask: sliding window + causal
        # Mask tokens that are too far in the past (beyond sliding window)
        mask_far_past = torch.triu(ones, diagonal=self.config.sliding_window).T
        mask_local = mask_global | mask_far_past
        return mask_global, mask_local

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) * (self.config.emb_dim ** 0.5)
        mask_global, mask_local = self._create_attention_masks(seq_len, x.device)
        for layer in self.layers:
            x = layer(x, mask_global, mask_local, self.cos_global, self.sin_global, self.cos_local, self.sin_local)
        x = self.final_norm(x)
        logits = self.lm_head(x.to(self.config.dtype))
        return logits

    def count_parameters(self) -> tuple[int, int]:
        """Count total and unique parameters (accounting for weight tying)"""
        total = sum(p.numel() for p in self.parameters())
        unique = total - self.token_embedding.weight.numel()
        return total, unique


class GemmaTokenizer:
    """Simple wrapper around HuggingFace tokenizer for Gemma 3"""
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_token = "<end_of_turn>"
        self.pad_token = self.eos_token

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids"""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def apply_chat_template(user_message: str) -> str:
        """Apply Gemma 3 chat template"""
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def load_pretrained_weights(model: Gemma3Model, weights_dict: dict[str, torch.Tensor]) -> None:
    """Load pretrained weights from HuggingFace format into our model"""
    def assign_weight(module_param: nn.Parameter, weight_tensor: Tensor, name: str = "unknown") -> nn.Parameter:
        """Helper to assign weights with shape validation"""
        if module_param.shape != weight_tensor.shape:
            raise ValueError(f"Shape mismatch for {name}: expected {module_param.shape}, got {weight_tensor.shape}")
        return nn.Parameter(weight_tensor.clone().detach())

    # Token embeddings
    if "model.embed_tokens.weight" in weights_dict:
        model.token_embedding.weight = assign_weight(
            model.token_embedding.weight, 
            weights_dict["model.embed_tokens.weight"], 
            "token_embedding"
        )

    # Load weights for each transformer layer
    for layer_idx in range(model.config.n_layers):
        layer = model.layers[layer_idx]
        prefix = f"model.layers.{layer_idx}"

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

        # QK normalization (if present)
        if layer.attention.q_norm is not None:
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

        # Pre/post feedforward norms (if present)
        pre_ff_key = f"{prefix}.pre_feedforward_layernorm.weight"
        post_ff_key = f"{prefix}.post_feedforward_layernorm.weight"

        if pre_ff_key in weights_dict:
            layer.pre_feedforward_layernorm.scale = assign_weight(
                layer.pre_feedforward_layernorm.scale,
                weights_dict[pre_ff_key],
                f"layer_{layer_idx}_pre_feedforward_layernorm"
            )
        if post_ff_key in weights_dict:
            layer.post_feedforward_layernorm.scale = assign_weight(
                layer.post_feedforward_layernorm.scale,
                weights_dict[post_ff_key],
                f"layer_{layer_idx}_post_feedforward_layernorm"
            )

    # Final layer norm
    if "model.norm.weight" in weights_dict:
        model.final_norm.scale = assign_weight(
            model.final_norm.scale,
            weights_dict["model.norm.weight"],
            "final_norm"
        )

    # Output projection (with potential weight tying)
    if "lm_head.weight" in weights_dict:
        model.lm_head.weight = assign_weight(
            model.lm_head.weight,
            weights_dict["lm_head.weight"],
            "lm_head"
        )
    elif "model.embed_tokens.weight" in weights_dict:
        # Weight tying: share embedding weights with output layer
        model.lm_head.weight = model.token_embedding.weight


def generate_text_stream(model: Gemma3Model, tokenizer: GemmaTokenizer, prompt: str, max_new_tokens: int = 500, device: str = "cuda", temperature: float = 1.0) -> str:
    """Generate text with streaming output"""
    input_text = GemmaTokenizer.apply_chat_template(prompt)
    input_ids = torch.tensor(tokenizer.encode(input_text), device=device).unsqueeze(0)
    eos_token_id = tokenizer.encode("<end_of_turn>")[-1]

    model.eval()
    generated_text = ""
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits.to(torch.float32)
            if temperature != 1.0:
                logits = logits / temperature
            next_token_id = torch.multinomial(torch.softmax(logits[0, -1], dim=-1), 1)
            next_token_id = next_token_id.unsqueeze(0)
            next_token_id = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            if next_token_id.item() == eos_token_id:
                break

            token_text = tokenizer.decode([next_token_id.item()])
            generated_text += token_text
            print(token_text, end="", flush=True)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    return generated_text

def main():
    """Example usage of the Gemma 3 model"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Explain the Mixture of Experts architecture and why they are used in large language models.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")

    config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(device)

    print("Initialising Gemma 3 Model")
    model = Gemma3Model(config).to(device); print(model)
    total_params, unique_params = model.count_parameters()
    print(f"Total parameters: {total_params:,} || Unique parameters: {unique_params:,}")

    try:
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download, snapshot_download


        repo_id = f"google/gemma-3-270m-it"
    
        local_dir = Path(repo_id).parts[-1]

        weights_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=local_dir)
        weights_dict = load_file(weights_file)
        load_pretrained_weights(model, weights_dict)
        model.to(device)
        del weights_dict

        tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file_path):
            try:
                tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
            except Exception as e:
                print(f"Warning: failed to download tokenizer.json: {e}")
                tokenizer_file_path = "tokenizer.json"

        tokenizer = GemmaTokenizer(tokenizer_file_path)

        print("Model loaded successfully")
        print(f"\nUser: {args.prompt}")
        print("Model: ", end="")
        generate_text_stream(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=device, temperature=args.temperature)
        print("\n")
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    main()

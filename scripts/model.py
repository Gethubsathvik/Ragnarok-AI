"""GPT-style decoder-only Transformer (12 layers / 768 dim / 12 heads ~ 124M params).

Training uses the full-sequence forward (optionally with gradient checkpointing).
Inference uses `forward_cached` (KV-cache): one new token per step instead of re-running the context.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if past is not None:                      # decoding: one new token attends to all cached keys
            assert T == 1, "cached decoding processes one token at a time"
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:                                     # training / prefill: fused causal attention
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return (y, (k, v)) if use_cache else y


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = FeedForward(cfg)

    def forward(self, x, past=None, use_cache=False):
        if use_cache:
            a, kv = self.attn(self.ln1(x), past, use_cache=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), kv
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                 # weight tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():               # GPT-2 residual scaling
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)   # recompute activations in backward
            else:
                x = blk(x)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None              # inference: last position only

    @torch.no_grad()
    def forward_cached(self, idx, past=None):
        """KV-cache forward for generation (eval mode).

        idx  : (B, T) tokens. Prefill: past=None, any T <= block_size. Decode: past given, T == 1.
        Returns (logits for the last position, shape (B, vocab), new_past).
        past is a list (one (k, v) pair per layer); its length is past[0][0].size(2).
        """
        B, T = idx.shape
        start = 0 if past is None else past[0][0].size(2)
        assert start + T <= self.cfg.block_size, "context window exceeded - re-prefill a shorter window"
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        new_past = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, None if past is None else past[i], use_cache=True)
            new_past.append(kv)
        return self.lm_head(self.ln_f(x[:, -1])), new_past

    def num_params(self):
        return sum(p.numel() for p in self.parameters())      # tied weights counted once

    def configure_optimizer(self, lr, weight_decay=0.1, betas=(0.9, 0.95), device_type="cuda"):
        """Adam with decoupled weight decay (AdamW). No decay on biases / LayerNorm."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    def flops_per_token(self):
        """Approx. training FLOPs/token (PaLM appendix B): 6N + 12*L*H*Q*T."""
        c = self.cfg
        n = self.num_params() - self.wpe.weight.numel()
        return 6 * n + 12 * c.n_layer * c.n_embd * c.block_size


def print_summary(model: GPT):
    c = model.cfg
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    rows = [
        ("Token embedding (tied with output head)", model.wte.weight.numel()),
        ("Position embedding", model.wpe.weight.numel()),
        (f"Transformer blocks  x{c.n_layer}", per_block * c.n_layer),
        ("  - per block: attention", sum(p.numel() for p in model.blocks[0].attn.parameters())),
        ("  - per block: feed-forward", sum(p.numel() for p in model.blocks[0].mlp.parameters())),
        ("  - per block: 2 x LayerNorm", sum(p.numel() for n in (model.blocks[0].ln1, model.blocks[0].ln2)
                                              for p in n.parameters())),
        ("Final LayerNorm", sum(p.numel() for p in model.ln_f.parameters())),
    ]
    print("=" * 66)
    print(f"GPT  layers={c.n_layer}  hidden={c.n_embd}  heads={c.n_head}  "
          f"context={c.block_size}  vocab={c.vocab_size}")
    print("=" * 66)
    for name, n in rows:
        print(f"{name:<46}{n:>16,}")
    print("-" * 66)
    total = model.num_params()
    print(f"{'TOTAL PARAMETERS':<46}{total:>16,}  ({total / 1e6:.1f}M)")
    print(f"{'fp32 weights':<46}{total * 4 / 1024**2:>13.0f} MB")
    print(f"{'+ AdamW states + grads (fp32)':<46}{total * 12 / 1024**2:>13.0f} MB")
    print("=" * 66)
    print("\nOne block:\n", model.blocks[0])

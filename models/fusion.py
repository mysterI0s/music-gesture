"""Audio-visual fusion module.

Two fusion schemes are selectable via ``fusion_mode``:

* ``transformer`` -- the repo's efficient default. Audio bottleneck tokens are
  concatenated with per-source gesture tokens and passed through a stack of
  self-attention Transformer encoder layers; the refined audio tokens are
  returned.

* ``paper_attn`` -- the Music Gesture cross-modal attention (Eq. 2-3). The
  *sound* feature is the query; the *visual* (gesture) features are the keys /
  values. Softmax attention over the visual tokens yields an attended visual
  context which is concatenated with the sound feature and passed through a
  small MLP, added back to the sound feature as a residual. Stacked ``depth``
  times.

Note on the paper's equations: the OCR of Eq. (2)-(3) swaps the s/v subscripts
in places; we follow the textual description "the sound feature attends to the
visual features", i.e. query = sound, key/value = visual.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalBlock(nn.Module):
    """Paper-faithful cross-modal attention: sound queries attend to visual.

    query  = sound (audio) tokens
    key/val = visual (gesture) tokens
    The attended visual context is concatenated with the sound feature and
    projected by a 2-layer MLP, then added back to the sound feature.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        # MLP consumes concat[sound, attended_visual] (2*dim) -> dim.
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(audio)
        kv = self.norm_kv(visual)
        # nn.MultiheadAttention softmaxes over the key (visual) axis.
        attended, _ = self.attn(q, kv, kv, need_weights=False)  # [B, Na, D]
        fused = torch.cat([audio, attended], dim=-1)             # [B, Na, 2D]
        return audio + self.mlp(fused)                           # residual on sound


class AudioVisualFusion(nn.Module):
    def __init__(self, dim: int = 512, depth: int = 3, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 fusion_mode: str = "transformer"):
        super().__init__()
        if fusion_mode not in ("transformer", "paper_attn"):
            raise ValueError(
                f"fusion_mode must be 'transformer' or 'paper_attn', got {fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode
        self.audio_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, dim))
        if fusion_mode == "paper_attn":
            self.blocks = nn.ModuleList(
                [CrossModalBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio_tokens: torch.Tensor,
                visual_tokens: torch.Tensor) -> torch.Tensor:
        """audio_tokens: [B, Na, D] ; visual_tokens: [B, Nv, D] -> [B, Na, D]."""
        na = audio_tokens.shape[1]
        a = audio_tokens + self.audio_type
        v = visual_tokens + self.visual_type
        if self.fusion_mode == "paper_attn":
            for block in self.blocks:
                a = block(a, v)
            return self.norm(a)
        x = torch.cat([a, v], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, :na]  # refined audio tokens only

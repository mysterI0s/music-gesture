"""Context-aware Graph CNN (CT-GCN) over body + hand keypoints.

Spatial-temporal graph convolution (ST-GCN style) over the human skeleton
(body 18 + 2x hand 21 = 60 joints), conditioned on the semantic context feature.

Two context-conditioning modes are supported (``context_mode``):

* ``film``  -- feature-wise linear modulation applied after *every* graph block
               (the repo's original, efficient default).
* ``concat``-- the paper-faithful scheme: the semantic context vector is
               broadcast over the (time, joint) grid and concatenated onto the
               input node features once, so downstream layers get raw access to
               the context channels (Music Gesture, Sec. 3.1: "we concatenated
               the visual appearance context features to each ... node feature").

The output is a temporal sequence of gesture tokens used by the audio-visual
fusion module.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.pose import build_skeleton_adjacency


class GraphConv(nn.Module):
    """Spatial graph convolution: aggregates over normalized adjacency A."""

    def __init__(self, in_c: int, out_c: int, num_subsets: int):
        super().__init__()
        self.num_subsets = num_subsets
        self.conv = nn.Conv2d(in_c, out_c * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V] ; A: [num_subsets, V, V]
        b, c, t, v = x.shape
        x = self.conv(x)
        x = x.view(b, self.num_subsets, -1, t, v)
        # einsum over the partition subsets and joints
        x = torch.einsum("bkctv,kvw->bctw", x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_subsets: int,
                 temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, num_subsets)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (temporal_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout, inplace=True),
        )
        if in_c == out_c and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, (stride, 1)),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class FiLM(nn.Module):
    """Feature-wise linear modulation from the semantic context vector."""

    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V] ; context: [B, context_dim]
        gamma = self.to_gamma(context)[:, :, None, None]
        beta = self.to_beta(context)[:, :, None, None]
        return (1 + gamma) * x + beta


class ContextAwareGraphCNN(nn.Module):
    def __init__(self, in_channels: int = 3, graph_layers=(64, 64, 64, 128, 128, 256),
                 temporal_kernel: int = 9, context_dim: int = 512,
                 embed_dim: int = 512, dropout: float = 0.1,
                 body_joints: int = 18, hand_joints: int = 21,
                 context_mode: str = "film", stride_layers=(2, 4)):
        super().__init__()
        if context_mode not in ("film", "concat"):
            raise ValueError(f"context_mode must be 'film' or 'concat', got {context_mode!r}")
        self.context_mode = context_mode
        self.stride_layers = tuple(stride_layers)
        A = build_skeleton_adjacency(body_joints, hand_joints)
        self.register_buffer("A", A)
        num_subsets = A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
        # In 'concat' mode the context vector is concatenated onto the input node
        # channels once (paper-faithful); the first graph block absorbs the extra
        # context_dim channels. In 'film' mode the input is just the keypoints.
        first_in = in_channels + (context_dim if context_mode == "concat" else 0)
        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        c_prev = first_in
        for i, c in enumerate(graph_layers):
            stride = 2 if (i in self.stride_layers) else 1
            self.blocks.append(
                STGCNBlock(c_prev, c, num_subsets, temporal_kernel, stride, dropout)
            )
            if context_mode == "film":
                # One FiLM per block; unused (and not created) in concat mode.
                self.films.append(FiLM(context_dim, c))
            c_prev = c
        self.to_tokens = nn.Conv2d(c_prev, embed_dim, kernel_size=1)

    def forward(self, keypoints: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """keypoints: [B, C, T, V] ; context: [B, context_dim].

        Returns gesture tokens [B, T', embed_dim] (mean-pooled over joints).
        """
        b, c, t, v = keypoints.shape
        x = keypoints.permute(0, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x).view(b, v, c, t).permute(0, 2, 3, 1).contiguous()
        if self.context_mode == "concat":
            # Broadcast context [B, Cctx] over T and V, concat on the channel axis
            # of the input node features (paper Sec. 3.1). Injected once, so the
            # context is not double-counted per block as it would be with FiLM.
            ctx = context[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, ctx], dim=1)
            for block in self.blocks:
                x = block(x, self.A)
        else:
            for block, film in zip(self.blocks, self.films):
                x = block(x, self.A)
                x = film(x, context)
        x = self.to_tokens(x)                 # [B, D, T', V]
        tokens = x.mean(dim=3).transpose(1, 2)  # [B, T', D]
        return tokens

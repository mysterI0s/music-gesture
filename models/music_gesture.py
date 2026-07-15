"""Full Music Gesture model.

Given a mixture spectrogram and, for each source, that source's keypoints and a
context crop, predict one separation mask per source.

Architectural variants (context conditioning, U-Net depth/kernel, fusion
scheme, GCN depth/strides, context width) are all driven by the config so the
same class can instantiate either the repo's efficient default or the
paper-faithful configuration (see configs/paper_faithful.yaml).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_net import AudioUNet
from .context_net import ContextNet
from .pose_net import ContextAwareGraphCNN
from .fusion import AudioVisualFusion
from .synthesizer import MaskHead


class MusicGesture(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        dim = m["fusion"]["dim"]
        a = m["audio"]

        self.audio_net = AudioUNet(
            ngf=a["ngf"], num_downs=a["num_downs"],
            input_nc=a["input_nc"], output_nc=a["output_nc"],
            bottleneck_dim=dim,
            conv_kernel=a.get("conv_kernel", 4), up_kernel=a.get("up_kernel", 4),
            dilation=a.get("dilation", 1),
        )
        self.context_net = ContextNet(
            backbone=m["context"]["backbone"], pretrained=m["context"]["pretrained"],
            feat_dim=m["context"]["feat_dim"],
        )
        self.pose_net = ContextAwareGraphCNN(
            in_channels=m["pose"]["in_channels"], graph_layers=tuple(m["pose"]["graph_layers"]),
            temporal_kernel=m["pose"]["temporal_kernel"], context_dim=m["context"]["feat_dim"],
            embed_dim=m["pose"]["embed_dim"], dropout=m["pose"]["dropout"],
            body_joints=cfg["video"]["body_joints"], hand_joints=cfg["video"]["hand_joints"],
            context_mode=m["pose"].get("context_mode", "film"),
            stride_layers=tuple(m["pose"].get("stride_layers", (2, 4))),
            context_inject_after=m["pose"].get("context_inject_after", 1),
            context_proj_dim=m["pose"].get("context_proj_dim", 0),
        )
        self.fusion = AudioVisualFusion(
            dim=dim, depth=m["fusion"]["depth"], heads=m["fusion"]["heads"],
            mlp_ratio=m["fusion"]["mlp_ratio"], dropout=m["fusion"]["dropout"],
            fusion_mode=m["fusion"].get("mode", "transformer"),
        )
        self.mask_head = MaskHead(mask_type=cfg["audio"]["mask_type"])
        # Vision-gated output head (Sound-of-Pixels / Music Gesture synthesizer).
        # The decoder emits `synth_channels` feature maps; the per-source visual
        # embedding is projected to per-channel weights that combine them into the
        # mask. This gives the visual signal a DIRECT, per-pixel path to the
        # output instead of only entering the (skip-bypassed) bottleneck.
        self.synth_channels = a["output_nc"]
        self.vis_gate_w = nn.Linear(dim, self.synth_channels)
        self.vis_gate_b = nn.Linear(dim, 1)

    def separate_one(self, spec: torch.Tensor, keypoints: torch.Tensor,
                     context_frame: torch.Tensor) -> torch.Tensor:
        """Predict a single source mask.

        spec:          [B, 1, F, T]
        keypoints:     [B, C, Tk, V]
        context_frame: [B, 3, H, W]
        """
        # The U-Net down/up-samples by 2**num_downs; pad F/T up to a multiple of
        # that so encoder skips and decoder feature maps align, then crop back.
        _, _, Fdim, Tdim = spec.shape
        m = 2 ** self.audio_net.num_downs
        pad_f = (m - Fdim % m) % m
        pad_t = (m - Tdim % m) % m
        if pad_f or pad_t:
            spec = F.pad(spec, (0, pad_t, 0, pad_f))
        audio_tokens, hw, skips = self.audio_net.encode(spec)
        context = self.context_net(context_frame)
        gesture_tokens = self.pose_net(keypoints, context)   # [B, T', D]
        fused = self.fusion(audio_tokens, gesture_tokens)    # paper cross-modal attn
        featmap = self.audio_net.decode(fused, hw, skips)    # [B, K, F, T]
        featmap = featmap[..., :Fdim, :Tdim]  # crop back to the true spectrogram size
        # Vision-gated combination: pool the gesture tokens into one per-source
        # visual vector and use it to weight the K synthesizer channels per pixel.
        vis_vec = gesture_tokens.mean(dim=1)                 # [B, D]
        w = self.vis_gate_w(vis_vec)                          # [B, K]
        b = self.vis_gate_b(vis_vec)                          # [B, 1]
        logits = torch.einsum("bkft,bk->bft", featmap, w) + b.unsqueeze(-1)  # [B, F, T]
        logits = logits.unsqueeze(1)                          # [B, 1, F, T]
        return self.mask_head(logits)

    def forward(self, mixture_spec: torch.Tensor,
                keypoints: List[torch.Tensor],
                context_frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return one mask per source."""
        masks = []
        for kp, ctx in zip(keypoints, context_frames):
            masks.append(self.separate_one(mixture_spec, kp, ctx))
        return masks

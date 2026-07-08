"""Full Music Gesture model.
​
Given a mixture spectrogram and, for each source, that source's keypoints and a
context crop, predict one separation mask per source.
"""
from __future__ import annotations
​
from typing import List
​
import torch
import torch.nn as nn
import torch.nn.functional as F
​
from .audio_net import AudioUNet
from .context_net import ContextNet
from .pose_net import ContextAwareGraphCNN
from .fusion import AudioVisualFusion
from .synthesizer import MaskHead
​
​
class MusicGesture(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        dim = m["fusion"]["dim"]
​
        self.audio_net = AudioUNet(
            ngf=m["audio"]["ngf"], num_downs=m["audio"]["num_downs"],
            input_nc=m["audio"]["input_nc"], output_nc=m["audio"]["output_nc"],
            bottleneck_dim=dim,
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
        )
        self.fusion = AudioVisualFusion(
            dim=dim, depth=m["fusion"]["depth"], heads=m["fusion"]["heads"],
            mlp_ratio=m["fusion"]["mlp_ratio"], dropout=m["fusion"]["dropout"],
        )
        self.mask_head = MaskHead(mask_type=cfg["audio"]["mask_type"])
​
    def separate_one(self, spec: torch.Tensor, keypoints: torch.Tensor,
                     context_frame: torch.Tensor) -> torch.Tensor:
        """Predict a single source mask.
​
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
        gesture_tokens = self.pose_net(keypoints, context)
        fused = self.fusion(audio_tokens, gesture_tokens)
        logits = self.audio_net.decode(fused, hw, skips)
        logits = logits[..., :Fdim, :Tdim]  # crop back to the true spectrogram size
        return self.mask_head(logits)
​
    def forward(self, mixture_spec: torch.Tensor,
                keypoints: List[torch.Tensor],
                context_frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return one mask per source."""
        masks = []
        for kp, ctx in zip(keypoints, context_frames):
            masks.append(self.separate_one(mixture_spec, kp, ctx))
        return masks
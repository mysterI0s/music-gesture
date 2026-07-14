"""Semantic context extractor.

The paper conditions the gesture graph network on a semantic *context* feature
of each musician so the model knows *which* instrument the keypoints belong to.
We use a ResNet-50 backbone over a cropped frame of each musician.

The paper keeps the raw 2048-d ResNet-50 pooled feature. Set ``feat_dim`` equal
to the backbone's pooled dimension (2048 for ResNet-50) to propagate it
unprojected (paper-faithful); any smaller ``feat_dim`` inserts a linear
projection to that width (the efficient default used elsewhere in the repo).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ContextNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 feat_dim: int = 512):
        super().__init__()
        net = getattr(torchvision.models, backbone)(pretrained=pretrained)
        modules = list(net.children())[:-1]  # drop the classifier fc
        self.backbone = nn.Sequential(*modules)
        in_dim = net.fc.in_features
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        # Paper-faithful: when feat_dim == the backbone's pooled width (2048 for
        # ResNet-50) keep the feature unprojected; otherwise project to feat_dim.
        if feat_dim == in_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, feat_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, 3, H, W] crop of the musician -> [B, feat_dim]."""
        feat = self.backbone(frames).flatten(1)
        return self.proj(feat)

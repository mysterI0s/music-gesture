"""Skeleton graph construction for body (COCO-18) + two hands (21 each).

Builds the normalized adjacency tensor A of shape [num_subsets, V, V] used by the
spatial graph convolution. We use the standard 3-subset spatial partitioning
(self, centripetal, centrifugal) from ST-GCN.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

# COCO-18 body edges (OpenPose ordering).
BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (0, 15),
    (14, 16), (15, 17),
]

# Hand edges (21 keypoints): wrist -> 5 fingers of 4 joints each.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Which body joints the two hand-roots (wrists) attach to (right wrist=4, left=7).
RIGHT_WRIST = 4
LEFT_WRIST = 7


def _edges(body_joints: int, hand_joints: int) -> Tuple[int, List[Tuple[int, int]]]:
    edges = list(BODY_EDGES)
    v = body_joints
    # right hand block
    r0 = v
    edges += [(a + r0, b + r0) for a, b in HAND_EDGES]
    edges.append((RIGHT_WRIST, r0))
    v += hand_joints
    # left hand block
    l0 = v
    edges += [(a + l0, b + l0) for a, b in HAND_EDGES]
    edges.append((LEFT_WRIST, l0))
    v += hand_joints
    return v, edges


def _normalize(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(0)
    deg_inv = np.zeros_like(deg)
    deg_inv[deg > 0] = deg[deg > 0] ** -1
    return adj * deg_inv[None, :]


def build_skeleton_adjacency(body_joints: int = 18,
                             hand_joints: int = 21) -> torch.Tensor:
    v, edges = _edges(body_joints, hand_joints)
    adj = np.zeros((v, v), dtype=np.float32)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    identity = np.eye(v, dtype=np.float32)

    # distance-1 neighborhood partitioning into self / inward / outward.
    center = 1  # neck as skeleton center
    dist = _bfs_distance(adj, center)
    inward = np.zeros((v, v), dtype=np.float32)
    outward = np.zeros((v, v), dtype=np.float32)
    for i in range(v):
        for j in range(v):
            if adj[i, j] > 0:
                if dist[j] < dist[i]:
                    inward[i, j] = 1.0
                elif dist[j] > dist[i]:
                    outward[i, j] = 1.0
    A = np.stack([identity, _normalize(inward), _normalize(outward)], axis=0)
    return torch.from_numpy(A)


def _bfs_distance(adj: np.ndarray, root: int) -> np.ndarray:
    v = adj.shape[0]
    dist = np.full(v, np.inf)
    dist[root] = 0
    frontier = [root]
    while frontier:
        nxt = []
        for node in frontier:
            for j in range(v):
                if adj[node, j] > 0 and dist[j] == np.inf:
                    dist[j] = dist[node] + 1
                    nxt.append(j)
        frontier = nxt
    dist[dist == np.inf] = dist[dist != np.inf].max() + 1
    return dist


def normalize_keypoints(kp: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Normalize (x, y) to [-1, 1] and keep confidence as the 3rd channel.

    kp: [T, V, 3] -> [3, T, V]
    """
    out = kp.astype(np.float32).copy()
    out[..., 0] = out[..., 0] / frame_w * 2 - 1
    out[..., 1] = out[..., 1] / frame_h * 2 - 1
    return np.transpose(out, (2, 0, 1))


def augment_keypoints(kp: np.ndarray, translate: float = 0.05,
                      scale: float = 0.1, rotate_deg: float = 10.0,
                      rng=None) -> np.ndarray:
    """Random rigid + scale jitter of the (x, y) coordinates for training.

    Paper Sec. 4 (implementation details) applies keypoint augmentation. The
    same random rotation / scale / translation is applied to every frame of a
    clip (temporally consistent) and only to channels 0-1; the confidence
    channel is left untouched.

    kp: [3, T, V] already normalized to [-1, 1] (output of normalize_keypoints).
    """
    if rng is None:
        rng = np.random
    out = kp.astype(np.float32).copy()
    x = out[0]
    y = out[1]
    theta = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = cos_t * x - sin_t * y
    yr = sin_t * x + cos_t * y
    s = 1.0 + rng.uniform(-scale, scale)
    xr *= s
    yr *= s
    xr += rng.uniform(-translate, translate)
    yr += rng.uniform(-translate, translate)
    out[0] = xr
    out[1] = yr
    return out

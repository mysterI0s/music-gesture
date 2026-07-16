"""Train Music Gesture with Mix-and-Separate self-supervision.

Supports both the repo's efficient defaults and the paper-faithful recipe via
config flags:

* optimizer:   Adam (default) or SGD momentum 0.9 with per-module LR groups
               (paper: 1e-2 audio+fusion, 1e-3 gcn+appearance).
* loss_mode:   energy-weighted BCE (default) or plain per-pixel BCE (paper).
* mask_target: dominant-source IBM (default) or literal S_k >= S_mix (paper).
* stages:      an optional list of curriculum stages (paper's two-stage
               hetero-musical pretrain -> homo-musical finetune). Each stage
               may override the mix policy, epoch count and LR scale, and is
               initialised from the previous stage's weights.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.music_dataset import MusicMixDataset, collate
from models import MusicGesture
from models.synthesizer import apply_mask
from utils.audio import ideal_ratio_mask, ideal_binary_mask, ideal_binary_mask_literal


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_targets(batch, cfg):
    mask_type = cfg["audio"]["mask_type"]
    mask_target = cfg["audio"].get("mask_target", "dominant")
    mix = batch["mixture_mag"]
    src_mags = batch["source_mags"]
    targets = []
    for i, src in enumerate(src_mags):
        if mask_type == "ratio":
            targets.append(ideal_ratio_mask(src, mix).clamp(0, 1))
        elif mask_target == "literal_mix":
            # Literal Music Gesture Eq. 4: this source is >= the mixture itself.
            targets.append(ideal_binary_mask_literal(src, mix))
        else:
            # Ideal binary mask (dominant source): 1 where this source is the
            # loudest at that time-frequency bin. Compare against the per-bin
            # max of the *other* sources so the targets are balanced ~50/50 and
            # a constant prediction can no longer win.
            other = None
            for j, s in enumerate(src_mags):
                if j == i:
                    continue
                other = s if other is None else torch.maximum(other, s)
            targets.append(ideal_binary_mask(src, other))
    return targets


def plain_bce(masks, targets):
    """Uniform per-pixel binary cross-entropy over *all* time-frequency bins.

    This is the paper's plain BCE objective. The mask head already applies a
    sigmoid, so the predictions are probabilities and we use
    ``F.binary_cross_entropy`` (not the with-logits variant).
    """
    total = 0.0
    for m, t in zip(masks, targets):
        total = total + F.binary_cross_entropy(m, t)
    return total / len(masks)


def energy_weighted_bce(masks, targets, energy, floor):
    """Per-pixel BCE restricted to time-frequency bins that carry energy.

    ``energy`` is the (log-freq) mixture magnitude [B, 1, F, T]. Bins below
    ``floor`` times each sample's peak magnitude are near-silent: their ideal
    binary label is essentially a coin flip and, because they dominate the bin
    count, they let the model minimise BCE by predicting a constant ~0.5. Zeroing
    their weight makes the gradient come from the informative, energetic bins.
    """
    b = energy.shape[0]
    peak = energy.reshape(b, -1).amax(dim=1).clamp_min(1e-8).reshape(b, 1, 1, 1)
    weight = (energy >= floor * peak).float()
    denom = weight.sum().clamp_min(1.0)
    total = 0.0
    for m, t in zip(masks, targets):
        bce = F.binary_cross_entropy(m, t, reduction="none")
        total = total + (bce * weight).sum() / denom
    return total / len(masks)


def build_optimizer(model, cfg):
    """Build the optimizer per config.

    Adam (default): base LR for everything except the ResNet context backbone,
    which gets ``lr_backbone``.

    SGD (paper): momentum 0.9 with two LR groups -- a higher LR on the
    audio-separation U-Net + fusion transformer, a lower LR on the ST-GCN pose
    net + appearance (context) net (and the mask head).
    """
    tcfg = cfg["train"]
    opt_name = tcfg.get("optimizer", "adam").lower()
    wd = tcfg.get("weight_decay", 0.0)
    if opt_name == "sgd":
        groups = tcfg.get("param_groups", {}) or {}
        lr_af = groups.get("audio_fusion", 1e-2)
        lr_ga = groups.get("gcn_appearance", 1e-3)
        audio_fusion, gcn_appearance = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("audio_net.") or name.startswith("fusion."):
                audio_fusion.append(p)
            else:  # pose_net (GCN) + context_net (appearance) + mask_head
                gcn_appearance.append(p)
        params = [
            {"params": audio_fusion, "lr": lr_af},
            {"params": gcn_appearance, "lr": lr_ga},
        ]
        momentum = tcfg.get("momentum", 0.9)
        print(f"[optim] SGD(momentum={momentum}) "
              f"audio+fusion: {len(audio_fusion)} params @ lr {lr_af}; "
              f"gcn+appearance: {len(gcn_appearance)} params @ lr {lr_ga}")
        return torch.optim.SGD(params, momentum=momentum, weight_decay=wd)

    backbone_ids = set(id(p) for p in model.context_net.parameters())
    params = [
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
         "lr": tcfg["lr"]},
        {"params": list(model.context_net.parameters()),
         "lr": tcfg["lr_backbone"]},
    ]
    print(f"[optim] Adam base lr {tcfg['lr']}, backbone lr {tcfg['lr_backbone']}")
    return torch.optim.Adam(params, weight_decay=wd)


def _wrap_data_parallel(model, cfg, device):
    """Optionally wrap ``model`` in ``nn.DataParallel`` for multi-GPU training.

    Enabled only when ``train.data_parallel`` is true, CUDA is available, and
    more than one GPU is visible (e.g. Kaggle's free 2x T4). ``train.gpu_ids``
    may pin specific device ids; ``null`` uses every visible CUDA device.

    When disabled (the default) the model is returned unchanged, so single-GPU
    runs stay byte-for-byte identical to before. DataParallel is transparent to
    the training loop: it scatters the batch (and the per-source keypoint/
    context lists) across GPUs along dim 0 and gathers the per-source mask list
    back onto the primary device, where the loss is computed exactly as before.
    """
    tcfg = cfg["train"]
    if not tcfg.get("data_parallel", False):
        return model
    if device.type != "cuda" or torch.cuda.device_count() < 2:
        n = torch.cuda.device_count() if device.type == "cuda" else 0
        print(f"[data_parallel] requested but only {n} CUDA device(s) visible; "
              "running on a single device (results match the 1-GPU config).")
        return model
    gpu_ids = tcfg.get("gpu_ids") or list(range(torch.cuda.device_count()))
    print(f"[data_parallel] DataParallel across {len(gpu_ids)} GPUs {gpu_ids}; "
          f"global batch {tcfg['batch_size']} split ~{tcfg['batch_size'] // len(gpu_ids)}/GPU")
    return nn.DataParallel(model, device_ids=gpu_ids)


def train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch):
    model.train()
    running = 0.0
    mask_type = cfg["audio"]["mask_type"]
    loss_mode = cfg["audio"].get("loss_mode", "bce_energy_weighted")
    for step, batch in enumerate(loader):
        net_input = batch["net_input"].to(device)
        energy = batch["mixture_mag"].to(device)
        keypoints = [k.to(device) for k in batch["keypoints"]]
        contexts = [c.to(device) for c in batch["contexts"]]
        targets = [t.to(device) for t in build_targets(batch, cfg)]

        masks = model(net_input, keypoints, contexts)
        if mask_type == "ratio":
            loss = sum(criterion(m, t) for m, t in zip(masks, targets)) / len(masks)
        elif loss_mode == "bce_plain":
            loss = plain_bce(masks, targets)
        else:
            loss = energy_weighted_bce(masks, targets, energy,
                                       cfg["audio"].get("loss_energy_floor", 0.0))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["audio"]["clip_grad"])
        optimizer.step()

        running += loss.item()
        if step % cfg["train"]["log_interval"] == 0:
            avg = running / (step + 1)
            print(f"epoch {epoch} step {step}/{len(loader)} loss {avg:.4f}")
    return running / max(1, len(loader))


def train_model(cfg, device, out_dir, init_from=None, resume=None):
    """Run a full training run for ``cfg`` into ``out_dir``; return last.pth path.

    ``init_from`` loads *model weights only* from a previous checkpoint (fresh
    optimizer -- used to chain curriculum stages). ``resume`` restores model +
    optimizer + epoch to continue an interrupted run.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_set = MusicMixDataset(cfg["data"]["train_index"], cfg, split="train")
    loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], collate_fn=collate,
                        drop_last=True)

    model = MusicGesture(cfg).to(device)
    # Build the optimizer on the raw module BEFORE any DataParallel wrapping so
    # the SGD param-group name matching ("audio_net."/"fusion." ...) and the
    # model.context_net lookup keep working. The parameters are shared, so the
    # optimizer stays correct after wrapping.
    optimizer = build_optimizer(model, cfg)

    # Optional multi-GPU data parallelism (opt-in via train.data_parallel).
    # ``raw_model`` is always the underlying MusicGesture module: we save/load
    # its state_dict so checkpoints stay identical to single-GPU runs and load
    # cleanly in eval_diag.py / resume_stage1.py (no "module." prefix).
    raw_model = model
    model = _wrap_data_parallel(model, cfg, device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["train"]["lr_steps"], gamma=cfg["train"]["lr_gamma"])
    criterion = nn.L1Loss() if cfg["audio"]["mask_type"] == "ratio" else nn.BCELoss()

    start_epoch = 0
    best_loss = float("inf")
    if init_from and os.path.isfile(init_from):
        ckpt = torch.load(init_from, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        print(f"[init] loaded model weights from {init_from}")
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[resume] continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch)
        scheduler.step()
        print(f"epoch {epoch} done avg_loss {loss:.4f}")
        # Keep only a rolling last.pth (+ best.pth) so the output dir does not
        # fill up with one ~150MB file per epoch. Atomic write via tmp+rename.
        state = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_loss": min(best_loss, loss), "cfg": cfg}
        tmp = os.path.join(out_dir, "last.pth.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(out_dir, "last.pth"))
        if loss < best_loss:
            best_loss = loss
            shutil.copyfile(os.path.join(out_dir, "last.pth"),
                            os.path.join(out_dir, "best.pth"))
    return os.path.join(out_dir, "last.pth")


def apply_stage(cfg, stage):
    """Return a deep-copied cfg with a curriculum stage's overrides applied."""
    cfg = copy.deepcopy(cfg)
    if "mix_policy" in stage:
        cfg["data"]["mix_policy"] = stage["mix_policy"]
    if "epochs" in stage:
        cfg["train"]["epochs"] = stage["epochs"]
    if "lr_steps" in stage:
        cfg["train"]["lr_steps"] = stage["lr_steps"]
    if "lr_scale" in stage:
        s = stage["lr_scale"]
        t = cfg["train"]
        if "lr" in t:
            t["lr"] = t["lr"] * s
        if "lr_backbone" in t:
            t["lr_backbone"] = t["lr_backbone"] * s
        pg = t.get("param_groups")
        if pg:
            for k in list(pg.keys()):
                pg[k] = pg[k] * s
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_out = cfg["experiment"]["output_dir"]
    os.makedirs(base_out, exist_ok=True)

    stages = cfg["train"].get("stages")
    if stages:
        # Paper two-stage curriculum: run each stage sequentially, chaining
        # weights from the previous stage's last checkpoint.
        prev_ckpt = None
        for si, stage in enumerate(stages):
            name = stage.get("name", f"stage{si}")
            stage_cfg = apply_stage(cfg, stage)
            out_dir = os.path.join(base_out, f"stage{si}_{name}")
            print(f"=== curriculum stage {si}: {name} "
                  f"(mix_policy={stage_cfg['data'].get('mix_policy')}, "
                  f"epochs={stage_cfg['train']['epochs']}) ===")
            prev_ckpt = train_model(stage_cfg, device, out_dir, init_from=prev_ckpt)
        # Surface the final stage checkpoint at the top level for convenience.
        if prev_ckpt and os.path.isfile(prev_ckpt):
            shutil.copyfile(prev_ckpt, os.path.join(base_out, "last.pth"))
    else:
        train_model(cfg, device, base_out, resume=args.resume)


if __name__ == "__main__":
    main()

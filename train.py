"""Train Music Gesture with Mix-and-Separate self-supervision."""
from __future__ import annotations

import argparse
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.music_dataset import MusicMixDataset, collate
from models import MusicGesture
from models.synthesizer import apply_mask
from utils.audio import ideal_ratio_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_targets(batch, mask_type: str):
    mix = batch["mixture_mag"]
    targets = []
    for src in batch["source_mags"]:
        if mask_type == "ratio":
            targets.append(ideal_ratio_mask(src, mix).clamp(0, 1))
        else:
            other = mix - src
            targets.append((src > other).float())
    return targets


def train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch):
    model.train()
    running = 0.0
    for step, batch in enumerate(loader):
        mix = batch["mixture_mag"].to(device)
        keypoints = [k.to(device) for k in batch["keypoints"]]
        contexts = [c.to(device) for c in batch["contexts"]]
        targets = [t.to(device) for t in build_targets(batch, cfg["audio"]["mask_type"])]

        masks = model(mix, keypoints, contexts)
        loss = sum(criterion(m, t) for m, t in zip(masks, targets)) / len(masks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["audio"]["clip_grad"])
        optimizer.step()

        running += loss.item()
        if step % cfg["train"]["log_interval"] == 0:
            avg = running / (step + 1)
            print(f"epoch {epoch} step {step}/{len(loader)} loss {avg:.4f}")
    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)

    train_set = MusicMixDataset(cfg["data"]["train_index"], cfg, split="train")
    loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], collate_fn=collate,
                        drop_last=True)

    model = MusicGesture(cfg).to(device)
    backbone_ids = set(id(p) for p in model.context_net.parameters())
    params = [
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
         "lr": cfg["train"]["lr"]},
        {"params": list(model.context_net.parameters()),
         "lr": cfg["train"]["lr_backbone"]},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["train"]["lr_steps"], gamma=cfg["train"]["lr_gamma"])
    criterion = nn.L1Loss() if cfg["audio"]["mask_type"] == "ratio" else nn.BCELoss()

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, criterion, device, cfg, epoch)
        scheduler.step()
        print(f"epoch {epoch} done avg_loss {loss:.4f}")
        # Keep only a rolling last.pth (+ best.pth) so /kaggle/working does not
        # fill up with one ~150MB file per epoch. Atomic write via tmp+rename.
        out_dir = cfg["experiment"]["output_dir"]
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_loss": min(best_loss, loss), "cfg": cfg}
        tmp = os.path.join(out_dir, "last.pth.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(out_dir, "last.pth"))
        if loss < best_loss:
            best_loss = loss
            shutil.copyfile(os.path.join(out_dir, "last.pth"),
                            os.path.join(out_dir, "best.pth"))


if __name__ == "__main__":
    main()

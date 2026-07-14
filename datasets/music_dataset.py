"""Mix-and-Separate dataset for MUSIC-21.

Each index file row points to a preprocessed solo clip:
    audio_path, pose_path, context_frame_path, category

``__getitem__`` samples ``num_mix`` solos, mixes their audio, and returns the
mixture spectrogram plus per-source keypoints/context/target so the model learns
to separate a source conditioned on its gestures.
"""
from __future__ import annotations

import csv
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import audio as A
from utils.pose import normalize_keypoints, augment_keypoints

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


class MusicMixDataset(Dataset):
    def __init__(self, index_file: str, cfg: dict, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.num_mix = cfg["data"]["num_mix"]
        self.sr = cfg["audio"]["sample_rate"]
        self.clip_len = int(cfg["audio"]["clip_seconds"] * self.sr)
        self.frame_size = cfg["video"]["frame_size"]
        self.samples = self._read_index(index_file)
        # Mixing policy for Mix-and-Separate (curriculum). 'random' = any other
        # solo; 'hetero' = a solo of a *different* instrument category; 'homo' =
        # a solo of the *same* category (the hard, homo-musical case the paper's
        # two-stage curriculum finetunes on). Built here as a category->indices
        # map so __getitem__ can restrict the sampling pool.
        self.mix_policy = cfg["data"].get("mix_policy", "random")
        self.by_category: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            self.by_category.setdefault(s.get("category", ""), []).append(i)
        # Keypoint augmentation (train split only).
        v = cfg["video"]
        self.pose_augment = bool(v.get("pose_augment", False))
        self.pose_jitter_translate = float(v.get("pose_jitter_translate", 0.05))
        self.pose_jitter_scale = float(v.get("pose_jitter_scale", 0.1))
        self.pose_jitter_rotate_deg = float(v.get("pose_jitter_rotate_deg", 10.0))
        self.log_freq = cfg["audio"].get("log_freq", False)
        self.warp = None
        if self.log_freq:
            mat = A.build_log_freq_matrix(
                cfg["audio"]["n_freq"], cfg["audio"]["n_log_freq"], self.sr)
            self.warp = torch.from_numpy(mat)

    @staticmethod
    def _read_index(path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: str) -> torch.Tensor:
        if sf is None:
            raise ImportError("soundfile is required to load audio")
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) < self.clip_len:
            wav = np.pad(wav, (0, self.clip_len - len(wav)))
        start = random.randint(0, len(wav) - self.clip_len) if self.split == "train" else 0
        return torch.from_numpy(wav[start:start + self.clip_len])

    def _load_pose(self, path: str) -> torch.Tensor:
        kp = np.load(path)  # [T, V, 3]
        kp = normalize_keypoints(kp, self.frame_size, self.frame_size)
        if self.split == "train" and self.pose_augment:
            kp = augment_keypoints(
                kp, translate=self.pose_jitter_translate,
                scale=self.pose_jitter_scale,
                rotate_deg=self.pose_jitter_rotate_deg)
        return torch.from_numpy(kp).float()

    def _sample_others(self, idx: int) -> List[int]:
        """Pick num_mix-1 partner indices according to self.mix_policy.

        Falls back to the unrestricted pool when the policy-restricted pool is
        too small (e.g. a category with a single solo under 'homo').
        """
        anchor_cat = self.samples[idx].get("category", "")
        if self.mix_policy == "homo":
            pool = [j for j in self.by_category.get(anchor_cat, []) if j != idx]
        elif self.mix_policy == "hetero":
            pool = [j for j in range(len(self.samples))
                    if self.samples[j].get("category", "") != anchor_cat]
        else:
            pool = [j for j in range(len(self.samples)) if j != idx]
        if len(pool) < self.num_mix - 1:
            pool = [j for j in range(len(self.samples)) if j != idx]
        return random.sample(pool, self.num_mix - 1)

    def _load_context(self, path: str) -> torch.Tensor:
        if cv2 is None:
            raise ImportError("opencv-python is required to load frames")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.frame_size, self.frame_size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))

    def _spec(self, wav: torch.Tensor) -> torch.Tensor:
        c = self.cfg["audio"]
        spec = A.stft(wav, c["n_fft"], c["hop_length"], c["win_length"])
        return spec

    def __getitem__(self, idx: int) -> Dict[str, object]:
        chosen = [self.samples[idx]]
        others = self._sample_others(idx)
        chosen += [self.samples[o] for o in others]

        waveforms = [self._load_audio(s["audio_path"]) for s in chosen]
        mixture, sources = A.mix_and_separate(waveforms)

        c = self.cfg["audio"]
        mix_spec = self._spec(mixture)
        mix_mag = mix_spec.abs()                       # [F, T] linear-frequency
        src_mags = [self._spec(s).abs() for s in sources]
        if self.warp is not None:
            # Warp to a log-frequency grid so the empty high-frequency bins are
            # compressed and resolution is concentrated where energy lives.
            mix_mag = A.warp_freq(mix_mag, self.warp)
            src_mags = [A.warp_freq(s, self.warp) for s in src_mags]
        # Network input is the log-magnitude of the (optionally warped) mixture;
        # mixture_mag stays linear-amplitude for the energy-weighted loss.
        net_input = A.log_magnitude(mix_mag).unsqueeze(0)
        mix_mag = mix_mag.unsqueeze(0)
        src_mags = [s.unsqueeze(0) for s in src_mags]

        keypoints = [self._load_pose(s["pose_path"]) for s in chosen]
        contexts = [self._load_context(s["context_frame_path"]) for s in chosen]

        return {
            "net_input": net_input,
            "mixture_mag": mix_mag,
            "mixture_wav": mixture,
            "source_mags": src_mags,
            "source_wavs": sources,
            "keypoints": keypoints,
            "contexts": contexts,
            "categories": [s["category"] for s in chosen],
        }


def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack a batch of variable-source samples (fixed num_mix)."""
    num_mix = len(batch[0]["keypoints"])
    out: Dict[str, object] = {
        "net_input": torch.stack([b["net_input"] for b in batch]),
        "mixture_mag": torch.stack([b["mixture_mag"] for b in batch]),
        "mixture_wav": torch.stack([b["mixture_wav"] for b in batch]),
        "source_mags": [torch.stack([b["source_mags"][i] for b in batch]) for i in range(num_mix)],
        "source_wavs": [torch.stack([b["source_wavs"][i] for b in batch]) for i in range(num_mix)],
        "keypoints": [torch.stack([b["keypoints"][i] for b in batch]) for i in range(num_mix)],
        "contexts": [torch.stack([b["contexts"][i] for b in batch]) for i in range(num_mix)],
    }
    return out

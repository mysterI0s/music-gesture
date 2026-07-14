"""Audio utilities: STFT / iSTFT, log-frequency warping, masks, mix-and-separate."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def stft(waveform: torch.Tensor, n_fft: int, hop_length: int,
         win_length: int) -> torch.Tensor:
    """Return complex STFT [..., F, T] using a Hann window."""
    window = torch.hann_window(win_length, device=waveform.device)
    return torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=True)


def istft(spec: torch.Tensor, n_fft: int, hop_length: int, win_length: int,
          length: int | None = None) -> torch.Tensor:
    window = torch.hann_window(win_length, device=spec.device)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, length=length)


def magnitude_phase(spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return spec.abs(), torch.angle(spec)


def log_magnitude(mag: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(mag + eps)


def ideal_ratio_mask(source_mag: torch.Tensor,
                     mixture_mag: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Ground-truth ratio mask for Mix-and-Separate supervision."""
    return source_mag / (mixture_mag + eps)


def ideal_binary_mask(source_mag: torch.Tensor,
                      other_mag: torch.Tensor) -> torch.Tensor:
    """1 where this source is at least as loud as the other sources.

    This is the ideal binary mask used as the ground-truth target in Music
    Gesture (Eq. 4): the label is the dominant source at each time-frequency
    bin. Because the two per-source masks are (near-)complementary, the targets
    are balanced ~50/50, so a constant prediction cannot minimise the loss.
    """
    return (source_mag >= other_mag).float()


def ideal_binary_mask_literal(source_mag: torch.Tensor,
                              mixture_mag: torch.Tensor) -> torch.Tensor:
    """Literal reading of Music Gesture Eq. 4: 1 where the source is at least
    as loud as the *mixture* itself (S_k >= S_mix), rather than the dominant
    source among the others. Because the mixture is the sum of magnitudes, this
    target is only 1 where a single source overwhelmingly dominates, so the
    labels are sparse. Selectable via ``audio.mask_target: literal_mix``.
    """
    return (source_mag >= mixture_mag).float()


def mix_and_separate(waveforms: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Given N solo waveforms, return (mixture, sources).

    The mixture is the sum of the sources (standard Mix-and-Separate, as used
    in Sound of Pixels / Music Gesture); each source is the individual
    waveform. This is the self-supervised signal used to train separation.
    """
    stacked = torch.stack(waveforms, dim=0)
    mixture = stacked.sum(dim=0)
    return mixture, waveforms


def build_log_freq_matrix(n_freq: int, n_log_freq: int,
                          sample_rate: int) -> np.ndarray:
    """Linear->log frequency resampling matrix, shape [n_log_freq, n_freq].

    Each log-spaced centre is a linear interpolation of the two bracketing
    linear-frequency bins, so the matrix is row-stochastic (rows sum to 1) and
    concentrates resolution on the low/mid band where instrument energy lives.
    """
    f_max = sample_rate / 2
    lin = np.linspace(0, f_max, n_freq)
    f0 = max(lin[1], 1.0)
    log_pts = np.logspace(np.log10(f0), np.log10(f_max), n_log_freq)
    mat = np.zeros((n_log_freq, n_freq), dtype=np.float32)
    for i, center in enumerate(log_pts):
        j = int(np.searchsorted(lin, center))
        if j <= 0:
            mat[i, 0] = 1.0
        elif j >= n_freq:
            mat[i, n_freq - 1] = 1.0
        else:
            lo, hi = lin[j - 1], lin[j]
            w = (center - lo) / (hi - lo + 1e-12)
            mat[i, j - 1] = 1.0 - w
            mat[i, j] = w
    return mat


def build_inv_log_freq_matrix(n_freq: int, n_log_freq: int,
                              sample_rate: int) -> np.ndarray:
    """Log->linear frequency resampling matrix, shape [n_freq, n_log_freq].

    Inverse of ``build_log_freq_matrix``: maps a predicted mask back from the
    log-frequency grid to the linear STFT grid for reconstruction. Each linear
    bin linearly interpolates the two bracketing log-spaced points.
    """
    f_max = sample_rate / 2
    lin = np.linspace(0, f_max, n_freq)
    f0 = max(lin[1], 1.0)
    log_pts = np.logspace(np.log10(f0), np.log10(f_max), n_log_freq)
    mat = np.zeros((n_freq, n_log_freq), dtype=np.float32)
    for i, f in enumerate(lin):
        j = int(np.searchsorted(log_pts, f))
        if j <= 0:
            mat[i, 0] = 1.0
        elif j >= n_log_freq:
            mat[i, n_log_freq - 1] = 1.0
        else:
            lo, hi = log_pts[j - 1], log_pts[j]
            w = (f - lo) / (hi - lo + 1e-12)
            mat[i, j - 1] = 1.0 - w
            mat[i, j] = w
    return mat


def warp_freq(mag: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Resample the frequency axis of ``mag`` [..., Fin, T] with ``matrix`` [Fout, Fin].

    Returns [..., Fout, T]. ``matrix`` broadcasts over any leading (batch/
    channel) dimensions, so this works for [F, T], [C, F, T] and [B, C, F, T].
    """
    return torch.matmul(matrix.to(mag.dtype), mag)

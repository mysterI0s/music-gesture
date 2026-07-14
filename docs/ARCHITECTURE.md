# Architecture notes

This document maps each paper component to the code that implements it.

| Paper component | Module | File |
| --- | --- | --- |
| Audio analysis/synthesis U-Net | `AudioUNet` | `models/audio_net.py` |
| Semantic context (ResNet-50) | `ContextNet` | `models/context_net.py` |
| Context-aware Graph CNN (CT-GCN) over body+hand keypoints | `ContextAwareGraphCNN` | `models/pose_net.py` |
| Audio-visual self-attention fusion | `AudioVisualFusion` | `models/fusion.py` |
| Mask prediction + masking | `MaskHead`, `apply_mask` | `models/synthesizer.py` |
| Full pipeline | `MusicGesture` | `models/music_gesture.py` |
| Mix-and-Separate data | `MusicMixDataset` | `datasets/music_dataset.py` |
| Skeleton graph (body 18 + 2x hand 21) | `build_skeleton_adjacency` | `utils/pose.py` |

## Data flow

1. **Audio branch.** The mixture waveform is converted to a (log-)magnitude
   spectrogram and encoded by the U-Net encoder into a grid of bottleneck
   tokens.
2. **Visual branch.** For each musician, per-frame body + hand keypoints form a
   spatio-temporal graph. `ContextAwareGraphCNN` runs ST-GCN blocks modulated by
   the ResNet-50 semantic context of that musician (FiLM, or channel-concat
   when `context_mode: concat`), producing gesture tokens.
3. **Fusion.** `AudioVisualFusion` concatenates audio tokens with the gesture
   tokens of the target source and applies Transformer self-attention so the
   audio representation is conditioned on that source's motion.
4. **Synthesis.** The U-Net decoder maps fused tokens back to a mask; the mask
   is applied to the mixture magnitude and inverted with the mixture phase.

## Why gestures solve same-category separation

Appearance features cannot distinguish two same-type instruments. Fine-grained
finger/body motion is discriminative even when appearance is identical, so
conditioning separation on per-musician gestures resolves the homo-musical case
that *The Sound of Pixels* fails on, while *The Sound of Motions* addresses it
with dense motion instead of sparse keypoints.

## Differences from the (unreleased) original

The paper leaves several details unspecified. Documented choices here:

- 3-subset ST-GCN spatial partitioning for the skeleton graph.
- Semantic context conditioning defaults to FiLM (per graph block); the
  paper-faithful channel-concatenation scheme is selectable (`context_mode`).
- Fusion defaults to a Transformer encoder over concatenated tokens; the
  paper's sound-queries-visual cross-attention is selectable (`fusion.mode`).
- Binary ideal mask + BCE by default (ratio mask + L1 is selectable via
  `mask_type: ratio`). The binary target defaults to the balanced
  dominant-source IBM; the literal `S_k >= S_mix` target and a plain
  (non-energy-weighted) BCE are selectable (`mask_target`, `loss_mode`).

## Paper-faithful config flags

Every deviation below is guarded by a flag so the repo's efficient default is
preserved. `configs/paper_faithful.yaml` flips them all at once; toggle them
individually for ablations.

| Aspect | Default | Paper value | Flag |
| --- | --- | --- | --- |
| Context conditioning | FiLM | channel concat | `model.pose.context_mode: concat` |
| Optimizer | Adam | SGD momentum 0.9 | `train.optimizer: sgd` (+ `param_groups`) |
| Separation loss | energy-weighted BCE | plain per-pixel BCE | `audio.loss_mode: bce_plain` |
| Binary mask target | dominant-source IBM | literal `S_k >= S_mix` | `audio.mask_target: literal_mix` |
| U-Net depth / kernel | 7 downs, 4x4 | 4 downs, 3x3 | `model.audio.num_downs: 4`, `conv_kernel/up_kernel: 3` |
| Fusion | Transformer self-attn | sound->visual cross-attn | `model.fusion.mode: paper_attn` |
| GCN depth / strides | 6 layers, strides [2,4] | 11 layers, strides [3,6] | `model.pose.graph_layers`, `stride_layers` |
| Context feature width | 512 (projected) | 2048 (raw ResNet-50) | `model.context.feat_dim: 2048` |
| Context frame | middle of clip | first frame | `prepare_urmp --context_frame first` |
| Keypoint augmentation | off | on | `video.pose_augment: true` |
| Mixing curriculum | random | hetero -> homo two-stage | `train.stages` (see `configs/curriculum.yaml`) |

Data-prep scripts: `scripts/extract_pose.py` provides a real MediaPipe /
AlphaPose / OpenPose backend dispatcher (`--backend`); `scripts/prepare_atinpiano.py`
prepares the solo-piano AtinPiano dataset used by the paper alongside
MUSIC-21 and URMP.

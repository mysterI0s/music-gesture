#!/usr/bin/env python3
"""Adapt the AtinPiano solo-piano dataset into the repo's expected layout.

AtinPiano is a collection of solo-piano performance videos (single instrument,
top-down / side view of the pianist's hands and body). The paper trains and
evaluates on AtinPiano alongside MUSIC-21 and URMP. Because every AtinPiano clip
is already a *solo*, each segmented clip becomes one Mix-and-Separate item with
category ``piano``.

Writes, under --out (default datasets/processed):
    audio/<clip>.wav   mono, --sr
    pose/<clip>.npy    float32 [num_frames, 60, 3]  (body18 + Rhand21 + Lhand21)
    frames/<clip>.jpg  context crop, frame_size x frame_size
    meta.csv           clip,category   (category == 'piano')

Pose backend mirrors scripts/extract_pose.py:
    --pose zeros       zeros pose + full-frame context (fast smoke test)
    --pose mediapipe   MediaPipe Pose + Hands per key-frame (CPU)

Usage:
    python scripts/prepare_atinpiano.py --videos_dir data/atinpiano \
        --out datasets/processed --pose mediapipe
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

BODY, HAND = 18, 21
VJ = BODY + 2 * HAND  # 60
CATEGORY = "piano"

# MediaPipe BlazePose(33) -> COCO-18; neck (idx 1) = shoulder midpoint.
BP2COCO = {0: 0, 12: 2, 14: 3, 16: 4, 11: 5, 13: 6, 15: 7, 24: 8, 26: 9,
           28: 10, 23: 11, 25: 12, 27: 13, 5: 14, 2: 15, 8: 16, 7: 17}


def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_videos(root):
    vids = []
    for ext in ("*.mp4", "*.mkv", "*.webm", "*.avi"):
        vids += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(vids)


def extract_frames(video, frames_dir, fps):
    os.makedirs(frames_dir, exist_ok=True)
    if not glob.glob(os.path.join(frames_dir, "*.jpg")):
        sh(["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps},scale=-2:480",
            os.path.join(frames_dir, "%06d.jpg")])
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))


def extract_audio(video, wav_path, sr):
    if not os.path.exists(wav_path):
        sh(["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", str(sr), wav_path])


def build_pose_detectors():
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                  min_detection_confidence=0.3)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     min_detection_confidence=0.3)
    return mp, pose, hands


def pose_for_frames(frame_paths, detectors, frame_size, stride):
    T = len(frame_paths)
    poses = np.zeros((T, VJ, 3), np.float32)
    crops = [None] * T
    mp, pose_det, hand_det = detectors
    key_ts = list(range(0, T, max(1, stride)))
    if T and key_ts[-1] != T - 1:
        key_ts.append(T - 1)
    for t in key_ts:
        bgr = cv2.imread(frame_paths[t])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pres = pose_det.process(rgb)
        if not pres.pose_landmarks:
            continue
        lms = pres.pose_landmarks.landmark
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(w, int(max(xs))), min(h, int(max(ys)))
        bw_, bh_ = max(1, x1 - x0), max(1, y1 - y0)
        for bp, ci in BP2COCO.items():
            lm = lms[bp]
            poses[t, ci, 0] = (lm.x * w - x0) / bw_ * frame_size
            poses[t, ci, 1] = (lm.y * h - y0) / bh_ * frame_size
            poses[t, ci, 2] = getattr(lm, "visibility", 1.0)
        ls, rs = lms[11], lms[12]
        poses[t, 1, 0] = ((ls.x + rs.x) / 2 * w - x0) / bw_ * frame_size
        poses[t, 1, 1] = ((ls.y + rs.y) / 2 * h - y0) / bh_ * frame_size
        poses[t, 1, 2] = min(getattr(ls, "visibility", 1.0), getattr(rs, "visibility", 1.0))
        crop = bgr[y0:y1, x0:x1]
        if crop.size:
            crop_rs = cv2.resize(crop, (frame_size, frame_size))
            cpath = frame_paths[t] + ".crop.jpg"
            cv2.imwrite(cpath, crop_rs)
            crops[t] = cpath
            hres = hand_det.process(cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB))
            if hres.multi_hand_landmarks:
                for hi, hl in enumerate(hres.multi_hand_landmarks):
                    label = "right"
                    if hres.multi_handedness:
                        label = hres.multi_handedness[hi].classification[0].label.lower()
                    base = BODY if label.startswith("r") else BODY + HAND
                    for j, lm in enumerate(hl.landmark):
                        poses[t, base + j, 0] = lm.x * frame_size
                        poses[t, base + j, 1] = lm.y * frame_size
                        poses[t, base + j, 2] = 1.0
    # interpolate skipped frames + forward/backward fill crops
    if stride > 1 and T > 1 and key_ts:
        key_arr = np.asarray(key_ts)
        all_t = np.arange(T)
        flat = poses[key_arr].reshape(len(key_arr), -1)
        filled = np.empty((T, flat.shape[1]), np.float32)
        for c in range(flat.shape[1]):
            filled[:, c] = np.interp(all_t, key_arr, flat[:, c])
        poses = filled.reshape(T, VJ, 3).astype(np.float32)
        last = None
        for t in range(T):
            if crops[t] is not None:
                last = crops[t]
            elif last is not None:
                crops[t] = last
        nxt = None
        for t in range(T - 1, -1, -1):
            if crops[t] is not None:
                nxt = crops[t]
            elif nxt is not None:
                crops[t] = nxt
    return poses, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--tmp", default="/tmp/_atinpiano_tmp")
    ap.add_argument("--pose", choices=["zeros", "mediapipe"], default="mediapipe")
    ap.add_argument("--sr", type=int, default=11025)
    ap.add_argument("--clip_seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--seg_hop", type=float, default=6.0)
    ap.add_argument("--pose_stride", type=int, default=2)
    ap.add_argument("--context_frame", choices=["first", "middle"], default="middle")
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if sf is None or cv2 is None:
        raise SystemExit("pip install soundfile opencv-python (and mediapipe for --pose mediapipe)")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    audio_dir = os.path.join(args.out, "audio")
    pose_dir = os.path.join(args.out, "pose")
    ctx_dir = os.path.join(args.out, "frames")
    for d in (audio_dir, pose_dir, ctx_dir):
        os.makedirs(d, exist_ok=True)

    videos = find_videos(args.videos_dir)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise SystemExit(f"no videos found under {args.videos_dir}")
    print(f"videos: {len(videos)} | pose mode: {args.pose}")

    detectors = None
    if args.pose == "mediapipe":
        detectors = build_pose_detectors()

    clip_len = int(args.clip_seconds * args.sr)
    seg_frames = args.num_frames
    fhop = int(args.seg_hop * args.fps)
    ahop = int(args.seg_hop * args.sr)

    meta_path = os.path.join(args.out, "meta.csv")
    write_header = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "category"])
        if write_header:
            writer.writeheader()
        for vi, video in enumerate(videos):
            name = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
            frames_dir = os.path.join(args.tmp, name)
            frame_paths = extract_frames(video, frames_dir, args.fps)
            wav_rs = os.path.join(args.tmp, f"{name}.wav")
            extract_audio(video, wav_rs, args.sr)
            wav, _ = sf.read(wav_rs, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if args.pose == "mediapipe":
                poses, crops = pose_for_frames(frame_paths, detectors, args.frame_size,
                                               args.pose_stride)
            else:
                poses, crops = None, None

            n_seg = 1 + max(0, (len(wav) - clip_len) // ahop)
            written = 0
            for si in range(n_seg):
                a0, a1 = si * ahop, si * ahop + clip_len
                f0 = si * fhop
                if a1 > len(wav) or f0 + seg_frames > len(frame_paths):
                    break
                clip = f"{name}__s{si:03d}"
                sf.write(os.path.join(audio_dir, clip + ".wav"), wav[a0:a1], args.sr)
                if poses is None:
                    pose = np.zeros((seg_frames, VJ, 3), np.float32)
                else:
                    pose = poses[f0:f0 + seg_frames]
                    if len(pose) < seg_frames:
                        pose = np.pad(pose, ((0, seg_frames - len(pose)), (0, 0), (0, 0)))
                np.save(os.path.join(pose_dir, clip + ".npy"), pose.astype(np.float32))
                ctx_idx = f0 if args.context_frame == "first" else f0 + seg_frames // 2
                ctx_idx = min(ctx_idx, len(frame_paths) - 1)
                if crops is not None and crops[ctx_idx]:
                    src_img = crops[ctx_idx]
                else:
                    src_img = frame_paths[ctx_idx]
                img = cv2.imread(src_img)
                img = cv2.resize(img, (args.frame_size, args.frame_size))
                cv2.imwrite(os.path.join(ctx_dir, clip + ".jpg"), img)
                writer.writerow({"clip": clip, "category": CATEGORY})
                written += 1
            print(f"[{vi+1}/{len(videos)}] {name}: {written} clips")
    print("meta.csv written ->", meta_path)


if __name__ == "__main__":
    main()

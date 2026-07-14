#!/usr/bin/env python3
"""Extract body + hand keypoints per frame from videos.

Produces, per input video, a numpy array of shape [T, V, 3] where
V = body(18) + right-hand(21) + left-hand(21) = 60 and the last channel is the
detection confidence. Coordinates are in pixels of the (per-person) crop scaled
to ``--frame_size`` so they match what ``utils.pose.normalize_keypoints``
expects downstream.

Backends (``--backend``):
    mediapipe   MediaPipe Pose + Hands (CPU-friendly, easiest to install).
    alphapose   RMPE / AlphaPose Halpe-136 wholebody (paper-faithful, GPU). Runs
                the AlphaPose CLI and maps Halpe-136 -> the 60-joint layout.
    openpose    OpenPose BODY_25 + hand keypoints, parsed from its JSON output.

This dispatcher extracts the *primary* (largest / most confident) person per
frame -- it is intended for solo clips (MUSIC-21 / AtinPiano). For multi-player
pieces such as URMP, use scripts/prepare_urmp.py, which handles per-player
assignment and cropping.

Usage:
    python scripts/extract_pose.py --videos_dir data/videos \
        --out_dir datasets/processed/pose --backend mediapipe
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

BODY = 18
HAND = 21
V = BODY + 2 * HAND  # 60

# MediaPipe BlazePose(33) -> COCO-18 (OpenPose ordering); neck (idx 1) is the
# midpoint of the shoulders, computed separately.
BP2COCO = {0: 0, 12: 2, 14: 3, 16: 4, 11: 5, 13: 6, 15: 7, 24: 8, 26: 9,
           28: 10, 23: 11, 25: 12, 27: 13, 5: 14, 2: 15, 8: 16, 7: 17}

# OpenPose BODY_25 -> COCO-18. BODY_25 already has an explicit neck (idx 1).
BODY25_2COCO = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 8,
                10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15,
                17: 16, 18: 17}

# Halpe-136 -> COCO-18 and hand block offsets (matches prepare_urmp.py).
HALPE2COCO = {0: 0, 18: 1, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 8,
              14: 9, 16: 10, 11: 11, 13: 12, 15: 13, 2: 14, 1: 15, 4: 16, 3: 17}
HALPE_RHAND0, HALPE_LHAND0 = 115, 94


def _read_frames(path: str, fps: int):
    """Decode a video to a list of BGR frames sampled at ``fps``."""
    if cv2 is None:
        raise ImportError("opencv-python is required to read videos")
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(src_fps / fps)))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


# --------------------------------------------------------------------------
# MediaPipe backend
# --------------------------------------------------------------------------
def _extract_mediapipe(path: str, fps: int, frame_size: int) -> np.ndarray:
    import mediapipe as mp
    frames = _read_frames(path, fps)
    T = len(frames)
    out = np.zeros((T, V, 3), np.float32)
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.3) as pose, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.3) as hands:
        for t, bgr in enumerate(frames):
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pres = pose.process(rgb)
            if not pres.pose_landmarks:
                continue
            lms = pres.pose_landmarks.landmark
            for bp, ci in BP2COCO.items():
                lm = lms[bp]
                out[t, ci, 0] = lm.x * frame_size
                out[t, ci, 1] = lm.y * frame_size
                out[t, ci, 2] = getattr(lm, "visibility", 1.0)
            ls, rs = lms[11], lms[12]
            out[t, 1, 0] = (ls.x + rs.x) / 2 * frame_size
            out[t, 1, 1] = (ls.y + rs.y) / 2 * frame_size
            out[t, 1, 2] = min(getattr(ls, "visibility", 1.0),
                               getattr(rs, "visibility", 1.0))
            hres = hands.process(rgb)
            if hres.multi_hand_landmarks:
                for hi, hl in enumerate(hres.multi_hand_landmarks):
                    label = "right"
                    if hres.multi_handedness:
                        label = hres.multi_handedness[hi].classification[0].label.lower()
                    base = BODY if label.startswith("r") else BODY + HAND
                    for j, lm in enumerate(hl.landmark):
                        out[t, base + j, 0] = lm.x * frame_size
                        out[t, base + j, 1] = lm.y * frame_size
                        out[t, base + j, 2] = 1.0
    return out


# --------------------------------------------------------------------------
# OpenPose backend
# --------------------------------------------------------------------------
def _extract_openpose(path: str, fps: int, frame_size: int, args) -> np.ndarray:
    """Run the OpenPose demo binary with hand keypoints and parse its JSON."""
    frames = _read_frames(path, fps)
    T = len(frames)
    out = np.zeros((T, V, 3), np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, "img")
        json_dir = os.path.join(tmp, "json")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        for t, bgr in enumerate(frames):
            cv2.imwrite(os.path.join(img_dir, f"{t:06d}.jpg"), bgr)
        cmd = [args.openpose_bin, "--image_dir", img_dir,
               "--write_json", json_dir, "--hand", "--render_pose", "0",
               "--display", "0", "--model_pose", "BODY_25"]
        subprocess.run(cmd, check=True, cwd=args.openpose_root or None)
        for t in range(T):
            jpath = os.path.join(json_dir, f"{t:06d}_keypoints.json")
            if not os.path.exists(jpath):
                continue
            people = json.load(open(jpath)).get("people", [])
            if not people:
                continue
            person = max(people, key=_openpose_person_score)
            body = np.asarray(person.get("pose_keypoints_2d", []), np.float32).reshape(-1, 3)
            bw = bh = 1.0
            xs = body[body[:, 2] > 0, 0]
            ys = body[body[:, 2] > 0, 1]
            if xs.size:
                bw = max(1.0, xs.max() - xs.min())
                bh = max(1.0, ys.max() - ys.min())
                x0, y0 = xs.min(), ys.min()
            else:
                x0 = y0 = 0.0
            for op, ci in BODY25_2COCO.items():
                if op < len(body):
                    out[t, ci, 0] = (body[op, 0] - x0) / bw * frame_size
                    out[t, ci, 1] = (body[op, 1] - y0) / bh * frame_size
                    out[t, ci, 2] = body[op, 2]
            rhand = np.asarray(person.get("hand_right_keypoints_2d", []), np.float32).reshape(-1, 3)
            lhand = np.asarray(person.get("hand_left_keypoints_2d", []), np.float32).reshape(-1, 3)
            for j in range(min(HAND, len(rhand))):
                out[t, BODY + j, 0] = (rhand[j, 0] - x0) / bw * frame_size
                out[t, BODY + j, 1] = (rhand[j, 1] - y0) / bh * frame_size
                out[t, BODY + j, 2] = rhand[j, 2]
            for j in range(min(HAND, len(lhand))):
                out[t, BODY + HAND + j, 0] = (lhand[j, 0] - x0) / bw * frame_size
                out[t, BODY + HAND + j, 1] = (lhand[j, 1] - y0) / bh * frame_size
                out[t, BODY + HAND + j, 2] = lhand[j, 2]
    return out


def _openpose_person_score(person) -> float:
    body = np.asarray(person.get("pose_keypoints_2d", []), np.float32).reshape(-1, 3)
    return float(body[:, 2].sum()) if body.size else 0.0


# --------------------------------------------------------------------------
# AlphaPose backend
# --------------------------------------------------------------------------
def _extract_alphapose(path: str, fps: int, frame_size: int, args) -> np.ndarray:
    """Run AlphaPose (Halpe-136) on decoded frames and map to the 60-joint layout."""
    frames = _read_frames(path, fps)
    T = len(frames)
    out = np.zeros((T, V, 3), np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, "img")
        out_dir = os.path.join(tmp, "ap")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        name_of = {}
        for t, bgr in enumerate(frames):
            fn = f"{t:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, fn), bgr)
            name_of[fn] = t
        demo = os.path.join(args.alphapose_root, "scripts", "demo_inference.py")
        argv = ["--cfg", args.ap_cfg, "--checkpoint", args.ap_ckpt,
                "--indir", img_dir, "--outdir", out_dir, "--sp",
                "--detector", args.ap_detector, "--gpus", args.ap_gpus]
        env = os.environ.copy()
        env["PYTHONPATH"] = args.alphapose_root + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(["python", demo] + argv, check=True,
                       cwd=args.alphapose_root, env=env)
        results = os.path.join(out_dir, "alphapose-results.json")
        data = json.load(open(results))
        best = {}
        for e in data:
            t = name_of.get(os.path.basename(str(e.get("image_id", ""))))
            if t is None:
                continue
            score = float(e.get("score", 0.0))
            if t not in best or score > best[t][0]:
                best[t] = (score, e)
        for t, (_, e) in best.items():
            kp = np.asarray(e["keypoints"], np.float32).reshape(-1, 3)
            xs = kp[:26, 0]
            ys = kp[:26, 1]
            x0, y0 = float(xs.min()), float(ys.min())
            bw = max(1.0, float(xs.max()) - x0)
            bh = max(1.0, float(ys.max()) - y0)
            for hp, ci in HALPE2COCO.items():
                out[t, ci, 0] = (kp[hp, 0] - x0) / bw * frame_size
                out[t, ci, 1] = (kp[hp, 1] - y0) / bh * frame_size
                out[t, ci, 2] = kp[hp, 2]
            for j in range(HAND):
                out[t, BODY + j, 0] = (kp[HALPE_RHAND0 + j, 0] - x0) / bw * frame_size
                out[t, BODY + j, 1] = (kp[HALPE_RHAND0 + j, 1] - y0) / bh * frame_size
                out[t, BODY + j, 2] = kp[HALPE_RHAND0 + j, 2]
                out[t, BODY + HAND + j, 0] = (kp[HALPE_LHAND0 + j, 0] - x0) / bw * frame_size
                out[t, BODY + HAND + j, 1] = (kp[HALPE_LHAND0 + j, 1] - y0) / bh * frame_size
                out[t, BODY + HAND + j, 2] = kp[HALPE_LHAND0 + j, 2]
    return out


def estimate_pose_for_video(path: str, args) -> np.ndarray:
    if args.backend == "mediapipe":
        return _extract_mediapipe(path, args.fps, args.frame_size)
    if args.backend == "openpose":
        return _extract_openpose(path, args.fps, args.frame_size, args)
    if args.backend == "alphapose":
        return _extract_alphapose(path, args.fps, args.frame_size, args)
    raise ValueError(f"unknown backend {args.backend!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--backend", choices=["mediapipe", "alphapose", "openpose"],
                        default="mediapipe")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame_size", type=int, default=224)
    # OpenPose options
    parser.add_argument("--openpose_bin", default="./build/examples/openpose/openpose.bin")
    parser.add_argument("--openpose_root", default=None)
    # AlphaPose options
    parser.add_argument("--alphapose_root", default="/kaggle/working/AlphaPose")
    parser.add_argument("--ap_cfg",
                        default="configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml")
    parser.add_argument("--ap_ckpt",
                        default="pretrained_models/halpe136_fast50_regression_256x192.pth")
    parser.add_argument("--ap_detector", default="yolo")
    parser.add_argument("--ap_gpus", default="0")
    args = parser.parse_args()

    if cv2 is None:
        raise SystemExit("pip install opencv-python (and the chosen pose backend)")
    os.makedirs(args.out_dir, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    if not videos:
        raise SystemExit(f"no .mp4 files under {args.videos_dir}")
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        kp = estimate_pose_for_video(v, args)
        np.save(os.path.join(args.out_dir, f"{name}.npy"), kp)
        print(f"saved pose for {name}: {kp.shape} ({args.backend})")


if __name__ == "__main__":
    main()

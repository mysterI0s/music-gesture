#!/usr/bin/env python3
"""Adapt the URMP dataset into the Music-Gesture repo's expected layout.

Writes, under --out (default datasets/processed):
    audio/<clip>.wav   mono, --sr, --clip_seconds
    pose/<clip>.npy    float32 [num_frames, 60, 3]  (body18 + Rhand21 + Lhand21; x,y in px[0,frame_size], conf)
    frames/<clip>.jpg  per-player context crop, frame_size x frame_size
    meta.csv           clip,category
    debug/<piece>.jpg  overlay to verify player<->stem assignment (mediapipe mode)

Each "clip" is ONE player (its isolated AuSep stem) over ONE time segment, so the
repo's Mix-and-Separate loader treats every URMP stem-segment as a solo.

Pose modes:
    --pose zeros       zeros pose + full-frame context (fast smoke test of the loop)
    --pose mediapipe   real per-player body+hand keypoints via MediaPipe Tasks

player<->stem assignment = left->right detected people matched to AuSep index
order. Inspect debug/<piece>.jpg to confirm audio and crop line up per player.
"""
from __future__ import annotations
import argparse, csv, glob, os, subprocess, urllib.request
import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import cv2
except Exception:
    cv2 = None

BODY, HAND = 18, 21
VJ = BODY + 2 * HAND  # 60

INSTR = {"vn": "violin", "va": "viola", "vc": "cello", "db": "double_bass",
         "fl": "flute", "ob": "oboe", "cl": "clarinet", "sax": "saxophone",
         "bn": "bassoon", "tpt": "trumpet", "hn": "horn", "tbn": "trombone",
         "tba": "tuba"}

# BlazePose(33) index -> COCO-18 (OpenPose ordering). Neck (idx 1) computed sep.
BP2COCO = {0: 0, 12: 2, 14: 3, 16: 4, 11: 5, 13: 6, 15: 7, 24: 8, 26: 9,
           28: 10, 23: 11, 25: 12, 27: 13, 5: 14, 2: 15, 8: 16, 7: 17}

POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")


def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def find_pieces(root):
    seps = glob.glob(os.path.join(root, "**", "AuSep_*.wav"), recursive=True)
    return sorted({os.path.dirname(p) for p in seps})


def parse_stems(pdir):
    out = []
    for s in sorted(glob.glob(os.path.join(pdir, "AuSep_*.wav"))):
        b = os.path.basename(s).split("_")
        out.append((int(b[1]), INSTR.get(b[2], b[2]), s))  # (k, instrument, path)
    return sorted(out)


def piece_name(pdir):
    m = glob.glob(os.path.join(pdir, "AuMix_*.wav"))
    base = os.path.basename(m[0]) if m else os.path.basename(pdir)
    return base.replace("AuMix_", "").replace(".wav", "")


# ---------- mediapipe helpers ----------
def build_detectors(cache, max_people):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    pose_task = fetch(POSE_URL, os.path.join(cache, "pose.task"))
    hand_task = fetch(HAND_URL, os.path.join(cache, "hand.task"))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=pose_task),
        running_mode=vision.RunningMode.IMAGE, num_poses=max_people,
        min_pose_detection_confidence=0.3))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=hand_task),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3))
    return mp, pose, hands


def person_bbox(lms, w, h, pad=0.15):
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - pw)), max(0, int(y0 - ph)),
            min(w, int(x1 + pw)), min(h, int(y1 + ph)))


# ---------- core ----------
def extract_piece(pdir, args, detectors=None):
    """Return (players, frame_paths) where players[i] = dict(k, instr, stem,
    pose[T,60,3], crops[list of jpg paths per frame])."""
    name = piece_name(pdir)
    stems = parse_stems(pdir)
    vids = glob.glob(os.path.join(pdir, "Vid_*.mp4"))
    if not vids or not stems:
        return name, [], []
    frames_dir = os.path.join(args.tmp, name)
    os.makedirs(frames_dir, exist_ok=True)
    if not glob.glob(os.path.join(frames_dir, "*.jpg")):
        sh(["ffmpeg", "-y", "-i", vids[0], "-vf", f"fps={args.fps},scale=-2:480",
            os.path.join(frames_dir, "%06d.jpg")])
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    n_players = len(stems)

    # zeros mode: no detection, shared full-frame context, zero pose
    if args.pose == "zeros":
        players = [dict(k=k, instr=instr, stem=s, pose=None, crops=None)
                   for (k, instr, s) in stems]
        return name, players, frame_paths

    mp, pose_det, hand_det = detectors
    T = len(frame_paths)
    poses = [np.zeros((T, VJ, 3), np.float32) for _ in range(n_players)]
    crops = [[None] * T for _ in range(n_players)]
    size = args.frame_size
    overlay = None

    for t, fp in enumerate(frame_paths):
        bgr = cv2.imread(fp)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = pose_det.detect(img)
        people = res.pose_landmarks or []
        # order people left->right by mean x
        order = sorted(range(len(people)),
                       key=lambda i: np.mean([lm.x for lm in people[i]]))
        if t == 0:
            overlay = bgr.copy()
        for slot, pi in enumerate(order[:n_players]):
            lms = people[pi]
            x0, y0, x1, y1 = person_bbox(lms, w, h)
            bw_, bh_ = max(1, x1 - x0), max(1, y1 - y0)
            J = np.zeros((VJ, 3), np.float32)
            for bp, ci in BP2COCO.items():
                lm = lms[bp]
                J[ci, 0] = (lm.x * w - x0) / bw_ * size
                J[ci, 1] = (lm.y * h - y0) / bh_ * size
                J[ci, 2] = getattr(lm, "visibility", 1.0)
            # neck = midpoint of shoulders (bp 11,12)
            ls, rs = lms[11], lms[12]
            J[1, 0] = ((ls.x + rs.x) / 2 * w - x0) / bw_ * size
            J[1, 1] = ((ls.y + rs.y) / 2 * h - y0) / bh_ * size
            J[1, 2] = min(getattr(ls, "visibility", 1.0), getattr(rs, "visibility", 1.0))
            # crop + hands
            crop = bgr[y0:y1, x0:x1]
            if crop.size:
                crop_rs = cv2.resize(crop, (size, size))
                cpath = os.path.join(frames_dir, f"crop_s{slot}_{t:06d}.jpg")
                cv2.imwrite(cpath, crop_rs)
                crops[slot][t] = cpath
                himg = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB))
                hres = hand_det.detect(himg)
                for hi, hl in enumerate(hres.hand_landmarks or []):
                    label = hres.handedness[hi][0].category_name.lower()
                    base = BODY if label.startswith("r") else BODY + HAND
                    for j, lm in enumerate(hl):
                        J[base + j, 0] = lm.x * size
                        J[base + j, 1] = lm.y * size
                        J[base + j, 2] = 1.0
            poses[slot][t] = J
            if t == 0 and overlay is not None:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
                inst = stems[slot][1] if slot < len(stems) else "?"
                cv2.putText(overlay, f"slot{slot}:{inst}", (x0, max(20, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if overlay is not None:
        os.makedirs(os.path.join(args.out, "debug"), exist_ok=True)
        cv2.imwrite(os.path.join(args.out, "debug", f"{name}.jpg"), overlay)

    players = []
    for slot, (k, instr, s) in enumerate(stems):
        players.append(dict(k=k, instr=instr, stem=s, pose=poses[slot], crops=crops[slot]))
    return name, players, frame_paths


def segment_and_write(name, players, frame_paths, args, writer):
    clip_len = int(args.clip_seconds * args.sr)
    seg_frames = args.num_frames
    fhop = int(args.seg_hop * args.fps)
    audio_dir = os.path.join(args.out, "audio")
    pose_dir = os.path.join(args.out, "pose")
    ctx_dir = os.path.join(args.out, "frames")
    for d in (audio_dir, pose_dir, ctx_dir):
        os.makedirs(d, exist_ok=True)

    for p in players:
        wav_rs = os.path.join(args.tmp, f"{name}_k{p['k']}.wav")
        if not os.path.exists(wav_rs):
            sh(["ffmpeg", "-y", "-i", p["stem"], "-ac", "1", "-ar", str(args.sr), wav_rs])
        wav, _ = sf.read(wav_rs, dtype="float32")
        n_seg = 1 + max(0, (len(wav) - clip_len) // (int(args.seg_hop * args.sr)))
        for si in range(n_seg):
            a0 = si * int(args.seg_hop * args.sr)
            a1 = a0 + clip_len
            f0 = si * fhop
            if a1 > len(wav) or f0 + seg_frames > len(frame_paths):
                break
            clip = f"{name}__p{p['k']}_{p['instr']}__s{si:03d}"
            sf.write(os.path.join(audio_dir, clip + ".wav"), wav[a0:a1], args.sr)
            # pose
            if args.pose == "zeros" or p["pose"] is None:
                pose = np.zeros((seg_frames, VJ, 3), np.float32)
            else:
                pose = p["pose"][f0:f0 + seg_frames]
                if len(pose) < seg_frames:
                    pose = np.pad(pose, ((0, seg_frames - len(pose)), (0, 0), (0, 0)))
            np.save(os.path.join(pose_dir, clip + ".npy"), pose.astype(np.float32))
            # context frame (middle of segment)
            mid = f0 + seg_frames // 2
            if args.pose != "zeros" and p["crops"] and p["crops"][mid]:
                src_img = p["crops"][mid]
            else:
                src_img = frame_paths[min(mid, len(frame_paths) - 1)]
            img = cv2.imread(src_img)
            img = cv2.resize(img, (args.frame_size, args.frame_size))
            cv2.imwrite(os.path.join(ctx_dir, clip + ".jpg"), img)
            writer.writerow({"clip": clip, "category": p["instr"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urmp_root", default="/kaggle/input")
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--tmp", default="/kaggle/working/_urmp_tmp")
    ap.add_argument("--pose", choices=["zeros", "mediapipe"], default="mediapipe")
    ap.add_argument("--sr", type=int, default=11025)
    ap.add_argument("--clip_seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--seg_hop", type=float, default=6.0)
    ap.add_argument("--max_people", type=int, default=5)
    ap.add_argument("--max_pieces", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if sf is None or cv2 is None:
        raise SystemExit("pip install soundfile opencv-python (and mediapipe for --pose mediapipe)")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    pieces = find_pieces(args.urmp_root)
    if args.max_pieces:
        pieces = pieces[:args.max_pieces]
    if not pieces:
        raise SystemExit(f"No AuSep_*.wav found under {args.urmp_root}")
    print(f"pieces: {len(pieces)} | pose mode: {args.pose}")

    detectors = None
    if args.pose == "mediapipe":
        detectors = build_detectors(args.tmp, args.max_people)

    meta_path = os.path.join(args.out, "meta.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "category"])
        writer.writeheader()
        for i, pdir in enumerate(pieces):
            name, players, frames = extract_piece(pdir, args, detectors)
            if not players:
                print(f"[{i+1}/{len(pieces)}] {name}: skipped (no video/stems)")
                continue
            segment_and_write(name, players, frames, args, writer)
            print(f"[{i+1}/{len(pieces)}] {name}: {len(players)} players done")
    print("meta.csv written ->", meta_path)


if __name__ == "__main__":
    main()

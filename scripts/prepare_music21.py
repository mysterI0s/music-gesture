#!/usr/bin/env python3
"""Adapt the MUSIC-21 dataset into the Music-Gesture repo's expected layout.

MUSIC-21 is a collection of instrument-performance videos crawled from YouTube
spanning 21 categories. This script consumes the *solo* videos: every video is a
single-instrument performance, so each segmented clip becomes one
Mix-and-Separate item tagged with its instrument category. The loader then
builds hetero (different-instrument) or homo (same-instrument) mixtures on the
fly via ``data.mix_policy`` -- exactly what the paper's 2-stage curriculum needs.

Writes, under --out (default datasets/processed):
    audio/<clip>.wav   mono, --sr, --clip_seconds
    pose/<clip>.npy    float32 [num_frames, 60, 3]  (body18 + Rhand21 + Lhand21)
    frames/<clip>.jpg  context crop, frame_size x frame_size
    meta.csv           clip,category

Videos are read from <videos_root>/<category>/<id>.<ext> (the layout produced by
--download using the MUSIC21 solo JSON). Pass --download to fetch them with
yt-dlp first (skips ids already on disk).

Pose backends mirror scripts/prepare_urmp.py:
    --pose zeros       zeros pose + full-frame context (fast smoke test)
    --pose alphapose   paper-faithful AlphaPose Halpe-136 wholebody on GPU
    --pose mediapipe   MediaPipe Pose + Hands per key-frame (CPU fallback)

Each MUSIC-21 solo has ONE musician, so -- unlike URMP -- we keep the single
highest-confidence detected person per frame (no player<->stem matching).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, shutil, subprocess, urllib.request
from collections import defaultdict
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

# Halpe-136 index -> COCO-18 (OpenPose ordering). Halpe has an explicit neck (18).
HALPE2COCO = {0: 0, 18: 1, 6: 2, 8: 3, 10: 4, 5: 5, 7: 6, 9: 7, 12: 8,
              14: 9, 16: 10, 11: 11, 13: 12, 15: 13, 2: 14, 1: 15, 4: 16, 3: 17}
HALPE_RHAND0, HALPE_LHAND0 = 115, 94  # 21 keypoints each in Halpe-136

# MediaPipe BlazePose(33) -> COCO-18; neck (idx 1) = shoulder midpoint.
BP2COCO = {0: 0, 12: 2, 14: 3, 16: 4, 11: 5, 13: 6, 15: 7, 24: 8, 26: 9,
           28: 10, 23: 11, 25: 12, 27: 13, 5: 14, 2: 15, 8: 16, 7: 17}
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")

VID_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.avi", "*.m4v")


def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_frames(video, frames_dir, args):
    """Decode frames at args.fps. Honors args._max_seconds to cap how much of the
    video is read (-t as an input option), which is the main speed lever: long
    videos no longer cost minutes of pose inference on frames we never use."""
    cmd = ["ffmpeg", "-y"]
    if getattr(args, "_max_seconds", 0):
        cmd += ["-t", str(args._max_seconds)]
    cmd += ["-i", video, "-vf", f"fps={args.fps},scale=-2:480",
            os.path.join(frames_dir, "%06d.jpg")]
    sh(cmd)


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


# ---------- MUSIC21 solo JSON + yt-dlp download ----------
def load_solo_json(path):
    """Return {category: [youtube_id, ...]} from a MUSIC21 solo JSON.
    Accepts both {"videos": {cat: [...]}} and a flat {cat: [...]}."""
    obj = json.load(open(path))
    cats = obj.get("videos", obj)
    out = {}
    for cat, ids in cats.items():
        clean = []
        for v in ids:
            # entries may be bare ids or full URLs
            v = str(v).strip()
            if "watch?v=" in v:
                v = v.split("watch?v=")[-1].split("&")[0]
            elif "youtu.be/" in v:
                v = v.split("youtu.be/")[-1].split("?")[0]
            clean.append(v)
        out[cat] = clean
    return out


def download_videos(cat_ids, videos_root, max_per_cat=0, height=480, cookiefile=None):
    """Fetch each youtube id into <videos_root>/<category>/<id>.mp4 via yt-dlp.
    Skips ids already present; never raises on a single failed video."""
    fmt = f"bv*[height<={height}]+ba/b[height<={height}]"
    for cat, ids in cat_ids.items():
        outdir = os.path.join(videos_root, cat)
        os.makedirs(outdir, exist_ok=True)
        todo = ids[:max_per_cat] if max_per_cat else ids
        for vid in todo:
            if glob.glob(os.path.join(outdir, vid + ".*")):
                continue
            cmd = ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
                   "-o", os.path.join(outdir, vid + ".%(ext)s"),
                   "--no-warnings", "--ignore-errors", "--no-playlist"]
            if cookiefile:
                cmd += ["--cookies", cookiefile]
            cmd += [f"https://www.youtube.com/watch?v={vid}"]
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"  [skip] {cat}/{vid}: {e}")


def find_videos_by_category(videos_root, cat_ids=None):
    """Return list of (category, video_path). Category is the parent dir name,
    optionally filtered to the categories present in cat_ids."""
    wanted = set(cat_ids) if cat_ids else None
    out = []
    for cat in sorted(os.listdir(videos_root)):
        cdir = os.path.join(videos_root, cat)
        if not os.path.isdir(cdir):
            continue
        if wanted is not None and cat not in wanted:
            continue
        vids = []
        for ext in VID_EXTS:
            vids += glob.glob(os.path.join(cdir, ext))
        for v in sorted(vids):
            out.append((cat, v))
    return out


class AlphaPoseWorker:
    """Persistent AlphaPose process: loads the YOLO detector + pose model ONCE
    and answers many frame-dir requests over a pipe, instead of reloading both
    models for every video (the ~10 s/video overhead visible as repeated
    'Loading YOLO model..' / 'Loading pose model...' lines). run_alphapose
    disables it automatically on any error, so a version mismatch just falls
    back to the per-video subprocess."""
    READY = "__AP_READY__"
    RESULT = "__AP_RESULT__"

    def __init__(self, args):
        self.args = args
        self.proc = None

    def start(self):
        worker_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "alphapose_worker.py")
        if not os.path.exists(worker_py):
            raise FileNotFoundError(worker_py)
        env = os.environ.copy()
        env["PYTHONPATH"] = (self.args.alphapose_root + os.pathsep
                             + env.get("PYTHONPATH", ""))
        cmd = ["python", worker_py,
               "--cfg", self.args.ap_cfg, "--checkpoint", self.args.ap_ckpt,
               "--detector", self.args.ap_detector, "--gpus", self.args.ap_gpus]
        self.proc = subprocess.Popen(
            cmd, cwd=self.args.alphapose_root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in self.proc.stdout:
            if line.startswith(self.READY):
                return self
        raise RuntimeError("AlphaPose worker did not become ready")

    def infer(self, frames_dir):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("AlphaPose worker is not running")
        outdir = os.path.join(self.args.tmp, "_ap",
                              os.path.basename(frames_dir.rstrip("/")))
        os.makedirs(outdir, exist_ok=True)
        self.proc.stdin.write(json.dumps({"indir": frames_dir,
                                          "outdir": outdir}) + "\n")
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            if line.startswith(self.RESULT):
                resp = json.loads(line[len(self.RESULT):].strip())
                if not resp.get("ok"):
                    raise RuntimeError(resp.get("error", "worker error"))
                return os.path.join(outdir, "alphapose-results.json")
        raise RuntimeError("AlphaPose worker closed unexpectedly")

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ---------- AlphaPose (RMPE) : paper-faithful GPU wholebody ----------
def run_alphapose(frames_dir, args):
    worker = getattr(args, "_ap_worker", None)
    if worker is not None:
        try:
            return worker.infer(frames_dir)
        except Exception as e:
            print(f"  [ap-worker] failed ({type(e).__name__}: {e}); "
                  f"falling back to per-video subprocess")
            try:
                worker.close()
            except Exception:
                pass
            args._ap_worker = None
    outdir = os.path.join(args.tmp, "_ap", os.path.basename(frames_dir.rstrip("/")))
    os.makedirs(outdir, exist_ok=True)
    demo = os.path.join(args.alphapose_root, "scripts", "demo_inference.py")
    argv = ["--cfg", args.ap_cfg, "--checkpoint", args.ap_ckpt,
            "--indir", frames_dir, "--outdir", outdir, "--sp",
            "--detector", args.ap_detector, "--gpus", args.ap_gpus]
    boot = (
        "import numpy as _np\n"
        "for _n,_t in (('float',float),('int',int),('bool',bool),('object',object),"
        "('str',str),('complex',complex),('long',int),('unicode',str)):\n"
        "    hasattr(_np,_n) or setattr(_np,_n,_t)\n"
        "import runpy,sys\n"
        "sys.argv=[{d!r}]+{a!r}\n"
        "runpy.run_path({d!r},run_name='__main__')\n"
    ).format(d=demo, a=argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = args.alphapose_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(["python", "-c", boot], check=True,
                   cwd=args.alphapose_root, env=env)
    return os.path.join(outdir, "alphapose-results.json")


def pts_bbox(xs, ys, w, h, pad=0.15):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0, int(x0 - pw)), max(0, int(y0 - ph)),
            min(w, int(x1 + pw)), min(h, int(y1 + ph)))


def _halpe_to_J(kp, x0, y0, bw_, bh_, size):
    """kp: [136,3] absolute px -> J [60,3] crop-relative px in [0,size]."""
    J = np.zeros((VJ, 3), np.float32)
    for hp, ci in HALPE2COCO.items():
        J[ci, 0] = (kp[hp, 0] - x0) / bw_ * size
        J[ci, 1] = (kp[hp, 1] - y0) / bh_ * size
        J[ci, 2] = kp[hp, 2]
    for j in range(HAND):
        J[BODY + j, 0] = (kp[HALPE_RHAND0 + j, 0] - x0) / bw_ * size
        J[BODY + j, 1] = (kp[HALPE_RHAND0 + j, 1] - y0) / bh_ * size
        J[BODY + j, 2] = kp[HALPE_RHAND0 + j, 2]
    for j in range(HAND):
        J[BODY + HAND + j, 0] = (kp[HALPE_LHAND0 + j, 0] - x0) / bw_ * size
        J[BODY + HAND + j, 1] = (kp[HALPE_LHAND0 + j, 1] - y0) / bh_ * size
        J[BODY + HAND + j, 2] = kp[HALPE_LHAND0 + j, 2]
    return J


def _alphapose_pose_from_results(results, frame_paths, args):
    """Map an alphapose-results.json onto pose[T,60,3] + crops[T] for the given
    ordered frame_paths, keeping the single highest-confidence person per frame."""
    data = json.load(open(results))
    size = args.frame_size
    T = len(frame_paths)
    idx_of = {os.path.basename(fp): t for t, fp in enumerate(frame_paths)}
    by_frame = defaultdict(list)
    for e in data:
        t = idx_of.get(os.path.basename(str(e.get("image_id", ""))))
        if t is not None:
            by_frame[t].append(e)
    pose = np.zeros((T, VJ, 3), np.float32)
    crops = [None] * T
    for t in range(T):
        people = by_frame.get(t, [])
        if not people:
            continue
        kps = [np.asarray(e["keypoints"], np.float32).reshape(-1, 3) for e in people]
        # solo: keep the most confident person (mean body-keypoint score)
        pi = int(np.argmax([float(np.mean(k[:26, 2])) for k in kps]))
        kp = kps[pi]
        bgr = cv2.imread(frame_paths[t])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = pts_bbox(kp[:26, 0], kp[:26, 1], w, h)
        bw_, bh_ = max(1, x1 - x0), max(1, y1 - y0)
        pose[t] = _halpe_to_J(kp, x0, y0, bw_, bh_, size)
        crop = bgr[y0:y1, x0:x1]
        if crop.size:
            crop_rs = cv2.resize(crop, (size, size))
            cpath = os.path.join(os.path.dirname(frame_paths[t]), f"crop_{t:06d}.jpg")
            cv2.imwrite(cpath, crop_rs)
            crops[t] = cpath
    return pose, crops


def extract_video_alphapose(video, args):
    """Solo backend: AlphaPose Halpe-136 on every frame, keep the single
    highest-confidence person per frame. Returns (frame_paths, pose[T,60,3],
    crops[list])."""
    name = os.path.splitext(os.path.basename(video))[0]
    frames_dir = os.path.join(args.tmp, "frames", name)
    os.makedirs(frames_dir, exist_ok=True)
    if not glob.glob(os.path.join(frames_dir, "*.jpg")):
        extract_frames(video, frames_dir, args)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    T = len(frame_paths)
    size = args.frame_size
    if T == 0:
        return [], None, None

    results = run_alphapose(frames_dir, args)
    pose, crops = _alphapose_pose_from_results(results, frame_paths, args)
    return frame_paths, pose, crops


def select_windows(wav, args):
    """Choose up to args.max_clips_per_video clip windows SPREAD across the whole
    track, skipping (near-)silent windows. Returns sorted start-seconds.

    This is the key difference from a naive 'first N clips' cap: concert intros,
    tuning, applause and silent gaps are NOT over-sampled -- clips are drawn from
    voiced regions across the entire video, so the dataset stays representative.
    """
    sr = args.sr
    clip_len = int(args.clip_seconds * sr)
    if len(wav) < clip_len:
        return []
    cands = []
    s = 0
    while s + clip_len <= len(wav):
        seg = wav[s:s + clip_len].astype(np.float64)
        rms = float(np.sqrt(np.mean(seg * seg)) + 1e-9)
        cands.append((s / sr, rms))
        s += clip_len  # non-overlapping candidate tiles across the whole video
    if not cands:
        return []
    voiced = [c for c in cands if c[1] >= args.min_rms]
    pool = voiced if voiced else cands  # if the whole clip is quiet, keep it
    N = args.max_clips_per_video
    if N <= 0 or len(pool) <= N:
        return [round(c[0], 3) for c in pool]
    # pick N positions evenly spaced across the surviving (voiced) tiles
    idxs = sorted(set(int(i) for i in np.linspace(0, len(pool) - 1, N).round()))
    return [round(pool[i][0], 3) for i in idxs]


def process_alphapose_windows(vi, total, category, video, args, writer):
    """Sampled-window AlphaPose path (used when --max_clips_per_video > 0).
    Extracts frames for only a handful of windows spread across the video and
    poses them in ONE AlphaPose call, so a long concert costs a fraction of a
    full-video pass and clips come from voiced regions -- never just the intro.
    """
    name = os.path.splitext(os.path.basename(video))[0]
    # 1) full audio: needed both to pick voiced windows and to cut each clip wav
    wav_path = os.path.join(args.tmp, name + ".wav")
    if not os.path.exists(wav_path):
        sh(["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", str(args.sr), wav_path])
    wav, _ = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    starts = select_windows(wav, args)
    if not starts:
        print(f"[{vi+1}/{total}] {category}/{name}: skipped (no voiced audio)")
        return 0
    # 2) extract only the sampled windows' frames into one dir (prefixed so we
    #    can regroup them after a single pose pass)
    frames_dir = os.path.join(args.tmp, "frames", name)
    os.makedirs(frames_dir, exist_ok=True)
    for wi, start in enumerate(starts):
        sh(["ffmpeg", "-y", "-ss", str(start), "-t", str(args.clip_seconds),
            "-i", video, "-vf", f"fps={args.fps},scale=-2:480",
            os.path.join(frames_dir, f"w{wi:03d}_%06d.jpg")])
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "w*.jpg")))
    if not frame_paths:
        print(f"[{vi+1}/{total}] {category}/{name}: skipped (no frames)")
        return 0
    # 3) ONE pose pass over all sampled frames
    results = run_alphapose(frames_dir, args)
    pose, crops = _alphapose_pose_from_results(results, frame_paths, args)
    # 4) write one clip per window
    clip_len = int(args.clip_seconds * args.sr)
    seg_frames = args.num_frames
    audio_dir = os.path.join(args.out, "audio")
    pose_dir = os.path.join(args.out, "pose")
    ctx_dir = os.path.join(args.out, "frames")
    written = 0
    for wi, start in enumerate(starts):
        pref = f"w{wi:03d}_"
        idxs = [t for t, fp in enumerate(frame_paths)
                if os.path.basename(fp).startswith(pref)]
        if len(idxs) < seg_frames // 2:
            continue
        idxs = idxs[:seg_frames]
        a0 = int(round(start * args.sr))
        a1 = a0 + clip_len
        if a1 > len(wav):
            continue
        clip = f"{category}__{name}__s{wi:03d}"
        sf.write(os.path.join(audio_dir, clip + ".wav"), wav[a0:a1], args.sr)
        seg_pose = pose[idxs]
        if len(seg_pose) < seg_frames:
            seg_pose = np.pad(seg_pose,
                              ((0, seg_frames - len(seg_pose)), (0, 0), (0, 0)))
        np.save(os.path.join(pose_dir, clip + ".npy"), seg_pose.astype(np.float32))
        cti = idxs[len(idxs) // 2] if args.context_frame == "middle" else idxs[0]
        src_img = crops[cti] if crops[cti] else frame_paths[cti]
        img = cv2.imread(src_img)
        img = cv2.resize(img, (args.frame_size, args.frame_size))
        cv2.imwrite(os.path.join(ctx_dir, clip + ".jpg"), img)
        writer.writerow({"clip": clip, "category": category})
        written += 1
    print(f"[{vi+1}/{total}] {category}/{name}: {written} clips "
          f"(sampled {len(starts)} windows across the video)")
    return written


# ---------- MediaPipe CPU fallback (single person) ----------
def build_mp_detectors(cache):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    pose_task = fetch(POSE_URL, os.path.join(cache, "pose.task"))
    hand_task = fetch(HAND_URL, os.path.join(cache, "hand.task"))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=pose_task),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=hand_task),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3))
    return mp, pose, hands


def extract_video_mediapipe(video, args, detectors):
    name = os.path.splitext(os.path.basename(video))[0]
    frames_dir = os.path.join(args.tmp, "frames", name)
    os.makedirs(frames_dir, exist_ok=True)
    if not glob.glob(os.path.join(frames_dir, "*.jpg")):
        extract_frames(video, frames_dir, args)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    T = len(frame_paths)
    size = args.frame_size
    if T == 0:
        return [], None, None
    mp, pose_det, hand_det = detectors
    pose = np.zeros((T, VJ, 3), np.float32)
    crops = [None] * T
    stride = max(1, args.pose_stride)
    key_ts = list(range(0, T, stride))
    if T and key_ts[-1] != T - 1:
        key_ts.append(T - 1)
    for t in key_ts:
        bgr = cv2.imread(frame_paths[t])
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = pose_det.detect(img)
        people = res.pose_landmarks or []
        if not people:
            continue
        lms = people[0]
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(w, int(max(xs))), min(h, int(max(ys)))
        bw_, bh_ = max(1, x1 - x0), max(1, y1 - y0)
        for bp, ci in BP2COCO.items():
            lm = lms[bp]
            pose[t, ci, 0] = (lm.x * w - x0) / bw_ * size
            pose[t, ci, 1] = (lm.y * h - y0) / bh_ * size
            pose[t, ci, 2] = getattr(lm, "visibility", 1.0)
        ls, rs = lms[11], lms[12]
        pose[t, 1, 0] = ((ls.x + rs.x) / 2 * w - x0) / bw_ * size
        pose[t, 1, 1] = ((ls.y + rs.y) / 2 * h - y0) / bh_ * size
        pose[t, 1, 2] = min(getattr(ls, "visibility", 1.0), getattr(rs, "visibility", 1.0))
        crop = bgr[y0:y1, x0:x1]
        if crop.size:
            crop_rs = cv2.resize(crop, (size, size))
            cpath = os.path.join(frames_dir, f"crop_{t:06d}.jpg")
            cv2.imwrite(cpath, crop_rs)
            crops[t] = cpath
            himg = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB))
            hres = hand_det.detect(himg)
            for hi, hl in enumerate(hres.hand_landmarks or []):
                label = hres.handedness[hi][0].category_name.lower()
                base = BODY if label.startswith("r") else BODY + HAND
                for j, lm in enumerate(hl):
                    pose[t, base + j, 0] = lm.x * size
                    pose[t, base + j, 1] = lm.y * size
                    pose[t, base + j, 2] = 1.0
    if stride > 1 and T > 1 and key_ts:
        key_arr = np.asarray(key_ts)
        all_t = np.arange(T)
        flat = pose[key_arr].reshape(len(key_arr), -1)
        filled = np.empty((T, flat.shape[1]), np.float32)
        for c in range(flat.shape[1]):
            filled[:, c] = np.interp(all_t, key_arr, flat[:, c])
        pose = filled.reshape(T, VJ, 3).astype(np.float32)
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
    return frame_paths, pose, crops


# ---------- segmentation / writing ----------
def segment_and_write(name, category, video, frame_paths, pose, crops, args, writer):
    clip_len = int(args.clip_seconds * args.sr)
    seg_frames = args.num_frames
    ahop = int(args.seg_hop * args.sr)
    fhop = int(args.seg_hop * args.fps)
    audio_dir = os.path.join(args.out, "audio")
    pose_dir = os.path.join(args.out, "pose")
    ctx_dir = os.path.join(args.out, "frames")

    wav_rs = os.path.join(args.tmp, f"{name}.wav")
    if not os.path.exists(wav_rs):
        wcmd = ["ffmpeg", "-y"]
        if getattr(args, "_max_seconds", 0):
            wcmd += ["-t", str(args._max_seconds)]
        wcmd += ["-i", video, "-ac", "1", "-ar", str(args.sr), wav_rs]
        sh(wcmd)
    wav, _ = sf.read(wav_rs, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    n_seg = 1 + max(0, (len(wav) - clip_len) // ahop)
    written = 0
    for si in range(n_seg):
        a0, a1 = si * ahop, si * ahop + clip_len
        f0 = si * fhop
        if a1 > len(wav) or f0 + seg_frames > len(frame_paths):
            break
        clip = f"{category}__{name}__s{si:03d}"
        sf.write(os.path.join(audio_dir, clip + ".wav"), wav[a0:a1], args.sr)
        if pose is None:
            seg_pose = np.zeros((seg_frames, VJ, 3), np.float32)
        else:
            seg_pose = pose[f0:f0 + seg_frames]
            if len(seg_pose) < seg_frames:
                seg_pose = np.pad(seg_pose, ((0, seg_frames - len(seg_pose)), (0, 0), (0, 0)))
        np.save(os.path.join(pose_dir, clip + ".npy"), seg_pose.astype(np.float32))
        ctx_idx = f0 if args.context_frame == "first" else f0 + seg_frames // 2
        ctx_idx = min(ctx_idx, len(frame_paths) - 1)
        if crops is not None and crops[ctx_idx]:
            src_img = crops[ctx_idx]
        else:
            src_img = frame_paths[ctx_idx]
        img = cv2.imread(src_img)
        img = cv2.resize(img, (args.frame_size, args.frame_size))
        cv2.imwrite(os.path.join(ctx_dir, clip + ".jpg"), img)
        writer.writerow({"clip": clip, "category": category})
        written += 1
    return written


def cleanup_tmp(name, args):
    """Remove all per-video temp artifacts (extracted frames, crops, AlphaPose
    json, resampled wav) so streaming preprocessing keeps disk usage bounded."""
    shutil.rmtree(os.path.join(args.tmp, "frames", name), ignore_errors=True)
    shutil.rmtree(os.path.join(args.tmp, "_ap", name), ignore_errors=True)
    try:
        os.remove(os.path.join(args.tmp, name + ".wav"))
    except OSError:
        pass


def process_one(vi, total, category, video, args, detectors, writer):
    """Extract pose/audio/frames for a single video and append its clips.
    Never raises: a corrupt or undecodable video is skipped and 0 is returned."""
    name = os.path.splitext(os.path.basename(video))[0]
    # Skip corrupt / truncated downloads (yt-dlp sometimes reports success for a
    # broken file that ffmpeg cannot decode).
    try:
        size = os.path.getsize(video)
    except OSError:
        size = 0
    if size < 200 * 1024:
        print(f"[{vi+1}/{total}] {category}/{name}: skipped (bad download, {size} bytes)")
        return 0
    try:
        if args.pose == "alphapose":
            if args.max_clips_per_video > 0:
                return process_alphapose_windows(vi, total, category, video,
                                                 args, writer)
            frames, pose, crops = extract_video_alphapose(video, args)
        elif args.pose == "mediapipe":
            frames, pose, crops = extract_video_mediapipe(video, args, detectors)
        else:
            # zeros: still need frames for context + timing
            frames_dir = os.path.join(args.tmp, "frames", name)
            os.makedirs(frames_dir, exist_ok=True)
            if not glob.glob(os.path.join(frames_dir, "*.jpg")):
                extract_frames(video, frames_dir, args)
            frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            pose, crops = None, None
    except Exception as e:
        print(f"[{vi+1}/{total}] {category}/{name}: skipped ({type(e).__name__}: {e})")
        return 0
    if not frames:
        print(f"[{vi+1}/{total}] {category}/{name}: skipped (no frames)")
        return 0
    n = segment_and_write(name, category, video, frames, pose, crops, args, writer)
    print(f"[{vi+1}/{total}] {category}/{name}: {n} clips")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="MUSIC21 solo videos JSON (for --download and category filtering)")
    ap.add_argument("--videos_root", default="/kaggle/working/music21_videos")
    ap.add_argument("--download", action="store_true", help="fetch videos with yt-dlp before preprocessing")
    ap.add_argument("--cookiefile", default="", help="yt-dlp cookies.txt (helps on Kaggle)")
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--tmp", default="/kaggle/working/_music21_tmp")
    ap.add_argument("--pose", choices=["zeros", "alphapose", "mediapipe"], default="alphapose")
    ap.add_argument("--sr", type=int, default=11025)
    ap.add_argument("--clip_seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=48)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--seg_hop", type=float, default=6.0)
    ap.add_argument("--context_frame", choices=["first", "middle"], default="middle")
    ap.add_argument("--pose_stride", type=int, default=1)
    ap.add_argument("--max_per_cat", type=int, default=0, help="0 = all ids per category (download)")
    ap.add_argument("--max_videos", type=int, default=0, help="0 = all videos (preprocess)")
    ap.add_argument("--max_clips_per_video", type=int, default=0,
                    help="0 = whole video. Set e.g. 6 to keep up to N clips per video, "
                         "SAMPLED evenly across the WHOLE video and skipping "
                         "(near-)silent windows (see --min_rms) -- NOT the first N. "
                         "Biggest speed lever: pose runs on ~N*num_frames frames "
                         "instead of every frame of a long concert (alphapose only).")
    ap.add_argument("--min_rms", type=float, default=0.01,
                    help="skip candidate windows whose audio RMS (float32 wav in "
                         "[-1,1]) is below this, so silent concert intros/gaps are "
                         "not sampled. Only used with --max_clips_per_video.")
    ap.add_argument("--no_persistent_ap", dest="persistent_ap",
                    action="store_false",
                    help="disable the persistent AlphaPose worker and load the "
                         "models per video (slower; use only if the worker errors "
                         "on your AlphaPose build).")
    ap.set_defaults(persistent_ap=True)
    ap.add_argument("--shard", default="",
                    help="process only shard idx/count of the video list, e.g. '0/2'. "
                         "Run two processes (one per GPU) to use both T4s; each writes "
                         "meta.part<idx>.csv -- concat them into meta.csv afterward.")
    ap.add_argument("--dl_height", type=int, default=480)
    ap.add_argument("--keep_videos", action="store_true",
                    help="keep source videos on disk (default: delete each after "
                         "processing so the full set fits without overflowing disk)")
    # AlphaPose (RMPE) backend -- paper-faithful GPU wholebody keypoints.
    ap.add_argument("--alphapose_root", default="/kaggle/working/AlphaPose")
    ap.add_argument("--ap_cfg",
                    default="configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml")
    ap.add_argument("--ap_ckpt",
                    default="pretrained_models/halpe136_fast50_regression_256x192.pth")
    ap.add_argument("--ap_detector", default="yolo")
    ap.add_argument("--ap_gpus", default="0")
    args = ap.parse_args()

    # Cap how much of each video we decode/pose (huge speedup for long videos).
    args._max_seconds = 0
    if args.max_clips_per_video:
        args._max_seconds = ((args.max_clips_per_video - 1) * args.seg_hop
                             + args.clip_seconds + 1.0)
    # Optional sharding across processes/GPUs.
    args._shard_idx, args._shard_cnt = 0, 1
    if args.shard:
        _pi, _pc = args.shard.split("/")
        args._shard_idx, args._shard_cnt = int(_pi), int(_pc)

    if sf is None or cv2 is None:
        raise SystemExit("pip install soundfile opencv-python (and mediapipe for --pose mediapipe)")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    for d in ("audio", "pose", "frames"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    cat_ids = load_solo_json(args.json) if args.json else None

    detectors = None
    if args.pose == "mediapipe":
        detectors = build_mp_detectors(args.tmp)

    # Persistent AlphaPose worker: load detector + pose model ONCE for the whole
    # run instead of per video. Auto-falls back to the per-video subprocess.
    args._ap_worker = None
    if args.pose == "alphapose" and args.persistent_ap:
        try:
            args._ap_worker = AlphaPoseWorker(args).start()
            print("AlphaPose persistent worker: ready (models loaded once)")
        except Exception as e:
            print(f"AlphaPose persistent worker unavailable "
                  f"({type(e).__name__}: {e}); using per-video subprocess")
            args._ap_worker = None

    tag = ""
    if args.pose == "alphapose":
        tag = f" | detector: {args.ap_detector} | gpus: {args.ap_gpus}"
    elif args.pose == "mediapipe":
        tag = f" | stride: {args.pose_stride}"

    meta_name = "meta.csv" if args._shard_cnt == 1 else f"meta.part{args._shard_idx}.csv"
    meta_path = os.path.join(args.out, meta_name)
    write_header = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "category"])
        if write_header:
            writer.writeheader()

        if args.download:
            # ---- STREAMING mode: download ONE video, process it, then DELETE it ----
            # Peak disk stays at ~one source video at a time; only the small processed
            # outputs (audio/pose/frames) are kept. This is what lets the full MUSIC-21
            # set be preprocessed on Kaggle without overflowing storage.
            if not cat_ids:
                raise SystemExit("--download requires --json")
            os.makedirs(args.videos_root, exist_ok=True)
            pairs = []
            for cat, ids in cat_ids.items():
                todo = ids[:args.max_per_cat] if args.max_per_cat else ids
                for vid in todo:
                    pairs.append((cat, vid))
            if args.max_videos:
                pairs = pairs[:args.max_videos]
            if args._shard_cnt > 1:
                pairs = pairs[args._shard_idx::args._shard_cnt]
            if not pairs:
                raise SystemExit("no video ids found in --json")
            print(f"videos: {len(pairs)} (streaming) | pose mode: {args.pose}{tag} | "
                  f"keep_videos: {args.keep_videos}")
            for vi, (category, vid) in enumerate(pairs):
                # download just this one id (skips if already on disk)
                download_videos({category: [vid]}, args.videos_root, max_per_cat=0,
                                height=args.dl_height, cookiefile=args.cookiefile or None)
                found = glob.glob(os.path.join(args.videos_root, category, vid + ".*"))
                video = found[0] if found else None
                if not video:
                    print(f"[{vi+1}/{len(pairs)}] {category}/{vid}: skipped (unavailable)")
                    continue
                name = os.path.splitext(os.path.basename(video))[0]
                process_one(vi, len(pairs), category, video, args, detectors, writer)
                # free disk immediately: drop the source video + its temp files
                if not args.keep_videos:
                    try:
                        os.remove(video)
                    except OSError:
                        pass
                cleanup_tmp(name, args)
        else:
            # ---- on-disk mode: videos already present under --videos_root ----
            # (source videos are left in place; only temp files are cleaned)
            videos = find_videos_by_category(args.videos_root, cat_ids)
            if args.max_videos:
                videos = videos[:args.max_videos]
            if args._shard_cnt > 1:
                videos = videos[args._shard_idx::args._shard_cnt]
            if not videos:
                raise SystemExit(f"no videos found under {args.videos_root}")
            print(f"videos: {len(videos)} | pose mode: {args.pose}{tag}")
            for vi, (category, video) in enumerate(videos):
                name = os.path.splitext(os.path.basename(video))[0]
                process_one(vi, len(videos), category, video, args, detectors, writer)
                cleanup_tmp(name, args)
    print("meta.csv written ->", meta_path)
    if getattr(args, "_ap_worker", None) is not None:
        args._ap_worker.close()


if __name__ == "__main__":
    main()

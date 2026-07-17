#!/usr/bin/env python3
"""Persistent AlphaPose worker.

Loads the person detector + pose model ONCE, then serves many frame-directory
requests over stdin/stdout, so preprocessing does not pay the model-load cost
for every video (the repeated 'Loading YOLO model..' / 'Loading pose model...'
lines). It mirrors AlphaPose's own scripts/demo_inference.py image path, reusing
its public modules, so results are identical to the per-video subprocess path.

Protocol (line-delimited, on stdout only):
    parent -> worker : {"indir": "...", "outdir": "..."}   or  {"cmd": "quit"}
    worker -> parent : "__AP_READY__" once, then "__AP_RESULT__ {json}" per req
All of AlphaPose's own logging + tqdm goes to stderr (shown in the cell); stdout
carries only those two markers so the parent can sync reliably.
"""
import sys, os, json, time, contextlib

# AlphaPose targets old numpy; restore removed aliases before importing it.
import numpy as _np
for _n, _t in (("float", float), ("int", int), ("bool", bool), ("object", object),
               ("str", str), ("complex", complex), ("long", int), ("unicode", str)):
    hasattr(_np, _n) or setattr(_np, _n, _t)

import argparse
import torch
from types import SimpleNamespace


def build_opt(a):
    """Assemble the Namespace AlphaPose's DetectionLoader/DataWriter/detector expect."""
    use_cuda = torch.cuda.is_available() and a.gpus not in ("", "-1")
    first = a.gpus.split(",")[0] if use_cuda else "-1"
    device = torch.device("cuda:" + first if use_cuda else "cpu")
    gpu_list = [int(x) for x in a.gpus.split(",")] if use_cuda else [-1]
    return SimpleNamespace(
        cfg=a.cfg, checkpoint=a.checkpoint, detector=a.detector,
        gpus=gpu_list, device=device, sp=True, tracking=False,
        pose_flow=False, pose_track=False, detbatch=a.detbatch,
        posebatch=a.posebatch, qsize=1024, flip=False, min_box_area=0,
        eval=False, format=None, debug=False, vis=False, showbox=False,
        profile=False, save_img=False, save_video=False, vis_fast=False,
        inputpath="", inputlist="", inputimg="", outputpath="",
        video="", webcam=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--detbatch", type=int, default=5)
    ap.add_argument("--posebatch", type=int, default=64)
    a = ap.parse_args()

    from alphapose.utils.config import update_config
    from alphapose.models import builder
    from alphapose.utils.detector import DetectionLoader
    from alphapose.utils.writer import DataWriter
    from alphapose.utils.pPose_nms import write_json
    from detector.apis import get_detector

    opt = build_opt(a)
    cfg = update_config(opt.cfg)

    # ---- load the heavy models ONCE (logs -> stderr so they show in the cell) ----
    with contextlib.redirect_stdout(sys.stderr):
        detector = get_detector(opt)
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        pose_model.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))
        pose_model.to(opt.device)
        pose_model.eval()

    def infer(indir, outdir):
        os.makedirs(outdir, exist_ok=True)
        opt.inputpath = indir
        opt.outputpath = outdir
        names = sorted(f for f in os.listdir(indir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        input_source = [os.path.join(indir, n) for n in names]
        if not input_source:
            write_json([], outdir, form=opt.format, for_eval=False)
            return
        det_loader = DetectionLoader(input_source, detector, cfg, opt,
                                     batchSize=opt.detbatch, mode="image",
                                     queueSize=opt.qsize)
        det_loader.start()
        writer = DataWriter(cfg, opt, save_video=False,
                            queueSize=opt.qsize).start()
        batch = opt.posebatch
        for _ in range(det_loader.length):
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids,
                 cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                inps = inps.to(opt.device)
                dl = inps.size(0)
                leftover = 1 if (dl % batch) else 0
                nb = dl // batch + leftover
                hm = []
                for j in range(nb):
                    hm.append(pose_model(inps[j * batch:min((j + 1) * batch, dl)]))
                hm = torch.cat(hm).cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
        while writer.running():
            time.sleep(0.2)
        writer.stop()
        det_loader.stop()
        write_json(writer.results(), outdir, form=opt.format, for_eval=False)

    print("__AP_READY__", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print("__AP_RESULT__ " + json.dumps({"ok": False, "error": f"bad request: {e}"}), flush=True)
            continue
        if req.get("cmd") == "quit":
            break
        try:
            with contextlib.redirect_stdout(sys.stderr):
                infer(req["indir"], req["outdir"])
            print("__AP_RESULT__ " + json.dumps({"ok": True, "outdir": req["outdir"]}), flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("__AP_RESULT__ " + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()

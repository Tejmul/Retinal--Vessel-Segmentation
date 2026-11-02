"""Export a small set of images for the results viewer.

For each chosen test image this writes the fundus photograph, the human label
and one mask per ablation row, all cropped to the original 584x565 frame and
sized for the web. Images are chosen by how much the proposed model and the
baseline disagree, so the viewer opens on cases where there is something to
see rather than on image 01 by default.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

ORIG_H, ORIG_W = 584, 565
BASE, PROP = "A_sgl_baseline", "D_sgl_cldice"


def _gt(root, idx):
    ve = np.array(Image.open(os.path.join(
        root, "test", "1st_manual", f"{idx}_manual1.gif")))
    ma = np.array(Image.open(os.path.join(root, "test", "mask", f"{idx}_mask.gif")))
    return ((ve > 200) & (ma > 125))[:ORIG_H, :ORIG_W]


def _pred(pred_root, row, idx):
    p = os.path.join(pred_root, row, f"{idx}_pred.png")
    if not os.path.exists(p):
        return None
    return np.array(Image.open(p))[:ORIG_H, :ORIG_W] > 127


def pick(root, pred_root, rows, n):
    """Rank images by how much the proposed model recovers over the baseline."""
    scores = []
    for i in range(1, 21):
        idx = f"{i:02d}"
        gt = _gt(root, idx)
        a, d = _pred(pred_root, BASE, idx), _pred(pred_root, PROP, idx)
        if a is None or d is None:
            continue
        gained = int((d & ~a & gt).sum())      # true vessel the baseline missed
        lost = int((a & ~d & gt).sum())
        scores.append((gained - lost, idx))
    scores.sort(reverse=True)
    chosen = [idx for _, idx in scores[:n - 1]]
    if scores:
        chosen.append(scores[-1][1])           # keep the worst case, honestly
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/DRIVE")
    ap.add_argument("--pred-root", default="../outputs/ablation_predictions")
    ap.add_argument("--out", default="../outputs/web_assets")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    rows = sorted(d for d in os.listdir(args.pred_root)
                  if os.path.isdir(os.path.join(args.pred_root, d)))
    ids = pick(args.root, args.pred_root, rows, args.n)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"ids": ids, "rows": rows, "width": ORIG_W, "height": ORIG_H,
                "images": {}}

    for idx in ids:
        rec = {}
        fundus = Image.open(os.path.join(
            args.root, "test", "images", f"{idx}_training.tif")).convert("RGB")
        fundus = fundus.crop((0, 0, ORIG_W, ORIG_H))
        fundus.save(os.path.join(args.out, f"{idx}_fundus.jpg"),
                    quality=args.quality, optimize=True)
        rec["fundus"] = f"{idx}_fundus.jpg"

        gt = _gt(args.root, idx)
        Image.fromarray((gt * 255).astype(np.uint8)).save(
            os.path.join(args.out, f"{idx}_gt.png"), optimize=True)
        rec["gt"] = f"{idx}_gt.png"

        for row in rows:
            m = _pred(args.pred_root, row, idx)
            if m is None:
                continue
            Image.fromarray((m * 255).astype(np.uint8)).save(
                os.path.join(args.out, f"{idx}_{row}.png"), optimize=True)
            rec[row] = f"{idx}_{row}.png"
        manifest["images"][idx] = rec
        print(f"  [assets] {idx}")

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=2)
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"[assets] {len(ids)} images -> {args.out}  ({total/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

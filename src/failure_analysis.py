"""Failure analysis for the ablation, as promised in the literature review:

  * place the baseline and the proposed model side by side on the same images,
  * count broken branches,
  * and look at where the errors gather.

The central number here is sensitivity split by vessel calibre. The hypothesis
is specifically about thin branches, so a gain that shows up only in the
aggregate Se is much weaker evidence than one concentrated in the thin stratum.
Calibre comes from the Euclidean distance transform of the label: a vessel
pixel whose distance to the background is <= 1.5 px sits in a vessel at most
about three pixels across.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize

ORIG_H, ORIG_W = 584, 565
THIN_RADIUS = 1.5


def _load_gt(root, idx):
    ve = np.array(Image.open(
        os.path.join(root, "test", "1st_manual", f"{idx}_manual1.gif"))).astype(np.float32)
    ma = np.array(Image.open(
        os.path.join(root, "test", "mask", f"{idx}_mask.gif"))).astype(np.float32)
    gt = ((ve > 200) & (ma > 125))[:ORIG_H, :ORIG_W]
    return gt, (ma > 125)[:ORIG_H, :ORIG_W]


def _load_pred(pred_dir, idx):
    p = os.path.join(pred_dir, f"{idx}_pred.png")
    if not os.path.exists(p):
        return None
    return np.array(Image.open(p)).astype(np.uint8)[:ORIG_H, :ORIG_W] > 127


def breaks(gt, pred):
    """Number of distinct missed centre-line segments ("broken branches").

    The label's skeleton is the vessel tree reduced to one-pixel lines. Every
    connected run of skeleton that the prediction fails to cover is one break.
    Single-pixel misses are ignored as annotation jitter.
    """
    skel = skeletonize(gt)
    missed = skel & ~pred
    lab = cc_label(missed, connectivity=2)
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= 2).sum()), int(skel.sum()), int(missed.sum())


def stratified(gt, pred, fov):
    """Sensitivity on thin vs thick vessel pixels, plus false-positive rate."""
    radius = distance_transform_edt(gt)
    thin = gt & (radius <= THIN_RADIUS)
    thick = gt & (radius > THIN_RADIUS)
    bg = fov & ~gt
    f = lambda m: float((pred & m).sum() / max(m.sum(), 1))
    return {
        "se_thin": f(thin), "se_thick": f(thick),
        "n_thin": int(thin.sum()), "n_thick": int(thick.sum()),
        "fpr": f(bg),
    }


def analyse(root, pred_dir, ids):
    out = []
    for idx in ids:
        gt, fov = _load_gt(root, idx)
        pred = _load_pred(pred_dir, idx)
        if pred is None:
            continue
        nb, n_skel, n_missed = breaks(gt, pred)
        rec = {"id": idx, "breaks": nb, "skel_px": n_skel, "missed_skel_px": n_missed,
               "cc_pred": int(cc_label(pred, connectivity=2).max()),
               "cc_gt": int(cc_label(gt, connectivity=2).max())}
        rec.update(stratified(gt, pred, fov))
        out.append(rec)
    return out


def summarise(recs):
    if not recs:
        return {}
    keys = ["breaks", "missed_skel_px", "cc_pred", "se_thin", "se_thick", "fpr"]
    return {k: float(np.mean([r[k] for r in recs])) for k in keys}


def panels(root, rows, ids, out_dir, pred_root):
    """Side-by-side figure per image: fundus | label | baseline | proposed | diff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    base_name, prop_name = "A_sgl_baseline", "D_sgl_cldice"
    for idx in ids:
        gt, _ = _load_gt(root, idx)
        a = _load_pred(os.path.join(pred_root, base_name), idx)
        d = _load_pred(os.path.join(pred_root, prop_name), idx)
        if a is None or d is None:
            continue
        fundus = np.array(Image.open(
            os.path.join(root, "test", "images", f"{idx}_training.tif")))[:ORIG_H, :ORIG_W]

        # where the two models disagree, coloured by who is right
        diff = np.zeros((ORIG_H, ORIG_W, 3), np.uint8)
        diff[gt] = (70, 70, 70)
        diff[d & ~a & gt] = (0, 220, 0)       # proposed recovers a true vessel
        diff[a & ~d & gt] = (0, 140, 255)     # baseline had it, proposed lost it
        diff[d & ~a & ~gt] = (230, 0, 0)      # proposed adds a false vessel

        fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
        for axis, im, title in [
            (ax[0], fundus, f"{idx}  fundus"),
            (ax[1], gt, "label (1st_manual)"),
            (ax[2], a, "A: SGL baseline"),
            (ax[3], d, "D: SGL + clDice"),
            (ax[4], diff, "disagreement"),
        ]:
            axis.imshow(im, cmap=None if im.ndim == 3 else "gray")
            axis.set_title(title, fontsize=11)
            axis.axis("off")
        fig.suptitle("green = clDice recovers a true vessel   |   blue = baseline only   "
                     "|   red = clDice false positive", fontsize=10, y=0.02)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{idx}_compare.png"), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  [panel] {idx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/DRIVE")
    ap.add_argument("--pred-root", default="../outputs/ablation_predictions")
    ap.add_argument("--out", default="../outputs/failure_analysis.json")
    ap.add_argument("--panel-dir", default="../outputs/failure_analysis")
    ap.add_argument("--panel-ids", default="01,03,08,19",
                    help="images to render side by side")
    args = ap.parse_args()

    ids = [f"{i:02d}" for i in range(1, 21)]
    rows = sorted(d for d in os.listdir(args.pred_root)
                  if os.path.isdir(os.path.join(args.pred_root, d)))
    report = {}
    for name in rows:
        recs = analyse(args.root, os.path.join(args.pred_root, name), ids)
        report[name] = {"mean": summarise(recs), "per_image": recs}
        m = report[name]["mean"]
        if m:
            print(f"{name:>16}  breaks {m['breaks']:7.1f}  "
                  f"Se_thin {m['se_thin']:.4f}  Se_thick {m['se_thick']:.4f}  "
                  f"FPR {m['fpr']:.4f}  CC_pred {m['cc_pred']:.0f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[failure] wrote {args.out}")

    if args.panel_ids:
        panels(args.root, rows, args.panel_ids.split(","), args.panel_dir, args.pred_root)


if __name__ == "__main__":
    main()

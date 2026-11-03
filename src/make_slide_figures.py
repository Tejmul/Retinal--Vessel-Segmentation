"""Figures for the phase-2 slide deck.

Two of them:

  * ``gap.png``   -- the didactic picture of why a pixel-wise loss is blind to
    a break: a two-pixel gap costs almost nothing on a pixel count and costs
    the vessel its connectivity.
  * ``crop_<id>.png`` -- the same real region under the baseline and under the
    proposed loss, chosen as the window where the proposed model recovers the
    most centre-line that the baseline missed.

Run from the repository root:  .venv/bin/python src/make_slide_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize

ORIG_H, ORIG_W = 584, 565
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "DRIVE")
PRED = os.path.join(ROOT, "outputs", "ablation_predictions")
OUT = os.path.join(ROOT, "outputs", "slides")

INK, CRIMSON, TEAL, MUTED = "#1A1614", "#B3202C", "#1F6F78", "#6E675F"


# --------------------------------------------------------------- the gap idea
def gap_figure():
    """A vessel with a two-pixel gap, scored both ways."""
    h, w = 34, 150
    vessel = np.zeros((h, w), bool)
    vessel[15:19, 8:142] = True                 # a 4-px-wide vessel
    pred = vessel.copy()
    pred[:, 72:74] = False                      # ... cut in half by a 2-px gap

    wrong = int((vessel ^ pred).sum())
    total = int(vessel.sum())
    n_true = cc_label(vessel, connectivity=2).max()
    n_pred = cc_label(pred, connectivity=2).max()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 2.9))
    for ax, (title, sub, colour) in zip(axes, [
        ("What a pixel-wise loss sees",
         f"{wrong} wrong pixels out of {total}   →   {100*(1-wrong/total):.1f}% correct",
         TEAL),
        ("What the vessel tree sees",
         f"{n_true} connected vessel   →   {n_pred} disconnected pieces",
         CRIMSON),
    ]):
        canvas = np.ones((h, w, 3))
        canvas[pred] = [0.10, 0.09, 0.08]
        ax.imshow(canvas, interpolation="nearest")
        ax.set_title(title, fontsize=13.5, color=INK, pad=11, weight="bold", loc="left")
        ax.text(0, h + 10, sub, fontsize=12, color=colour, va="top")
        ax.axis("off")

    # ring the gap on both panels so the eye goes straight to it
    for ax in axes:
        ax.add_patch(plt.Circle((73, 17), 11, fill=False, ec=CRIMSON, lw=2.0))

    fig.subplots_adjust(left=.02, right=.98, top=.80, bottom=.26, wspace=.12)
    fig.savefig(os.path.join(OUT, "gap.png"), dpi=220, facecolor="white")
    plt.close(fig)
    print("gap.png")


# ------------------------------------------------------- the real comparison
def _load(idx):
    base = os.path.join(DATA, "test")
    hr = np.array(Image.open(os.path.join(base, "images", f"{idx}_training.tif")))[:ORIG_H, :ORIG_W]
    ve = np.array(Image.open(os.path.join(base, "1st_manual", f"{idx}_manual1.gif")))[:ORIG_H, :ORIG_W] > 200
    a = np.array(Image.open(os.path.join(PRED, "A_sgl_baseline", f"{idx}_pred.png")))[:ORIG_H, :ORIG_W] > 127
    d = np.array(Image.open(os.path.join(PRED, "D_sgl_cldice", f"{idx}_pred.png")))[:ORIG_H, :ORIG_W] > 127
    return hr, ve, a, d


def _best_window(gt, a, d, size=150):
    """Window where D recovers the most label skeleton that A missed."""
    skel = skeletonize(gt)
    gain = (skel & d & ~a).astype(np.float32) - 0.6 * (skel & a & ~d).astype(np.float32)
    best, by, bx = -1e9, 0, 0
    for y in range(0, ORIG_H - size, 20):
        for x in range(0, ORIG_W - size, 20):
            s = gain[y:y + size, x:x + size].sum()
            if s > best:
                best, by, bx = s, y, x
    return by, bx, size


def _overlay(hr, gt, pred):
    """Grey where both agree, blue where the label was missed, red where invented."""
    out = (hr.astype(np.float32) * 0.42).astype(np.uint8)
    out[gt & pred] = [205, 205, 205]
    out[gt & ~pred] = [70, 150, 215]
    out[~gt & pred] = [205, 60, 66]
    return out


def crop_figure(idx):
    hr, gt, a, d = _load(idx)
    y, x, s = _best_window(gt, a, d)
    sl = (slice(y, y + s), slice(x, x + s))
    skel = skeletonize(gt)

    def breaks(p):
        lab = cc_label(skel[sl] & ~p[sl], connectivity=2)
        sizes = np.bincount(lab.ravel())[1:]
        return int((sizes >= 2).sum())

    panels = [
        (hr[sl], "Fundus", ""),
        (_overlay(hr, gt, a)[sl], "A  SGL baseline", f"{breaks(a)} broken branches here"),
        (_overlay(hr, gt, d)[sl], "D  SGL + clDice", f"{breaks(d)} broken branches here"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1))
    for ax, (im, title, sub) in zip(axes, panels):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=13, color=INK, pad=9, weight="bold")
        if sub:
            ax.text(.5, -.045, sub, fontsize=11.5, color=CRIMSON, ha="center",
                    va="top", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#D8D3CC")
    fig.text(.5, .018, "blue = vessel the human traced and the model missed      "
                       "red = pixels the model added", fontsize=10.5,
             color=MUTED, ha="center")
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.14, wspace=.07)
    fig.savefig(os.path.join(OUT, f"crop_{idx}.png"), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"crop_{idx}.png  window=({y},{x})  A={breaks(a)}  D={breaks(d)}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gap_figure()
    for i in ("03", "08", "19"):
        crop_figure(i)

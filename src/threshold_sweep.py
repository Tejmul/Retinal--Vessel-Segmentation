"""Is the F1 gap between the rows just a thresholding artefact?

The ablation scores every row at a fixed 100/255. That threshold was chosen
for a cross-entropy model, and a clDice-trained model puts its probability
mass somewhere else -- so a fair reviewer will ask whether row D is being
judged at the wrong operating point. This sweeps the threshold per row and
reports each row's own best mean F1, which settles the question either way.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ORIG_H, ORIG_W, PAD_H, PAD_W   # noqa: E402
from model import build_model                       # noqa: E402
from utils import get_device, predict               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "DRIVE", "test")
CKPT = os.path.join(ROOT, "checkpoints", "ablation")
ROWS = ["B_no_sgl", "C_cldice", "A_sgl_baseline", "D_sgl_cldice"]
IDS = [f"{i:02d}" for i in range(1, 21)]


def load_case(idx):
    hr = np.array(Image.open(os.path.join(DATA, "images", f"{idx}_training.tif")))[:ORIG_H, :ORIG_W]
    ve = np.array(Image.open(os.path.join(DATA, "1st_manual", f"{idx}_manual1.gif")))[:ORIG_H, :ORIG_W]
    ma = np.array(Image.open(os.path.join(DATA, "mask", f"{idx}_mask.gif")))[:ORIG_H, :ORIG_W]
    return hr, (ve > 200), (ma > 125)


@torch.no_grad()
def probs_for(row, device):
    net = build_model()
    state = torch.load(os.path.join(CKPT, f"{row}_last.pt"), map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state), strict=False)
    net.to(device).eval()
    out = []
    for idx in IDS:
        hr, gt, fov = load_case(idx)
        x = np.zeros((PAD_H, PAD_W, 3), np.float32)
        x[:ORIG_H, :ORIG_W] = hr
        t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
        p = predict(net, t, self_ensemble=True)[0, 0].cpu().numpy()[:ORIG_H, :ORIG_W]
        out.append((p * fov, gt, fov))
        print(f"  {row} {idx}", flush=True)
    return out


def mean_dice(cases, t):
    vals = []
    for p, gt, fov in cases:
        m = p > (t / 255.0)
        tp = np.sum(m & gt); fp = np.sum(m & ~gt & fov); fn = np.sum(~m & gt)
        vals.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(vals))


if __name__ == "__main__":
    device = get_device("auto")
    print(f"device={device}", flush=True)
    report = {}
    for row in ROWS:
        cases = probs_for(row, device)
        grid = {t: mean_dice(cases, t) for t in range(20, 231, 5)}
        best_t = max(grid, key=grid.get)
        report[row] = {"dice_at_100": mean_dice(cases, 100),
                       "best_threshold": best_t, "best_dice": grid[best_t],
                       "grid": grid}
        r = report[row]
        print(f"{row}: fixed@100 {r['dice_at_100']:.4f}   "
              f"best {r['best_dice']:.4f} @ {best_t}/255", flush=True)
    with open(os.path.join(ROOT, "outputs", "threshold_sweep.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote outputs/threshold_sweep.json")

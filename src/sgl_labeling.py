"""Study Group Learning: K-fold pseudo-label (teacher) generation.

The 20 DRIVE training images are divided into K groups. For each group we train
a CON model on all the *other* groups and use it (with 8-way self-ensemble) to
predict soft vessel maps for the held-out group. Because every image is labelled
by a model that never saw it, the resulting pseudo-labels act as a denoised /
cross-validated "study group" consensus, which later supervises the final model
alongside the human annotations.
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
import torch

from dataset import ORIG_H, ORIG_W, PAD_H, PAD_W
from train import train_model
from utils import get_device, predict


def all_training_ids(root):
    img_dir = os.path.join(root, "training", "images")
    return sorted(os.path.basename(p).split("_")[0]
                  for p in glob.glob(os.path.join(img_dir, "*.tif")))


def make_folds(ids, k):
    """Split ids into k contiguous groups (sizes differ by at most one)."""
    n = len(ids)
    folds, start = [], 0
    for i in range(k):
        size = (n - start) // (k - i)
        folds.append(ids[start:start + size])
        start += size
    return [f for f in folds if f]


def _load_padded(root, idx, device):
    hr = np.array(Image.open(
        os.path.join(root, "training", "images", f"{idx}_training.tif"))).astype(np.float32)
    canvas = np.zeros((PAD_H, PAD_W, 3), np.float32)
    canvas[:hr.shape[0], :hr.shape[1]] = hr
    t = torch.from_numpy(canvas.transpose(2, 0, 1))[None].to(device)
    return t


@torch.no_grad()
def save_pseudo_label(model, root, idx, out_dir, device, self_ensemble=True):
    os.makedirs(out_dir, exist_ok=True)
    t = _load_padded(root, idx, device)
    prob = predict(model, t, self_ensemble=self_ensemble)[0, 0].cpu().numpy()
    prob = prob[:ORIG_H, :ORIG_W]
    img = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(out_dir, f"{idx}_training_x1_teacher.png"))


def generate_pseudo_labels(root, out_dir, *, k=5, epochs=20, batch_size=4,
                           patch_size=128, lr=1e-4, repeat=40, device=None,
                           self_ensemble=True, ckpt_dir="checkpoints/folds"):
    device = device or get_device("auto")
    ids = all_training_ids(root)
    folds = make_folds(ids, k)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[sgl] {len(ids)} images, {len(folds)} folds: "
          + " | ".join("".join(f) if len(f) < 4 else f"{f[0]}..{f[-1]}" for f in folds))

    for fi, holdout in enumerate(folds):
        train_ids = [i for i in ids if i not in set(holdout)]
        print(f"\n[sgl] fold {fi+1}/{len(folds)}  holdout={holdout}  "
              f"train on {len(train_ids)} images")
        model = train_model(root, train_indices=train_ids, teacher_dir=None,
                            epochs=epochs, batch_size=batch_size,
                            patch_size=patch_size, lr=lr, repeat=repeat,
                            device=device,
                            out_ckpt=os.path.join(ckpt_dir, f"fold{fi+1}.pt"),
                            seed=fi)
        for idx in holdout:
            save_pseudo_label(model, root, idx, out_dir, device, self_ensemble)
            print(f"  [sgl] pseudo-label saved: {idx}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    print(f"\n[sgl] all pseudo-labels written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DRIVE")
    ap.add_argument("--out-dir", default="data/DRIVE/training/teacher_sgl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    generate_pseudo_labels(args.root, args.out_dir, k=args.k, epochs=args.epochs,
                           batch_size=args.batch_size, patch_size=args.patch_size,
                           repeat=args.repeat, device=get_device(args.device))


if __name__ == "__main__":
    main()

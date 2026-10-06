"""Evaluate a trained CON model on the DRIVE test set."""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DriveDataset
from metrics import evaluate_prediction, mean_metrics
from model import build_model
from utils import get_device, predict


@torch.no_grad()
def evaluate_model(model, root, device, self_ensemble=True, save_dir=None):
    ds = DriveDataset(root, split="test", patch_size=None, repeat=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model.to(device).eval()
    per_image = []
    for batch in loader:
        hr = batch["hr"].to(device)
        prob = predict(model, hr, self_ensemble=self_ensemble)[0, 0].cpu().numpy()
        ve = batch["ve"][0, 0].numpy()
        ma = batch["ma"][0, 0].numpy()
        m = evaluate_prediction(prob, ve, ma)
        per_image.append(m)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            from PIL import Image
            out = (prob[:584, :565] * ma[:584, :565] > 100 / 255.0).astype(np.uint8) * 255
            Image.fromarray(out).save(os.path.join(save_dir, f"{batch['id'][0]}_pred.png"))
    return mean_metrics(per_image), per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DRIVE")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-self-ensemble", action="store_true")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    model = build_model()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    mean, per = evaluate_model(model, args.root, device,
                               self_ensemble=not args.no_self_ensemble,
                               save_dir=args.save_dir)
    print("\n=== DRIVE test results ===")
    for k, v in mean.items():
        print(f"{k:>11}: {v:.4f}")
    print(f"\nAccuracy {mean['acc']*100:.2f}%  (target > 95%)")


if __name__ == "__main__":
    main()

"""End-to-end Study Group Learning pipeline on DRIVE.

Stages
------
1. (optional) synthesise noisy labels to stress-test robustness.
2. K-fold study-group labelling -> soft pseudo-labels (teacher maps).
3. Final CON model trained on human labels + pseudo-labels (dual BCE).
4. Evaluation on the DRIVE test set (accuracy target > 95%).
"""
import argparse
import json
import os

import torch

from evaluate import evaluate_model
from sgl_labeling import generate_pseudo_labels
from train import train_model
from utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DRIVE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=5, help="number of study groups")
    ap.add_argument("--fold-epochs", type=int, default=15)
    ap.add_argument("--final-epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--teacher-dir", default="data/DRIVE/training/teacher_sgl")
    ap.add_argument("--skip-labeling", action="store_true",
                    help="reuse existing pseudo-labels in --teacher-dir")
    ap.add_argument("--out", default="checkpoints/sgl_final.pt")
    ap.add_argument("--results", default="outputs/results.json")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"=== SGL pipeline on {device} ===")

    # Stage 2: study-group pseudo-labels
    if not args.skip_labeling:
        generate_pseudo_labels(args.root, args.teacher_dir, k=args.k,
                               epochs=args.fold_epochs, batch_size=args.batch_size,
                               patch_size=args.patch_size, repeat=args.repeat,
                               device=device)
    else:
        print(f"[run] skipping labeling, using {args.teacher_dir}")

    # Stage 3: final model on GT + pseudo-labels (keeps best-accuracy checkpoint)
    print("\n=== Final SGL training (human labels + study-group pseudo-labels) ===")
    model = train_model(args.root, train_indices=None, teacher_dir=args.teacher_dir,
                        epochs=args.final_epochs, batch_size=args.batch_size,
                        patch_size=args.patch_size, repeat=args.repeat,
                        device=device, out_ckpt=args.out, seed=123,
                        eval_every=5, eval_self_ensemble=False)

    # Stage 4: evaluation with self-ensemble on the best checkpoint
    print("\n=== Evaluation on DRIVE test set (best checkpoint, 8-way self-ensemble) ===")
    state = torch.load(args.out, map_location="cpu")
    model.load_state_dict(state["model"])
    mean, per = evaluate_model(model, args.root, device, self_ensemble=True,
                               save_dir="outputs/predictions")
    for k, v in mean.items():
        print(f"{k:>11}: {v:.4f}")
    print(f"\nAccuracy {mean['acc']*100:.2f}%  (target > 95%)")

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w") as f:
        json.dump({"mean": mean, "per_image": per}, f, indent=2)
    print(f"[run] metrics written to {args.results}")


if __name__ == "__main__":
    main()

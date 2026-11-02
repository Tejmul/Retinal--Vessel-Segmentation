"""Ablation: SGL pseudo-labels x clDice loss on DRIVE.

Four models are trained with an identical network, split, schedule, optimiser,
augmentation and seed. The only differences are (a) whether the study-group
pseudo-labels supervise the model alongside the human annotations and (b)
whether the pixel-wise objective is mixed with the centre-line Dice loss of
Shit et al. (CVPR 2021). That makes the comparison a 2x2 factorial:

    row              SGL teacher   clDice    loss key
    -------------------------------------------------------
    B  no-SGL         no            no       bce
    C  clDice only    no            yes      cldice
    A  SGL baseline   yes           no       sgl          (reproduced paper)
    D  SGL + clDice   yes           yes      sgl_cldice   (proposed)

Row "B" matters because SGL already repairs missing thin vessels by fixing the
labels, so it is possible that little is left for a topology loss to win; the
factorial shows that instead of leaving it as a guess.

Each row is scored twice: on the final-epoch weights (primary -- no checkpoint
selection, so no test-set leakage) and on the best-by-Dice checkpoint
(secondary, matching the original pipeline's protocol).
"""
import argparse
import json
import os
import time

import torch

from evaluate import evaluate_model
from losses import NEEDS_TEACHER
from sgl_labeling import generate_pseudo_labels
from train import train_model
from utils import get_device

ROWS = [
    ("B_no_sgl", "bce", "no-SGL control (human labels, BCE)"),
    ("C_cldice", "cldice", "clDice without SGL"),
    ("A_sgl_baseline", "sgl", "SGL baseline (reproduced paper)"),
    ("D_sgl_cldice", "sgl_cldice", "SGL + clDice (proposed)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/DRIVE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--fold-epochs", type=int, default=10)
    ap.add_argument("--final-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--teacher-dir", default="../data/DRIVE/training/teacher_sgl")
    ap.add_argument("--skip-labeling", action="store_true")
    ap.add_argument("--ckpt-dir", default="../checkpoints/ablation")
    ap.add_argument("--out", default="../outputs/ablation.json")
    ap.add_argument("--pred-dir", default="../outputs/ablation_predictions")
    ap.add_argument("--only", default=None,
                    help="comma-separated row names to run (default: all)")
    args = ap.parse_args()

    device = get_device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"=== Ablation on {device} | seed={args.seed} "
          f"epochs={args.final_epochs} patch={args.patch_size} ===", flush=True)

    # ---------------------------------------------------- study-group labelling
    if not args.skip_labeling:
        t0 = time.time()
        generate_pseudo_labels(args.root, args.teacher_dir, k=args.k,
                               epochs=args.fold_epochs, batch_size=args.batch_size,
                               patch_size=args.patch_size, repeat=args.repeat,
                               device=device,
                               ckpt_dir=os.path.join(args.ckpt_dir, "folds"))
        print(f"[ablation] labelling took {(time.time()-t0)/60:.1f} min", flush=True)
    else:
        print(f"[ablation] reusing pseudo-labels in {args.teacher_dir}", flush=True)

    wanted = set(args.only.split(",")) if args.only else None
    results = {}
    if os.path.exists(args.out):           # resume-friendly
        results = json.load(open(args.out)).get("rows", {})

    for name, loss_key, desc in ROWS:
        if wanted and name not in wanted:
            continue
        print(f"\n=== {name}: {desc}  [loss={loss_key}] ===", flush=True)
        t0 = time.time()
        best_ckpt = os.path.join(args.ckpt_dir, f"{name}_best.pt")
        last_ckpt = os.path.join(args.ckpt_dir, f"{name}_last.pt")
        # rows without an SGL term never see the teacher maps
        teacher = args.teacher_dir if loss_key in NEEDS_TEACHER else None

        model = train_model(
            args.root, train_indices=None, teacher_dir=teacher,
            epochs=args.final_epochs, batch_size=args.batch_size,
            patch_size=args.patch_size, repeat=args.repeat, device=device,
            out_ckpt=best_ckpt, out_last=last_ckpt, seed=args.seed,
            eval_every=5, eval_self_ensemble=False,
            loss=loss_key, select_metric="dice")
        train_min = (time.time() - t0) / 60

        row = {"loss": loss_key, "desc": desc, "train_minutes": round(train_min, 1),
               "uses_teacher": teacher is not None}
        for tag, ckpt in (("last", last_ckpt), ("best_dice", best_ckpt)):
            if not os.path.exists(ckpt):
                continue
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            save_dir = os.path.join(args.pred_dir, name) if tag == "last" else None
            mean, per = evaluate_model(model, args.root, device,
                                       self_ensemble=True, save_dir=save_dir)
            row[tag] = {"mean": mean, "per_image": per}
            print(f"  [{name}/{tag}] acc={mean['acc']:.4f} auc={mean['auc']:.4f} "
                  f"dice={mean['dice']:.4f} cldice={mean['cldice']:.4f} "
                  f"se={mean['se']:.4f} sp={mean['sp']:.4f} "
                  f"beta0={mean['beta0_ratio']:.2f}", flush=True)

        results[name] = row
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "rows": results}, f, indent=2)
        print(f"[ablation] {name} done in {train_min:.1f} min -> {args.out}", flush=True)

    print("\n=== Ablation complete ===", flush=True)


if __name__ == "__main__":
    main()

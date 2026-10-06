"""Train a single CON (Enhance + Seg) model.

Used both for the K study-group fold models (teacher_dir=None -> supervise on
ground truth only) and for the final SGL model (teacher_dir set -> dual loss
against ground truth and the pseudo-labels).
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import DriveDataset
from losses import SGLoss
from model import build_model
from utils import get_device


def train_model(root, train_indices=None, teacher_dir=None, *, epochs=20,
                batch_size=4, patch_size=128, lr=1e-4, repeat=40,
                device=None, out_ckpt=None, augment=True, elastic=True,
                log_every=20, seed=0, eval_every=0, eval_self_ensemble=False):
    torch.manual_seed(seed)
    device = device or get_device("auto")
    print(f"[train] device={device} epochs={epochs} bs={batch_size} "
          f"patch={patch_size} lr={lr} teacher={'yes' if teacher_dir else 'no'} "
          f"images={train_indices or 'all'}")

    ds = DriveDataset(root, split="training", indices=train_indices,
                      teacher_dir=teacher_dir, patch_size=patch_size,
                      repeat=repeat, augment=augment, elastic=elastic)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model = build_model().to(device)
    criterion = SGLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    best_acc = -1.0

    def _save(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": model.state_dict()}, path)

    for epoch in range(1, epochs + 1):
        model.train()
        running, n, t0 = 0.0, 0, time.time()
        for it, batch in enumerate(loader):
            hr = batch["hr"].to(device)
            ve = batch["ve"].to(device)
            te = batch["te"].to(device)
            optimizer.zero_grad()
            _, seg = model(hr)
            loss = criterion(seg, ve, te)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
            if (it + 1) % log_every == 0:
                print(f"  epoch {epoch:3d} it {it+1:4d}/{len(loader)} "
                      f"loss {running/n:.4f}", flush=True)
        print(f"[epoch {epoch:3d}/{epochs}] mean_loss={running/max(n,1):.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

        if eval_every and (epoch % eval_every == 0 or epoch == epochs):
            from evaluate import evaluate_model  # local import avoids cycle
            mean, _ = evaluate_model(model, root, device,
                                     self_ensemble=eval_self_ensemble)
            print(f"  [eval @ep{epoch}] acc={mean['acc']:.4f} auc={mean['auc']:.4f} "
                  f"dice={mean['dice']:.4f} se={mean['se']:.4f} sp={mean['sp']:.4f}",
                  flush=True)
            if out_ckpt and mean["acc"] > best_acc:
                best_acc = mean["acc"]
                _save(out_ckpt)
                print(f"  [eval] new best acc={best_acc:.4f} -> saved {out_ckpt}",
                      flush=True)

    if out_ckpt and best_acc < 0:  # no eval-based saving happened
        _save(out_ckpt)
        print(f"[train] saved checkpoint -> {out_ckpt}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DRIVE")
    ap.add_argument("--teacher-dir", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="checkpoints/con_model.pt")
    ap.add_argument("--no-elastic", action="store_true")
    args = ap.parse_args()

    train_model(args.root, teacher_dir=args.teacher_dir, epochs=args.epochs,
                batch_size=args.batch_size, patch_size=args.patch_size,
                lr=args.lr, repeat=args.repeat, device=get_device(args.device),
                out_ckpt=args.out, elastic=not args.no_elastic)


if __name__ == "__main__":
    main()

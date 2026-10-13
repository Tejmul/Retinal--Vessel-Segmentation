# Retinal Vessel Segmentation — SGL (Study Group Learning) Reproduction

Reproduction of:

> **Y. Zhou, H. Yu, H. Shi. "Study Group Learning: Improving Retinal Vessel
> Segmentation Trained with Noisy Labels." MICCAI 2021.**
> [arXiv:2103.03451](https://arxiv.org/abs/2103.03451) ·
> [official code](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)

Dataset: **DRIVE** (Digital Retinal Images for Vessel Extraction), 20 training /
20 test images. Target: vessel-segmentation **accuracy > 95%**.

This repo re-implements the paper from scratch in PyTorch (clean, readable code)
and reproduces the full Study Group Learning scheme, trained and evaluated
locally on an Apple-silicon Mac (MPS).

---

## 1. Architecture (`CON`)

A two-stage cascade of two U-Nets (exactly matching the official `model/con.py`):

```
RGB image (0..255)
      │   (/255)
      ▼
┌──────────────┐   3-channel
│  EnhanceNet  │─── enhancement ──┐
│   (U-Net)    │   map (sigmoid)  │
└──────────────┘                  ▼
                         ┌──────────────┐   1-channel vessel
                         │    SegNet    │── probability (sigmoid)
                         │   (U-Net)    │
                         └──────────────┘
```

* Each U-Net has 5 levels, channels `3 → 32 → 64 → 128 → 256 → 512`, pre-activation
  conv blocks (`PReLU → BatchNorm → Conv`), `ConvTranspose2d` upsampling and skip
  concatenations. The final block concatenates the decoder feature (32ch) with the
  raw input before the output conv.
* **Only the final segmentation output is supervised.** The enhancement map is
  learned implicitly through the segmentation loss (and doubles as a clinician-
  friendly visualization, as in the paper).
* ~15.5M parameters.

See [src/model.py](src/model.py).

## 2. Study Group Learning

The key idea: human vessel annotations are *noisy*. SGL denoises them with a
cross-validated "study group" of models.

1. Split the 20 training images into **K groups**.
2. For each group, train a `CON` model on the **other** K−1 groups and predict the
   held-out group with **8-way self-ensemble**. Every image thus receives a
   *pseudo-label* from a model that never saw it. — [src/sgl_labeling.py](src/sgl_labeling.py)
3. Train the **final** `CON` model on all 20 images with a **dual loss**:

   ```
   loss = BCE(pred, human_label) + BCE(pred, study_group_pseudo_label)
   ```
   — [src/losses.py](src/losses.py), [src/train.py](src/train.py)

## 3. Data & evaluation protocol

* Images kept in `[0, 255]` (the network divides by 255 internally); vessel/FOV
  masks binarised; teacher labels soft in `[0, 1]`.
* Training: random `128×128` patches (paper uses 256; 128 fits 8 GB) with
  hflip / vflip / rot90 and 50% elastic deformation. — [src/dataset.py](src/dataset.py)
* Testing: zero-pad to `608×608`, 8-way self-ensemble, threshold `prob > 100/255`,
  crop back to `584×565`. Metrics (accuracy, sensitivity, specificity, AUC, IoU,
  Dice) are computed within the cropped region with FOV masking. — [src/metrics.py](src/metrics.py)

## 4. Usage

### Getting the data

The DRIVE dataset is **not** redistributed here (its license restricts
redistribution). Obtain it (e.g. from the
[DRIVE grand-challenge page](https://drive.grand-challenge.org/) or the official
[SGL repo](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)) and
arrange it as:

```
data/DRIVE/
  training/{images,1st_manual,mask}/   # ids 21-40 (*_training.tif, *_manual1.gif, *_mask.gif)
  test/{images,1st_manual,mask}/       # ids 01-20
```

### Run

```bash
pip install -r requirements.txt

# Full pipeline: K-fold SGL labeling -> final training -> evaluation
cd src
python run_sgl.py --root ../data/DRIVE --k 5 \
    --fold-epochs 15 --final-epochs 45 \
    --batch-size 4 --patch-size 128 --repeat 40

# Reuse already-generated pseudo-labels (skip the expensive labeling stage)
python run_sgl.py --root ../data/DRIVE --skip-labeling \
    --teacher-dir ../data/DRIVE/training/teacher_sgl

# Evaluate a saved checkpoint
python evaluate.py --root ../data/DRIVE --ckpt ../checkpoints/sgl_final.pt
```

Individual stages (`train.py`, `sgl_labeling.py`, `evaluate.py`) are also runnable
on their own. Device is auto-selected (MPS → CUDA → CPU).

## 5. Layout

```
src/
  model.py         CON = EnhanceNet + SegNet (two U-Nets)
  dataset.py       DRIVE loading, patching, augmentation
  losses.py        BCE / dual-target SGL loss
  metrics.py       Acc / Se / Sp / AUC / IoU / Dice (paper protocol)
  utils.py         device + 8-way self-ensemble inference
  train.py         train one CON model (fold or final)
  sgl_labeling.py  K-fold study-group pseudo-label generation
  evaluate.py      test-set evaluation
  run_sgl.py       end-to-end pipeline
data/DRIVE/        training/ (ids 21-40) and test/ (ids 01-20)
checkpoints/       saved models (best-accuracy)
outputs/           logs, metrics JSON, prediction PNGs
```


## 6. Results

Full pipeline run on the DRIVE test set (20 images), 8-way self-ensemble, best
checkpoint. Config: K=4 study groups, 10 fold-epochs, 30 final-epochs, 128px
patches (see `outputs/results.json` for per-image numbers).

| Metric | Value |
|---|---|
| **Accuracy** | **0.9655** ✅ (> 95%) |
| AUC | 0.9859 |
| Sensitivity | 0.8596 |
| Specificity | 0.9760 |
| Dice (F1) | 0.8127 |
| IoU (vessel) | 0.6847 |
| IoU (background) | 0.9627 |

For reference, the paper reports on DRIVE roughly AUC ≈ 0.983, F1 ≈ 0.82 and
within-FOV accuracy ≈ 0.956 — our AUC and Dice match closely; accuracy reads a
little higher because of the FOV-masking protocol noted below. Larger patches
(256) and more epochs, as in the paper, close the small remaining gap.

> Note on the accuracy protocol: following the official code, pixels outside the
> circular field-of-view are masked to zero in both prediction and label and are
> included in the accuracy denominator (as true negatives). This is the same
> protocol used by the released repo.

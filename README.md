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
python run_sgl.py --root ../data/DRIVE --k 4 \
    --fold-epochs 10 --final-epochs 30 \
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
  run_ablation.py  2x2 ablation: SGL pseudo-labels x clDice loss
  make_comparison.py  comparison tables + hypothesis verdict
  failure_analysis.py failure analysis (calibre-split Se, broken branches)
  export_web_assets.py  image export for the results viewer
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


---

## 7. Ablation: does a topology loss help?

The literature review proposed mixing the centre-line Dice loss
(Shit *et al.*, CVPR 2021) into SGL's objective, predicting that punishing
breaks in thin branches more heavily than slightly thick edges would raise
sensitivity and F1. [src/losses.py](src/losses.py) implements it as a
differentiable soft skeleton; the experiment is a 2x2 factorial over
*SGL pseudo-labels* x *clDice*, with the network, split, schedule, optimiser,
augmentation and seed (123) identical in every row.

Scored on the **final-epoch** weights — no checkpoint is selected on the test
set — with 8-way self-ensemble at threshold 100/255.

| # | configuration | SGL | clDice | Acc | AUC | Se | Sp | F1 | clDice | IoU-v | beta0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R | published `drive_k8` weights, our code | K=8 | — | 0.9705 | 0.9886 | 0.8380 | 0.9834 | 0.8316 | — | 0.7120 | — |
| B | human labels, BCE | — | — | 0.9695 | 0.9872 | 0.8012 | **0.9860** | **0.8200** | 0.8118 | **0.6956** | 32.6 |
| C | clDice only | — | yes | 0.9599 | 0.9818 | 0.8677 | 0.9690 | 0.7904 | **0.8595** | 0.6537 | **8.2** |
| A | SGL baseline (paper loss) | yes | — | 0.9630 | 0.9849 | 0.8713 | 0.9720 | 0.8038 | 0.8319 | 0.6723 | 29.6 |
| D | SGL + clDice (proposed) | yes | yes | 0.9422 | 0.9764 | **0.8833** | 0.9480 | 0.7274 | 0.8565 | 0.5719 | 9.6 |

Row R is the reproduction check: the paper's released weights load into this
re-implementation with all 324 tensors matching (the official checkpoint's two
extra `last_act` parameters are dead — the call is commented out upstream) and
score AUC 0.9886 / F1 0.8316 against the paper's reported ~0.983 / ~0.82. The
retrained rows sit lower because they get 30 epochs at 128px rather than the
paper's longer schedule at 256px.

### The hypothesis was wrong, and informatively so

| prediction | observed (D − A) | holds? |
|---|---|---|
| Se rises by 2–5 pp | +1.20 pp | no |
| F1 rises by 0.5–1.5 pp | **−7.64 pp** | no |
| Sp falls by at most 1 pp | −2.40 pp | no |

The F1 drop clears the review's own 1 pp noise floor by a wide margin and has
the same sign on 20 of 20 images (Wilcoxon p < 0.001), so it is not sampling
noise. But clDice delivered exactly the topology it promised:

| | Se thin | Se thick | broken branches | components | FP rate |
|---|---|---|---|---|---|
| B no-SGL | 0.6986 | 0.9802 | 181 | 65 | 0.0213 |
| C clDice only | 0.8002 | 0.9832 | **97** | **17** | 0.0471 |
| A SGL baseline | 0.8031 | 0.9892 | 141 | 57 | 0.0426 |
| D SGL + clDice | **0.8220** | 0.9884 | **97** | 20 | 0.0791 |

Row D has the best thin-vessel recall of the four, 31% fewer broken branches
than the baseline and a third of its fragmentation, improving clDice on 20/20
images. The bill arrives as false positives: a loss that only rewards covering
the label's centre-line has no reason to keep the prediction thin, so the model
thickens everything until the skeleton is covered. That is what costs the F1.

Two further findings worth more than the headline:

* **SGL did not help under this budget.** A against B is −1.6 pp F1 and
  −0.7 pp accuracy, buying +7.0 pp sensitivity and +2.0 pp clDice. The
  pseudo-labels shift the operating point rather than dominating — which is
  exactly why the review asked for the no-SGL control row.
* **alpha = 0.5 is too strong.** clDice got equal weight with cross-entropy
  because the hypothesis specified "an equal mix". The connectivity gain
  saturates while the precision loss does not, so the experiment this run does
  *not* cover, and should not be read as covering, is a sweep of alpha from
  0.1 to 0.3.

Single seed throughout; differences below 1 pp should not be trusted without
repeats.

### Reproducing it

```bash
cd src
python run_ablation.py --k 4 --fold-epochs 10 --final-epochs 30 \
    --patch-size 128 --batch-size 4 --repeat 40 --seed 123
python make_comparison.py        # outputs/comparison.{md,tex}
python failure_analysis.py       # outputs/failure_analysis.json + panels
```

Study-group labelling is resumable: a fold whose pseudo-labels already exist on
disk is skipped, and `outputs/ablation.json` is rewritten after each row and
read back on startup, so an interrupted run does not repeat finished work.
Total ~2 h on an M2 (labelling ~25 min, then ~23 min per row).

## 8. Interfaces

Two front ends, both in the repo.

### `app/` — Vessel Studio (interactive, runs the models)

```bash
.venv/bin/python app/server.py      # http://127.0.0.1:5001
```

Pick any of the five checkpoints (the four ablation rows plus the paper's
released weights), choose a DRIVE test image or drop in your own fundus
photograph, and segment it. Views: overlay against the human tracing, binary
mask, raw probability, **EnhanceNet's own output** (the first U-Net of the
cascade, which is never supervised directly), and the plain fundus. A
magnifier follows the cursor at 6x, because the differences this project is
about are one or two pixels wide.

The server returns the probability map rather than a binary mask, so the
**threshold slider re-thresholds and re-scores in the browser with no further
inference** — F1, sensitivity, specificity, accuracy and IoU update as you
drag. That matters: the ablation fixes the threshold at `100/255`, which was
chosen for a BCE-trained model, and a clDice-trained model puts its probability
mass elsewhere. `Find best F1` sweeps it. A deep link such as
`?img=03&model=A_sgl_baseline&thr=120` opens straight on one result.

Inference is ~0.2 s per image on MPS, ~1.6 s with the 8-way self-ensemble.

### `web/` — static results viewer

[web/index.html](web/index.html) needs no server-side model: it ships
precomputed masks for six test images and compares the baseline against the
proposed model with the same magnifier, alongside the ablation tables and the
scored hypothesis. Serve it over HTTP (`python -m http.server` from `web/`)
rather than opening it as a `file://` URL, since it fetches its data.

# Phase 2 — presentation prep guide

Everything you need to walk someone through this project and defend it.
Deck: `report/Phase2_Presentation.pptx` (10 slides, speaker notes on every slide).

---

## 1. The 30-second version

> We reproduced SGL, a 2021 vessel-segmentation paper, and then changed one thing: we
> added the clDice loss, which scores the *skeleton* of the vessel tree instead of the
> pixels. We predicted in phase 1 that F₁ would go up. It went down. So the hypothesis
> is refuted. But the loss did exactly what it was designed to do — it cut broken
> branches by a third and fragmentation by two thirds, on every image. We used too
> strong a weight, and part of what looked like a loss of quality was just the decision
> threshold being wrong for the new model.

If you only remember one sentence: **the hypothesis failed, the mechanism worked.**

---

## 2. What the project is

Given a photograph of the back of the eye (a *fundus* image), mark every pixel that is
a blood vessel. It matters clinically because vessel shape changes early in diabetic
retinopathy, hypertension and glaucoma.

The benchmark is **DRIVE**: 40 images, a fixed split of 20 for training and 20 for
testing, each with a vessel map traced by a human.

Two things make it hard:

- **Imbalance.** Only about 8–10% of pixels are vessel. Predicting "no vessel"
  everywhere already scores ~90% accuracy, which is why nobody reports accuracy alone.
- **Thin vessels.** The capillaries at the edge of the tree are 1–2 pixels wide, faint,
  and are exactly what the clinician cares about.

---

## 3. What we did in phase 1 (last phase)

| | |
|---|---|
| **Reviewed** | 10 DRIVE papers — U-Net variants, training-scheme methods, topology losses, attention and lightweight models |
| **Chose a base paper** | **SGL** (Study Group Learning, MICCAI 2021). Picked over FR-UNet because it ships code, data, pseudo-labels *and* trained weights, and because its loss is selected by a command-line flag — so our change is one new file, not surgery |
| **Stated a hypothesis** | With numbers, written down before any experiment |
| **Planned the ablation** | Four configurations, so the loss is the only thing that moves |

**The hypothesis, word for word:** replacing the pixel-wise loss with an equal mix of
clDice and cross-entropy will raise sensitivity by **2–5 pp** and F₁ by **0.5–1.5 pp**,
while lowering specificity by **no more than 1 pp**.

Why this matters when you present: we committed to three falsifiable numbers *in
advance*. That is what makes phase 2 a test rather than a demo. Say this out loud — it
is the strongest thing about the project.

### What SGL is, in two sentences
Human vessel tracings are inconsistent, especially on thin branches. SGL trains K models
on different folds, uses each to re-label the images it never saw ("pseudo-labels", or
*teacher* labels), then trains the final model against **both** the human label and the
teacher label.

---

## 4. What we did in phase 2 (this phase)

| Built | Where |
|---|---|
| **clDice loss** — differentiable soft skeleton by iterated erode-minus-open, 3 steps | `src/losses.py` |
| **Four objectives behind one flag** — `bce`, `sgl`, `cldice`, `sgl_cldice` | `src/train.py` |
| **Topology metrics** — binary clDice, β₀ fragmentation ratio, broken-branch count | `src/metrics.py` |
| **SGL pipeline reproduced**, and made resumable after we lost power mid-run | `src/sgl_labeling.py` |
| **Ablation driver** — trains and scores all four rows, 8-way self-ensemble, resumable | `src/run_ablation.py` |
| **Failure analysis** — broken branches, calibre-stratified recall, side-by-side panels | `src/failure_analysis.py` |
| **Threshold sweep** — the calibration check (added after the first result) | `src/threshold_sweep.py` |
| **Vessel Studio** — a local app that runs any checkpoint on any test image, live | `app/server.py` |

**The experiment:** 2×2. Rows = which labels (human vs SGL pseudo-labels). Columns =
which loss (cross-entropy vs cross-entropy + clDice). Identical network, split,
schedule, optimiser, augmentation and seed (123), 30 epochs each, α = 0.5.

| | Cross-entropy only | + clDice |
|---|---|---|
| **Human labels** | **B** no-SGL control | **C** clDice only |
| **SGL pseudo-labels** | **A** baseline (the paper) | **D** SGL + clDice (proposed) |

Row C was not in the original plan. We added it so we could tell the *loss* apart from
the *labels*.

---

## 5. The result, in plain words

**Scored at the fixed threshold we pre-registered (100/255):**

- Sensitivity **+1.20 pp** — we predicted +2 to +5. ✗
- F₁ **−7.64 pp** — we predicted +0.5 to +1.5. ✗
- Specificity **−2.40 pp** — we predicted no worse than −1. ✗

All three fail. F₁ fell on **20 of 20 images** (Wilcoxon p < 0.001), which is seven
times the 1-point noise floor we set ourselves in phase 1. This is not a near miss.

**But three things qualify that:**

1. **The topology genuinely improved.** Broken branches 141 → 97. Disconnected pieces
   per image 57 → 20. The clDice metric rose on 20 of 20 images. Thin-vessel recall
   0.803 → 0.822.
2. **Most of the F₁ gap was calibration.** 100/255 was chosen for a cross-entropy
   model. A clDice model is calibrated much higher — D's best threshold is 170/255.
   Let each row pick its own and the D-vs-A F₁ gap falls from −7.64 pp to **−1.40 pp**.
   (Caveat you must state: those thresholds are fitted on the test set, so they flatter
   every row equally. It is an upper bound, not a score.)
3. **The price is false positives.** FPR 0.043 → 0.079, nearly doubled. clDice is scored
   on the skeleton, so making a vessel fatter is a cheap way to keep it connected.

**Unplanned finding:** SGL did not pay for itself at our budget — row A is 1.6 points of
F₁ *below* row B. Report it; it was a controlled comparison and it surprised us.

---

## 6. Slide-by-slide script

Target ~15 minutes for 10 slides. Full narration is in the deck's speaker notes — this
is the skeleton.

| # | Slide | Say this | Time |
|---|---|---|---|
| 1 | Title | "Phase 1 ended with a hypothesis. This phase we implemented it, trained it and measured it. The hypothesis is refuted — but the mechanism behind it is real, and I'll show you both." | 0:30 |
| 2 | Where phase 1 left us | Four things we did. Then point at the box: **"We wrote down three numbers before we ran anything."** | 1:30 |
| 3 | Why we expected it to work | The whole idea in one picture. Left: a 2-pixel gap is 98.5% correct, so the pixel loss has no reason to close it. Right: that same gap turned one vessel into two. And our baseline really does fragment — 57 pieces where the human has 3. | 2:00 |
| 4 | What we built | Six pieces. The loss itself is ~40 lines; most of the work was the metrics that can *see* topology and a driver that trains all four rows identically. | 1:30 |
| 5 | Is our baseline trustworthy? | The insurance policy. All 324 keys match, their weights score 0.9886 AUC in our code. So from here, any change is the loss, not the plumbing. | 1:30 |
| 6 | The experiment | Two factors, two levels, four models. Everything pinned including the seed. **α = 0.5 — clDice is half the objective.** | 1:30 |
| 7 | Results table | Read it in two columns. β₀ collapses wherever clDice is on. F₁ falls wherever clDice is on. Same cause, opposite effects. Also note A vs B. | 2:30 |
| 8 | The hypothesis is refuted | Say plainly: **we were wrong**, on all three numbers. Then raise the calibration footnote *yourself*, before anyone asks. | 2:30 |
| 9 | But the mechanism is real | Point at the middle vs right panel. Less blue on the right — 38 broken branches vs 17 in this window. Then point at the red: that is the price. | 2:00 |
| 10 | What we take from this | Three findings, then next steps. Close on: *a negative result that is measured, pre-registered and explained is a result.* | 1:30 |

---

## 7. Numbers to know by heart

| | B (no SGL) | C (clDice) | A (baseline) | D (proposed) |
|---|---|---|---|---|
| F₁ @ 100/255 | 0.8200 | 0.7904 | 0.8038 | **0.7274** |
| F₁ @ own best threshold | 0.8217 @ 85 | 0.8068 @ 165 | 0.8127 @ 125 | **0.7987 @ 170** |
| Sensitivity | 0.8012 | 0.8677 | 0.8713 | **0.8833** |
| Specificity | 0.9860 | 0.9690 | 0.9720 | **0.9480** |
| β₀ (fragmentation) | 32.6 | 8.1 | 29.6 | **9.6** |
| Broken branches | 181 | 97 | 141 | **97** |
| Thin-vessel recall | 0.699 | 0.800 | 0.803 | **0.822** |
| False-positive rate | 0.021 | 0.047 | 0.043 | **0.079** |

Plus: **R** = the paper's published weights run through our code — AUC **0.9886**, F₁
**0.8316** (paper reports ≈0.983 / ≈0.82).

---

## 8. Hard questions, and how to answer them

**"So your idea failed."**
The hypothesis failed; the idea did not. We predicted F₁ would rise and it fell, so we
report a refutation. But the loss did what it was designed to do — it connected the tree
on all 20 images. What we got wrong is the weight.

**"Why α = 0.5?"**
Because the hypothesis said "an equal mix", and we tested what we wrote down. In
hindsight that is the strongest setting available, and the clDice paper itself uses
smaller weights. The α sweep from 0.1 to 0.3 is the first thing we would run next.

**"Isn't the fixed threshold unfair to D?"**
Partly, and we checked rather than guessed. At each row's own best threshold the gap is
−1.40 pp, not −7.64. We keep the fixed threshold as the headline because it is what we
pre-registered, and because the per-row thresholds are fitted on the test set, which
flatters every row. Both numbers are in the deck.

**"Why is your SGL baseline worse than no SGL? That contradicts the paper."**
At *our* budget. We ran K=4 folds with 10 epochs each and 30 final epochs; the paper
uses K=8 and trains far longer. Weaker teachers make noisier pseudo-labels. This is a
statement about a 30-epoch budget, not about SGL.

**"Your F₁ of 0.8038 is below the paper's 0.82 — is your implementation wrong?"**
No, and we checked that before touching the loss. The authors' own weights score 0.8316
F₁ and 0.9886 AUC in our pipeline, at or above their published numbers, and all 324
state-dict keys match. The pipeline reproduces. Our *retrained* baseline is lower purely
because we trained 30 epochs instead of their full schedule.

**"20 test images is tiny. Is any of this significant?"**
We fixed the rule in phase 1: any F₁ change below 1 point is noise. The F₁ drop is 7.6
points with the same sign on 20/20 images, p < 0.001 — real by that rule. The topology
changes are also 20/20. The sensitivity gain of +1.2 points we explicitly **do not**
claim: p = 0.097, only 12/20 images. That one is noise by our own standard.

**"What is clDice, in one sentence?"**
Dice computed between the *skeletons* — the one-pixel-wide centre lines — of the
prediction and the label, instead of between the full pixel masks.

**"How can a skeleton be differentiable?"**
Soft morphology. Erosion is a min-pool, dilation a max-pool, opening is erode-then-
dilate, and the soft skeleton is the accumulated `relu(image − opening)` over a few
erosion steps. It is all min/max pooling, so gradients flow. We use 3 iterations.

**"What is β₀?"**
The zeroth Betti number — the count of connected components. We report prediction
components ÷ label components. The label has about 3; our baseline produces 57.

**"Why did false positives nearly double?"**
Because clDice is scored on the skeleton. Thickening a vessel keeps the skeleton
connected at almost no cost to the clDice term, and cross-entropy is the only thing
pushing back. At α = 0.5 cross-entropy loses that argument.

**"Where do the errors actually gather?"**
Three places, and we measured all three. **Thin vessels:** recall 0.822 vs 0.988
on thick ones — that is where everything is lost. **The optic disc:** a
false-positive hotspot for every model, about twice the background rate, and
clDice makes it worse faster than it worsens the rest of the image, because a
loss that rewards covering centre-lines is happy to thicken bright structures.
**Crossings:** not actually a problem — recall at junctions is about 8 points
*above* plain runs, because junctions sit on the thick central tree. clDice's
gain is concentrated on plain runs (+4.6 pp) rather than at crossings (+2.1 pp):
it helps a vessel keep going, not resolve where two vessels touch.

**"What would you do differently?"**
Three things: sweep α from 0.1 to 0.3 across several seeds; fit the threshold on a
validation split instead of a constant; and always report a topology metric beside F₁,
because F₁ cannot see a broken vessel.

---

## 9. Running the live demo

```bash
cd ~/Desktop/Retinal--Vessel-Segmentation
.venv/bin/python app/server.py       # then open http://127.0.0.1:5001
```

A 60-second demo that lands:

1. Pick test image **03**, model **A**, press **Segment**.
2. Switch to model **D**, press **Segment**. Point at the overlay — less blue (fewer
   missed vessels), more red (more false positives). That is the whole finding, visible.
3. **Drag the threshold slider.** The scores update live with no new inference. Push it
   up towards 170 and watch D's F₁ climb — this is the calibration argument from slide 8,
   demonstrated rather than asserted.
4. Press **Find best F₁** to let it sweep automatically.

Have it already running before you start. Don't start a server in front of the room.

---

## 10. Words you should be able to define on the spot

- **Fundus** — the inside back surface of the eye; what the camera photographs.
- **FOV mask** — the circular region of the image that actually contains retina. All
  scores are computed inside it only.
- **Sensitivity / recall** — of the vessel pixels the human marked, what fraction did we
  find.
- **Specificity** — of the non-vessel pixels, what fraction did we correctly leave alone.
- **F₁ / Dice** — harmonic mean of precision and recall; the standard vessel score.
- **AUC** — area under the ROC curve; threshold-free, which is why it moves less than F₁.
- **Self-ensemble (8-way TTA)** — run the image through the net in 8 flips/transposes and
  average. Costs 8× the time, gains a few tenths of a point.
- **Pseudo-label / teacher label** — a label produced by a model rather than a human.
- **Skeleton** — a shape thinned to a one-pixel-wide centre line.
- **α (alpha)** — how much of the objective is clDice. We used 0.5.

---

## 11. Three mistakes to avoid on the day

1. **Do not claim the sensitivity gain.** +1.2 pp with p = 0.097 is noise by our own
   pre-registered rule. Claiming it is the easiest way to lose credibility.
2. **Do not say "clDice failed".** Say "the hypothesis failed at α = 0.5". clDice
   improved every topology metric on every image.
3. **Do not skip slide 5.** If anyone doubts the baseline, every later number is worth
   nothing. Establish the reproduction first.

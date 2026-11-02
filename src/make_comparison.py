"""Turn ``outputs/ablation.json`` into the comparison tables and the verdict
on the hypothesis stated in the literature review.

The hypothesis under test (report/literature_review.tex, sec. "Hypothesis"):

    replacing the pixel-wise loss with an equal mix of the centre-line clDice
    loss and cross-entropy will raise sensitivity by 2 to 5 percentage points
    and F1 by 0.5 to 1.5 points ... while lowering specificity by no more than
    1 point.

Because the DRIVE test set has only 20 images, every headline difference is
also reported as a *paired* per-image difference with a Wilcoxon signed-rank
test, and the review's own 0.01 F1 noise floor is applied.
"""
import argparse
import json
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# display name, json key, number of decimals, higher-is-better
METRICS = [
    ("Acc", "acc", 4, True),
    ("AUC", "auc", 4, True),
    ("Se", "se", 4, True),
    ("Sp", "sp", 4, True),
    ("Dice/F1", "dice", 4, True),
    ("clDice", "cldice", 4, True),
    ("IoU-v", "iou_vessel", 4, True),
    ("beta0", "beta0_ratio", 2, False),
]

ROW_ORDER = ["B_no_sgl", "C_cldice", "A_sgl_baseline", "D_sgl_cldice"]
LABELS = {
    "B_no_sgl":       ("B", "human labels, BCE", "no", "no"),
    "C_cldice":       ("C", "clDice only",       "no", "yes"),
    "A_sgl_baseline": ("A", "SGL baseline (paper loss)", "yes", "no"),
    "D_sgl_cldice":   ("D", "SGL + clDice (proposed)",   "yes", "yes"),
}


def _fmt(v, nd):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def paired(rows, a, b, key, tag="last"):
    """Per-image difference b - a for one metric."""
    if a not in rows or b not in rows:
        return None
    pa = [m[key] for m in rows[a][tag]["per_image"]]
    pb = [m[key] for m in rows[b][tag]["per_image"]]
    d = np.array(pb) - np.array(pa)
    out = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
           "wins": int((d > 0).sum()), "n": len(d), "p": None}
    if _HAS_SCIPY and len(d) > 1 and np.any(d != 0):
        try:
            out["p"] = float(wilcoxon(pb, pa).pvalue)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="../outputs/ablation.json")
    ap.add_argument("--official", default="../outputs/results_row1_official.json")
    ap.add_argument("--tag", default="last", choices=["last", "best_dice"],
                    help="which checkpoint to tabulate (default: final epoch)")
    ap.add_argument("--out-md", default="../outputs/comparison.md")
    ap.add_argument("--out-tex", default="../outputs/comparison.tex")
    args = ap.parse_args()

    rows = json.load(open(args.ablation))["rows"]
    official = None
    if os.path.exists(args.official):
        official = json.load(open(args.official))["mean"]

    present = [r for r in ROW_ORDER if r in rows and args.tag in rows[r]]
    L = []
    L.append(f"## Ablation: SGL pseudo-labels x clDice loss ({args.tag} checkpoint)\n")
    L.append("Identical network, split, schedule, optimiser, augmentation and seed "
             "(123) across all four rows; the loss is the only change.\n")

    head = "| # | configuration | SGL | clDice | " + \
           " | ".join(m[0] for m in METRICS) + " |"
    L.append(head)
    L.append("|---|---|---|---|" + "---|" * len(METRICS))

    if official is not None:
        L.append("| R | published weights in our code | yes (K=8) | no | "
                 + " | ".join(_fmt(official.get(k), nd) for _, k, nd, _ in METRICS)
                 + " |")

    for name in present:
        tagd, desc, sgl, cld = LABELS[name]
        mean = rows[name][args.tag]["mean"]
        L.append(f"| {tagd} | {desc} | {sgl} | {cld} | "
                 + " | ".join(_fmt(mean.get(k), nd) for _, k, nd, _ in METRICS) + " |")

    # ----------------------------------------------------------- paired deltas
    L.append("\n### Paired per-image differences (20 test images)\n")
    L.append("| comparison | metric | mean delta | std | images improved | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|")
    pairs = [("A_sgl_baseline", "D_sgl_cldice", "D - A  (clDice on top of SGL)"),
             ("B_no_sgl", "C_cldice", "C - B  (clDice without SGL)"),
             ("B_no_sgl", "A_sgl_baseline", "A - B  (SGL pseudo-labels alone)")]
    for a, b, lbl in pairs:
        for disp, key, nd, _ in METRICS:
            st = paired(rows, a, b, key, args.tag)
            if st is None:
                continue
            p = "n/a" if st["p"] is None else f"{st['p']:.3f}"
            L.append(f"| {lbl} | {disp} | {st['mean']:+.4f} | {st['std']:.4f} | "
                     f"{st['wins']}/{st['n']} | {p} |")

    # ------------------------------------------------------------- hypothesis
    L.append("\n### Verdict on the stated hypothesis\n")
    if "A_sgl_baseline" in rows and "D_sgl_cldice" in rows:
        a = rows["A_sgl_baseline"][args.tag]["mean"]
        d = rows["D_sgl_cldice"][args.tag]["mean"]
        d_se = (d["se"] - a["se"]) * 100
        d_f1 = (d["dice"] - a["dice"]) * 100
        d_sp = (d["sp"] - a["sp"]) * 100
        d_cl = (d["cldice"] - a["cldice"]) * 100
        checks = [
            ("Se rises by 2-5 pp", d_se, 2.0 <= d_se <= 5.0, f"{d_se:+.2f} pp"),
            ("F1 rises by 0.5-1.5 pp", d_f1, 0.5 <= d_f1 <= 1.5, f"{d_f1:+.2f} pp"),
            ("Sp falls by at most 1 pp", d_sp, d_sp >= -1.0, f"{d_sp:+.2f} pp"),
        ]
        L.append("| prediction | observed | holds? |")
        L.append("|---|---|---|")
        for lbl, _, ok, shown in checks:
            L.append(f"| {lbl} | {shown} | {'yes' if ok else 'no'} |")
        L.append(f"\nclDice metric changed by {d_cl:+.2f} pp. "
                 "The review set a 0.01 (1 pp) F1 noise floor on this 20-image "
                 f"test set, so an F1 move of {abs(d_f1):.2f} pp is "
                 f"{'above' if abs(d_f1) >= 1.0 else 'below'} that floor and "
                 f"{'should' if abs(d_f1) >= 1.0 else 'should not'} be read as real "
                 "without repeating it across seeds.")

    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    open(args.out_md, "w").write(md)
    print(md)

    # ------------------------------------------------------------- LaTeX table
    tex = [r"\begin{table}[!t]",
           r"  \caption{Ablation on DRIVE: study-group pseudo-labels against the",
           r"  clDice loss. Network, split, schedule, optimiser, augmentation and",
           r"  seed are identical in all rows; only the objective changes.}",
           r"  \label{tab:ablation}", r"  \centering",
           r"  \renewcommand{\arraystretch}{1.15}", r"  \setlength{\tabcolsep}{4pt}",
           r"  \begin{tabular}{@{} l l c c r r r r r @{}}", r"    \toprule",
           r"    & \textbf{Configuration} & \textbf{SGL} & \textbf{clDice} & "
           r"\textbf{Acc} & \textbf{AUC} & \textbf{Se} & \textbf{$F_1$} & "
           r"\textbf{clDice} \\", r"    \midrule"]
    keys = ["acc", "auc", "se", "dice", "cldice"]
    _tex_cell = lambda d, k: (f"{d[k]:.4f}" if d.get(k) is not None else "--")
    if official is not None:
        tex.append("    R & published weights & yes & no & "
                   + " & ".join(_tex_cell(official, k) for k in keys) + r" \\")
    for name in present:
        tagd, desc, sgl, cld = LABELS[name]
        mean = rows[name][args.tag]["mean"]
        desc_tex = desc.replace("_", r"\_")
        tex.append(f"    {tagd} & {desc_tex} & {sgl} & {cld} & "
                   + " & ".join(_tex_cell(mean, k) for k in keys) + r" \\")
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    open(args.out_tex, "w").write("\n".join(tex) + "\n")
    print(f"[comparison] wrote {args.out_md} and {args.out_tex}")


if __name__ == "__main__":
    main()

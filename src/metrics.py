"""Segmentation metrics matching the official SGL evaluation protocol.

For each test image the probability map is masked by the FOV, cropped to the
original 584x565 region and thresholded (prob > 100/255). Metrics are then
computed over the cropped region (out-of-FOV pixels are zero in both the
prediction and the label, counting as true negatives -- this is the protocol
used in the paper).
"""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

ORIG_H, ORIG_W = 584, 565
THRESH = 100 / 255.0


def _acc_from_confusion(conf):
    """conf = confusion_matrix(pred, gt); rows = pred, cols = gt."""
    conf = np.asarray(conf, np.float64)
    tp = np.diag(conf)
    acc = tp.sum() / conf.sum()
    # Se = TP / (TP + FN); with rows=pred, cols=gt: FN = conf[0][1]
    se = conf[1][1] / (conf[1][1] + conf[0][1] + 1e-12)
    sp = conf[0][0] / (conf[0][0] + conf[1][0] + 1e-12)
    # per-class IoU
    iu = []
    for i in range(2):
        denom = conf[i].sum() + conf[:, i].sum() - conf[i][i]
        iu.append(conf[i][i] / (denom + 1e-12))
    return acc, se, sp, iu


def evaluate_prediction(prob, vessel_gt, fov_mask):
    """Compute metrics for a single image.

    Parameters
    ----------
    prob : (H, W) float in [0, 1]      -- raw segmentation probability
    vessel_gt : (H, W) float {0, 1}
    fov_mask : (H, W) float {0, 1}
    """
    prob = prob[:ORIG_H, :ORIG_W] * fov_mask[:ORIG_H, :ORIG_W]
    gt = (vessel_gt[:ORIG_H, :ORIG_W] * fov_mask[:ORIG_H, :ORIG_W] > 0.5).astype(np.uint8)
    pred_bin = (prob > THRESH).astype(np.uint8)

    conf = confusion_matrix(pred_bin.reshape(-1), gt.reshape(-1), labels=[0, 1])
    acc, se, sp, iu = _acc_from_confusion(conf)
    try:
        auc = roc_auc_score(gt.reshape(-1), prob.reshape(-1))
    except ValueError:
        auc = float("nan")

    inter = np.sum(pred_bin * gt)
    dice = 2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-12)
    return {"acc": acc, "se": se, "sp": sp, "auc": auc,
            "iou_bg": iu[0], "iou_vessel": iu[1], "dice": dice}


def mean_metrics(per_image):
    keys = ["acc", "se", "sp", "auc", "iou_bg", "iou_vessel", "dice"]
    return {k: float(np.nanmean([m[k] for m in per_image])) for k in keys}

"""Segmentation metrics matching the official SGL evaluation protocol.

For each test image the probability map is masked by the FOV, cropped to the
original 584x565 region and thresholded (prob > 100/255). Metrics are then
computed over the cropped region (out-of-FOV pixels are zero in both the
prediction and the label, counting as true negatives -- this is the protocol
used in the paper).
"""
import numpy as np
from skimage.measure import label as cc_label
from skimage.morphology import skeletonize
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


def _cldice(pred_bin, gt):
    """Binary centre-line Dice (Shit et al., CVPR 2021) -- the topology metric
    matching the clDice training loss. Unlike Dice it is driven by whether the
    two masks share a connected centre-line, not by how thick they are."""
    if pred_bin.sum() == 0 or gt.sum() == 0:
        return 0.0
    skel_pred = skeletonize(pred_bin.astype(bool))
    skel_gt = skeletonize(gt.astype(bool))
    tprec = (skel_pred & gt.astype(bool)).sum() / (skel_pred.sum() + 1e-12)
    tsens = (skel_gt & pred_bin.astype(bool)).sum() / (skel_gt.sum() + 1e-12)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def _connectivity(pred_bin, gt):
    """Fragmentation of the predicted vessel tree.

    The retinal vessel tree is (almost) one connected object, so a prediction
    that breaks thin branches shows up as many extra connected components.
    ``beta0_ratio`` > 1 means the prediction is more fragmented than the label.
    """
    n_pred = cc_label(pred_bin, connectivity=2).max()
    n_gt = cc_label(gt, connectivity=2).max()
    return int(n_pred), int(n_gt), float(n_pred / max(n_gt, 1))


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
    cldice = _cldice(pred_bin, gt)
    n_pred, n_gt, beta0 = _connectivity(pred_bin, gt)
    return {"acc": acc, "se": se, "sp": sp, "auc": auc,
            "iou_bg": iu[0], "iou_vessel": iu[1], "dice": dice,
            "cldice": cldice, "beta0_ratio": beta0,
            "n_cc_pred": n_pred, "n_cc_gt": n_gt}


def mean_metrics(per_image):
    keys = ["acc", "se", "sp", "auc", "iou_bg", "iou_vessel", "dice",
            "cldice", "beta0_ratio", "n_cc_pred", "n_cc_gt"]
    return {k: float(np.nanmean([m[k] for m in per_image])) for k in keys}

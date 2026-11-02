"""Loss functions for SGL training.

The official model is trained with plain binary cross-entropy on the sigmoid
segmentation output (``base_bce_loss`` in loss/bceloss.py). The SGL final model
sums two BCE terms -- one against the human label, one against the teacher
(pseudo) label:

    loss = BCE(pred, vessel_gt) + BCE(pred, teacher)

On top of that this module provides the centre-line Dice (clDice) loss of

    Shit et al. "clDice -- a Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation." CVPR 2021.

clDice scores the overlap between the *skeleton* of one mask and the volume of
the other, so a break in a thin branch costs far more than a slightly thick
edge. ``build_loss`` exposes the four combinations used by the ablation
(SGL pseudo-labels on/off x clDice on/off).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-14


class BCELoss(nn.Module):
    """Mean binary cross-entropy over all pixels (matches base_bce_loss)."""

    def forward(self, pred, target):
        bce = -target * torch.log(pred + EPS) - (1 - target) * torch.log(1 - pred + EPS)
        return bce.mean()


# --------------------------------------------------------------- soft morphology
def soft_erode(img):
    """Greyscale erosion with a cross-shaped structuring element."""
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeleton(img, iters=3):
    """Differentiable skeleton (Shit et al., eq. 5): iterated erode-minus-open."""
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class ClDiceLoss(nn.Module):
    """Soft centre-line Dice loss, ``1 - clDice``.

    ``target`` may be soft (the SGL teacher maps are in [0, 1]); the skeleton
    of a soft map is well defined because the morphology is greyscale.
    """

    def __init__(self, iters=3, smooth=1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target):
        skel_pred = soft_skeleton(pred, self.iters)
        skel_true = soft_skeleton(target, self.iters)
        s = self.smooth
        # topology precision: does the predicted centre-line lie inside the label?
        tprec = (torch.sum(skel_pred * target) + s) / (torch.sum(skel_pred) + s)
        # topology sensitivity: is the label's centre-line covered by the prediction?
        tsens = (torch.sum(skel_true * pred) + s) / (torch.sum(skel_true) + s)
        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


# -------------------------------------------------------------------- objectives
class SGLoss(nn.Module):
    """Study-Group dual-target loss: BCE against GT + BCE against teacher."""

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()

    def forward(self, pred, vessel_gt, teacher):
        return self.bce(pred, vessel_gt) + self.bce(pred, teacher)


class PlainBCELoss(nn.Module):
    """Single-target BCE on the human label; ignores the teacher map.

    This is the no-SGL control: the same network and schedule trained on the
    original (noisy) DRIVE annotations only.
    """

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()

    def forward(self, pred, vessel_gt, teacher=None):
        return self.bce(pred, vessel_gt)


class ClDiceBCELoss(nn.Module):
    """Equal mix of BCE and clDice against the human label (no SGL teacher)."""

    def __init__(self, alpha=0.5, iters=3):
        super().__init__()
        self.alpha = alpha
        self.bce = BCELoss()
        self.cldice = ClDiceLoss(iters=iters)

    def forward(self, pred, vessel_gt, teacher=None):
        return ((1 - self.alpha) * self.bce(pred, vessel_gt)
                + self.alpha * self.cldice(pred, vessel_gt))


class SGLClDiceLoss(nn.Module):
    """Proposed loss: SGL's dual target, with clDice mixed into each term.

        loss = (1-a) * [BCE(gt) + BCE(teacher)]
             +    a  * [clDice(gt) + clDice(teacher)]

    Keeping both targets in both terms means the only change with respect to
    the reproduced baseline is the *kind* of agreement being measured, not what
    the model is asked to agree with.
    """

    def __init__(self, alpha=0.5, iters=3):
        super().__init__()
        self.alpha = alpha
        self.bce = BCELoss()
        self.cldice = ClDiceLoss(iters=iters)

    def forward(self, pred, vessel_gt, teacher):
        ce = self.bce(pred, vessel_gt) + self.bce(pred, teacher)
        cl = self.cldice(pred, vessel_gt) + self.cldice(pred, teacher)
        return (1 - self.alpha) * ce + self.alpha * cl


_LOSSES = {
    "bce": PlainBCELoss,          # row 3: no SGL, no clDice
    "sgl": SGLoss,                # row 2: paper baseline
    "cldice": ClDiceBCELoss,      # row 5: clDice without SGL
    "sgl_cldice": SGLClDiceLoss,  # row 4: proposed
}

#: losses that make use of the teacher (pseudo-label) target
NEEDS_TEACHER = {"sgl", "sgl_cldice"}


def build_loss(name="sgl", **kwargs):
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(_LOSSES)}")
    cls = _LOSSES[name]
    if name in ("cldice", "sgl_cldice"):
        return cls(**kwargs)
    return cls()

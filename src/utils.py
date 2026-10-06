"""Shared helpers: device selection and self-ensemble (8-way TTA) inference."""
import itertools

import torch


def get_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _transform(x, hflip, vflip, transpose):
    if hflip:
        x = torch.flip(x, dims=[-1])
    if vflip:
        x = torch.flip(x, dims=[-2])
    if transpose:
        x = x.transpose(-1, -2)
    return x


def _inverse(x, hflip, vflip, transpose):
    # invert in reverse order
    if transpose:
        x = x.transpose(-1, -2)
    if vflip:
        x = torch.flip(x, dims=[-2])
    if hflip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict(model, hr, self_ensemble=True):
    """Run the segmentation model on a padded (B, 3, H, W) tensor.

    Returns the averaged probability map (B, 1, H, W). With ``self_ensemble``
    the prediction averages the 8 flip/transpose variants (matching the
    official ``--self_ensemble`` option).
    """
    model.eval()
    combos = list(itertools.product([False, True], repeat=3)) if self_ensemble \
        else [(False, False, False)]
    acc = None
    for hflip, vflip, transpose in combos:
        xt = _transform(hr, hflip, vflip, transpose)
        _, seg = model(xt)
        seg = _inverse(seg, hflip, vflip, transpose)
        acc = seg if acc is None else acc + seg
    return acc / len(combos)

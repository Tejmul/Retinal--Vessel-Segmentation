"""DRIVE dataset loading, patch sampling and augmentation for SGL.

Reproduces the data pipeline of the official repo (sgl_training/data):
  * images kept in [0, 255] (the network divides by 255 internally),
  * vessel / FOV masks binarised, teacher (pseudo) labels kept soft in [0, 1],
  * training: random crop + hflip/vflip/rot90 (+ optional elastic deform),
  * testing : zero-pad to a multiple of 16 (608x608), crop back at eval time.
"""
import os
import glob
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:  # elastic deformation is optional (only used for training augmentation)
    from scipy.ndimage import gaussian_filter, map_coordinates
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

ORIG_H, ORIG_W = 584, 565
PAD_H, PAD_W = 608, 608  # next multiple of 16 that covers the image


def _read(path):
    return np.array(Image.open(path))


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation (Simard 2003) applied jointly to stacked channels."""
    import cv2

    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size,
    ])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)


class DriveDataset(Dataset):
    """DRIVE images with vessel labels, FOV masks and optional teacher labels.

    Parameters
    ----------
    root : str
        Path to ``data/DRIVE``.
    split : {"training", "test"}
    indices : list[str] or None
        Two-digit image ids to include (e.g. ``["21", "22"]``). ``None`` = all.
    teacher_dir : str or None
        Directory with soft pseudo-labels ``{id}_training_x1_teacher.png``.
        When ``None`` the vessel ground truth is reused as the teacher target
        (so the dual-term loss degenerates to supervision on the GT only).
    patch_size, repeat : training patch configuration.
    augment, elastic : enable geometric augmentation / elastic deform.
    """

    def __init__(self, root, split="training", indices=None, teacher_dir=None,
                 patch_size=128, repeat=40, augment=True, elastic=True):
        self.root = root
        self.split = split
        self.train = split == "training"
        self.patch_size = patch_size
        self.repeat = repeat if self.train else 1
        self.augment = augment and self.train
        self.elastic = elastic and self.train and _HAS_SCIPY
        self.teacher_dir = teacher_dir

        img_dir = os.path.join(root, split, "images")
        self.ids = sorted(os.path.basename(p).split("_")[0]
                          for p in glob.glob(os.path.join(img_dir, "*.tif")))
        if indices is not None:
            keep = set(indices)
            self.ids = [i for i in self.ids if i in keep]
        if not self.ids:
            raise RuntimeError(f"No images found for split={split} indices={indices}")

        self.samples = [self._load_one(i) for i in self.ids]

    # ------------------------------------------------------------------ IO
    def _load_one(self, idx):
        base = os.path.join(self.root, self.split)
        hr = _read(os.path.join(base, "images", f"{idx}_training.tif")).astype(np.float32)
        ve = _read(os.path.join(base, "1st_manual", f"{idx}_manual1.gif")).astype(np.float32)
        ma = _read(os.path.join(base, "mask", f"{idx}_mask.gif")).astype(np.float32)
        ve = (ve > 200).astype(np.float32)
        ma = (ma > 125).astype(np.float32)
        if self.teacher_dir is not None:
            tp = os.path.join(self.teacher_dir, f"{idx}_training_x1_teacher.png")
            te = _read(tp).astype(np.float32) / 255.0
        else:
            te = ve.copy()
        return {"id": idx, "hr": hr, "ve": ve, "ma": ma, "te": te}

    def __len__(self):
        return len(self.samples) * self.repeat

    # ------------------------------------------------------------ transforms
    def _augment(self, hr, ve, ma, te):
        if random.random() < 0.5:  # hflip
            hr, ve, ma, te = hr[:, ::-1], ve[:, ::-1], ma[:, ::-1], te[:, ::-1]
        if random.random() < 0.5:  # vflip
            hr, ve, ma, te = hr[::-1], ve[::-1], ma[::-1], te[::-1]
        if random.random() < 0.5:  # rot90 (transpose spatial dims)
            hr = hr.transpose(1, 0, 2)
            ve, ma, te = ve.T, ma.T, te.T
        return hr, ve, ma, te

    def _crop(self, hr, ve, ma, te):
        ih, iw = hr.shape[:2]
        ip = self.patch_size
        iy = random.randrange(0, ih - ip + 1)
        ix = random.randrange(0, iw - ip + 1)
        sl = (slice(iy, iy + ip), slice(ix, ix + ip))
        return hr[sl], ve[sl], ma[sl], te[sl]

    def _getitem_train(self, s):
        hr, ve, ma, te = s["hr"], s["ve"], s["ma"], s["te"]
        if self.augment:
            hr, ve, ma, te = self._augment(hr, ve, ma, te)
        hr, ve, ma, te = self._crop(np.ascontiguousarray(hr),
                                    np.ascontiguousarray(ve),
                                    np.ascontiguousarray(ma),
                                    np.ascontiguousarray(te))
        if self.elastic and random.random() < 0.5:
            merged = np.concatenate([hr, ve[..., None], te[..., None]], axis=2)
            merged = elastic_transform(merged, merged.shape[1] * 3,
                                       merged.shape[1] * 0.07, merged.shape[1] * 0.09)
            hr = merged[..., 0:3]
            ve = (merged[..., 3] > 0.5).astype(np.float32)
            te = np.clip(merged[..., 4], 0, 1)
        return hr, ve, ma, te

    def _getitem_test(self, s):
        hr = np.zeros((PAD_H, PAD_W, 3), np.float32)
        ve = np.zeros((PAD_H, PAD_W), np.float32)
        ma = np.zeros((PAD_H, PAD_W), np.float32)
        h, w = s["hr"].shape[:2]
        hr[:h, :w] = s["hr"]
        ve[:h, :w] = s["ve"]
        ma[:h, :w] = s["ma"]
        return hr, ve, ma, ve

    def __getitem__(self, i):
        s = self.samples[i % len(self.samples)]
        if self.train:
            hr, ve, ma, te = self._getitem_train(s)
        else:
            hr, ve, ma, te = self._getitem_test(s)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        hr_t = to_t(hr.transpose(2, 0, 1))
        ve_t = to_t(ve)[None]
        ma_t = to_t(ma)[None]
        te_t = to_t(te)[None]
        return {"hr": hr_t, "ve": ve_t, "ma": ma_t, "te": te_t, "id": s["id"]}

"""Local web UI for the SGL / clDice retinal vessel models.

Run from the repository root:

    .venv/bin/python app/server.py          # then open http://127.0.0.1:5001

The browser asks for one inference and gets back the *probability* map rather
than a binary mask, so the threshold slider re-thresholds and re-scores in the
page with no further round trip. That matters here: the ablation's fixed
threshold of 100/255 was chosen for a BCE-trained model, and a clDice-trained
model puts its probability mass elsewhere, so being able to sweep the threshold
is the difference between reading the comparison and arguing about it.
"""
import base64
import io
import os
import sys
import time

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dataset import ORIG_H, ORIG_W, PAD_H, PAD_W  # noqa: E402
from model import build_model                      # noqa: E402
from utils import get_device, predict              # noqa: E402

DATA = os.path.join(ROOT, "data", "DRIVE")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

#: the four ablation rows, plus the paper's released weights
MODELS = {
    "D_sgl_cldice":   ("D  SGL + clDice (proposed)", "ablation/D_sgl_cldice_last.pt"),
    "A_sgl_baseline": ("A  SGL baseline (paper loss)", "ablation/A_sgl_baseline_last.pt"),
    "C_cldice":       ("C  clDice only", "ablation/C_cldice_last.pt"),
    "B_no_sgl":       ("B  no SGL, BCE", "ablation/B_no_sgl_last.pt"),
    "official_k8":    ("R  published drive_k8 weights", "official_drive_k8.pth"),
}

MAX_UPLOAD = 12 * 1024 * 1024
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

_device = get_device("auto")
_cache = {}          # key -> model, kept small on purpose (59 MB each)


def load_model(key):
    if key in _cache:
        return _cache[key]
    if key not in MODELS:
        raise KeyError(key)
    path = os.path.join(CKPT_DIR, MODELS[key][1])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    net = build_model()
    # the released checkpoint carries two dead `last_act` tensors we do not use
    net.load_state_dict(state, strict=False)
    net.to(_device).eval()
    if len(_cache) >= 2:                      # keep memory bounded on 8 GB
        _cache.pop(next(iter(_cache)))
    _cache[key] = net
    return net


def available_models():
    return [{"key": k, "label": lbl,
             "available": os.path.exists(os.path.join(CKPT_DIR, rel))}
            for k, (lbl, rel) in MODELS.items()]


def test_ids():
    d = os.path.join(DATA, "test", "images")
    if not os.path.isdir(d):
        return []
    return sorted(f.split("_")[0] for f in os.listdir(d) if f.endswith(".tif"))


# ----------------------------------------------------------------- image utils
def _png_b64(arr, mode="L"):
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpg_b64(arr, quality=86):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_test_image(idx):
    """Fundus, label and FOV mask for one DRIVE test image."""
    base = os.path.join(DATA, "test")
    hr = np.array(Image.open(os.path.join(base, "images", f"{idx}_training.tif")))
    ve = np.array(Image.open(os.path.join(base, "1st_manual", f"{idx}_manual1.gif")))
    ma = np.array(Image.open(os.path.join(base, "mask", f"{idx}_mask.gif")))
    return (hr[:ORIG_H, :ORIG_W].astype(np.uint8),
            (ve[:ORIG_H, :ORIG_W] > 200).astype(np.uint8),
            (ma[:ORIG_H, :ORIG_W] > 125).astype(np.uint8))


def load_upload(fs):
    """An arbitrary photograph, letterboxed into the 584x565 working frame.

    There is no field-of-view mask for an uploaded image, so one is estimated
    from image brightness -- the retina is the bright disc on a black surround.
    """
    im = Image.open(fs.stream).convert("RGB")
    im.thumbnail((ORIG_W, ORIG_H), Image.LANCZOS)
    canvas = np.zeros((ORIG_H, ORIG_W, 3), np.uint8)
    a = np.array(im)
    canvas[:a.shape[0], :a.shape[1]] = a
    fov = (canvas.max(axis=2) > 28).astype(np.uint8)
    return canvas, None, fov


@torch.no_grad()
def run(net, fundus, self_ensemble):
    """Pad to 608x608, run the cascade, return (probability, enhancement)."""
    x = np.zeros((PAD_H, PAD_W, 3), np.float32)
    x[:fundus.shape[0], :fundus.shape[1]] = fundus
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(_device)
    prob = predict(net, t, self_ensemble=self_ensemble)[0, 0].cpu().numpy()
    enh, _ = net(t)
    enh = enh[0].cpu().numpy().transpose(1, 2, 0)
    return prob[:ORIG_H, :ORIG_W], enh[:ORIG_H, :ORIG_W]


# ----------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def state():
    return jsonify({"models": available_models(), "test_ids": test_ids(),
                    "device": str(_device), "width": ORIG_W, "height": ORIG_H})


@app.get("/api/thumb/<idx>")
def thumb(idx):
    if idx not in test_ids():
        return jsonify({"error": "unknown image"}), 404
    hr, _, _ = load_test_image(idx)
    small = np.array(Image.fromarray(hr).resize((96, 99), Image.LANCZOS))
    return jsonify({"src": _jpg_b64(small, quality=72)})


@app.post("/api/infer")
def infer():
    key = request.form.get("model") or (request.json or {}).get("model", "")
    se = (request.form.get("self_ensemble", "0") == "1")
    source = request.form.get("source", "")

    try:
        net = load_model(key)
    except FileNotFoundError:
        return jsonify({"error": f"checkpoint for {key} is not on disk. "
                                 "Train it with src/run_ablation.py first."}), 400
    except KeyError:
        return jsonify({"error": f"unknown model {key!r}"}), 400

    if "file" in request.files and request.files["file"].filename:
        try:
            fundus, gt, fov = load_upload(request.files["file"])
        except Exception as exc:
            return jsonify({"error": f"could not read that image: {exc}"}), 400
        label = "uploaded image"
    elif source in test_ids():
        fundus, gt, fov = load_test_image(source)
        label = f"DRIVE test {source}"
    else:
        return jsonify({"error": "choose a test image or upload one"}), 400

    t0 = time.time()
    prob, enh = run(net, fundus, se)
    ms = int((time.time() - t0) * 1000)

    prob = prob * fov                      # the paper scores inside the FOV only
    enh8 = np.clip(enh * 255, 0, 255).astype(np.uint8)

    return jsonify({
        "label": label, "model": key, "ms": ms, "self_ensemble": se,
        "width": ORIG_W, "height": ORIG_H,
        "fundus": _jpg_b64(fundus),
        "prob": _png_b64((prob * 255).astype(np.uint8)),
        "enhance": _jpg_b64(enh8),
        "fov": _png_b64(fov * 255),
        "gt": _png_b64(gt * 255) if gt is not None else None,
        "device": str(_device),
    })


if __name__ == "__main__":
    print(f"[ui] device={_device}")
    print(f"[ui] checkpoints in {CKPT_DIR}")
    for m in available_models():
        print(f"      {'ok ' if m['available'] else '-- '} {m['label']}")
    print(f"[ui] {len(test_ids())} DRIVE test images")
    print("[ui] http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=False)

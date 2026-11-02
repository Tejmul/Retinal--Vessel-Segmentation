/* Vessel Studio.
 *
 * The server returns the probability map once; everything the threshold slider
 * drives -- the mask, the overlay, the scores -- is recomputed here from that
 * one array, so dragging it costs nothing.
 */
const $ = s => document.querySelector(s);
const ZOOM = 6, LOUPE = 182;

const cv = $("#cv"), lcv = $("#lcv");
const cx = cv.getContext("2d", { willReadFrequently: true });
const lx = lcv.getContext("2d");
lcv.width = lcv.height = LOUPE;
cx.imageSmoothingEnabled = false;
lx.imageSmoothingEnabled = false;

let W = 565, H = 584;
let view = "overlay", thr = 100, selectedId = null, uploadFile = null;
let modelKey = null, busy = false;

/* decoded per-pixel state for the current result */
let S = null;   // {prob:Uint8Array, gt:Uint8Array|null, fov:Uint8Array,
                //  fundus:ImageData, enhance:ImageData, mask:Uint8Array}

/* ------------------------------------------------------------ small helpers */
const f4 = v => v.toFixed(4);
const loadImg = src => new Promise((res, rej) => {
  const im = new Image();
  im.onload = () => res(im);
  im.onerror = () => rej(new Error("image decode failed"));
  im.src = src;
});
function grayOf(img){
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, W, H);
  const d = g.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = d[p];
  return out;
}
function rgbaOf(img){
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, W, H);
  return g.getImageData(0, 0, W, H);
}
function showErr(msg){
  const e = $("#err");
  e.hidden = !msg;
  e.textContent = msg || "";
}

/* ------------------------------------------------------------------ drawing */
function applyThreshold(){
  if (!S) return;
  const n = W * H, m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = (S.prob[i] > thr && S.fov[i]) ? 1 : 0;
  S.mask = m;
}

function draw(){
  if (!S) return;
  cv.width = W; cv.height = H;
  let img;
  if (view === "fundus"){
    img = new ImageData(new Uint8ClampedArray(S.fundus.data), W, H);
  } else if (view === "enhance"){
    img = new ImageData(new Uint8ClampedArray(S.enhance.data), W, H);
  } else if (view === "prob"){
    img = cx.createImageData(W, H);
    for (let i = 0, p = 0; i < W * H; i++, p += 4){
      const v = S.prob[i];
      /* black -> amber -> white, so faint vessels stay visible */
      img.data[p]     = Math.min(255, v * 1.9);
      img.data[p + 1] = Math.min(255, v * 1.25);
      img.data[p + 2] = Math.min(255, v * 0.72);
      img.data[p + 3] = 255;
    }
  } else if (view === "mask"){
    img = cx.createImageData(W, H);
    for (let i = 0, p = 0; i < W * H; i++, p += 4){
      const v = S.mask[i] ? 240 : 10;
      img.data[p] = img.data[p + 1] = img.data[p + 2] = v;
      img.data[p + 3] = 255;
    }
  } else {                                   /* overlay */
    img = new ImageData(new Uint8ClampedArray(S.fundus.data), W, H);
    const d = img.data;
    for (let i = 0, p = 0; i < W * H; i++, p += 4){
      const col = pixelClass(i);
      if (col){ d[p] = col[0]; d[p+1] = col[1]; d[p+2] = col[2]; d[p+3] = 255; }
    }
  }
  cx.putImageData(img, 0, 0);
}

const COL = { hit: [154,160,166], miss: [78,163,214], fp: [226,86,92], plain: [240,240,240] };
function pixelClass(i){
  const m = S.mask[i];
  if (!S.gt){ return m ? COL.plain : null; }
  const g = S.gt[i] > 127 ? 1 : 0;
  if (m && g) return COL.hit;
  if (!m && g) return COL.miss;
  if (m && !g) return COL.fp;
  return null;
}

/* ------------------------------------------------------------------- scores */
function score(){
  const box = $("#scores");
  if (!S || !S.gt){
    box.innerHTML = `<p class="note">Scores appear once a DRIVE test image is segmented —
      an uploaded photograph has no ground-truth tracing to compare against.</p>`;
    $("#best").disabled = true;
    $("#scoreSrc").textContent = S ? "no label" : "vs 1st_manual";
    return;
  }
  const s = stats(thr);
  $("#scoreSrc").textContent = "vs 1st_manual";
  $("#best").disabled = false;
  const row = (k, v, cls = "") => `<div class="k">${k}</div><div class="v ${cls}">${v}</div>`;
  box.innerHTML =
      row("F₁ (Dice)", f4(s.dice), "good")
    + `<div class="meter"><i style="width:${(s.dice * 100).toFixed(1)}%"></i></div>`
    + row("Sensitivity", f4(s.se))
    + row("Specificity", f4(s.sp))
    + row("Accuracy", f4(s.acc))
    + row("IoU (vessel)", f4(s.iou))
    + row("Vessel pixels", s.pred.toLocaleString())
    + row("Label pixels", s.gtn.toLocaleString());
}

function stats(t){
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < W * H; i++){
    const m = (S.prob[i] > t && S.fov[i]) ? 1 : 0;
    const g = S.gt[i] > 127 ? 1 : 0;
    if (m && g) tp++; else if (m) fp++; else if (g) fn++; else tn++;
  }
  return {
    dice: (2 * tp) / (2 * tp + fp + fn || 1),
    se: tp / (tp + fn || 1),
    sp: tn / (tn + fp || 1),
    acc: (tp + tn) / (tp + tn + fp + fn || 1),
    iou: tp / (tp + fp + fn || 1),
    pred: tp + fp, gtn: tp + fn,
  };
}

/* ------------------------------------------------------------------- loupe */
const viewport = $("#viewport"), loupe = $("#loupe");
function drawLoupe(ix, iy){
  const src = Math.round(LOUPE / ZOOM), half = src >> 1;
  const x0 = Math.max(0, Math.min(W - src, ix - half));
  const y0 = Math.max(0, Math.min(H - src, iy - half));
  const out = lx.createImageData(src, src);
  const fd = S.fundus.data, od = out.data;
  for (let y = 0; y < src; y++){
    for (let x = 0; x < src; x++){
      const si = (y0 + y) * W + (x0 + x), sp = si * 4, op = (y * src + x) * 4;
      let r = fd[sp] * .5, g = fd[sp+1] * .5, b = fd[sp+2] * .5;
      const col = pixelClass(si);
      if (col){ r = col[0]; g = col[1]; b = col[2]; }
      od[op] = r; od[op+1] = g; od[op+2] = b; od[op+3] = 255;
    }
  }
  const t = document.createElement("canvas");
  t.width = t.height = src;
  t.getContext("2d").putImageData(out, 0, 0);
  lx.clearRect(0, 0, LOUPE, LOUPE);
  lx.imageSmoothingEnabled = false;
  lx.drawImage(t, 0, 0, LOUPE, LOUPE);
  lx.strokeStyle = "rgba(255,255,255,.45)"; lx.lineWidth = 1;
  lx.beginPath();
  lx.moveTo(LOUPE/2, LOUPE/2 - 8); lx.lineTo(LOUPE/2, LOUPE/2 + 8);
  lx.moveTo(LOUPE/2 - 8, LOUPE/2); lx.lineTo(LOUPE/2 + 8, LOUPE/2);
  lx.stroke();
}
viewport.addEventListener("pointermove", ev => {
  if (!S) return;
  const r = cv.getBoundingClientRect();
  const ix = Math.round((ev.clientX - r.left) / r.width * W);
  const iy = Math.round((ev.clientY - r.top) / r.height * H);
  if (ix < 0 || iy < 0 || ix >= W || iy >= H){ loupe.classList.remove("on"); return; }
  loupe.classList.add("on");
  const vr = viewport.getBoundingClientRect();
  loupe.style.left = (ev.clientX - vr.left - LOUPE / 2) + "px";
  loupe.style.top  = (ev.clientY - vr.top  - LOUPE / 2) + "px";
  drawLoupe(ix, iy);
});
viewport.addEventListener("pointerleave", () => loupe.classList.remove("on"));

/* --------------------------------------------------------------- inference */
async function segment(){
  if (busy) return;
  if (!selectedId && !uploadFile){ showErr("Choose a test image or drop a photograph first."); return; }
  if (!modelKey){ showErr("No trained checkpoint available."); return; }
  busy = true; showErr("");
  $("#busy").hidden = false; $("#run").disabled = true;
  $("#timing").textContent = "running…"; $("#timing").classList.add("live");

  const fd = new FormData();
  fd.append("model", modelKey);
  fd.append("self_ensemble", $("#se").checked ? "1" : "0");
  if (uploadFile) fd.append("file", uploadFile);
  else fd.append("source", selectedId);

  try {
    const r = await fetch("/api/infer", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "inference failed");
    W = j.width; H = j.height;
    const [fun, prob, enh, fov, gt] = await Promise.all([
      loadImg(j.fundus), loadImg(j.prob), loadImg(j.enhance), loadImg(j.fov),
      j.gt ? loadImg(j.gt) : Promise.resolve(null),
    ]);
    S = { fundus: rgbaOf(fun), enhance: rgbaOf(enh), prob: grayOf(prob),
          fov: grayOf(fov), gt: gt ? grayOf(gt) : null, mask: null };
    $("#empty").hidden = true;
    applyThreshold(); draw(); score();
    $("#timing").textContent = `${j.label} · ${j.ms} ms${j.self_ensemble ? " · 8×TTA" : ""}`;
  } catch (e){
    showErr(e.message);
    $("#timing").textContent = "failed";
  } finally {
    busy = false;
    $("#busy").hidden = true; $("#run").disabled = false;
    $("#timing").classList.remove("live");
  }
}

/* --------------------------------------------------------------- UI wiring */
$("#run").onclick = segment;

$("#tabs").onclick = e => {
  const b = e.target.closest(".tab");
  if (!b) return;
  view = b.dataset.view;
  document.querySelectorAll(".tab").forEach(t =>
    t.setAttribute("aria-pressed", String(t === b)));
  $("#caption").textContent = ({
    overlay: "Prediction against the human tracing, over the fundus. Grey found, blue missed, red false positive.",
    mask: "The thresholded binary mask, exactly as the metrics see it.",
    prob: "Raw sigmoid output before thresholding. Faint capillaries sit in the mid-tones — this is what the threshold slider is cutting.",
    enhance: "EnhanceNet's own output — the first of the two cascaded U-Nets. It is never supervised directly; it is what the network learned to feed the segmentation stage.",
    fundus: "The input photograph, unmodified.",
  })[view];
  draw();
};

const thrEl = $("#thr");
thrEl.oninput = () => {
  thr = +thrEl.value;
  $("#thrOut").textContent = `${thr} / 255`;
  applyThreshold(); draw(); score();
};

$("#best").onclick = () => {
  if (!S || !S.gt) return;
  let bestT = thr, bestD = -1;
  for (let t = 10; t <= 245; t += 5){
    const d = stats(t).dice;
    if (d > bestD){ bestD = d; bestT = t; }
  }
  thrEl.value = bestT; thrEl.oninput();
};

/* file drop */
const drop = $("#drop"), fileEl = $("#file");
drop.onclick = () => fileEl.click();
fileEl.onchange = () => {
  if (!fileEl.files[0]) return;
  uploadFile = fileEl.files[0]; selectedId = null;
  document.querySelectorAll("#thumbs button").forEach(b => b.setAttribute("aria-pressed", "false"));
  drop.querySelector("strong").textContent = uploadFile.name;
  showErr(""); segment();
};
["dragenter", "dragover"].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev =>
  drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  uploadFile = f; selectedId = null;
  drop.querySelector("strong").textContent = f.name;
  segment();
});

/* ------------------------------------------------------------------- boot */
(async function init(){
  let st;
  try {
    st = await (await fetch("/api/state")).json();
  } catch {
    showErr("Cannot reach the server. Start it with: .venv/bin/python app/server.py");
    return;
  }
  $("#device").textContent = "device " + st.device;
  W = st.width; H = st.height;

  const avail = st.models.filter(m => m.available);
  $("#models").innerHTML = st.models.map((m, i) => {
    const on = m.available && m.key === (avail[0] && avail[0].key);
    return `<label class="${m.available ? (on ? "on" : "") : "off"}">
      <input type="radio" name="m" value="${m.key}" ${on ? "checked" : ""}
             ${m.available ? "" : "disabled"}>
      <span><span class="tagc">${m.label.slice(0, 2)}</span>${m.label.slice(2)}
      ${m.available ? "" : "<small>checkpoint not on disk</small>"}</span></label>`;
  }).join("");
  modelKey = avail[0] ? avail[0].key : null;
  $("#models").onchange = e => {
    modelKey = e.target.value;
    document.querySelectorAll("#models label").forEach(l =>
      l.classList.toggle("on", l.querySelector("input").checked));
    if (S) segment();
  };
  if (!avail.length) showErr("No checkpoints found. Run src/run_ablation.py first.");

  $("#thumbs").innerHTML = st.test_ids.map(id =>
    `<button data-id="${id}" aria-pressed="false" title="DRIVE test ${id}">
       <b>${id}</b></button>`).join("");
  document.querySelectorAll("#thumbs button").forEach(b => {
    b.onclick = () => pick(b.dataset.id);
    fetch(`/api/thumb/${b.dataset.id}`).then(r => r.json()).then(j => {
      const im = new Image();
      im.src = j.src; im.alt = "";
      b.prepend(im);
    }).catch(() => {});
  });

  /* ?img=03&model=A_sgl_baseline&thr=120 opens straight on that result */
  const q = new URLSearchParams(location.search);
  const qm = q.get("model");
  if (qm && avail.some(m => m.key === qm)){
    modelKey = qm;
    const r = document.querySelector(`#models input[value="${CSS.escape(qm)}"]`);
    if (r){
      r.checked = true;
      document.querySelectorAll("#models label").forEach(l =>
        l.classList.toggle("on", l.querySelector("input").checked));
    }
  }
  const qt = parseInt(q.get("thr"), 10);
  if (qt >= 1 && qt <= 254){ thrEl.value = qt; thr = qt; $("#thrOut").textContent = `${qt} / 255`; }
  const qi = q.get("img");
  if (qi && st.test_ids.includes(qi)) pick(qi);
})();

function pick(id){
  selectedId = id; uploadFile = null;
  drop.querySelector("strong").textContent = "Drop a fundus photo";
  document.querySelectorAll("#thumbs button").forEach(o =>
    o.setAttribute("aria-pressed", String(o.dataset.id === id)));
  segment();
}

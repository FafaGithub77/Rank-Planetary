#!/usr/bin/env python3
"""
rank_planetary.py - ranks planetary images (TIF/PNG/FITS-like) by quality.

Usage:
    python3 rank_planetary.py folder_or_file [others...] [--csv out.csv] [--sheet sheet.png] [--top N]
                              [--copy N|top [--copy-dir DIR]] [--side OUT.png [--side-n N]] [--with-snr]
    python3 rank_planetary.py --compare raws.csv processed.csv     (before/after)

Example:
    python3 rank_planetary.py ~/Saturn/*.tif --csv ranking.csv --sheet sheet.png

Dependencies: numpy, opencv-python, tifffile
    pip install numpy opencv-python tifffile

Metrics (all independent of brightness/stretch, computed on the planet):
  detail  Fine-detail energy (DoG 1px-2px) in the disk, normalised on brightness.
          Higher = more detail. DoG partially ignores pixel-level noise.
  limb    Steepness of the planet edge (normalised peak gradient).
          Higher = sharper edge = better seeing/stack.
  hf      Fraction of spectral power at high frequency (above 0.25 of Nyquist).
          Higher = more effective resolution.
  noise   Estimated sky noise (MAD of DoG), relative to brightness. Lower = better.
  snr     detail / noise. Higher = real detail, not amplified noise.
  halo    Dark/bright ring around the planet (ringing from deconvolution). Lower = better.
  rgb     Residual misalignment between R/G/B channels in pixels (max of R-G and B-G,
          sub-pixel ECC registration). Lower = better RGB alignment. Informational only.

score = relative score (ranks, depends on the set; differences below TOL = tied).
top   = top group (*): images indistinguishable from the best on all score metrics.
z     = absolute score (robust z-score vs fixed reference, stable across runs).

Relative score = mean of ranks for detail, limb, hf, snr, minus penalty for high halo.
Note: compare only images at the same scale (same focal length/crop/binning).
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tifffile

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
BEST_DIR = "best"  # folder where copy_best() copies the top images (excluded from searches)


def load_gray(path):
    if path.lower().endswith((".tif", ".tiff")):
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # OpenCV returns BGR(A)
        if img is not None and img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., [2, 1, 0] + list(range(3, img.shape[2]))]  # -> RGB(A)
    img = np.asarray(img).astype(np.float64)
    if img.ndim == 3:
        if img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
            img = np.moveaxis(img, 0, -1)
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        rgb = img[..., :3]
    else:
        lum, rgb = img, None
    return lum, rgb


def planet_mask(lum):
    """Planet mask (rings included) via Otsu thresholding on smoothed luminance."""
    sm = cv2.GaussianBlur(lum, (0, 0), 2)
    u8 = np.clip(255 * sm / max(sm.max(), 1e-9), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (u8 > t * 0.5).astype(np.uint8)  # low threshold to include faint edges
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise ValueError("planet not found")
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    m = (lab == k).astype(np.uint8)
    # fill holes (Cassini division etc.)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _ecc_shift(ref, mov):
    """Sub-pixel translation (px) between two channels via ECC; NaN if no convergence."""
    ref = cv2.GaussianBlur(ref, (0, 0), 1.0).astype(np.float32)
    mov = cv2.GaussianBlur(mov, (0, 0), 1.0).astype(np.float32)
    ref /= max(float(ref.max()), 1e-9)
    mov /= max(float(mov.max()), 1e-9)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), None, 5)
    except cv2.error:
        return float("nan")
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def chroma_offset(rgb, box):
    """Residual channel misalignment (px): max(|R-G|, |B-G|), on the crop around the planet."""
    if rgb is None:
        return float("nan")
    y0, y1, x0, x1 = box
    c = [rgb[y0:y1, x0:x1, i] for i in range(3)]
    v = [_ecc_shift(c[1], c[0]), _ecc_shift(c[1], c[2])]
    return float(np.nanmax(v)) if not np.all(np.isnan(v)) else float("nan")


def analyze(path):
    lum, rgb = load_gray(path)
    lum = lum - np.percentile(lum, 5)  # subtract sky background
    lum = np.clip(lum, 0, None)

    m = planet_mask(lum)
    inner = cv2.erode(m, disk(4)).astype(bool)             # interior, away from the edge
    band = (cv2.dilate(m, disk(4)) - cv2.erode(m, disk(4))).astype(bool)  # limb band
    sky_far = ~cv2.dilate(m, disk(25)).astype(bool)
    sky_near = (cv2.dilate(m, disk(10)) - cv2.dilate(m, disk(4))).astype(bool)

    ref = float(np.median(lum[inner])) if inner.sum() > 50 else float(lum[m.astype(bool)].mean())
    if ref < 1e-4 * float(lum.max()):
        raise ValueError("disk too faint or mask error: metrics unreliable")
    ref = max(ref, 1e-9)
    n = lum / ref  # brightness normalisation

    # --- detail: DoG (1-2 px band) ---
    dog = cv2.GaussianBlur(n, (0, 0), 1.0) - cv2.GaussianBlur(n, (0, 0), 2.0)
    detail = float(np.sqrt(np.mean(dog[inner] ** 2)))

    # --- noise from far sky ---
    if sky_far.sum() > 200:
        d = dog[sky_far]
        noise = float(1.4826 * np.median(np.abs(d - np.median(d))))
    else:
        noise = float("nan")
    snr = detail / noise if noise and noise > 0 else float("nan")

    # --- limb: peak gradient along the edge ---
    sm = cv2.GaussianBlur(n, (0, 0), 0.8)
    gx, gy = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3) / 8, cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3) / 8
    g = np.hypot(gx, gy)
    vals = g[band]
    limb = float(np.mean(np.sort(vals)[-max(10, len(vals) // 4):])) if len(vals) else float("nan")

    # --- hf: high-frequency spectral power (crop around planet, Hann window) ---
    ys, xs = np.where(m)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, n.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, n.shape[1])
    crop = n[y0:y1, x0:x1]
    win = np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2((crop - crop.mean()) * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(crop.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(crop.shape[1]))[None, :]
    r = np.hypot(fx, fy) / 0.5  # 1.0 = Nyquist
    hf = float(P[(r > 0.25) & (r <= 1.0)].sum() / P[(r > 0.02) & (r <= 1.0)].sum())

    # --- halo: near-limb sky offset vs far sky (ringing) ---
    if sky_near.sum() > 50 and sky_far.sum() > 200:
        halo = float(abs(np.mean(n[sky_near]) - np.mean(n[sky_far])))
    else:
        halo = float("nan")

    rgb_off = chroma_offset(rgb, (y0, y1, x0, x1))
    size = float(np.sqrt(m.sum()))  # sqrt(planet area, rings included): rotation-independent

    name = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
    return dict(file=name, path=path, detail=detail, limb=limb,
                hf=hf, noise=noise, snr=snr, halo=halo, rgb=rgb_off, size=size)


# Fixed reference for the absolute score "z": median and robust scale (1.4826*MAD) of log10(metric),
# computed on 21 processed images (AutoStakkert + AstroSurface) of Saturn on 2026-10-03, Barlow 1x6.
# Being constants, z does not depend on the analysed set and remains comparable across runs
# (for the same instrument/scale). z=0 -> typical image from that session; z>0 better, z<0 worse.
# For a different setup (focal length, camera) recompute these values on your own images.
REF = {  # metric: (median, scale)
    "detail": (-1.248, 0.041),
    "limb": (-0.594, 0.029),
    "hf": (-2.104, 0.135),
    "snr": (2.199, 0.082),
    "halo": (-2.441, 0.145),
}


def abs_scores(rows):
    """Absolute score: mean of robust z-scores for detail/limb/hf/snr (NaN excluded), minus halo penalty."""
    def zs(key):
        med, sc = REF[key]
        v = np.array([r[key] for r in rows], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log10(np.where(v > 0, v, np.nan)) - med) / sc

    z = np.nanmean([zs("detail"), zs("limb"), zs("hf"), zs("snr")], axis=0)
    return z - 0.25 * np.nan_to_num(np.maximum(zs("halo"), 0), nan=0.0)


# Relative tolerance below which two images are tied on a metric (in the relative score).
# Derived from variability measured by cropping the same image at different sizes and from
# comparisons of equivalent processings: detail/limb stable, hf slightly less, snr ~5-10%, halo ~25%.
TOL = {"detail": 0.02, "limb": 0.02, "hf": 0.03, "snr": 0.10, "halo": 0.25}


def rank_scores(rows, sharp=True, mode=None):
    def ranks(key, higher_better=True):
        """1 = best, 0 = worst. Values within TOL[key] (relative) are tied and share the mean rank.
        NaN stays NaN (metric excluded for that image)."""
        v = np.array([r[key] for r in rows], dtype=float)
        ok = ~np.isnan(v)
        rk = np.full(len(v), np.nan)
        k = int(ok.sum())
        if k == 1:
            rk[ok] = 1.0
        elif k > 1:
            idx = np.where(ok)[0]
            order = idx[np.argsort(-v[idx] if higher_better else v[idx])]
            pos = 1 - np.arange(k) / (k - 1)
            i = 0
            while i < k:
                j, ref = i + 1, v[order[i]]
                while j < k and abs(v[order[j]] - ref) <= TOL[key] * abs(ref):
                    j += 1
                rk[order[i:j]] = pos[i:j].mean()
                i = j
        return rk

    # sharp=True (default): score based on sharpness only (detail, limb, hf), without snr.
    # For raws, snr rewards frame count and works against sharpness.
    # sharp=False: includes snr (useful for processed images).
    metrics = ["detail", "limb", "hf"] + ([] if sharp else ["snr"])
    score = np.nanmean([ranks(k) for k in metrics], axis=0)
    score = score - 0.15 * np.nan_to_num(1 - ranks("halo", higher_better=False), nan=0.0)
    z = abs_scores(rows)
    for r, s, zz in zip(rows, score, z):
        r["score"] = float(s)
        r["z"] = float(zz)
    rows = sorted(rows, key=lambda r: -r["score"])
    # top group: not worse than the first by more than TOL on any score metric
    best = rows[0] if rows else None
    for r in rows:
        r["top"] = all(np.isnan(r[k]) or np.isnan(best[k]) or r[k] >= best[k] * (1 - TOL[k]) for k in metrics)
    mode = mode or ("raws" if sharp else "processed")  # if not specified: sharp=True -> raws
    for r, why in zip(rows, discard_flags(rows, mode)):
        r["discard"] = why
    return rows


def top_count(rows):
    return sum(1 for r in rows if r.get("top"))


def copy_best(rows, n, dest):
    """Copy the top n images (n=0: top group) to dest. Skips already copied files."""
    n = n or top_count(rows)
    os.makedirs(dest, exist_ok=True)
    out = []
    for r in rows[:n]:
        src = r["path"]
        base = os.path.basename(src)
        tgt = os.path.join(dest, base)
        k = 2
        while os.path.exists(tgt) and os.path.getsize(tgt) != os.path.getsize(src):
            stem, ext = os.path.splitext(base)
            tgt = os.path.join(dest, f"{stem}_{k}{ext}")
            k += 1
        if not os.path.exists(tgt):
            shutil.copy2(src, tgt)
        out.append(tgt)
    return out


# ---------- before/after comparison (raws vs processed) ----------
def read_csv_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for d in csv.DictReader(fh):
            r = {"file": d.get("file", ""), "path": d.get("file", "")}
            for k in ("detail", "limb", "hf", "noise", "snr", "halo", "rgb", "z", "size"):
                try:
                    r[k] = float(d[k])
                except (KeyError, ValueError, TypeError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def _stem(name):
    return os.path.splitext(name.replace("\\", "/").split("/")[-1])[0].lower()


def match_pairs(raws, procs, min_len=8):
    """Match each processed image to the raw whose name (without extension) is contained in its name."""
    stems = [_stem(r["file"]) for r in raws]
    best, unmatched, multi = {}, [], 0
    for p in procs:
        ps = _stem(p["file"])
        cand = [(len(s), i) for i, s in enumerate(stems) if len(s) >= min_len and s in ps]
        if not cand:
            unmatched.append(p)
            continue
        i = max(cand)[1]
        if i in best:
            multi += 1
            if p["z"] > best[i]["z"]:
                best[i] = p  # when multiple variants exist, keep the one with highest z
        else:
            best[i] = p
    return [(raws[i], best[i]) for i in sorted(best)], unmatched, multi


def compare_csvs(raw_csv, proc_csv, sharp=True):
    """Before/after table: rank of each raw and its processed counterpart (ranked among matched pairs only)."""
    raws, procs = read_csv_rows(raw_csv), read_csv_rows(proc_csv)
    pairs, unmatched, multi = match_pairs(raws, procs)
    out = [f"Raws: {len(raws)}  Processed: {len(procs)}  Matched: {len(pairs)}"
           + (f"  (multiple variants: kept best by z in {multi} cases)" if multi else "")]
    if len(pairs) < 2:
        out += ["Too few matches: the processed filename must contain the raw filename",
                "(e.g. raw.tif -> raw_wavelets.tif)."]
        return out
    R = [dict(a, _i=i) for i, (a, _) in enumerate(pairs)]
    P = [dict(b, _i=i) for i, (_, b) in enumerate(pairs)]
    ra, pa = rank_scores(R, sharp, "raws"), rank_scores(P, sharp, "processed")
    rpos = {r["_i"]: k + 1 for k, r in enumerate(ra)}
    ppos = {r["_i"]: k + 1 for k, r in enumerate(pa)}
    rtop = {r["_i"] for r in ra if r["top"]}
    ptop = {r["_i"] for r in pa if r["top"]}
    n = len(pairs)
    out += ["", f"{'raw':>6} {'proc.':>5} {'diff':>5} {'z raw':>8} {'z proc.':>8}  file (raw)"]
    for i in sorted(range(n), key=lambda i: rpos[i]):
        out.append(f"{rpos[i]:>5}{'*' if i in rtop else ' '} {ppos[i]:>4}{'*' if i in ptop else ' '} "
                   f"{ppos[i] - rpos[i]:>+5d} {R[i]['z']:8.1f} {P[i]['z']:8.1f}  {R[i]['file']}")
    first = ra[0]["_i"]
    out += ["", f"Top raw -> rank {ppos[first]} out of {n} among processed.",
            f"Top group of raws ({len(rtop)}): {len(rtop & ptop)} are also in the top group of processed ({len(ptop)})."]
    if n >= 3:
        rho = float(np.corrcoef([rpos[i] for i in range(n)], [ppos[i] for i in range(n)])[0, 1])
        verdict = ("raw selection holds" if rho >= 0.8 else
                   "partially holds" if rho >= 0.5 else "raw selection does not hold")
        out.append(f"Rank correlation (Spearman): {rho:.2f} -> {verdict}" + (" (small sample)" if n < 6 else "") + ".")
    if unmatched:
        out.append(f"Processed images with no matching raw: {len(unmatched)}")
    out.append("(* = top group; ranked on " + ("detail, limb, hf" if sharp else "detail, limb, hf, snr") + ")")
    return out


def contact_sheet(rows, out, top=None, thumb=240):
    tiles = []
    for i, r in enumerate(rows[: top or len(rows)]):
        lum, rgb = load_gray(r["path"])
        im = rgb if rgb is not None else np.dstack([lum] * 3)
        im = np.clip(im / max(np.percentile(im, 99.8), 1e-9), 0, 1)
        im = (im[..., ::-1] * 255).astype(np.uint8)
        h, w = im.shape[:2]
        s = thumb / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        tile = np.zeros((thumb + 22, thumb, 3), np.uint8)
        y0, x0 = (thumb - im.shape[0]) // 2, (thumb - im.shape[1]) // 2  # thumbnail centred in tile
        tile[22 + y0:22 + y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.putText(tile, f"#{i + 1}{'*' if r.get('top') else ''} {r['score']:.2f}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    cols = min(4, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(out, grid)


def side_by_side(rows, out, n=None, zoom=2, cols=3):
    """Single image with the top n images side by side (default: top group, at least 3), cropped
    around the planet, zoomed in and with the same stretch (same disk brightness)."""
    if n is None:
        n = top_count(rows)
        if n < 2:
            n = min(3, len(rows))
    sel = rows[: min(n, 8)]
    data = []
    for r in sel:
        lum, rgb = load_gray(r["path"])
        im = rgb if rgb is not None else np.dstack([lum] * 3)
        im = np.clip(im - np.percentile(lum, 5), 0, None)
        m = planet_mask(np.clip(lum - np.percentile(lum, 5), 0, None)).astype(bool)
        ys, xs = np.where(m)
        data.append((im, m, (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2,
                     ys.max() - ys.min(), xs.max() - xs.min()))
    hh = int(max(d[4] for d in data) / 2) + 10
    hw = int(max(d[5] for d in data) / 2) + 10
    tiles = []
    for i, (r, (im, m, cy, cx, _, _)) in enumerate(zip(sel, data)):
        im = np.clip(im / max(float(np.percentile(im[m], 99.5)), 1e-9), 0, 1)
        H, W = im.shape[:2]
        y0, x0 = int(round(cy)) - hh, int(round(cx)) - hw
        t = np.zeros((2 * hh, 2 * hw, 3))
        a0, a1, b0, b1 = max(y0, 0), min(y0 + 2 * hh, H), max(x0, 0), min(x0 + 2 * hw, W)
        t[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = im[a0:a1, b0:b1]
        t = cv2.resize(t, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        t = (np.clip(t, 0, 1)[..., ::-1] * 255).astype(np.uint8)
        band = np.zeros((22, t.shape[1], 3), np.uint8)
        head = f"#{i + 1}{'*' if r.get('top') else ''} z={r['z']:.1f} "
        stem = os.path.splitext(os.path.basename(r["path"]))[0]
        room = max(8, t.shape[1] // 7 - len(head))  # show the tail of the name, where differences are
        label = head + (stem if len(stem) <= room else "~" + stem[-(room - 1):])
        cv2.putText(band, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(np.vstack([band, t]))
    cols = max(1, min(cols, len(tiles)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    return len(sel)


def info_text(mode):
    """Quick guide: which parameters to look at, with thresholds derived from Saturn test data."""
    t = {k: int(round(v * 100)) for k, v in TOL.items()}
    if mode == "raws":
        return f"""## Raws: what to look at
## Colours
- Green = candidate (top group). Red = objective flaw (serious warning: corrupted file or different scale). Images at the bottom are not flagged: quality drops continuously with no natural cut-off.
## Read in this order
1. Top group (green rows): these are the candidates. Within {t['detail']}% on detail/limb and {t['hf']}% on hf they are tied.
2. limb and detail: edge and disk sharpness. These are the reference metrics.
3. hf: more sensitive but less stable. Use it to break ties within the same run only.
## When in doubt
- Among tied images, pick the stack with more frames (higher snr): less noise, holds up better under processing.
- Keep at least 500-1000 frames in the stack.
- If two stacks differ by less than {t['detail']}%, process both and decide on the processed image.
## Informational only — do not affect the score
- snr and noise: in raws these reward frame count, not sharpness (not in the score).
- z: around -10 is normal for raws, which are not sharpened. Use it to compare sessions, not to pick a winner.
- rgb: below ~0.5 px is fine. Above {RGB_WARN:g} px check the RGB alignment.
## Watch out
- Compare only stacks at the same scale and with the same AutoStakkert settings (AP, quality, RGB align).
- Do not mix stacks from different runs: the offset and hf change between runs.
## Parameters in detail
- detail  Fine-detail energy in the disk (DoG filter, difference of Gaussians at 1 and 2 px),
          normalised on the median disk brightness.
          In raws: measures how much fine structure the stack produced (seeing + frame count).
          In processed: always increases with wavelets and deconvolution; compare only between
          versions of the same stack processed with identical settings.

- limb    Steepness of the planet edge: mean of the top quartile of the Sobel gradient along
          the limb band, normalised. Measures profile sharpness: turbulence, poor focus or a
          badly selected stack lower limb.
          In raws: the primary reference metric, stable and not very sensitive to noise.
          In processed: increases with processing but less than detail; the most reliable of the three.

- hf      Fraction of spectral power above 0.25× Nyquist on the planet crop (Hann window).
          Measures how much resolution is present at the finest scales.
          In raws: sensitive to instantaneous seeing; can vary a lot between runs even with the
          same instrument. Use it to break ties within the same run, not across different nights.
          In processed: increases with processing more than the other two; the risk is confusing
          amplified noise with real resolution. Always check that snr does not drop.

- snr     detail / noise (signal-to-noise ratio of the detail).
          In raws: mainly increases with the number of stacked frames, not with sharpness.
          Not in the score: use it only as a tiebreaker among equal images to pick the most
          robust stack for processing.
          In processed: drops if deconvolution amplifies noise. A drop beyond {int((1 - SNR_LOW) * 100)}%
          below the group median triggers the "amplified noise" warning.

- noise   Median absolute deviation (MAD) of the DoG filter in the far sky, relative to
          brightness. Estimates the background noise of the frame. Informational only.

- halo    Mean offset between the near-limb sky and the far sky, relative to brightness.
          Measures ringing from excessive wavelets or deconvolution: a bright or dark ring
          around the planet raises halo.
          In raws: almost always negligible; a high value indicates a corrupted file.
          In processed: the primary control metric. Above {HALO_WARN:g} is suspicious (ringing);
          above {HALO_OUT:g} the file is likely out of scale (e.g. double white balance at 120%).

- rgb     Residual misalignment between R, G, B channels in pixels (max of |R-G| and |B-G|,
          estimated with sub-pixel ECC on the planet crop). Informational only, not in the score.
          Below 0.5 px is negligible; above {RGB_WARN:g} px it is worth checking RGB alignment
          in AutoStakkert or AstroSurface.

- score   Mean of the relative ranks of the metrics that determine the score (those in UPPERCASE
          in the column header), minus a penalty for high halo. Depends on the set of images
          analysed: meaningful only for comparing images processed in the same session.

- z       Robust z-score against a fixed reference calibrated on a specific setup
          (Saturn 2026-10-03, Barlow 1×6, 21 processed images). Unlike score, it does not
          depend on the set: z=0 is a typical image from that reference session.
          Useful for comparing different nights or keeping a log over time, but systematically
          shifted if your setup (focal length, camera, typical seeing) differs from the reference.
          To recalibrate on your own setup, see REF in rank_planetary.py.
"""
    return f"""## Processed: what to look at
## Colours
- Green = candidate (top group). Red = discard: high halo (over-processed), amplified noise or serious warning. Being less sharp is NOT a reason to discard.
## Key point
Stronger wavelets and deconvolution always raise detail, limb and hf. A higher score does NOT mean a better image: the script cannot judge processing strength — it is here to keep it under control.
## Between two versions of the same stack
- If detail/limb differ by less than {t['detail']}% they are tied: pick the less processed one.
- Stop at the step before detail gains less than 2-3% but snr drops by more than 10%.
- If halo rises above {HALO_WARN:g} or snr falls below {int(SNR_LOW * 100)}% of the median, you have pushed too far.
## Control metrics
- halo: 0.0001-0.002 in clean processed images. Above {HALO_WARN:g} is suspicious (ringing). Above {HALO_OUT:g} (e.g. 8) the file is out of scale (e.g. a second white balance applied at 120%).
- snr: a well-dosed deconvolution is expected to drop snr by 20-30% (from Wiener 0.5 tests). More than that means too much noise amplification.
- rgb: after alignment should stay below ~0.5 px.
- White balance: does not affect sharpness; differences between wb1 and wb2 are just noise.
## Between different stacks
- Process with identical settings, then compare score and z.
## Limitation
- The script does not see artefacts inside the disk (fake bands, double edges on Cassini). Check by eye at 100-200%, using the Side-by-side comparison.
## Parameters in detail
- detail  Fine-detail energy in the disk (DoG filter, difference of Gaussians at 1 and 2 px),
          normalised on the median disk brightness.
          In raws: measures how much fine structure the stack produced (seeing + frame count).
          In processed: always increases with wavelets and deconvolution; compare only between
          versions of the same stack processed with identical settings.

- limb    Steepness of the planet edge: mean of the top quartile of the Sobel gradient along
          the limb band, normalised. Measures profile sharpness: turbulence, poor focus or a
          badly selected stack lower limb.
          In raws: the primary reference metric, stable and not very sensitive to noise.
          In processed: increases with processing but less than detail; the most reliable of the three.

- hf      Fraction of spectral power above 0.25× Nyquist on the planet crop (Hann window).
          Measures how much resolution is present at the finest scales.
          In raws: sensitive to instantaneous seeing; can vary a lot between runs even with the
          same instrument. Use it to break ties within the same run, not across different nights.
          In processed: increases with processing more than the other two; the risk is confusing
          amplified noise with real resolution. Always check that snr does not drop.

- snr     detail / noise (signal-to-noise ratio of the detail).
          In raws: mainly increases with the number of stacked frames, not with sharpness.
          Not in the score: use it only as a tiebreaker among equal images to pick the most
          robust stack for processing.
          In processed: drops if deconvolution amplifies noise. A drop beyond {int((1 - SNR_LOW) * 100)}%
          below the group median triggers the "amplified noise" warning.

- noise   Median absolute deviation (MAD) of the DoG filter in the far sky, relative to
          brightness. Estimates the background noise of the frame. Informational only.

- halo    Mean offset between the near-limb sky and the far sky, relative to brightness.
          Measures ringing from excessive wavelets or deconvolution: a bright or dark ring
          around the planet raises halo.
          In raws: almost always negligible; a high value indicates a corrupted file.
          In processed: the primary control metric. Above {HALO_WARN:g} is suspicious (ringing);
          above {HALO_OUT:g} the file is likely out of scale (e.g. double white balance at 120%).

- rgb     Residual misalignment between R, G, B channels in pixels (max of |R-G| and |B-G|,
          estimated with sub-pixel ECC on the planet crop). Informational only, not in the score.
          Below 0.5 px is negligible; above {RGB_WARN:g} px it is worth checking RGB alignment
          in AutoStakkert or AstroSurface.

- score   Mean of the relative ranks of the metrics that determine the score (those in UPPERCASE
          in the column header), minus a penalty for high halo. Depends on the set of images
          analysed: meaningful only for comparing images processed in the same session.

- z       Robust z-score against a fixed reference calibrated on a specific setup
          (Saturn 2026-10-03, Barlow 1×6, 21 processed images). Unlike score, it does not
          depend on the set: z=0 is a typical image from that reference session.
          Useful for comparing different nights or keeping a log over time, but systematically
          shifted if your setup (focal length, camera, typical seeing) differs from the reference.
          To recalibrate on your own setup, see REF in rank_planetary.py.
"""


# ---------- warnings and verdict ----------
HALO_WARN = 0.005     # processed: halo above this threshold = suspicious (ringing)
HALO_OUT = 1.0        # halo above this threshold = out-of-scale file (all modes)
RGB_WARN = 1.0        # channel misalignment in px
SNR_LOW = 0.6         # processed: snr below this fraction of the median = amplified noise
SIZE_TOL = 0.20       # relative planet size difference = different scale


def short_name(name, n=46):
    stem = os.path.splitext(name.replace("\\", "/").split("/")[-1])[0]
    return stem if len(stem) <= n else "~" + stem[-(n - 1):]


def warnings_for(rows, mode):
    """For each row a list of (severe, text). `mode`: "raws" or "processed"."""
    def med(key):
        v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        return float(np.nanmedian(v)) if np.any(~np.isnan(v)) else float("nan")
    m_size, m_snr = med("size"), med("snr")
    out = []
    for r in rows:
        w = []
        h = r["halo"]
        if not np.isnan(h) and h > HALO_OUT:
            w.append((True, f"halo out of scale ({h:.2f}): file likely corrupted"))
        elif mode == "processed" and not np.isnan(h) and h > HALO_WARN:
            w.append((False, f"high halo ({h:.4f}): possible ringing"))
        if not np.isnan(r["rgb"]) and r["rgb"] > RGB_WARN:
            w.append((False, f"RGB channels misaligned ({r['rgb']:.1f} px)"))
        if mode == "processed":
            if np.isnan(r["snr"]):
                w.append((False, "snr not computable (too little sky in crop)"))
            elif not np.isnan(m_snr) and r["snr"] < SNR_LOW * m_snr:
                w.append((False, f"very low snr ({r['snr']:.0f} vs median {m_snr:.0f}): amplified noise"))
        sz = r.get("size", np.nan)
        if not np.isnan(sz) and not np.isnan(m_size) and abs(sz - m_size) / m_size > SIZE_TOL:
            w.append((True, f"different scale (planet {sz:.0f} px vs median {m_size:.0f}): not comparable"))
        out.append(w)
    return out


def discard_flags(rows, mode):
    """For each row the reason to discard it (objective flaw), or "" if it should be kept.
    Raws: serious warnings only (corrupted file, different scale). Processed: serious warning,
    high halo or amplified noise. Being at the bottom of the ranking is NOT a reason:
    quality drops continuously with no natural cut-off."""
    warns = warnings_for(rows, mode)
    m_snr = float(np.nanmedian([r["snr"] for r in rows])) if rows else float("nan")
    out = []
    for r, w in zip(rows, warns):
        why = ""
        grave = [t for g, t in w if g]
        if grave:
            why = "serious warning: " + grave[0]
        elif mode == "processed":
            if not np.isnan(r["halo"]) and r["halo"] > HALO_WARN:
                why = f"over-processed (halo {r['halo']:.4f})"
            elif not np.isnan(r["snr"]) and not np.isnan(m_snr) and r["snr"] < SNR_LOW * m_snr:
                why = f"amplified noise (snr {r['snr']:.0f})"
        out.append(why)
    return out


def verdict(rows, mode):
    """Recommendation in one or two sentences, applying the guide rules (top group, tiebreaker ...)."""
    if not rows:
        return ""
    warns = warnings_for(rows, mode)
    group = [i for i, r in enumerate(rows) if r.get("top")]
    ok = [i for i in group if not any(g for g, _ in warns[i])]
    if not ok:
        return "No reliable candidate in the top group: check the warnings."
    if mode == "raws":
        pick = max(ok, key=lambda i: -1 if np.isnan(rows[i]["snr"]) else rows[i]["snr"])  # more frames = higher snr
        rule = "among tied images I pick the one with more frames (highest snr)"
    else:
        pick = max(ok, key=lambda i: -1 if np.isnan(rows[i]["snr"]) else rows[i]["snr"])  # less processed = higher snr
        rule = "among tied images I pick the least processed (highest snr)"
    name = short_name(rows[pick]["file"])
    if len(ok) == 1:
        text = f"Recommendation: {name}. It stands out from the others on the score metrics."
    else:
        text = f"Recommendation: {name}. {len(ok)} images are tied: {rule}."
        if pick != 0:
            text += " It is not first in the ranking only due to negligible differences."
    if len(ok) < len(group):
        text += f" Excluded due to serious warning: {len(group) - len(ok)}."
    if mode == "processed":
        text += " Check by eye for artefacts inside the disk: the script cannot see them."
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--csv")
    ap.add_argument("--sheet", help="save a sorted contact sheet (PNG)")
    ap.add_argument("--top", type=int, help="show only the top N in the contact sheet")
    ap.add_argument("--copy", metavar="N|top", help="copy the top N images (or the top group: 'top') to --copy-dir")
    ap.add_argument("--copy-dir", default=BEST_DIR, help=f"destination folder for --copy (default: ./{BEST_DIR})")
    ap.add_argument("--compare", nargs=2, metavar=("RAWS.csv", "PROCESSED.csv"),
                    help="before/after comparison between two CSVs (processed filename must contain the raw filename)")
    ap.add_argument("--side", metavar="OUT.png", help="image with the top group side by side (zoomed crops, same stretch)")
    ap.add_argument("--side-n", type=int, help="how many images to place side by side (default: top group, at least 3; max 8)")
    ap.add_argument("--mode", choices=("raws", "processed"),
                    help="warning and discard rules (default: processed with --with-snr, otherwise raws); "
                         "useful to use --with-snr on raws: --with-snr --mode raws")
    ap.add_argument("--with-snr", action="store_true", help="include snr in the score (for processed images); default: sharpness only")
    a = ap.parse_args()

    if a.compare:
        print(*compare_csvs(a.compare[0], a.compare[1], not a.with_snr), sep="\n")
        return
    if not a.inputs:
        ap.error("specify at least one folder or file (or use --compare)")

    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d != BEST_DIR]
                files += [os.path.join(root, n) for n in names if n.lower().endswith(EXTS)]
        else:
            files += glob.glob(p, recursive=True)
    files = sorted({f for f in files if os.path.isfile(f) and f.lower().endswith(EXTS)
                    and not os.path.basename(f).startswith(".")})
    if not files:
        sys.exit("no files found")

    rows = []
    for f in files:
        try:
            rows.append(analyze(f))
        except Exception as e:
            print(f"! {os.path.basename(f)}: {e}", file=sys.stderr)
    mode = a.mode or ("processed" if a.with_snr else "raws")
    rows = rank_scores(rows, not a.with_snr, mode)

    print(f"{'#':>2} {'score':>5} {'z':>6} {'detail':>7} {'limb':>6} {'hf':>8} {'snr':>6} {'noise':>7} {'halo':>6} {'rgb':>5}  file")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2} {r['score']:5.2f} {r['z']:6.1f} {r['detail']:7.4f} {r['limb']:6.3f} {r['hf']:8.3g} "
              f"{r['snr']:6.1f} {r['noise']:7.4f} {r['halo']:6.4f} {r['rgb']:5.2f} {'*' if r['top'] else ('x' if r['discard'] else ' ')}{r['file']}")
    print(f"\nTop group (*): {top_count(rows)} images indistinguishable from the best on "
          + ("detail, limb, hf" if not a.with_snr else "detail, limb, hf, snr") + ".")
    print(verdict(rows, mode))
    nd = sum(1 for r in rows if r["discard"])
    if nd:
        print(f"To discard (x): {nd} out of {len(rows)}.")
    for i, w in enumerate(warnings_for(rows, mode), 1):
        for _, txt in w:
            print(f"! #{i} {txt}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "path"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if a.sheet:
        contact_sheet(rows, a.sheet, a.top)
    if a.side:
        k = side_by_side(rows, a.side, a.side_n)
        print(f"Side-by-side: {k} images saved to {a.side}")
    if a.copy:
        n = 0 if a.copy.lower() == "top" else int(a.copy)
        done = copy_best(rows, n, a.copy_dir)
        print(f"Copied {len(done)} images to {a.copy_dir}")


if __name__ == "__main__":
    main()

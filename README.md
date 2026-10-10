# Rank Planetary

Ranks planetary images (AutoStakkert stacks, AstroSurface outputs, …) with objective sharpness metrics, so you can discard the weaker ones quickly and judge only the best candidates by eye.

Developed and calibrated on Saturn, but it works on any image with a planet on a dark background.

<!-- Add a screenshot of the GUI here, e.g. docs/screenshot.png -->

## Features

- Reads TIFF (including 16-bit), PNG and JPG, from a folder or its subfolders.
- Finds the planet (rings included) and separates it from the sky.
- Measures sharpness, noise, halo and colour-channel alignment, independently of brightness and stretch.
- Produces a ranking with a relative `score`, an absolute `z`, and a **top group**: the images the script cannot tell apart from the best one.
- Gives a one-sentence **recommendation** and **warnings** for objective problems (out-of-scale file, over-sharpening, different scale…).
- Exports CSV and a contact sheet, builds a zoomed **side-by-side** of the best images, copies the best ones to a folder, and compares a raws run with a processed run (**before/after**).
- Desktop GUI (two tabs: *Raws* and *Processed*) and command line.

## Installation

Python 3 is required.

```bash
pip install numpy opencv-python tifffile
```

The GUI also needs `tkinter` (bundled with the Python installers for Windows and macOS; on Linux: `sudo apt install python3-tk`).

Keep both files in the same folder:

```
rank_planetary.py        # analysis engine and command line
rank_planetary_gui.py    # graphical interface
```

## Usage

### GUI

```bash
python3 rank_planetary_gui.py        # on Windows: py -3 rank_planetary_gui.py
```

Two tabs, each with its own folder and results:

| Tab | For | Score |
|---|---|---|
| **Raws** | AutoStakkert stacks | sharpness only (`detail`, `limb`, `hf`) |
| **Processed** | AstroSurface / edited images | sharpness + `snr` |

1. **Choose folder…** finds the images right away (tick *Include subfolders* to search deeper).
2. **Analyse** ranks them. Green rows are the top group; red rows have an objective defect.
3. Above the table you get the **recommendation** and any **warnings**. Columns are ordered by importance, left to right; the metrics that decide the score are in UPPERCASE.
4. **ℹ Info** opens a short guide on which parameters to look at in the current tab.
5. **Copy best** copies the first *N* images (0 = the top group) to a `best` folder inside the chosen folder.
6. **Side-by-side** builds one image with the best images zoomed, centred and with the same stretch.
7. **Before/after comparison…** (Processed tab) compares the raws CSV with the processed CSV.

Outputs are saved in the chosen folder: `ranking_planetary_raws.csv` / `ranking_planetary_processed.csv`, `ranking_planetary_sheet_<tab>.png`, `side_by_side_<tab>.png`. Double-click a row to open the image.

### Command line

```bash
python3 rank_planetary.py folder [more folders/files...] [options]
```

| Option | Effect |
|---|---|
| `--csv out.csv` | save the full table |
| `--sheet sheet.png` | sorted contact sheet (`--top N` to show only the first N) |
| `--copy N` / `--copy top` | copy the first N images (or the top group) to `--copy-dir` (default `./best`) |
| `--side out.png` | side-by-side of the best images (`--side-n N` to choose how many; default: the top group, or the first 3 if the top group is a single image; max 8) |
| `--with-snr` | include `snr` in the score (use for processed images); default is sharpness only |
| `--mode raws\|processed` | rules for warnings and discards (default: `processed` with `--with-snr`, otherwise `raws`) |
| `--compare raws.csv processed.csv` | before/after comparison between two CSVs |

Examples:

```bash
# screening of raw stacks: rank, save CSV, copy the top group
python3 rank_planetary.py ~/Saturn/stacks --csv raws.csv --copy top

# rank processed images, including snr
python3 rank_planetary.py ~/Saturn/processed --with-snr --csv processed.csv

# ** patterns work (use quotes)
python3 rank_planetary.py "~/Saturn/**/*.tif"
```

## How it works

For every image:

1. **Luminance and sky level.** Luminance is computed and the 5th percentile subtracted.
2. **Planet mask.** Gaussian blur (σ = 2 px), Otsu threshold lowered by half to include the faint limb, largest connected component, holes filled (the Cassini division does not split the mask).
3. **Measurement regions**, derived from the mask:
   - *interior*: the mask eroded by 4 px, away from the edge;
   - *limb band*: ±4 px around the edge;
   - *near sky*: 4–10 px outside the edge;
   - *far sky*: more than 25 px from the planet.
4. **Normalisation.** The image is divided by the median interior brightness, so metrics do not depend on gain, exposure or stretch.
5. **Metrics** (table below).
6. **Scores** to compare the images with each other.

## Metrics

| Metric | What it measures | How it is computed | Better |
|---|---|---|---|
| `detail` | fine detail inside the disk | RMS of a difference of Gaussians (σ 1 px − σ 2 px) over the interior of the normalised image. Partly ignores pixel-level noise. | higher |
| `limb` | sharpness of the planet edge | mean of the steepest quarter of the gradient magnitude (Sobel after a σ 0.8 px blur) in the limb band. The most direct proxy for seeing and stack quality. | higher |
| `hf` | effective resolution | fraction of the spectral power above 0.25 of the Nyquist frequency, on a crop around the planet with a Hann window. | higher |
| `noise` | sky noise | MAD × 1.4826 of the difference of Gaussians in the far sky. Not used directly in the score. | lower |
| `snr` | detail relative to noise | `detail / noise`. On raws it mostly reflects how many frames were stacked; on processed images it shows whether wavelets amplified noise. | higher |
| `halo` | ring around the planet (deconvolution ringing) | difference between the mean near-sky and far-sky brightness. | lower |
| `rgb` | residual misalignment of the colour channels | sub-pixel translation (ECC) of R and B relative to G on the planet crop; the larger of R−G and B−G, in pixels. Informational, not part of the score. | lower |
| `size` | planet size | square root of the planet area (rotation-independent). Only used to detect scale mismatches. | – |

If the crop is too tight and no far sky is left, `noise`, `snr` and `halo` cannot be computed: the metric is excluded for that image instead of penalising it.

## Reading the ranking

### `score` — relative score

For each metric, images get a rank from 0 (worst) to 1 (best). The score is the mean rank of `detail`, `limb`, `hf` (plus `snr` with `--with-snr` / in the Processed tab), minus a small penalty for high `halo`.

Small differences are **ties**: two images that differ by less than a tolerance on a metric get the average rank. Tolerances (`TOL` in the code) come from measured variability — cropping the same image in different ways and re-stacking:

| Metric | `detail` | `limb` | `hf` | `snr` | `halo` |
|---|---|---|---|---|---|
| Tolerance | 2% | 2% | 3% | 10% | 25% |

The score **depends on the set of images analysed**: adding one changes the others' scores. To compare different sessions, use `z`.

### `z` — absolute score

Mean of the robust z-scores of `detail`, `limb`, `hf`, `snr` (log scale) against a fixed reference (`REF` in the code), minus a penalty for halo. Because it is computed against constants it does not change when images are added or removed, and stays comparable across runs. `z = 0` is a typical image of the calibration set; higher is better.

The reference was calibrated on 21 processed Saturn images (Barlow, same scale). **For a different setup (focal length, camera) recompute it**: median and robust scale (1.4826 × MAD) of `log10(metric)` on your own images.

### Top group

Images that are not worse than the first by more than the tolerance `TOL` on any score metric. It tells you how many images are really indistinguishable from the best and need an eye check. In the GUI they are the green rows with `★`; in the CSV, the `top` column.

### Recommendation

- If one image stands out, it is recommended.
- If several are tied: on **raws** it picks the one with the most frames (highest `snr`); on **processed** images it picks the least pushed one (highest `snr`).
- Candidates with a serious warning are excluded.

### Warnings and red rows

| Warning | Rule | Serious |
|---|---|---|
| out-of-scale halo | `halo` > 1.0 | yes |
| different scale | planet size differs > 20% from the median | yes |
| high halo (processed) | `halo` > 0.005 | no, but the row is marked red |
| amplified noise (processed) | `snr` < 60% of the median | no, but the row is marked red |
| misaligned channels | `rgb` > 1 px | no |

Red rows (`✖`) mean an **objective defect**. Last places in the ranking are deliberately not marked: quality usually decreases continuously, so there is no natural cut-off.

The thresholds are constants at the top of the "warnings and verdict" section of `rank_planetary.py`.

### Before/after comparison

`--compare` (or the button in the Processed tab) takes the raws CSV and the processed CSV, matches the rows, and shows the position of every image in both rankings plus the rank correlation (Spearman). It answers: *does the choice I made on raws still hold after processing?*

Matching is by name: the processed filename must **contain** the raw filename (e.g. `X.tif` → `X_wavelets.tif`). With several variants of the same raw, the one with the highest `z` is kept.

## Suggested workflow

1. Stack at several frame percentages (e.g. 15, 25, 35, 50 %).
2. **Raws tab:** analyse everything, look at `limb`, `detail`, `hf`, keep the top group.
3. Process only those images, all with the same parameters.
4. **Processed tab:** analyse with `snr`; `snr` and `halo` warn you if wavelets were pushed too far.
5. Choose by eye among the first ones, helped by the side-by-side.

## Limitations

- **Compare only images at the same scale** (same focal length and binning). A different crop is fine: `detail`, `limb` and `hf` measure the planet and do not change; `noise`, `snr` and `halo` use the sky and may vary a little (about 5% and 25%).
- **Differences below about 2% are not significant.** Even re-stacking the same data moves the metrics slightly; a batch stack and a single stack can differ by about 1% on `limb`. Do not mix them in one comparison. `hf` is more sensitive but less stable between runs: compare it only within the same run.
- **It measures sharpness, not fidelity.** Stronger wavelets or deconvolution almost always raise `detail`, `limb` and `hf`; `halo` and `snr` limit the excess, but artefacts inside the disk (false bands, double edges on the Cassini division) are not detected. The last word belongs to your eyes.
- **Extra processing steps change the numbers.** For example, an additional Camera Raw or Photoshop pass raises the sharpness metrics, so compare only images that went through the same steps.
- **Do not mix raws and processed images in the same ranking.** Use the two tabs, and `z` to compare across them.
- **`snr` means different things.** On raws it rewards stacks with more frames, which is why the Raws score does not use it.
- JPEG works, but TIFF (16-bit) is preferable: compression and 8-bit quantisation add noise to the measurements.

## CSV columns

`file, detail, limb, hf, noise, snr, halo, rgb, size, score, z, top, discard`

## Project files

| File | Content |
|---|---|
| `rank_planetary.py` | analysis, scores, before/after comparison, copy, command line |
| `rank_planetary_gui.py` | two-tab graphical interface |
| `ranking_planetary_*.csv` | full table (written by the GUI) |
| `ranking_planetary_sheet_*.png` | sorted contact sheet |
| `side_by_side_*.png` | best images side by side, zoomed |
| `best/` | images copied by **Copy best** / `--copy` (ignored in searches) |

## License

Add the license of your choice here (e.g. MIT).

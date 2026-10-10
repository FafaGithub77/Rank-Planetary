#!/usr/bin/env python3
"""
rank_planetary_gui.py - graphical interface for rank_planetary.py

Must be placed in the same folder as rank_planetary.py.
Usage:  python3 rank_planetary_gui.py

Two tabs, each with its own folder and results:
  Raws      - AutoStakkert stacks. Sharpness-only score (detail, limb, hf). Copy top images.
  Processed - AstroSurface images. Sharpness + snr score (penalises noise and ringing from
              excessive wavelets). Before/after comparison with the raws CSV.
In each tab: "Choose folder..." finds images (TIF/PNG/JPG) immediately ("Include subfolders"
to search recursively), "Analyse" ranks them and saves a CSV and contact sheet in the chosen
folder (named after the tab: ranking_planetary_raws.csv / _processed.csv).
Double-click a result row to open the image.

Dependencies: those of rank_planetary.py (numpy, opencv-python, tifffile) + tkinter.
"""
import csv
import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import rank_planetary as rp

_NEEDED = ("BEST_DIR", "top_count", "copy_best", "side_by_side", "compare_csvs", "info_text", "verdict", "warnings_for")
if not all(hasattr(rp, n) for n in _NEEDED):
    raise SystemExit("rank_planetary.py is an old version: replace it with the updated one "
                     "(it must be in the same folder as this GUI) and restart.")

MODES = ("raws", "processed")
CSV_NAME = "ranking_planetary_{mode}.csv"
COMPARE_NAME = "side_by_side_{mode}.png"
SHEET_NAME = "ranking_planetary_sheet_{mode}.png"
GENERATED = {t.format(mode=m) for t in (SHEET_NAME, COMPARE_NAME) for m in MODES} | {
    "ranking_planetary_sheet.png", "confronto_affiancato.png", "side_by_side.png"}

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#1e1f26"   # main background
SURFACE   = "#2a2d38"   # card / frame
SURFACE2  = "#32364a"   # hover / alternate
BORDER    = "#3c4060"   # thin borders
ACCENT    = "#5b8af5"   # primary blue
ACCENT2   = "#7c5cd8"   # secondary violet
SUCCESS   = "#3ecf8e"   # success green
WARNING   = "#f5a623"   # warning orange
ERROR     = "#e05c5c"   # error red
FG        = "#e8eaf0"   # main text
FG_MUTED  = "#8891aa"   # muted text
TOP_ROW   = "#1e3a2a"   # top-group row background
TOP_FG    = "#3ecf8e"   # top-group row text
DISCARD_ROW = "#3b1f25" # discard row background
# ──────────────────────────────────────────────────────────────────────────────

FONT_FAMILY = "SF Pro Display" if platform.system() == "Darwin" else "Segoe UI"
FONT        = (FONT_FAMILY, 11)
FONT_SMALL  = (FONT_FAMILY, 10)
FONT_BOLD   = (FONT_FAMILY, 11, "bold")
FONT_MONO   = ("SF Mono" if platform.system() == "Darwin" else "Consolas", 10)


def _apply_theme(root: tk.Tk):
    """Configure ttk.Style with modern colours and fonts."""
    style = ttk.Style(root)
    style.theme_use("clam")

    # Global
    style.configure(".",
                    background=BG, foreground=FG, borderwidth=0,
                    font=FONT, relief="flat")

    # TFrame / TLabelFrame
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)
    style.configure("TLabelframe",
                    background=SURFACE, foreground=ACCENT,
                    borderwidth=1, relief="groove",
                    font=(FONT_FAMILY, 10, "bold"))
    style.configure("TLabelframe.Label",
                    background=SURFACE, foreground=ACCENT,
                    font=(FONT_FAMILY, 10, "bold"))

    # TNotebook
    style.configure("TNotebook",
                    background=BG, borderwidth=0, tabmargins=[2, 4, 0, 0])
    style.configure("TNotebook.Tab",
                    background=SURFACE, foreground=FG_MUTED,
                    padding=[18, 8], font=FONT_BOLD, borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT), ("active", SURFACE2)],
              foreground=[("selected", "#ffffff"), ("active", FG)])

    # TButton
    style.configure("TButton",
                    background=ACCENT, foreground="#ffffff",
                    padding=[14, 7], relief="flat",
                    font=FONT_BOLD, borderwidth=0)
    style.map("TButton",
              background=[("disabled", SURFACE2), ("active", "#7aa3f7"),
                          ("pressed", "#3d6be0")],
              foreground=[("disabled", FG_MUTED)])

    # Outline.TButton (secondary buttons)
    style.configure("Outline.TButton",
                    background=SURFACE, foreground=ACCENT,
                    padding=[12, 6], relief="flat",
                    font=FONT, borderwidth=1)
    style.map("Outline.TButton",
              background=[("disabled", SURFACE), ("active", SURFACE2)],
              foreground=[("disabled", FG_MUTED)])

    # TLabel
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Muted.TLabel", background=BG, foreground=FG_MUTED, font=FONT_SMALL)
    style.configure("Path.TLabel",  background=BG, foreground=FG, font=FONT_SMALL)
    style.configure("Score.TLabel", background=SURFACE, foreground=FG_MUTED, font=FONT_SMALL)

    # TCheckbutton
    style.configure("TCheckbutton",
                    background=BG, foreground=FG,
                    indicatorcolor=SURFACE2, indicatorbackground=SURFACE2,
                    font=FONT_SMALL)
    style.map("TCheckbutton",
              indicatorcolor=[("selected", ACCENT), ("!selected", SURFACE2)],
              background=[("active", BG)])

    # TSpinbox
    style.configure("TSpinbox",
                    background=SURFACE2, foreground=FG,
                    arrowcolor=ACCENT, borderwidth=1,
                    fieldbackground=SURFACE2, font=FONT_SMALL)

    # Progressbar
    style.configure("TProgressbar",
                    background=ACCENT, troughcolor=SURFACE2,
                    borderwidth=0, thickness=6)

    # Scrollbar
    style.configure("TScrollbar",
                    background=SURFACE2, troughcolor=BG,
                    arrowcolor=FG_MUTED, borderwidth=0, relief="flat")
    style.map("TScrollbar",
              background=[("active", BORDER), ("!active", SURFACE2)])

    # Treeview
    style.configure("Treeview",
                    background=SURFACE, foreground=FG,
                    fieldbackground=SURFACE, borderwidth=0,
                    font=FONT_SMALL, rowheight=24)
    style.configure("Treeview.Heading",
                    background=SURFACE2, foreground=FG_MUTED,
                    relief="flat", borderwidth=0,
                    font=(FONT_FAMILY, 10, "bold"))
    style.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.map("Treeview.Heading",
              background=[("active", BORDER)])


def images_in(folder, recursive=False):
    """Images in the folder (and subfolders if requested)."""
    out = []
    for cur, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if d != rp.BEST_DIR]
        out += [os.path.join(cur, n) for n in names
                if n.lower().endswith(rp.EXTS) and not n.startswith(".") and n not in GENERATED]
        if not recursive:
            break
    return sorted(out)


def open_path(path):
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)  # noqa
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Error", f"Cannot open {path}\n{e}")


# ── light horizontal separator ────────────────────────────────────────────────
def _sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=4)


class Panel(ttk.Frame):
    # Column order by importance (left to right), different per tab.
    # Those in UPPERCASE are the metrics that determine the score; others are informational.
    COL_ORDER = {
        "raws":      ("n", "score", "limb", "detail", "hf", "rgb", "snr", "z", "halo", "noise", "file"),
        "processed": ("n", "score", "detail", "limb", "hf", "snr", "halo", "rgb", "z", "noise", "file"),
    }
    COL_KEY = {"raws": ("limb", "detail", "hf"), "processed": ("detail", "limb", "hf", "snr")}
    COL_WIDTH = {"n": 45, "score": 60, "z": 45, "detail": 70, "limb": 60, "hf": 90, "snr": 55,
                 "noise": 80, "halo": 60, "rgb": 50, "file": 360}
    LEGEND = {
        "raws":      "Score metrics:  LIMB · DETAIL · HF        Informational:  rgb · snr · z · halo · noise",
        "processed": "Score metrics:  DETAIL · LIMB · HF · SNR        Control:  halo · rgb · z · noise",
    }

    def __init__(self, master, mode):
        super().__init__(master)
        self.configure(style="TFrame")
        self.mode = mode
        self.root_dir = None
        self.files = []
        self.result_paths = {}
        self.sheet_path = None
        self.rows = []
        self.q = queue.Queue()
        self.running = False
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        px, py = 12, 6

        # ── Section 1: folder selection ────────────────────────────────────
        row1 = ttk.Frame(self)
        row1.pack(fill="x", padx=px, pady=(14, 2))

        ttk.Button(row1, text="📂  Choose folder…", command=self.choose_root,
                   style="TButton").pack(side="left")
        self.root_lbl = ttk.Label(row1, text="(no folder selected)",
                                  style="Muted.TLabel")
        self.root_lbl.pack(side="left", padx=12)
        self.recursive = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Include subfolders",
                        variable=self.recursive, command=self.scan).pack(side="right")

        _sep(self)

        # ── Section 2: options + run ────────────────────────────────────────
        row2 = ttk.Frame(self)
        row2.pack(fill="x", padx=px, pady=py)

        self.save_csv   = tk.BooleanVar(value=True)
        self.save_sheet = tk.BooleanVar(value=True)
        self.top_n      = tk.IntVar(value=0)
        self.sharp      = tk.BooleanVar(value=(self.mode == "raws"))

        ttk.Checkbutton(row2, text="Save CSV",
                        variable=self.save_csv).pack(side="left")
        ttk.Checkbutton(row2, text="Save contact sheet",
                        variable=self.save_sheet).pack(side="left", padx=14)

        score_mode = "sharpness only" if self.mode == "raws" else "sharpness + SNR"
        ttk.Label(row2, text=f"Score: {score_mode}",
                  style="Score.TLabel",
                  foreground=ACCENT if self.mode == "raws" else ACCENT2
                  ).pack(side="left", padx=10)

        ttk.Label(row2, text="Thumbnails (0 = all):",
                  style="Muted.TLabel").pack(side="left", padx=(16, 2))
        ttk.Spinbox(row2, from_=0, to=500, textvariable=self.top_n,
                    width=5).pack(side="left")

        self.run_btn = ttk.Button(row2, text="▶  Analyse", command=self.run,
                                  style="TButton")
        self.run_btn.pack(side="right")
        self.sheet_btn = ttk.Button(row2, text="📋  Contact sheet",
                                    command=self.open_sheet, state="disabled",
                                    style="Outline.TButton")
        self.sheet_btn.pack(side="right", padx=8)

        _sep(self)

        # ── Section 3: progress bar + status ───────────────────────────────
        row3 = ttk.Frame(self)
        row3.pack(fill="x", padx=px, pady=(4, 2))

        self.status = ttk.Label(row3, text="", anchor="w", style="Muted.TLabel")
        self.status.pack(side="left", fill="x", expand=True)
        self.bar = ttk.Progressbar(row3, mode="determinate", length=220)
        self.bar.pack(side="right")

        # ── Section 4: actions ─────────────────────────────────────────────
        row4 = ttk.Frame(self)
        row4.pack(fill="x", padx=px, pady=(4, 2))
        ttk.Button(row4, text="ℹ  Info", command=self.info,
                   style="Outline.TButton").pack(side="right")

        self.copy_n  = tk.IntVar(value=0)
        self.copy_btn = ttk.Button(row4, text="📁  Copy best",
                                   command=self.copy_best, state="disabled",
                                   style="Outline.TButton")
        if self.mode == "raws":
            ttk.Label(row4, text=f"Copy best to '{rp.BEST_DIR}'  (0 = top group):",
                      style="Muted.TLabel").pack(side="left")
            ttk.Spinbox(row4, from_=0, to=500, textvariable=self.copy_n,
                        width=5).pack(side="left", padx=6)
            self.copy_btn.pack(side="left", padx=4)
        else:
            ttk.Button(row4, text="🔍  Before/after comparison…",
                       command=self.compare,
                       style="Outline.TButton").pack(side="right")

        self.side_btn = ttk.Button(row4, text="⬛  Side-by-side",
                                   command=self.side, state="disabled",
                                   style="Outline.TButton")
        self.side_btn.pack(side="right", padx=8)

        _sep(self)

        # ── Section 5: ranking ─────────────────────────────────────────────
        self.res_frame = ttk.LabelFrame(self, text="  Ranking")
        self.res_frame.pack(fill="both", expand=True, padx=px, pady=(4, 6))

        self.cols = self.COL_ORDER[self.mode]
        self.res = ttk.Treeview(self.res_frame, columns=self.cols, show="headings", height=13)
        for c in self.cols:
            head = "#" if c == "n" else (c.upper() if c in self.COL_KEY[self.mode] else c)
            self.res.heading(c, text=head)
            self.res.column(c, width=self.COL_WIDTH[c], anchor="w" if c == "file" else "e")
        sb = ttk.Scrollbar(self.res_frame, orient="vertical", command=self.res.yview)
        self.res.configure(yscrollcommand=sb.set)
        self.verdict_lbl = tk.Label(self.res_frame, text="", bg=BG, fg=SUCCESS, font=FONT_BOLD,
                                    anchor="w", justify="left", wraplength=900)
        self.verdict_lbl.pack(side="top", fill="x", padx=8, pady=(4, 0))
        self.warn_lbl = tk.Label(self.res_frame, text="", bg=BG, fg=WARNING, font=FONT_SMALL,
                                 anchor="w", justify="left", wraplength=900)
        self.warn_lbl.pack(side="top", fill="x", padx=8, pady=(0, 2))
        ttk.Label(self.res_frame, text=self.LEGEND[self.mode], style="Muted.TLabel").pack(
            side="top", fill="x", padx=6, pady=(2, 4))
        self.res.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        # tag "top": dark green background, light green text
        self.res.tag_configure("top", background=TOP_ROW, foreground=TOP_FG)
        self.res.tag_configure("discard", background=DISCARD_ROW, foreground=ERROR)
        self.res.bind("<Double-1>", self.open_result)

        # ── Section 6: error log ───────────────────────────────────────────
        self.log = tk.Text(self, height=3,
                           bg=SURFACE, fg=ERROR,
                           insertbackground=FG,
                           font=FONT_MONO,
                           relief="flat", bd=0,
                           state="disabled",
                           padx=8, pady=6)
        self.log.pack(fill="x", padx=px, pady=(0, 10))

    # ── folders ───────────────────────────────────────────────────────────────
    def choose_root(self):
        d = filedialog.askdirectory(title="Folder with images",
                                    initialdir=self.root_dir or "~")
        if d:
            self.root_dir = d
            # truncate path if too long
            display = d if len(d) <= 60 else "…" + d[-57:]
            self.root_lbl.config(text=display, foreground=FG,
                                 style="Path.TLabel")
            self.scan()

    def scan(self):
        if not self.root_dir:
            return self.choose_root()
        self.files = images_in(self.root_dir, self.recursive.get())
        if self.files:
            self.status.config(text=f"✔  {len(self.files)} images found",
                               foreground=SUCCESS)
        else:
            self.status.config(text="⚠  No images found",
                               foreground=WARNING)

    # ── analysis ──────────────────────────────────────────────────────────────
    def run(self):
        if self.running:
            return
        if not self.root_dir:
            return self.choose_root()
        files = images_in(self.root_dir, self.recursive.get())
        if not files:
            return messagebox.showinfo("Rank Planetary", "No images in the folder.")
        self.running = True
        self.run_btn.config(state="disabled")
        self.sheet_btn.config(state="disabled")
        self.res.delete(*self.res.get_children())
        self.result_paths.clear()
        self.sheet_path = None
        self.copy_btn.config(state="disabled")
        self.side_btn.config(state="disabled")
        self.res_frame.config(text="  Ranking")
        self.verdict_lbl.config(text="")
        self.warn_lbl.config(text="")
        self._log(clear=True)
        self.bar.config(maximum=len(files), value=0)
        opts = dict(csv=self.save_csv.get(), sheet=self.save_sheet.get(),
                    top=self.top_n.get() or None, sharp=self.sharp.get(), mode=self.mode)
        threading.Thread(target=self._worker, args=(files, opts), daemon=True).start()
        self.after(100, self._poll)

    def _worker(self, files, opts):
        rows = []
        try:
            for i, f in enumerate(files, 1):
                try:
                    r = rp.analyze(f)
                    r["file"] = os.path.relpath(f, self.root_dir)
                    rows.append(r)
                except Exception as e:
                    self.q.put(("err", f"{os.path.relpath(f, self.root_dir)}: {e}"))
                self.q.put(("prog", i, len(files)))
            if not rows:
                self.q.put(("fail", "No images analysed."))
                return
            rows = rp.rank_scores(rows, opts["sharp"], opts["mode"])
            out_csv = out_sheet = None
            if opts["csv"]:
                out_csv = os.path.join(self.root_dir, CSV_NAME.format(mode=self.mode))
                with open(out_csv, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "path"],
                                       extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
            if opts["sheet"]:
                self.q.put(("msg", "Building contact sheet…"))
                out_sheet = os.path.join(self.root_dir, SHEET_NAME.format(mode=self.mode))
                rp.contact_sheet(rows, out_sheet, opts["top"])
            self.q.put(("done", rows, out_csv, out_sheet))
        except Exception as e:
            self.q.put(("fail", str(e)))

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "prog":
                    self.bar.config(value=m[1])
                    self.status.config(text=f"Analysing {m[1]}/{m[2]}…",
                                       foreground=FG_MUTED)
                elif m[0] == "msg":
                    self.status.config(text=m[1], foreground=FG_MUTED)
                elif m[0] == "err":
                    self._log(m[1])
                elif m[0] == "fail":
                    self._finish()
                    self.status.config(text="✖  Error", foreground=ERROR)
                    messagebox.showerror("Rank Planetary", m[1])
                    return
                elif m[0] == "done":
                    self._show(m[1], m[2], m[3])
                    self._finish()
                    return
        except queue.Empty:
            pass
        if self.running:
            self.after(100, self._poll)

    def _finish(self):
        self.running = False
        self.run_btn.config(state="normal")

    def _show(self, rows, out_csv, out_sheet):
        self.rows = rows
        ntop = rp.top_count(rows)
        warns = rp.warnings_for(rows, self.mode)
        self.verdict_lbl.config(text="✔  " + rp.verdict(rows, self.mode))
        nd = sum(1 for r in rows if r["discard"])
        lines = [f"✖  {nd} to discard (red rows)"] if nd else []
        lines += [f"⚠  #{i}: {txt}" for i, w in enumerate(warns, 1) for _, txt in w]
        self.warn_lbl.config(text="\n".join(lines[:6]) + (f"\n…and {len(lines) - 6} more warnings" if len(lines) > 6 else ""))
        self.res_frame.config(
            text=f"  Ranking  —  top group: {ntop} / {len(rows)}  (green)"
                 + (f"  ·  to discard: {sum(1 for r in rows if r['discard'])}  (red)" if any(r["discard"] for r in rows) else ""))
        for i, (r, w) in enumerate(zip(rows, warns), 1):
            self.res.insert("", "end",
                            tags=("discard",) if r["discard"] else (("top",) if r["top"] else ()),
                            values=tuple(self._cell(c, i, r, bool(w)) for c in self.cols))
            self.result_paths[self.res.get_children()[-1]] = r["path"]
        saved = [os.path.basename(p) for p in (out_csv, out_sheet) if p]
        msg = f"✔  {len(rows)} images analysed"
        if saved:
            msg += f"  —  saved: {', '.join(saved)}"
        self.status.config(text=msg, foreground=SUCCESS)
        self.copy_btn.config(state="normal")
        self.side_btn.config(state="normal")
        if out_sheet:
            self.sheet_path = out_sheet
            self.sheet_btn.config(state="normal")

    @staticmethod
    def _cell(col, i, r, warn=False):
        """Cell text for column `col`, row `r` at position `i`."""
        return {
            "n": f"{i}{'★' if r['top'] else ''}{'✖' if r['discard'] else ('⚠' if warn else '')}",
            "score": f"{r['score']:.2f}", "z": f"{r['z']:.1f}",
            "detail": f"{r['detail']:.4f}", "limb": f"{r['limb']:.3f}",
            "hf": f"{r['hf']:.3g}", "snr": f"{r['snr']:.1f}",
            "noise": f"{r['noise']:.5f}", "halo": f"{r['halo']:.4f}",
            "rgb": f"{r['rgb']:.2f}", "file": r["file"],
        }[col]

    def copy_best(self):
        if not self.rows or not self.root_dir:
            return
        dest = os.path.join(self.root_dir, rp.BEST_DIR)
        done = rp.copy_best(self.rows, self.copy_n.get(), dest)
        self.status.config(text=f"✔  Copied {len(done)} images to '{rp.BEST_DIR}'",
                           foreground=SUCCESS)
        open_path(dest)

    def side(self):
        if not self.rows or not self.root_dir:
            return
        if len(self.rows) < 2:
            self.status.config(text="At least 2 images are needed for the side-by-side comparison")
            return
        out = os.path.join(self.root_dir, COMPARE_NAME.format(mode=self.mode))
        try:
            rp.side_by_side(self.rows, out)
        except Exception as e:
            return messagebox.showerror("Rank Planetary", str(e))
        self.status.config(text=f"✔  Saved {os.path.basename(out)}",
                           foreground=SUCCESS)
        open_path(out)

    def info(self):
        """Quick guide: which parameters to look at in this tab."""
        win = tk.Toplevel(self)
        win.title("Info - " + ("Raws" if self.mode == "raws" else "Processed"))
        win.geometry("780x680")
        win.configure(bg=BG)
        txt = tk.Text(win, wrap="word", font=FONT, bg=SURFACE, fg=FG, insertbackground=FG,
                      relief="flat", bd=0, padx=18, pady=12, spacing3=4)
        txt.tag_configure("h", font=FONT_BOLD, foreground=ACCENT, spacing1=12, spacing3=4)
        for line in rp.info_text(self.mode).splitlines():
            if line.startswith("## "):
                txt.insert("end", line[3:] + "\n", "h")
            else:
                txt.insert("end", line + "\n")
        txt.config(state="disabled")
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

    def compare(self):
        init = self.root_dir or "~"
        raw = filedialog.askopenfilename(
            title="CSV of RAWS", initialdir=init, filetypes=[("CSV", "*.csv")])
        if not raw:
            return
        proc = filedialog.askopenfilename(
            title="CSV of PROCESSED", initialdir=os.path.dirname(raw),
            filetypes=[("CSV", "*.csv")])
        if not proc:
            return
        try:
            lines = rp.compare_csvs(raw, proc, True)
        except Exception as e:
            return messagebox.showerror("Rank Planetary", str(e))

        win = tk.Toplevel(self)
        win.title("Before / after")
        win.geometry("980x460")
        win.configure(bg=BG)
        txt = tk.Text(win, wrap="none",
                      font=FONT_MONO,
                      bg=SURFACE, fg=FG,
                      insertbackground=FG,
                      relief="flat", bd=0,
                      padx=12, pady=8)
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")
        hs = ttk.Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(xscrollcommand=hs.set)
        hs.pack(side="bottom", fill="x")
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

    def open_result(self, event):
        iid = self.res.identify_row(event.y)
        if iid in self.result_paths:
            open_path(self.result_paths[iid])

    def open_sheet(self):
        if self.sheet_path:
            open_path(self.sheet_path)

    def _log(self, text=None, clear=False):
        self.log.config(state="normal")
        if clear:
            self.log.delete("1.0", "end")
        if text:
            self.log.insert("end", "⚠  " + text + "\n")
        self.log.config(state="disabled")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rank Planetary")
        self.geometry(f"1020x{min(820, self.winfo_screenheight() - 90)}")
        self.minsize(900, 640)
        self.configure(bg=BG)
        _apply_theme(self)

        # header
        hdr = tk.Frame(self, bg=SURFACE, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🪐  Rank Planetary",
                 bg=SURFACE, fg=FG,
                 font=(FONT_FAMILY, 16, "bold")).pack(side="left", padx=18, pady=10)
        tk.Label(hdr, text="Planetary image ranking",
                 bg=SURFACE, fg=FG_MUTED,
                 font=(FONT_FAMILY, 11)).pack(side="left", padx=4, pady=10)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.raws      = Panel(nb, "raws")
        self.processed = Panel(nb, "processed")
        nb.add(self.raws,      text="  Raws  ")
        nb.add(self.processed, text="  Processed  ")


if __name__ == "__main__":
    App().mainloop()

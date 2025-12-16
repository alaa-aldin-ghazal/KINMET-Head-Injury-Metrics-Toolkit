"""
KINMET — Head Injury Metrics Toolkit

KINMET is an open-source, research-oriented Python application with a graphical
user interface (GUI) for computing head injury metrics from 6-DOF kinematic data
(measured linear acceleration and angular velocity).

The tool supports interactive column mapping, unit conversion, signal filtering,
and automated report generation, and is intended for use in biomechanics,
automotive safety, sports injury research, and related domains.

Core capabilities:
- Interactive column mapping and unit selection with live signal preview
- Sampling-aware signal filtering (Butterworth, CFC-equivalent, Savitzky–Golay,
  Moving Average) with zero-phase implementation where applicable
- Automated computation of established head injury metrics:
    • HIC15 / HIC36
    • Gadd Severity Index (SI)
    • Brain Injury Criterion (BrIC)
    • Peak linear and angular kinematics
    • Estimated neck loads and Nij (Hybrid-III–based approximation)
- Generation of a multi-page, self-contained PDF report including figures,
  tables, and mathematical definitions
- Export of computed metrics and full processing metadata to JSON

IMPORTANT DISCLAIMER:
This software is provided for research and educational purposes only.
Computed injury metrics and injury risk estimates are NOT clinically validated
and must NOT be used for medical diagnosis, safety certification, regulatory
decision-making, or real-time injury prediction.

Implementation notes:
- CFC filtering is implemented as a zero-phase digital low-pass approximation
  consistent with common biomechanics practice and is not a strict analog
  SAE J211 implementation.
- Neck loads and Nij are estimated using rigid-body Newton–Euler dynamics with
  nominal Hybrid-III head properties; results should be interpreted as
  comparative indicators rather than absolute injury thresholds.

License:
MIT License

Version:
1.0.0
"""


from __future__ import annotations
import os
import math
import json
import logging
import datetime
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy import integrate
from scipy.spatial.transform import Rotation as R
# cumulative trapezoid fallback for older SciPy
try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except Exception:
    def _cumtrapz(y, x, initial=0.0):
        y = np.asarray(y); x = np.asarray(x)
        out = np.zeros(len(y))
        for i in range(1, len(y)):
            out[i] = out[i-1] + 0.5*(y[i] + y[i-1]) * (x[i] - x[i-1])
        if initial != 0.0:
            out = out + initial
        return out

# filters
from scipy.signal import butter, filtfilt, savgol_filter

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import matplotlib as mpl

mpl.rcParams.update({
    "pdf.fonttype": 42,          # TrueType embedding
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",  # NOT DejaVuSansDisplay
    "text.usetex": False
})
# ---------- CONFIG & CONSTANTS ----------
G2MSS = 9.80665
W_CX, W_CY, W_CZ = 66.25, 56.45, 42.87   # BrIC denominators (rad/s) default

# CFC presets (approximate digital cutoffs used as reasonable defaults)
CFC_PRESETS = {
    "CFC60": 100.0,
    "CFC180": 300.0,
    "CFC600": 1000.0,
    "CFC1000": 1650.0
}

M_HEAD = 4.54
I_HEAD = np.diag([0.016, 0.018, 0.022])
R_OC_TO_CG = np.array([0.0, 0.0, 0.017])
FZC_TENSION = 6806.0
FZC_COMPRESSION = -6160.0
MYC_FLEXION = 190.0
MYC_EXTENSION = 57.0

# Risk thresholds (for table coloring)
PEAK_ACCEL_LOW, PEAK_ACCEL_MOD, PEAK_ACCEL_HIGH = 50.0, 80.0, 120.0
PEAK_ANGVEL_LOW, PEAK_ANGVEL_MOD, PEAK_ANGVEL_HIGH = 30.0, 60.0, 100.0
HIC_LOW, HIC_MOD, HIC_HIGH = 250.0, 700.0, 1500.0
BRIC_LOW, BRIC_MOD, BRIC_HIGH = 0.45, 0.7, 1.0

__version__ = "1.0.0"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("head_injury_gui")


# ---------- Utility helpers ----------
def risk_label(val: float, low: float, mod: float, high: float) -> str:
    if val < low:
        return "Low"
    if val < mod:
        return "Moderate"
    if val < high:
        return "High"
    return "Very High"


def risk_color(r: str) -> str:
    mapping = {"Low": "#c8e6c9", "Moderate": "#fff9c4", "High": "#ffcc80", "Very High": "#ef9a9a"}
    return mapping.get(r, "white")


def recommend_cfc_for_fs(fs_hz: float) -> str:
    """Recommend a reasonable CFC preset for a given sampling frequency."""
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return "CFC180"
    if fs_hz < 250:
        return "CFC60"
    if fs_hz < 800:
        return "CFC180"
    if fs_hz < 2000:
        return "CFC600"
    return "CFC1000"


# ---------- Filtering / smoothing helpers ----------
def _design_butter(cutoff_hz: float, fs_hz: float, order: int = 4):
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"cutoff {cutoff_hz} must be between 0 and Nyquist ({nyq:.1f} Hz)")
    b, a = butter(order, cutoff_hz/nyq, btype="low", analog=False)
    return b, a

def apply_butter_zero_phase(signal: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if signal is None:
        return signal
    if len(signal) < max(8, order*3):
        return signal.copy()
    b, a = _design_butter(cutoff_hz, fs_hz, order)
    return filtfilt(b, a, signal, method="pad")

def apply_cfc(signal: np.ndarray, fs_hz: float, cfc_name: str = "CFC180") -> np.ndarray:
    cutoff = CFC_PRESETS.get(cfc_name)
    if cutoff is None:
        raise ValueError("Unknown CFC preset")
    return apply_butter_zero_phase(signal, fs_hz, cutoff, order=4)

def apply_savgol(signal: np.ndarray, window_length: int = 21, polyorder: int = 2) -> np.ndarray:
    if signal is None:
        return signal
    if window_length % 2 == 0:
        window_length += 1
    wl = min(window_length, len(signal) if len(signal) % 2 == 1 else max(1, len(signal)-1))
    if wl < 3:
        return signal.copy()
    return savgol_filter(signal, wl, polyorder, mode="interp")

def smooth_signal_wrapper(signal: np.ndarray, fs_hz: float, method: str = "none", **kwargs) -> np.ndarray:
    method = (method or "none").lower()
    if signal is None:
        return signal
    if method == "cfc":
        return apply_cfc(signal, fs_hz, kwargs.get("cfc_name", "CFC180"))
    if method == "butter":
        return apply_butter_zero_phase(signal, fs_hz, kwargs.get("cutoff", 300.0), order=kwargs.get("order", 4))
    if method == "savgol":
        return apply_savgol(signal, kwargs.get("window_length", 21), kwargs.get("polyorder", 2))
    if method == "ma":
        w = int(kwargs.get("ma_win", 5))
        if w <= 1:
            return signal.copy()
        kernel = np.ones(w) / w
        return np.convolve(signal, kernel, mode="same")
    return signal.copy()


# ---------- File preview & GUI helpers ----------
def parse_preview_rows(in_file: str, max_rows: int = 25):
    import csv
    rows = []
    with open(in_file, "r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= max_rows - 1:
                break
    return rows


def ask_header_row_with_preview(in_file: str) -> Optional[int]:
    """Show first rows and let user pick which row index contains header. Returns row index or None."""
    preview_rows = parse_preview_rows(in_file, max_rows=25)
    result = {"row": None}
    win = tk.Toplevel()
    win.title("Select Header Row (preview first rows)")
    win.geometry("900x420")
    win.attributes("-topmost", True)
    win.update()
    win.attributes("-topmost", False)

    tk.Label(win, text="Preview of the first rows. Select the row that contains column headers, then Confirm.", wraplength=850).pack(padx=8, pady=6)
    frame = tk.Frame(win); frame.pack(fill="both", expand=True, padx=8, pady=4)
    listbox = tk.Listbox(frame, width=140, height=20, activestyle='none')
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)
    for idx, row in enumerate(preview_rows):
        row_display = ", ".join([str(x) for x in row])
        if len(row_display) > 800:
            row_display = row_display[:800] + " ... (truncated)"
        listbox.insert("end", f"{idx:02d}: {row_display}")
    if preview_rows:
        listbox.selection_set(0); listbox.see(0)

    def do_confirm():
        sel = listbox.curselection()
        if not sel:
            messagebox.showerror("Selection required", "Please select a row index that contains the header.")
            return
        result["row"] = int(sel[0]); win.destroy()

    def on_double(event):
        do_confirm()

    listbox.bind("<Double-Button-1>", on_double)
    btn_frame = tk.Frame(win); btn_frame.pack(fill="x", pady=6)
    tk.Button(btn_frame, text="Confirm Selection", command=do_confirm).pack(side="left", padx=8)
    tk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side="right", padx=8)
    win.grab_set(); win.wait_window()
    return result["row"]

def build_filter_suggestion(fs_est: float, selected_filter: str):
    """
    Return (info_text, warn_text, suggestions_dict) giving sampling-aware hints
    for the chosen filter. Does NOT change widget states — only computes strings
    and numeric suggestions that you can show to the user.

    Parameters
    - fs_est: estimated sampling frequency in Hz (float or np.nan)
    - selected_filter: one of "none","cfc","butter","butterworth","savgol",
                       "savitzky","moving","ma" (case-insensitive)

    Returns
    - info_text (str): short blue info label text suitable for fs_label
    - warn_text (str): warning text (empty if none)
    - suggestions (dict): numeric suggestions for keys relevant to filter:
          { "viable_cfc": [...],
            "butter_cutoff": float,
            "butter_order": int,
            "savgol_win": int,
            "savgol_poly": int,
            "ma_win": int }
    """
    sel = (selected_filter or "none").strip().lower()
    suggestions = {"viable_cfc": [], "butter_cutoff": None, "butter_order": 4,
                   "savgol_win": None, "savgol_poly": 2, "ma_win": None}
    # safe guard
    if fs_est is None or (isinstance(fs_est, float) and np.isnan(fs_est)) or fs_est <= 0:
        info = "Sampling: N/A — cannot suggest filter parameters (invalid time column)."
        warn = ""
        return info, warn, suggestions

    nyq = 0.5 * fs_est

    # conservative viability check for CFC: require fs >= 6 * cutoff and cutoff < Nyquist
    viable = []
    for name, cutoff in CFC_PRESETS.items():
        if (fs_est >= 6.0 * cutoff) and (cutoff < nyq):
            viable.append(name)
    suggestions["viable_cfc"] = viable

    # Suggest Butterworth cutoff: choose a safe fraction of fs (not above Nyquist)
    # Use min(0.45*fs, nyq*0.9, 300) as a reasonable default upper bound for many crash tests.
    butter_cut = max(1.0, min(0.45 * fs_est, nyq * 0.9, 300.0))
    suggestions["butter_cutoff"] = round(butter_cut, 1)

    # Suggest Savitzky window: aim for ~40–60 ms smoothing region (adjustable)
    # Convert to samples: win_samples = ceil(0.05 * fs) and make odd and >=5
    win = max(5, int(round(0.05 * fs_est)))
    if win % 2 == 0:
        win += 1
    suggestions["savgol_win"] = win
    # poly order: small (2) or < window-1
    suggestions["savgol_poly"] = min(3, max(1, 2 if win >= 5 else 1))

    # Moving average suggestion: small window ~ 1-10 ms depending on sampling
    ma_win = max(3, int(round(0.01 * fs_est)))
    suggestions["ma_win"] = ma_win

    # Prepare info and warning texts depending on the selected filter
    if sel in ("none", ""):
        info = f"Sampling: ~{fs_est:.1f} Hz. No preview filter selected. Viable CFC presets: {', '.join(viable) if viable else 'none'}."
        warn = ""
    elif sel in ("cfc",):
        if viable:
            info = f"Sampling: ~{fs_est:.1f} Hz. Viable CFC presets: {', '.join(viable)}."
            warn = ""
        else:
            info = f"Sampling: ~{fs_est:.1f} Hz. No CFC preset looks viable at this sampling rate."
            warn = ("CFC not recommended: sampling too low for standard CFC presets. "
                    f"Consider Butterworth cutoff ≈ {suggestions['butter_cutoff']} Hz or "
                    f"Savitzky–Golay window ≈ {suggestions['savgol_win']} samples (poly {suggestions['savgol_poly']}).")
    elif "butter" in sel:
        info = (f"Sampling: ~{fs_est:.1f} Hz. Suggested Butterworth cutoff ≈ {suggestions['butter_cutoff']} Hz "
                f"(order {suggestions['butter_order']}).")
        warn = ""
    elif "savgol" in sel or "savitz" in sel:
        info = (f"Sampling: ~{fs_est:.1f} Hz. Suggested Savitzky–Golay window ≈ {suggestions['savgol_win']} samples "
                f"(poly {suggestions['savgol_poly']}).")
        warn = ""
    elif "moving" in sel or sel == "ma":
        info = (f"Sampling: ~{fs_est:.1f} Hz. Suggested moving-average window ≈ {suggestions['ma_win']} samples.")
        warn = ""
    else:
        info = f"Sampling: ~{fs_est:.1f} Hz. Viable CFC presets: {', '.join(viable) if viable else 'none'}."
        warn = ""

    return info, warn, suggestions


def get_column_mapping_and_units(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], dict]:
    """
    Dialog to map dataframe columns to signals and pick units.
    Also allows selecting a smoothing/filter option and previews raw vs filtered overlay.

    Returns:
        mapping: Dict[str,str] - mapping of signals ("Time","ax","ay","az","wx","wy","wz") to column names
        units: Dict[str,str] - selected units for each signal
        filter_opts: dict - keys:
            - method: "none"|"cfc"|"butter"|"savgol"|"ma"
            - cfc_name, butter_cutoff, butter_order, savgol_win, savgol_poly, ma_win
            - use_filtered_for_metrics: bool
    """
    mapping: Dict[str, str] = {}
    units: Dict[str, str] = {}
    # default filter options returned if user confirms
    filter_opts = {
        "method": "none",
        "cfc_name": "CFC180",
        "butter_cutoff": 300.0,
        "butter_order": 4,
        "savgol_win": 21,
        "savgol_poly": 2,
        "ma_win": 5,
        "use_filtered_for_metrics": False
    }

    # --- Setup window ---
    root = tk.Toplevel()
    root.title("Column & Unit Mapping (Preview + Filter)")
    root.geometry("1200x640")
    root.minsize(1000, 600)
    root.resizable(True, True)
    # keep window on top briefly so it appears in front
    root.attributes("-topmost", True)
    root.update()
    root.attributes("-topmost", False)

    # left panel = form, right panel = preview
    form_frame = tk.Frame(root)
    form_frame.pack(side="left", fill="y", padx=8, pady=8)
    preview_frame = tk.Frame(root)
    preview_frame.pack(side="right", fill="both", expand=True, padx=8, pady=8)

    tk.Label(form_frame, text="Select columns and units", font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=4, pady=6)

    # unit choices
    time_units = ["seconds", "milliseconds", "microseconds"]
    accel_units = ["g", "m/s²"]
    angvel_units = ["deg/s", "rad/s"]

    signals = [("Time", time_units), ("ax", accel_units), ("ay", accel_units), ("az", accel_units),
               ("wx", angvel_units), ("wy", angvel_units), ("wz", angvel_units)]
    dropdowns: Dict[str, ttk.Combobox] = {}
    unit_boxes: Dict[str, ttk.Combobox] = {}

    cols = list(df.columns)

    # choose numeric-like default columns where possible
    numeric_cols = []
    for c in cols:
        try:
            # check first few entries are numeric
            pd.to_numeric(df[c].dropna().iloc[:10])
            numeric_cols.append(c)
        except Exception:
            continue
    defaults_order = numeric_cols[:7] if len(numeric_cols) >= 7 else (cols[:7] + [""] * max(0, 7 - len(cols)))

    # create mapping controls
    for i, (sig, unit_list) in enumerate(signals, start=1):
        tk.Label(form_frame, text=f"{sig} column:").grid(row=i, column=0, sticky="e", padx=6, pady=4)
        col_box = ttk.Combobox(form_frame, values=cols, state="readonly", width=36)
        try:
            col_box.set(defaults_order[i-1])
        except Exception:
            pass
        col_box.grid(row=i, column=1, padx=6, pady=4, columnspan=2)
        dropdowns[sig] = col_box

        unit_box = ttk.Combobox(form_frame, values=unit_list, state="readonly", width=10)
        unit_box.set(unit_list[0])
        unit_box.grid(row=i, column=3, padx=6, pady=4)
        unit_boxes[sig] = unit_box

    # -------------------------
    # Filter controls
    # -------------------------
    tk.Label(form_frame, text="Preview Filter:", font=("Helvetica", 10, "bold")).grid(row=9, column=0, columnspan=4, pady=(10,2))

    # Use a StringVar so we can reliably watch changes across platforms
    filter_method_var = tk.StringVar(value="none")
    filter_method_box = ttk.Combobox(form_frame, textvariable=filter_method_var,
                                     values=["none", "CFC", "Butterworth", "Savitzky-Golay", "MovingAvg"],
                                     state="readonly", width=18)
    filter_method_box.grid(row=10, column=1, sticky="w", padx=6)

    tk.Label(form_frame, text="Params:").grid(row=11, column=0, sticky="e", padx=6)
    cfc_var = tk.StringVar(value=filter_opts["cfc_name"])
    cfc_box = ttk.Combobox(form_frame, textvariable=cfc_var, values=list(CFC_PRESETS.keys()),
                           state="readonly", width=12)
    cfc_box.grid(row=11, column=1, sticky="w", padx=6)

    tk.Label(form_frame, text="Butter cutoff (Hz):").grid(row=12, column=0, sticky="e")
    butter_cut_entry = tk.Entry(form_frame, width=8)
    butter_cut_entry.insert(0, str(filter_opts["butter_cutoff"]))
    butter_cut_entry.grid(row=12, column=1, sticky="w", padx=6)
    tk.Label(form_frame, text="Order:").grid(row=12, column=2, sticky="e")
    butter_order_entry = tk.Entry(form_frame, width=4)
    butter_order_entry.insert(0, str(filter_opts["butter_order"]))
    butter_order_entry.grid(row=12, column=3, sticky="w", padx=6)

    tk.Label(form_frame, text="Savgol window:").grid(row=13, column=0, sticky="e")
    savgol_win_entry = tk.Entry(form_frame, width=6)
    savgol_win_entry.insert(0, str(filter_opts["savgol_win"]))
    savgol_win_entry.grid(row=13, column=1, sticky="w", padx=6)
    tk.Label(form_frame, text="poly:").grid(row=13, column=2, sticky="e")
    savgol_poly_entry = tk.Entry(form_frame, width=4)
    savgol_poly_entry.insert(0, str(filter_opts["savgol_poly"]))
    savgol_poly_entry.grid(row=13, column=3, sticky="w", padx=6)

    tk.Label(form_frame, text="MA window:").grid(row=14, column=0, sticky="e")
    ma_win_entry = tk.Entry(form_frame, width=6)
    ma_win_entry.insert(0, str(filter_opts["ma_win"]))
    ma_win_entry.grid(row=14, column=1, sticky="w", padx=6)

    use_filtered_var = tk.BooleanVar(value=False)
    tk.Checkbutton(form_frame, text="Use filtered signals for metrics & plots", variable=use_filtered_var).grid(row=15, column=1, sticky="w", padx=6, pady=6)

    # sampling info and warning labels
    fs_label = tk.Label(form_frame, text="Sampling: N/A — Recommended CFC: N/A", wraplength=260, justify="left", fg="blue")
    fs_label.grid(row=16, column=0, columnspan=4, pady=(6,0), padx=4, sticky="w")
    warn_label = tk.Label(form_frame, text="", wraplength=260, justify="left", fg="red")
    warn_label.grid(row=17, column=0, columnspan=4, pady=(2,0), padx=4, sticky="w")

    # Preview X/Y choosers
    tk.Label(form_frame, text="Preview X:").grid(row=18, column=0, sticky="e", padx=6, pady=6)
    preview_x = ttk.Combobox(form_frame, values=["Time","ax","ay","az","wx","wy","wz"], state="readonly", width=12)
    preview_x.set("Time"); preview_x.grid(row=18, column=1, sticky="w", padx=6)
    tk.Label(form_frame, text="Preview Y:").grid(row=19, column=0, sticky="e", padx=6, pady=6)
    preview_y = ttk.Combobox(form_frame, values=["a_res","ax","ay","az","w_res","wx","wy","wz"], state="readonly", width=12)
    preview_y.set("a_res"); preview_y.grid(row=19, column=1, sticky="w", padx=6)
    autoscale_var = tk.BooleanVar(value=True)
    tk.Checkbutton(form_frame, text="Autoscale preview", variable=autoscale_var).grid(row=20, column=1, sticky="w", padx=6, pady=4)

    # --- Matplotlib preview area ---
    fig = Figure(figsize=(8.2, 8.0), dpi=100)
    grid = fig.add_gridspec(8, 8)
    ax_main = fig.add_subplot(grid[:, :4])
    ax_acc_overlay = fig.add_subplot(grid[0:4, 5:])
    ax_ang_overlay = fig.add_subplot(grid[5:8, 5:])

    ax_main.set_title("Preview"); ax_main.grid(True)
    ax_acc_overlay.set_title("Linear Accel (resultant)"); ax_acc_overlay.grid(True)
    ax_ang_overlay.set_title("Angular Vel (resultant)"); ax_ang_overlay.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=preview_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # initially disable parameter widgets until a filter is chosen
    cfc_box.config(state="disabled")
    butter_cut_entry.config(state="disabled"); butter_order_entry.config(state="disabled")
    savgol_win_entry.config(state="disabled"); savgol_poly_entry.config(state="disabled")
    ma_win_entry.config(state="disabled")

    # -----------------------------
    # Helper: fetch numeric arrays adapted from mapping selection
    # -----------------------------
    def get_preview_arrays():
        """Return (t_arr, ax, ay, az, a_res, wx, wy, wz, w_res) or None on failure."""
        sel_time = dropdowns["Time"].get()
        if not sel_time or sel_time not in df.columns:
            return None
        try:
            t_raw = pd.to_numeric(df[sel_time].dropna()).values.astype(float)
            # apply time unit conversion
            u = unit_boxes["Time"].get()
            if u == "milliseconds":
                t_arr = t_raw * 1e-3
            elif u == "microseconds":
                t_arr = t_raw * 1e-6
            else:
                t_arr = t_raw
        except Exception:
            return None

        def fetch(sig):
            col = dropdowns[sig].get()
            if not col or col not in df.columns:
                return None
            arr = pd.to_numeric(df[col].astype(float), errors='coerce').values
            u = unit_boxes[sig].get()
            if sig == "Time":
                return arr
            if u == "m/s²":
                arr = arr / G2MSS  # convert preview to g
            if sig in ("wx", "wy", "wz") and u == "deg/s":
                arr = np.deg2rad(arr)
            return arr

        ax_arr = fetch("ax"); ay_arr = fetch("ay"); az_arr = fetch("az")
        a_res = None
        if ax_arr is not None and ay_arr is not None and az_arr is not None:
            a_res = np.sqrt(ax_arr**2 + ay_arr**2 + az_arr**2)
        wx_arr = fetch("wx"); wy_arr = fetch("wy"); wz_arr = fetch("wz")
        w_res = None
        if wx_arr is not None and wy_arr is not None and wz_arr is not None:
            w_res = np.sqrt(wx_arr**2 + wy_arr**2 + wz_arr**2)
        return t_arr, ax_arr, ay_arr, az_arr, a_res, wx_arr, wy_arr, wz_arr, w_res

    # Apply selected preview filter safely on 1D arrays
    def apply_preview_filter(arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None
        method = (filter_method_var.get() or "none").strip().lower()
        # estimate fs
        fs = None
        try:
            dt = np.median(np.diff(t_arr))
            fs = 1.0 / dt if dt > 0 else None
        except Exception:
            fs = None
        try:
            if method == "none" or method == "":
                return arr
            if method == "cfc":
                if fs is None or np.isnan(fs):
                    return arr
                return smooth_signal_wrapper(arr, fs, method="cfc", cfc_name=cfc_var.get())
            if method == "butterworth" or "butter" in method:
                cutoff = float(butter_cut_entry.get())
                order = int(butter_order_entry.get())
                if fs is None or np.isnan(fs):
                    return arr
                return smooth_signal_wrapper(arr, fs, method="butter", cutoff=cutoff, order=order)
            if method == "savitzky-golay" or "savgol" in method:
                w = int(savgol_win_entry.get()); p = int(savgol_poly_entry.get())
                return smooth_signal_wrapper(arr, fs if fs is not None and not np.isnan(fs) else 1000.0,
                                             method="savgol", window_length=w, polyorder=p)
            if method == "movingavg" or method == "ma":
                w = int(ma_win_entry.get())
                return smooth_signal_wrapper(arr, fs if fs is not None and not np.isnan(fs) else 1000.0,
                                             method="ma", ma_win=w)
        except Exception as e:
            logger.warning("Preview filter failed: %s", e)
            return arr
        return arr

    # -----------------------------
    # Main preview update function
    # -----------------------------
    def update_preview(*args):
        data = get_preview_arrays()
        ax_main.clear()
        ax_main.grid(True)
        ax_main.set_title("Preview")

        ax_acc_overlay.clear()
        ax_acc_overlay.grid(True)
        ax_acc_overlay.set_title("Linear Accel (resultant)")

        ax_ang_overlay.clear()
        ax_ang_overlay.grid(True)
        ax_ang_overlay.set_title("Angular Vel (resultant)")

        if data is None:
            ax_main.text(0.5, 0.5, "Select valid Time column", ha="center")
            canvas.draw()
            return

        t_arr, ax_arr, ay_arr, az_arr, a_res, wx_arr, wy_arr, wz_arr, w_res = data
        name_x = preview_x.get(); name_y = preview_y.get()
        mapping_preview = {"Time": t_arr, "ax": ax_arr, "ay": ay_arr, "az": az_arr,
                           "a_res": a_res, "wx": wx_arr, "wy": wy_arr, "wz": wz_arr, "w_res": w_res}
        xdata = mapping_preview.get(name_x); ydata = mapping_preview.get(name_y)

        if xdata is None or ydata is None:
            ax_main.text(0.5, 0.5, "Preview unavailable for chosen columns", ha="center")
            canvas.draw()
            return

        # align lengths & mask invalid
        minlen = min(len(xdata), len(ydata))
        x_plot = xdata[:minlen]; y_plot = ydata[:minlen]
        mask = ~(np.isnan(x_plot) | np.isnan(y_plot))
        if np.count_nonzero(mask) < 2:
            ax_main.text(0.5, 0.5, "Not enough numeric data in selected columns", ha="center")
            canvas.draw()
            return
        x_plot = x_plot[mask]; y_plot = y_plot[mask]

        # Event detection parameters
        event_window_sec = 0.2   # total window length around detected peak (adjustable)
        min_window_s = 0.05      # fallback minimum window if detection finds narrow spike

        # Use a_res if available (full-rate)
        if a_res is not None and len(a_res) == len(t_arr):
            # compute robust baseline and detect peaks
            arr_full = a_res.copy()
            med = np.nanmedian(arr_full)
            mad = np.nanmedian(np.abs(arr_full - med)) or 1e-6
            # threshold: median + 6*MAD (robust)
            thresh = med + 6.0 * mad

            # indices where accel above threshold
            hits = np.where(arr_full > thresh)[0]
            if hits.size:
                i0, i1 = hits[0], hits[-1]
                # expand by margin (event_window_sec/2) but clipped to data bounds
                half_win = int(max((event_window_sec/2) / np.median(np.diff(t_arr)), 1))
                s_idx = max(0, i0 - half_win)
                e_idx = min(len(t_arr)-1, i1 + half_win)
                # create high-rate slices for accurate filtering
                t_slice = t_arr[s_idx:e_idx+1]
                y_slice = (mapping_preview.get(name_y))[s_idx:e_idx+1]
                use_event_slice = True
            else:
                use_event_slice = False
        else:
            use_event_slice = False

        # -------------------------
        # Plot main trace (event-slice if found) and acceleration resultant overlay
        # -------------------------
        # Prepare containers
        main_t = None; main_y = None; main_filt = None
        acc_t = None; acc_raw = None; acc_filt = None

        if use_event_slice:
            # t_slice, y_slice already defined earlier; s_idx/e_idx exist in this scope
            slice_len = len(t_slice)
            # compute fs on the slice
            try:
                dt_full = np.median(np.diff(t_slice))
                fs_full = 1.0 / dt_full if dt_full > 0 else None
            except Exception:
                fs_full = None

            # filter at original rate on the high-rate slice
            try:
                filt_slice = apply_preview_filter(y_slice, t_slice)
            except Exception:
                filt_slice = None

            # downsample the slice for plotting
            max_pts = 4000
            if slice_len > max_pts:
                step = max(1, slice_len // max_pts)
                main_t = t_slice[::step]; main_y = y_slice[::step]
                main_filt = filt_slice[::step] if (filt_slice is not None and len(filt_slice) == slice_len) else None
            else:
                main_t = t_slice; main_y = y_slice; main_filt = filt_slice

            # Accel resultant for the same slice:
            # If preview Y is a_res, use a_res slice; otherwise compute from ax/ay/az slice
            if name_y == "a_res":
                acc_raw_full = a_res[s_idx:e_idx+1]
            else:
                ax_slice = ax_arr[s_idx:e_idx+1] if ax_arr is not None else None
                ay_slice = ay_arr[s_idx:e_idx+1] if ay_arr is not None else None
                az_slice = az_arr[s_idx:e_idx+1] if az_arr is not None else None
                if ax_slice is not None and ay_slice is not None and az_slice is not None:
                    acc_raw_full = np.sqrt(ax_slice**2 + ay_slice**2 + az_slice**2)
                else:
                    acc_raw_full = None

            if acc_raw_full is not None:
                # filter the full-rate acc array similarly
                try:
                    acc_filt_full = apply_preview_filter(acc_raw_full, t_slice)
                except Exception:
                    acc_filt_full = None

                # downsample to plotting resolution (use same step as main plot if used)
                if slice_len > max_pts:
                    acc_t = t_slice[::step]; acc_raw = acc_raw_full[::step]
                    acc_filt = acc_filt_full[::step] if (acc_filt_full is not None and len(acc_filt_full) == slice_len) else None
                else:
                    acc_t = t_slice; acc_raw = acc_raw_full; acc_filt = acc_filt_full
            else:
                acc_t = None; acc_raw = None; acc_filt = None

        else:
            # fallback: decimate full-record (existing behavior)
            max_pts = 4000
            if len(x_plot) > max_pts:
                step = max(1, len(x_plot) // max_pts)
                main_t = x_plot[::step]; main_y = y_plot[::step]
            else:
                main_t = x_plot; main_y = y_plot
            main_filt = apply_preview_filter(main_y, main_t)

            # Acc resultant for full-record (use precomputed a_res if available)
            if a_res is not None:
                acc_full = a_res[:minlen][mask]
                # downsample similarly
                if len(acc_full) > max_pts:
                    step = max(1, len(acc_full) // max_pts)
                    acc_t = t_arr[:minlen][mask][::step]; acc_raw = acc_full[::step]
                else:
                    acc_t = t_arr[:minlen][mask]; acc_raw = acc_full
                acc_filt = apply_preview_filter(acc_raw, acc_t)
            else:
                acc_t = None; acc_raw = None; acc_filt = None

        # Draw main axes
        if main_t is not None and main_y is not None:
            ax_main.plot(main_t, main_y, linewidth=1.0, label="raw", alpha=0.6)
            if main_filt is not None and filter_method_var.get().lower() != "none":
                ax_main.plot(main_t, main_filt, linewidth=1.6, label="filtered")
            ax_main.set_xlabel(name_x); ax_main.set_ylabel(name_y)
            if autoscale_var.get():
                ax_main.relim(); ax_main.autoscale_view()
            ax_main.legend(loc="upper right", fontsize=8)

        # Draw acceleration resultant overlay
        ax_acc_overlay.clear(); ax_acc_overlay.grid(True); ax_acc_overlay.set_title("Linear Accel (resultant)")
        if acc_t is not None and acc_raw is not None:
            ax_acc_overlay.plot(acc_t, acc_raw, linewidth=1.0, alpha=0.6, label="raw")
            if acc_filt is not None and filter_method_var.get().lower() != "none":
                ax_acc_overlay.plot(acc_t, acc_filt, linewidth=1.6, label="filtered")
            ax_acc_overlay.set_xlabel("t (s)"); ax_acc_overlay.set_ylabel("g"); ax_acc_overlay.legend(fontsize=8)
            if autoscale_var.get():
                ax_acc_overlay.relim(); ax_acc_overlay.autoscale_view()
        else:
            ax_acc_overlay.text(0.5, 0.5, "Resultant unavailable for preview", ha="center")
            if autoscale_var.get():
                ax_acc_overlay.relim(); ax_acc_overlay.autoscale_view()

        # -------------------------
        # Angular resultant overlay (aligned with main plot / decimation)
        # -------------------------
        ax_ang_overlay.clear(); ax_ang_overlay.grid(True); ax_ang_overlay.set_title("Angular Vel (resultant)")
        t_w_plot = None; raw_w = None; filt_w = None
        try:
            if use_event_slice:
                # slice component angular arrays
                wx_slice = wx_arr[s_idx:e_idx+1] if wx_arr is not None else None
                wy_slice = wy_arr[s_idx:e_idx+1] if wy_arr is not None else None
                wz_slice = wz_arr[s_idx:e_idx+1] if wz_arr is not None else None
                if wx_slice is not None and wy_slice is not None and wz_slice is not None:
                    raw_w_full = np.sqrt(wx_slice**2 + wy_slice**2 + wz_slice**2)
                    if 'step' in locals() and step > 1:
                        t_w_plot = t_slice[::step]
                        raw_w = raw_w_full[::step]
                    else:
                        t_w_plot = t_slice
                        raw_w = raw_w_full
            else:
                # fallback use decimated full-record arrays
                if w_res is not None:
                    # reuse acc_t if available to align time axes; otherwise derive decimated time
                    if 'acc_t' in locals() and acc_t is not None:
                        t_w_plot = acc_t
                    else:
                        if len(t_arr[:minlen][mask]) > 0:
                            if 'step' in locals() and step > 1:
                                t_w_plot = t_arr[:minlen][mask][::step]
                            else:
                                t_w_plot = t_arr[:minlen][mask]
                    # compute resultant for decimated components
                    try:
                        raw_w_full = np.sqrt((wx_arr[:minlen][mask])**2 + (wy_arr[:minlen][mask])**2 + (wz_arr[:minlen][mask])**2)
                        if 'step' in locals() and step > 1:
                            raw_w = raw_w_full[::step]
                        else:
                            raw_w = raw_w_full
                    except Exception:
                        raw_w = None
        except Exception:
            t_w_plot = None; raw_w = None

        if t_w_plot is not None and raw_w is not None:
            try:
                filt_w = apply_preview_filter(raw_w, t_w_plot)
            except Exception:
                filt_w = None
            ax_ang_overlay.plot(t_w_plot, raw_w, linewidth=1.0, alpha=0.6, label="raw")
            if filt_w is not None and filter_method_var.get().lower() != "none":
                ax_ang_overlay.plot(t_w_plot, filt_w, linewidth=1.6, label="filtered")
            ax_ang_overlay.set_xlabel("t (s)"); ax_ang_overlay.set_ylabel("rad/s"); ax_ang_overlay.legend(fontsize=8)
            if autoscale_var.get():
                ax_ang_overlay.relim(); ax_ang_overlay.autoscale_view()
        else:
            ax_ang_overlay.text(0.5, 0.5, "Angular resultant unavailable for preview", ha="center")
            if autoscale_var.get():
                ax_ang_overlay.relim(); ax_ang_overlay.autoscale_view()

        # Robust sampling estimate computed from cleaned time vector (non-NaN, increasing)
        try:
            # prefer to compute fs from the cleaned t_arr (drop NaN and duplicates)
            t_clean = t_arr[np.isfinite(t_arr)]
            if len(t_clean) > 1:
                # ensure strictly increasing by taking differences > small eps
                diffs = np.diff(t_clean)
                diffs_pos = diffs[diffs > 1e-12]
                if len(diffs_pos) > 0:
                    dt_med = float(np.median(diffs_pos))
                else:
                    dt_med = float(np.median(diffs))  # fallback
                fs_est = 1.0 / dt_med if dt_med > 0 else float("nan")
            else:
                fs_est = float("nan")
        except Exception:
            fs_est = float("nan")

        # Call helper to build suggestion + warning text
        try:
            info_text, warn_text, suggestions = build_filter_suggestion(fs_est, filter_method_var.get())
            fs_label.config(text=info_text)
            warn_label.config(text=warn_text)
        except Exception:
            try:
                fs_label.config(text=f"Sampling: N/A")
                warn_label.config(text="")
            except Exception:
                pass

        canvas.draw()

    # --- Enabling/disabling filter parameter widgets ---
    def on_filter_method_change_var(*_):
        m = (filter_method_var.get() or "").strip().lower()
        is_none = (m == "" or m == "none")
        is_cfc = (m == "cfc")
        is_butter = ("butter" in m)
        is_savgol = ("savgol" in m or "savitz" in m or "savitzky" in m)
        is_ma = ("moving" in m or m == "ma" or "movingavg" in m)

        # set states
        cfc_box.config(state="readonly" if is_cfc else "disabled")
        butter_cut_entry.config(state="normal" if is_butter else "disabled")
        butter_order_entry.config(state="normal" if is_butter else "disabled")
        savgol_win_entry.config(state="normal" if is_savgol else "disabled")
        savgol_poly_entry.config(state="normal" if is_savgol else "disabled")
        ma_win_entry.config(state="normal" if is_ma else "disabled")

        # immediate preview update if possible
        try:
            update_preview()
        except Exception:
            pass

    # Bindings: variable trace + combobox selection
    filter_method_var.trace_add("write", on_filter_method_change_var)
    filter_method_box.bind("<<ComboboxSelected>>", lambda e: on_filter_method_change_var())

    # Bind mapping/unit/combo events to update preview
    for w in list(dropdowns.values()) + list(unit_boxes.values()) + [preview_x, preview_y]:
        w.bind("<<ComboboxSelected>>", lambda e: update_preview())
    autoscale_var.trace_add("write", lambda *args: update_preview())

    # default populate mapping choices then draw preview after a short delay
    for i, (sig, _) in enumerate(signals):
        try:
            dropdowns[sig].set(defaults_order[i])
        except Exception:
            pass
    root.after(150, update_preview)

    # Confirm handler: validate and return values
    def confirm():
        # required mappings
        for sig, _ in signals:
            sel = dropdowns[sig].get()
            if not sel:
                messagebox.showerror("Missing", f"Please select column for {sig}")
                return
            mapping[sig] = sel
            units[sig] = unit_boxes[sig].get()

        # collect filter options
        method_name = filter_method_var.get().lower()
        if method_name in ("", "none"):
            filter_opts["method"] = "none"
        elif method_name == "cfc":
            filter_opts["method"] = "cfc"
        elif "butter" in method_name:
            filter_opts["method"] = "butter"
        elif "savgol" in method_name or "savitz" in method_name:
            filter_opts["method"] = "savgol"
        elif "moving" in method_name or "ma" in method_name:
            filter_opts["method"] = "ma"
        else:
            filter_opts["method"] = "none"

        filter_opts["cfc_name"] = cfc_var.get()
        # parse numeric params safely
        try:
            filter_opts["butter_cutoff"] = float(butter_cut_entry.get())
        except Exception:
            pass
        try:
            filter_opts["butter_order"] = int(butter_order_entry.get())
        except Exception:
            pass
        try:
            filter_opts["savgol_win"] = int(savgol_win_entry.get())
        except Exception:
            pass
        try:
            filter_opts["savgol_poly"] = int(savgol_poly_entry.get())
        except Exception:
            pass
        try:
            filter_opts["ma_win"] = int(ma_win_entry.get())
        except Exception:
            pass

        filter_opts["use_filtered_for_metrics"] = bool(use_filtered_var.get())

        root.destroy()

    # Buttons: Update preview and Confirm
    btn_frame = tk.Frame(form_frame)
    btn_frame.grid(row=21, column=0, columnspan=4, pady=8)
    tk.Button(btn_frame, text="Update preview", command=update_preview, width=16).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Confirm", command=confirm, width=16).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", command=root.destroy, width=10).pack(side="left", padx=6)

    # Make dialog modal
    root.grab_set()
    root.wait_window()

    return mapping, units, filter_opts

# ---------- Core calculations ----------
def compute_hic(a_g: np.ndarray, t_s: np.ndarray, max_win_s: float) -> Tuple[float, float, float]:
    n = len(t_s)
    hic_best = 0.0; t1_best = t_s[0] if n>0 else 0.0; t2_best = t_s[0] if n>0 else 0.0
    if n < 2:
        return 0.0, t1_best, t2_best
    a_mid = 0.5 * (a_g[:-1] + a_g[1:])
    dt = np.diff(t_s)
    cum_int = np.concatenate(([0.0], np.cumsum(a_mid * dt)))
    j = 0
    for i in range(n-1):
        while j < n-1 and (t_s[j] - t_s[i]) <= max_win_s:
            j += 1
        if j-1 > i:
            t1 = t_s[i]; t2 = t_s[j-1]; dur = t2 - t1
            if dur <= 0: continue
            integral = cum_int[j-1] - cum_int[i]
            avg_a = integral / dur
            hic_val = dur * (avg_a ** 2.5)
            if hic_val > hic_best:
                hic_best = hic_val; t1_best = t1; t2_best = t2
    return float(hic_best), float(t1_best), float(t2_best)


def compute_GSI(t_s: np.ndarray, a_g: np.ndarray) -> float:
    # Uses trapezoid (scipy.integrate.trapezoid if available)
    try:
        return float(integrate.trapezoid(np.abs(a_g) ** 2.5, t_s))
    except Exception:
        return float(np.trapz(np.abs(a_g) ** 2.5, t_s))


def compute_BrIC(wx_rad_peak: float, wy_rad_peak: float, wz_rad_peak: float, wcx=W_CX, wcy=W_CY, wcz=W_CZ) -> float:
    return float(math.sqrt((wx_rad_peak / wcx) ** 2 + (wy_rad_peak / wcy) ** 2 + (wz_rad_peak / wcz) ** 2))


def integrate_translation(t: np.ndarray, ax_m_s2: np.ndarray, ay_m_s2: np.ndarray, az_m_s2: np.ndarray, method: str="trapezoid"):
    if method == "trapezoid":
        vx = _cumtrapz(ax_m_s2, t, initial=0.0)
        vy = _cumtrapz(ay_m_s2, t, initial=0.0)
        vz = _cumtrapz(az_m_s2, t, initial=0.0)
        x = _cumtrapz(vx, t, initial=0.0)
        y = _cumtrapz(vy, t, initial=0.0)
        z = _cumtrapz(vz, t, initial=0.0)
    else:
        dt = np.mean(np.diff(t)) if len(t)>1 else 0.01
        vx = np.cumsum(ax_m_s2 * dt)
        vy = np.cumsum(ay_m_s2 * dt)
        vz = np.cumsum(az_m_s2 * dt)
        x = np.cumsum(vx * dt) - np.cumsum(vx * dt)[0]
        y = np.cumsum(vy * dt) - np.cumsum(vy * dt)[0]
        z = np.cumsum(vz * dt) - np.cumsum(vz * dt)[0]
    return x, y, z, vx, vy, vz


def integrate_rotation(t: np.ndarray, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> R:
    n = len(t)
    rots = [R.identity()]
    for i in range(1, n):
        omega = np.array([wx[i], wy[i], wz[i]])
        dt_i = t[i] - t[i-1]
        rots.append(rots[-1] * R.from_rotvec(omega * dt_i))
    return R.from_quat([r.as_quat() for r in rots])


def compute_neck_loads_and_nij(t_s: np.ndarray, ax_g: np.ndarray, ay_g: np.ndarray, az_g: np.ndarray, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray):
    ax = ax_g * G2MSS; ay = ay_g * G2MSS; az = az_g * G2MSS
    a_cg = np.vstack([ax, ay, az]).T
    omega = np.vstack([wx, wy, wz]).T
    alpha = np.vstack([np.gradient(wx, t_s), np.gradient(wy, t_s), np.gradient(wz, t_s)]).T
    I = I_HEAD
    Iomega = omega @ I
    gyro = np.cross(omega, Iomega)
    Ialpha = alpha @ I
    M_cg = Ialpha + gyro
    F = M_HEAD * a_cg
    r = R_OC_TO_CG
    M_offset = np.cross(r, F)
    M = M_cg + M_offset
    Fz = F[:, 2]; My = M[:, 1]
    Nij_A = (Fz / FZC_TENSION) + (np.maximum(My, 0.0) / MYC_FLEXION)
    Nij_B = (Fz / FZC_TENSION) + (np.maximum(-My, 0.0) / MYC_EXTENSION)
    Nij_C = (Fz / FZC_COMPRESSION) + (np.maximum(My, 0.0) / MYC_FLEXION)
    Nij_D = (Fz / FZC_COMPRESSION) + (np.maximum(-My, 0.0) / MYC_EXTENSION)
    Nij = np.vstack([Nij_A, Nij_B, Nij_C, Nij_D]).max(axis=0)
    return F, M, Nij


# ---------- AIS helpers ----------
def p_ais3p_hic(hic: float) -> float:
    if hic <= 0: return 0.0
    return 1.0 / (1.0 + math.exp(3.39 + 200.0 / hic - 0.00372 * hic))

def p_ais4p_hic(hic: float) -> float:
    z = (hic - 1434.0) / 430.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def p_ais3p_bric(bric: float) -> float:
    if bric <= 0: return 0.0
    return 1.0 - math.exp(- (bric / 0.947) ** 2.94)

def p_ais4p_bric(bric: float) -> float:
    if bric <= 0: return 0.0
    return 1.0 - math.exp(- (bric / 1.024) ** 2.94)

def discrete_ais_from_probs(p3: float, p4: float) -> str:
    if p4 >= 0.50: return "4"
    if p3 >= 0.50: return "3"
    if p4 >= 0.20: return "3 (borderline 4)"
    if p3 >= 0.20: return "2–3 (borderline 3)"
    return "≤2"

def ais_rank(label: str) -> int:
    order = {"≤2": 0, "2–3 (borderline 3)": 1, "3": 2, "3 (borderline 4)": 3, "4": 4}
    return order.get(label, 0)

def map_value_to_AIS(val: float, breaks: list) -> int:
    for i in range(len(breaks)-1):
        if breaks[i] <= val < breaks[i+1]:
            return i
    return max(0, len(breaks)-2)

#--------------------------------------------------------------------
def draw_multiline_text(ax, lines, x=0.05, y_start=0.96, line_pad=0.006, y_min=0.18):
    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    y = y_start
    for item in lines:
        if y < y_min:
            break

        y -= item.get("gap_before", 0.0)
        txt = ax.text(
            x, y,
            item["text"],
            fontsize=item.get("fontsize", 10),
            fontweight=item.get("weight", "normal"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            wrap=True,
            usetex=False,
            bbox=dict(boxstyle="square,pad=0", facecolor="none", edgecolor="none")
        )
        txt.set_wrap(True)
            

        fig.canvas.draw_idle()
        bbox = txt.get_window_extent(renderer=renderer)

        inv = ax.transAxes.inverted()
        _, h_axes = inv.transform((0, bbox.height)) - inv.transform((0, 0))

        y -= (h_axes + line_pad)

    return y  # return last y-position


# ---------- Report generation ----------
def generate_report(time: np.ndarray, ax_g: np.ndarray, ay_g: np.ndarray, az_g: np.ndarray,
                    wx: np.ndarray, wy: np.ndarray, wz: np.ndarray,
                    out_pdf: str, in_file_name: str = "", processing_metadata: dict = None) -> Dict[str, Any]:
    """
    Generate multi-page PDF report and return metrics dict.
    """
    t_s = time - time[0]
    dt = np.diff(t_s)
    dt = dt[dt > 0]                          # remove zeros
    if len(dt) == 0:
        fs_est = np.nan
    else:
        fs_est = 1.0 / np.median(dt)

    # angular to degrees for reporting plots
    wx_deg, wy_deg, wz_deg = np.rad2deg(wx), np.rad2deg(wy), np.rad2deg(wz)
    a_res_g = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
    w_res_deg = np.sqrt(wx_deg**2 + wy_deg**2 + wz_deg**2)

    # angular acceleration
    alpha_x = np.gradient(wx, t_s); alpha_y = np.gradient(wy, t_s); alpha_z = np.gradient(wz, t_s)
    alpha_res_deg = np.rad2deg(np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2))

    # metrics
    hic15, h15_t1, h15_t2 = compute_hic(a_res_g, t_s, 0.015)
    hic36, h36_t1, h36_t2 = compute_hic(a_res_g, t_s, 0.036)
    bric_val = compute_BrIC(np.max(np.abs(wx)), np.max(np.abs(wy)), np.max(np.abs(wz)))
    peak_a_g = float(np.max(a_res_g)) if len(a_res_g)>0 else 0.0
    peak_w_deg = float(np.max(w_res_deg)) if len(w_res_deg)>0 else 0.0
    peak_alpha_deg = float(np.max(alpha_res_deg)) if len(alpha_res_deg)>0 else 0.0
    SI_val = compute_GSI(t_s, a_res_g) if len(a_res_g)>0 else 0.0

    # WSTC probability mapping (logistic-ish)
    try:
        xv = 0.006 * (SI_val - 1000.0)
        if xv >= 0:
            z = math.exp(-xv); wstc_prob = 1.0 / (1.0 + z)
        else:
            z = math.exp(xv); wstc_prob = z / (1.0 + z)
    except Exception:
        wstc_prob = 0.0

    # neck loads
    F_neck, M_neck, Nij = compute_neck_loads_and_nij(t_s, ax_g, ay_g, az_g, wx, wy, wz)
    Fx_peak, Fy_peak, Fz_peak = np.max(np.abs(F_neck), axis=0)
    Mx_peak, My_peak, Mz_peak = np.max(np.abs(M_neck), axis=0)
    Nij_peak = float(np.max(Nij)) if len(Nij)>0 else 0.0

    # AIS mapping
    hic_ais = map_value_to_AIS(hic15, [0, 250, 500, 1000, 1500, 2000])
    bric_ais = map_value_to_AIS(bric_val, [0.0, 0.25, 0.35, 0.5, 0.75, 1.0])
    pa_ais = map_value_to_AIS(peak_a_g, [0, 50, 75, 100, 150, 200])
    av_ais = map_value_to_AIS(peak_w_deg, [0, 30, 60, 100, 150, 250])
    composite_ais = int(round(np.mean([hic_ais, bric_ais, pa_ais, av_ais])))

    P_AIS3p_HIC = p_ais3p_hic(hic15); P_AIS4p_HIC = p_ais4p_hic(hic15)
    P_AIS3p_BrIC = p_ais3p_bric(bric_val); P_AIS4p_BrIC = p_ais4p_bric(bric_val)
    AIS_HIC = discrete_ais_from_probs(P_AIS3p_HIC, P_AIS4p_HIC)
    AIS_BrIC = discrete_ais_from_probs(P_AIS3p_BrIC, P_AIS4p_BrIC)
    AIS_Combined = AIS_HIC if ais_rank(AIS_HIC) >= ais_rank(AIS_BrIC) else AIS_BrIC

    # Build PDF
    pdf = PdfPages(out_pdf)

    # Page 1: Summary table
    fig, axp = plt.subplots(figsize=(8.5, 11)); axp.axis("off")
    table_data = [
        ["Metric", "Value", "Unit", "Risk", "AIS"],
        ["Samples", len(t_s), "", "", ""],
        ["fs", f"{fs_est:.1f}" if not np.isnan(fs_est) else "N/A", "Hz", "", ""],
        ["Peak Linear Accel", f"{peak_a_g:.2f}", "g", risk_label(peak_a_g, PEAK_ACCEL_LOW, PEAK_ACCEL_MOD, PEAK_ACCEL_HIGH), pa_ais],
        ["HIC15", f"{hic15:.1f}", "", risk_label(hic15, HIC_LOW, HIC_MOD, HIC_HIGH), hic_ais],
        ["HIC36", f"{hic36:.1f}", "", risk_label(hic36, HIC_LOW, HIC_MOD, HIC_HIGH), ""],
        ["BrIC", f"{bric_val:.3f}", "", risk_label(bric_val, BRIC_LOW, BRIC_MOD, BRIC_HIGH), bric_ais],
        ["Peak Angular Vel", f"{peak_w_deg:.1f}", "deg/s", risk_label(peak_w_deg, PEAK_ANGVEL_LOW, PEAK_ANGVEL_MOD, PEAK_ANGVEL_HIGH), av_ais],
        ["Peak Angular Accel", f"{peak_alpha_deg:.1f}", "deg/s²", "", ""],
        ["Gadd SI", f"{SI_val:.1f}", "SI units", "", ""],
        ["WSTC Prob", f"{(wstc_prob*100):.1f}%", "", "", ""],
        ["Composite AIS", composite_ais, "", "", ""],
        ["Peak |Fz| (axial)", f"{abs(Fz_peak):.0f}", "N", "", ""],
        ["Peak |My| (sagittal)", f"{abs(My_peak):.1f}", "Nm", "", ""],
        ["Nij (peak)", f"{Nij_peak:.2f}", "(limit 1.0)", "High" if Nij_peak>=1.0 else ("Moderate" if Nij_peak>=0.5 else "Low"), ""],
    ]
    table = axp.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.28,0.18,0.15,0.2,0.1])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.5)
    # colorize risk column
    for i, row in enumerate(table_data):
        if i==0: continue
        risk = row[3]
        if risk:
            table[(i,3)].set_facecolor(risk_color(risk))
    axp.set_title(f"Head Injury Report\nFile: {in_file_name}\nDate: {datetime.date.today()}", fontsize=12, weight="bold")
    pdf.savefig(fig); plt.close()

    # Methods & processing (fixed, mathtext-safe)
    fig, axm = plt.subplots(figsize=(8.5, 11))
    axm.axis("off")
    
    methods_lines = [
        {"text": "Methods & Definitions", "fontsize": 13, "weight": "bold"},
    
        {"text": "Head Injury Criterion (HIC)", "fontsize": 11, "weight": "bold", "gap_before": 0.012},
        {"text": r"$\mathrm{HIC} = \max_{t_1,t_2}(t_2-t_1)\left(\frac{1}{t_2-t_1}\int_{t_1}^{t_2} a(t)\,dt\right)^{2.5}$",
         "fontsize": 10},
        {"text": "Evaluated over windows of 15 ms (HIC15) and 36 ms (HIC36).", "fontsize": 9},
    
        {"text": "Brain Injury Criterion (BrIC)", "fontsize": 11, "weight": "bold", "gap_before": 0.012},
        {"text": r"$\mathrm{BrIC}=\sqrt{\left(\frac{\omega_x}{W_{cx}}\right)^2+\left(\frac{\omega_y}{W_{cy}}\right)^2+\left(\frac{\omega_z}{W_{cz}}\right)^2}$",
         "fontsize": 10},
        {"text": rf"with $W_{{cx}}={W_CX:.2f}$, $W_{{cy}}={W_CY:.2f}$, $W_{{cz}}={W_CZ:.2f}$.", "fontsize": 9},
    
        {"text": "Gadd Severity Index (SI)", "fontsize": 11, "weight": "bold", "gap_before": 0.012},
        {"text": r"$\mathrm{SI}=\int a(t)^{2.5}\,dt$", "fontsize": 10},
    
        {"text": "Angular acceleration is computed from filtered $\omega(t)$ with smoothing prior to differentiation.", "fontsize": 9},
        {"text": "Neck loads and Nij are estimated using Newton–Euler rigid-body dynamics with Hybrid-III head properties.", "fontsize": 9},
    
        {"text": "Processing includes user-selected filtering (Butterworth, CFC, Savitzky–Golay, or Moving Average).", "fontsize": 9},
        {"text": "Numerical integration uses the trapezoidal rule with optional detrending.", "fontsize": 9},

    ]
    
    
    draw_multiline_text(axm, methods_lines)


    # ---- Footer: processing metadata  ----
    if processing_metadata:
        footer_y = 0.14
        footer_lines = [
            f"Sampling rate (estimated): {processing_metadata.get('fs_est', fs_est):.1f} Hz",
            f"Filter applied: {processing_metadata.get('filter',{}).get('method','none').upper()}",
            f"Integration: {processing_metadata.get('integration_method','trapezoidal')}",
            f"Detrending: {processing_metadata.get('detrend', False)}",
            f"Metrics computed on filtered signals: {processing_metadata.get('use_filtered_for_metrics', False)}",
        ]
    
        # light separator line
        axm.plot([0.04, 0.96], [footer_y + 0.02, footer_y + 0.02],
                 transform=axm.transAxes, color="0.7", linewidth=0.8)
    
        for ln in footer_lines:
            axm.text(0.05, footer_y, ln, fontsize=9, va="top")
            footer_y -= 0.028
# To show the equations please un comment the following line -------------

#    pdf.savefig(fig); plt.close(fig)

    # Time-series pages: linear accel, angular vel, angular accel, HIC windows, BrIC, SI curve
    style_kwargs = dict(linewidth=1.2)
    # Linear accel time series
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(t_s*1000, ax_g, label='ax', **style_kwargs)
    ax.plot(t_s*1000, ay_g, label='ay', **style_kwargs)
    ax.plot(t_s*1000, az_g, label='az', **style_kwargs)
    ax.plot(t_s*1000, a_res_g, label='Resultant', color='k', linewidth=1.6)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("g"); ax.set_title("Linear Acceleration")
    ax.legend(); ax.grid(True)
    pdf.savefig(fig); plt.close()

    # Angular velocity
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(t_s*1000, wx_deg, label='wx', **style_kwargs)
    ax.plot(t_s*1000, wy_deg, label='wy', **style_kwargs)
    ax.plot(t_s*1000, wz_deg, label='wz', **style_kwargs)
    ax.plot(t_s*1000, w_res_deg, label='Resultant', color='k', linewidth=1.6)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("deg/s"); ax.set_title("Angular Velocity")
    ax.legend(); ax.grid(True)
    pdf.savefig(fig); plt.close()

    # Angular accel
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(t_s*1000, alpha_res_deg, color='purple', linewidth=1.4)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("deg/s²"); ax.set_title("Angular Acceleration (derived)")
    ax.grid(True)
    pdf.savefig(fig); plt.close()

    # HIC windows
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(t_s*1000, a_res_g, color='k', linewidth=1.2)
    ax.axvspan(h15_t1*1000, h15_t2*1000, color='orange', alpha=0.3, label="HIC15")
    ax.axvspan(h36_t1*1000, h36_t2*1000, color='purple', alpha=0.2, label="HIC36")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("g"); ax.set_title("HIC Windows")
    ax.legend(); ax.grid(True)
    pdf.savefig(fig); plt.close()

    # BrIC over time
    fig, ax = plt.subplots(figsize=(11,4))
    bric_curve = np.sqrt((np.abs(wx)/W_CX)**2 + (np.abs(wy)/W_CY)**2 + (np.abs(wz)/W_CZ)**2)
    ax.plot(t_s*1000, bric_curve, linewidth=1.2)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("BrIC"); ax.set_title("BrIC Over Time")
    ax.grid(True)
    pdf.savefig(fig); plt.close()

    # SI probability / WSTC curve
    fig, ax = plt.subplots(figsize=(11,4))
    si_range = np.linspace(0, max(SI_val*1.2, 2000), 300)
    si_prob = [( (math.exp(-0.006*(s-1000.0))) / (1+math.exp(-0.006*(s-1000.0))) ) * 100 for s in si_range]
    ax.plot(si_range, si_prob)
    ax.axvline(SI_val, color='r', linestyle='--', label=f"Observed SI={SI_val:.1f}")
    ax.set_xlabel("Gadd SI"); ax.set_ylabel("Probability (%)"); ax.set_title("WSTC Probability Curve")
    ax.legend(); ax.grid(True)
    pdf.savefig(fig); plt.close()

    # Neck forces & moments & Nij time histories
    fig, axp = plt.subplots(figsize=(11,6))
    axp.plot(t_s, F_neck[:,0], label="Fx (N)", **style_kwargs)
    axp.plot(t_s, F_neck[:,1], label="Fy (N)", **style_kwargs)
    axp.plot(t_s, F_neck[:,2], label="Fz (N)", **style_kwargs)
    axp.set_xlabel("Time (s)"); axp.set_ylabel("Force (N)"); axp.set_title("Estimated Neck Reaction Forces at OC")
    axp.legend(); axp.grid(True)
    pdf.savefig(fig); plt.close()

    fig, axp = plt.subplots(figsize=(11,6))
    axp.plot(t_s, M_neck[:,0], label="Mx (Nm)", **style_kwargs)
    axp.plot(t_s, M_neck[:,1], label="My (Nm)", **style_kwargs)
    axp.plot(t_s, M_neck[:,2], label="Mz (Nm)", **style_kwargs)
    axp.set_xlabel("Time (s)"); axp.set_ylabel("Moment (Nm)"); axp.set_title("Estimated Neck Moments at OC")
    axp.legend(); axp.grid(True)
    pdf.savefig(fig); plt.close()

    fig, axp = plt.subplots(figsize=(11,6))
    axp.plot(t_s, Nij, label="Nij", **style_kwargs)
    axp.axhline(1.0, linestyle="--", label="Limit = 1.0")
    axp.set_xlabel("Time (s)"); axp.set_ylabel("Nij (-)"); axp.set_title(f"Nij Time History (peak = {Nij_peak:.2f})")
    axp.legend(); axp.grid(True)
    pdf.savefig(fig); plt.close()

    # AIS equations summary & color bar
    fig, axq = plt.subplots(figsize=(8.5,11)); axq.axis("off")
    lines = [
        "Equation-based AIS Estimates",
        "",
        f"HIC15 = {hic15:.1f}, BrIC = {bric_val:.3f}",
        f"P(AIS3+) from HIC15: {P_AIS3p_HIC*100:.1f}%",
        f"P(AIS4+) from HIC15: {P_AIS4p_HIC*100:.1f}%",
        f"P(AIS3+) from BrIC:  {P_AIS3p_BrIC*100:.1f}%",
        f"P(AIS4+) from BrIC:  {P_AIS4p_BrIC*100:.1f}%",
        f"Calculated AIS (HIC): {AIS_HIC}",
        f"Calculated AIS (BrIC): {AIS_BrIC}",
        f"Final Combined AIS: {AIS_Combined}"
    ]
    y = 0.95
    for ln in lines:
        axq.text(0.05, y, ln, fontsize=10); y -= 0.04
    pdf.savefig(fig); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    cats = ["AIS3+ (HIC)", "AIS4+ (HIC)", "AIS3+ (BrIC)", "AIS4+ (BrIC)"]
    vals = [P_AIS3p_HIC*100, P_AIS4p_HIC*100, P_AIS3p_BrIC*100, P_AIS4p_BrIC*100]
    def color_for_prob(p):
        if p < 20: return "#c8e6c9"
        if p < 40: return "#fff9c4"
        if p < 70: return "#ffcc80"
        return "#ef9a9a"
    colors = [color_for_prob(v) for v in vals]
    ax.bar(cats, vals, color=colors)
    ax.set_ylim(0,100); ax.set_ylabel("Probability (%)"); ax.set_title("Equation-based AIS Probabilities")
    for i, v in enumerate(vals):
        ax.text(i, v+1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    pdf.savefig(fig); plt.close()

    pdf.close()
    logger.info("Saved PDF report: %s", out_pdf)

    metrics = {
        "HIC15": hic15, "HIC36": hic36, "GSI": SI_val, "BrIC": bric_val,
        "peak_acc_g": peak_a_g, "peak_omega_deg_s": peak_w_deg, "Nij_peak": Nij_peak
    }
    return metrics


# ---------- Main application ----------
def main():
    root = tk.Tk(); root.withdraw()
    in_file = filedialog.askopenfilename(title="Select head impact CSV/TXT file", filetypes=[("CSV/TXT","*.csv *.txt"), ("All files","*.*")])
    if not in_file:
        messagebox.showerror("No file", "No input file selected. Exiting."); return

    header_row = ask_header_row_with_preview(in_file)
    if header_row is None:
        messagebox.showerror("Header", "No header row selected. Exiting."); return

    try:
        df = pd.read_csv(in_file, header=header_row)
    except Exception as e:
        messagebox.showerror("Read error", f"Failed to read CSV: {e}"); return

    mapping, units, filter_opts = get_column_mapping_and_units(df)

    # extract arrays and convert units to internal representation:
    try:
        time = pd.to_numeric(df[mapping["Time"]].values).astype(float)
    except Exception as e:
        messagebox.showerror("Column error", f"Time column extraction failed: {e}"); return

    if units["Time"] == "milliseconds":
        time = time * 1e-3
    elif units["Time"] == "microseconds":
        time = time * 1e-6
    time = time.astype(float)

    def get_acc(sig):
        arr = pd.to_numeric(df[mapping[sig]].values, errors='coerce').astype(float)
        if units[sig] == "m/s²":
            arr = arr / G2MSS  # convert to g for our metric functions
        return arr

    ax = get_acc("ax"); ay = get_acc("ay"); az = get_acc("az")

    def get_ang(sig):
        arr = pd.to_numeric(df[mapping[sig]].values, errors='coerce').astype(float)
        if units[sig] == "deg/s":
            arr = np.deg2rad(arr)
        return arr

    wx = get_ang("wx"); wy = get_ang("wy"); wz = get_ang("wz")

    # ask integration/detrend preferences
    integration_method = simpledialog.askstring("Integration", "Integration method: 'trapezoid' (recommended) or 'cumsum' (visual)", initialvalue="trapezoid")
    if integration_method not in ("trapezoid","cumsum"): integration_method = "trapezoid"
    do_detrend = messagebox.askyesno("Detrend", "Remove mean bias from accelerations before integration? (recommended)")

    if do_detrend:
        ax_d, ay_d, az_d = ax - np.nanmean(ax), ay - np.nanmean(ay), az - np.nanmean(az)
    else:
        ax_d, ay_d, az_d = ax, ay, az

    # estimate sampling frequency (safe)
    try:
        t_s = time - time[0]
        fs_est = 1.0 / np.median(np.diff(t_s)) if len(t_s) > 1 else float("nan")
    except Exception:
        fs_est = float("nan")

    # Ensure filter_opts exists
    if not filter_opts:
        filter_opts = {"method":"none", "use_filtered_for_metrics": False}

    # quick sampling check for CFC presets (warn)
    if filter_opts.get("method") == "cfc":
        cutoff = CFC_PRESETS.get(filter_opts.get("cfc_name","CFC180"), None)
        if np.isnan(fs_est) or (cutoff is not None and fs_est < 6.0 * cutoff):
            messagebox.showwarning("Sampling warning",
                f"Sampling ~{fs_est:.1f} Hz may be low for {filter_opts.get('cfc_name')} (cutoff {cutoff} Hz). Filter may distort results.")

    # keep raw copies and compute filtered copies
    ax_raw, ay_raw, az_raw = ax_d.copy(), ay_d.copy(), az_d.copy()
    wx_raw, wy_raw, wz_raw = wx.copy(), wy.copy(), wz.copy()

    method = filter_opts.get("method", "none")
    if method != "none":
        if method == "cfc":
            ax_f = smooth_signal_wrapper(ax_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
            ay_f = smooth_signal_wrapper(ay_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
            az_f = smooth_signal_wrapper(az_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
            wx_f = smooth_signal_wrapper(wx_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
            wy_f = smooth_signal_wrapper(wy_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
            wz_f = smooth_signal_wrapper(wz_raw, fs_est, method="cfc", cfc_name=filter_opts.get("cfc_name","CFC180"))
        elif method == "butter":
            ax_f = smooth_signal_wrapper(ax_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
            ay_f = smooth_signal_wrapper(ay_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
            az_f = smooth_signal_wrapper(az_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
            wx_f = smooth_signal_wrapper(wx_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
            wy_f = smooth_signal_wrapper(wy_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
            wz_f = smooth_signal_wrapper(wz_raw, fs_est, method="butter", cutoff=filter_opts.get("butter_cutoff",300.0), order=filter_opts.get("butter_order",4))
        elif method == "savgol":
            ax_f = smooth_signal_wrapper(ax_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
            ay_f = smooth_signal_wrapper(ay_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
            az_f = smooth_signal_wrapper(az_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
            wx_f = smooth_signal_wrapper(wx_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
            wy_f = smooth_signal_wrapper(wy_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
            wz_f = smooth_signal_wrapper(wz_raw, fs_est, method="savgol", window_length=filter_opts.get("savgol_win",21), polyorder=filter_opts.get("savgol_poly",2))
        else:
            ax_f = smooth_signal_wrapper(ax_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
            ay_f = smooth_signal_wrapper(ay_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
            az_f = smooth_signal_wrapper(az_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
            wx_f = smooth_signal_wrapper(wx_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
            wy_f = smooth_signal_wrapper(wy_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
            wz_f = smooth_signal_wrapper(wz_raw, fs_est, method="ma", ma_win=filter_opts.get("ma_win",5))
    else:
        ax_f, ay_f, az_f = ax_raw.copy(), ay_raw.copy(), az_raw.copy()
        wx_f, wy_f, wz_f = wx_raw.copy(), wy_raw.copy(), wz_raw.copy()

    # choose signals for metrics & plots
    use_filtered = bool(filter_opts.get("use_filtered_for_metrics", False))
    ax_for_metrics, ay_for_metrics, az_for_metrics = (ax_f, ay_f, az_f) if use_filtered else (ax_raw, ay_raw, az_raw)
    wx_for_metrics, wy_for_metrics, wz_for_metrics = (wx_f, wy_f, wz_f) if use_filtered else (wx_raw, wy_raw, wz_raw)

    # integrate (positions and rotations) for visualization only (no animation saved)
    try:
        ax_m = ax_for_metrics * G2MSS; ay_m = ay_for_metrics * G2MSS; az_m = az_for_metrics * G2MSS
        x, y, z, vx, vy, vz = integrate_translation(time, ax_m, ay_m, az_m, method=integration_method)
        rotations = integrate_rotation(time, wx_for_metrics, wy_for_metrics, wz_for_metrics)
    except Exception as e:
        messagebox.showerror("Integration error", f"Integration failed: {e}"); return

    # prepare processing metadata
    processing_metadata = {
        "fs_est": float(fs_est) if not np.isnan(fs_est) else None,
        "filter": {
            "method": filter_opts.get("method", "none"),
            "params": {k:v for k,v in filter_opts.items() if k not in ("method",)}
        },
        "integration_method": integration_method,
        "detrend": bool(do_detrend),
        "units": units,
        "use_filtered_for_metrics": use_filtered
    }

    # compute report (metrics) using chosen arrays
    try:
        out_pdf = filedialog.asksaveasfilename(title="Save injury report as...", defaultextension=".pdf", filetypes=[("PDF files","*.pdf")])
        if not out_pdf:
            messagebox.showerror("No output", "No output file selected. Exiting."); return
        metrics = generate_report(time, ax_for_metrics, ay_for_metrics, az_for_metrics,
                                  wx_for_metrics, wy_for_metrics, wz_for_metrics,
                                  out_pdf = out_pdf,
                                  in_file_name = os.path.basename(in_file),
                                  processing_metadata = processing_metadata)
    except Exception as e:
        logger.exception("Report generation failed"); messagebox.showerror("Error", f"Report generation failed: {e}"); return

    # Save metrics JSON next to the PDF
    metrics_path = os.path.splitext(os.path.basename(out_pdf))[0] + "_metrics.json"
    metrics["processing"] = processing_metadata
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics JSON: %s", metrics_path)
    except Exception:
        logger.warning("Failed to save metrics JSON")

    messagebox.showinfo("Success", f"Report saved: {out_pdf}\nMetrics JSON: {metrics_path}")

    root.destroy()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Output tab — output directory, file options, and post-processing smoother."""
import tkinter as tk
from tkinter import ttk
from beamontarget.gui.gui_widgets import make_card, browse_directory

# SMOOTHING_METHOD config value -> label shown in the GUI
_METHOD_LABELS = {
    "radius": "Radius average",
    "physical": "Physical (energy-conserving)",
}
_METHOD_KEYS = {v: k for k, v in _METHOD_LABELS.items()}


class OutputTab(ttk.Frame):
    """Output paths, save options, and batch smoother configuration."""

    def __init__(self, parent, cfg, colours, *, open_extract_fn):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._open_extract_dialog = open_extract_fn
        self._build()

    def _build(self):
        # --- Paths card ---
        card = make_card(self, "Output Settings", pady=(12, 10))
        ttk.Label(card, text="Output directory:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_outdir = tk.StringVar(value=self.cfg.get("DETAILED_OUTPUT_DIR", "OUTPUT"))
        ttk.Entry(card, textvariable=self.var_outdir, width=30).grid(
            row=0, column=1, sticky="we", padx=(8, 4))
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=lambda: browse_directory(self.var_outdir)).grid(
            row=0, column=2)

        ttk.Label(card, text="Summary CSV filename:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        self.var_summary = tk.StringVar(value=self.cfg.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
        ttk.Entry(card, textvariable=self.var_summary, width=40).grid(
            row=1, column=1, sticky="we", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- File options card ---
        opts_card = make_card(self, "Save Options")
        checkboxes = [
            ("SAVE_PARAVIEW_FILES", "Save ParaView (.vtp) files"),
            ("SAVE_CSV_REPORTS", "Save CSV reports"),
        ]
        for key, label in checkboxes:
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            ttk.Checkbutton(opts_card, text=label, variable=v,
                             style="Card.TCheckbutton").pack(anchor="w", pady=2)
            setattr(self, f"var_{key}", v)

        ttk.Button(opts_card, text="Extract results data...",
                    style="Secondary.TButton",
                    command=self._open_extract_dialog).pack(
                        anchor="w", pady=(8, 0))

        # --- Post-Processing Smoother card ---
        sm_card = make_card(self, "Post-Processing Smoother")
        self.var_smoother = tk.BooleanVar(value=self.cfg.get("RUN_SMOOTHER_AFTER_SIM", False))
        ttk.Checkbutton(sm_card, text="Run batch smoother after simulation",
                         variable=self.var_smoother,
                         style="Card.TCheckbutton").pack(anchor="w", pady=(0, 8))

        method_row = ttk.Frame(sm_card, style="Card.TFrame")
        method_row.pack(fill="x", pady=(0, 4))
        ttk.Label(method_row, text="Method:", style="Card.TLabel").pack(side="left")
        self.var_sm_method = tk.StringVar(
            value=_METHOD_LABELS.get(self.cfg.get("SMOOTHING_METHOD", "radius"),
                                     _METHOD_LABELS["radius"]))
        ttk.Combobox(method_row, textvariable=self.var_sm_method,
                     values=list(_METHOD_LABELS.values()),
                     state="readonly", width=30).pack(side="left", padx=(8, 0))

        # --- Physical method parameters ---
        self._phys_grid = ttk.Frame(sm_card, style="Card.TFrame")
        ttk.Label(self._phys_grid, text="Target rel. error (0.05 = 5 %):",
                  style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.var_sm_target_err = tk.DoubleVar(
            value=self.cfg.get("SMOOTHING_TARGET_REL_ERROR", 0.05))
        ttk.Entry(self._phys_grid, textvariable=self.var_sm_target_err, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(self._phys_grid, text="Max bandwidth (m):",
                  style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.var_sm_max_bw = tk.DoubleVar(
            value=self.cfg.get("SMOOTHING_MAX_BANDWIDTH_M", 0.04))
        ttk.Entry(self._phys_grid, textvariable=self.var_sm_max_bw, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Label(self._phys_grid,
                  text="Energy-conserving; also writes per-cell error arrays "
                       "(sigma, bias, p95). Slower than the radius method.",
                  style="Card.TLabel", foreground=self._colours.get("dim", "#64748b"),
                  wraplength=520, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # --- Radius method parameters ---
        sm_grid = ttk.Frame(sm_card, style="Card.TFrame")
        self._radius_grid = sm_grid

        def _on_method(*_):
            if self.smoothing_method() == "physical":
                self._radius_grid.pack_forget()
                self._phys_grid.pack(fill="x")
            else:
                self._phys_grid.pack_forget()
                self._radius_grid.pack(fill="x")

        self.var_sm_method.trace_add("write", _on_method)
        _on_method()

        # Row 0: Radius
        ttk.Label(sm_grid, text="Smoothing radius (m):", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_sm_radius = tk.DoubleVar(value=self.cfg.get("SMOOTHING_RADIUS", 0.02))
        ttk.Entry(sm_grid, textvariable=self.var_sm_radius, width=12).grid(
            row=0, column=1, sticky="w", padx=(8, 0))

        # Row 1: Max Cell Area
        ttk.Label(sm_grid, text="Max cell area (m², empty=None):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=5)
        mca = self.cfg.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca = tk.StringVar(value=str(mca) if mca is not None else "")
        ttk.Entry(sm_grid, textvariable=self.var_sm_mca, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0))

        # Row 2: Normal Threshold Angle (NEW)
        ttk.Label(sm_grid, text="Normal threshold (deg):", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=5)
        self.var_sm_angle = tk.DoubleVar(value=self.cfg.get("SMOOTHING_NORMAL_THRESHOLD_DEG", 7.0))
        ttk.Entry(sm_grid, textvariable=self.var_sm_angle, width=12).grid(
            row=2, column=1, sticky="w", padx=(8, 0))

    def smoothing_method(self):
        """Return the selected smoothing method key ("radius" or "physical")."""
        return _METHOD_KEYS.get(self.var_sm_method.get(), "radius")

    def collect(self, d):
        """Write output keys into *d*."""
        d["DETAILED_OUTPUT_DIR"] = self.var_outdir.get()
        d["SAVE_PARAVIEW_FILES"] = self.var_SAVE_PARAVIEW_FILES.get()
        d["SAVE_CSV_REPORTS"] = self.var_SAVE_CSV_REPORTS.get()
        d["SUMMARY_CSV_FILENAME"] = self.var_summary.get()
        d["RUN_SMOOTHER_AFTER_SIM"] = self.var_smoother.get()
        d["SMOOTHING_RADIUS"] = self.var_sm_radius.get()
        d["SMOOTHING_NORMAL_THRESHOLD_DEG"] = self.var_sm_angle.get()
        mca = self.var_sm_mca.get().strip()
        d["SMOOTHING_MAX_CELL_AREA"] = float(mca) if mca else None
        d["SMOOTHING_METHOD"] = self.smoothing_method()
        d["SMOOTHING_TARGET_REL_ERROR"] = self.var_sm_target_err.get()
        d["SMOOTHING_MAX_BANDWIDTH_M"] = self.var_sm_max_bw.get()

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        self.var_outdir.set(c.get("DETAILED_OUTPUT_DIR", "OUTPUT"))
        self.var_SAVE_PARAVIEW_FILES.set(c.get("SAVE_PARAVIEW_FILES", True))
        self.var_SAVE_CSV_REPORTS.set(c.get("SAVE_CSV_REPORTS", False))
        self.var_summary.set(c.get("SUMMARY_CSV_FILENAME", "power_summary_by_object.csv"))
        self.var_smoother.set(c.get("RUN_SMOOTHER_AFTER_SIM", False))
        self.var_sm_radius.set(c.get("SMOOTHING_RADIUS", 0.02))
        self.var_sm_angle.set(c.get("SMOOTHING_NORMAL_THRESHOLD_DEG", 7.0))
        mca = c.get("SMOOTHING_MAX_CELL_AREA")
        self.var_sm_mca.set(str(mca) if mca is not None else "")
        self.var_sm_method.set(_METHOD_LABELS.get(c.get("SMOOTHING_METHOD", "radius"),
                                                  _METHOD_LABELS["radius"]))
        self.var_sm_target_err.set(c.get("SMOOTHING_TARGET_REL_ERROR", 0.05))
        self.var_sm_max_bw.set(c.get("SMOOTHING_MAX_BANDWIDTH_M", 0.04))

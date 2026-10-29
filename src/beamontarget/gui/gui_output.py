# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Output tab — output directory, file options, and post-processing smoother."""
import tkinter as tk
from tkinter import ttk
from beamontarget.gui.gui_widgets import make_card, browse_directory

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

        sm_grid = ttk.Frame(sm_card, style="Card.TFrame")
        sm_grid.pack(fill="x")

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

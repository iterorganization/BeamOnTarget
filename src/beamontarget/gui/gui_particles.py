# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Particles tab — beam source configuration and beamlet file listing."""
import os
import glob
import tkinter as tk
from tkinter import ttk

from beamontarget.gui.gui_widgets import (
    make_card,
    _SCRIPT_DIR,
    resolve_path as _resolve_path,
    browse_directory,
)


class ParticlesTab(ttk.Frame):
    """Beam source parameters and .bl file browser."""

    def __init__(self, parent, cfg, colours, *,
                 view_sources_o3d_fn):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._view_sources_o3d = view_sources_o3d_fn
        self._build()

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self):
        # --- Beam source card ---
        card = make_card(self, "Beam Source", pady=(12, 10))

        row = 0
        ttk.Label(card, text="Beam config directory:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_src_dir = tk.StringVar(value=self.cfg.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIG_EXAMPLE"))
        ttk.Entry(card, textvariable=self.var_src_dir, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 4))
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=lambda: browse_directory(self.var_src_dir)).grid(
            row=row, column=2)

        row += 1
        ttk.Label(card, text="Particles per beamlet:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_npb = tk.IntVar(value=self.cfg.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        ttk.Entry(card, textvariable=self.var_npb, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Beamlet grid radius (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_radius = tk.DoubleVar(value=self.cfg.get("BEAMLET_RADIUS_M", 0.007))
        ttk.Entry(card, textvariable=self.var_radius, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Particle batch size:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_batch = tk.IntVar(value=self.cfg.get("PARTICLE_BATCH_SIZE", 2_500_000))
        ttk.Entry(card, textvariable=self.var_batch, width=12).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- Beam files card ---
        bl_card = make_card(self, "Beam Configuration Files")

        self.bl_listbox = tk.Listbox(bl_card, height=8, width=50,
                                      font=("Segoe UI", 10),
                                      bg="white", fg=self._colours["fg"],
                                      selectbackground=self._colours["accent"],
                                      selectforeground="white",
                                      highlightthickness=0, bd=1,
                                      relief="solid")
        self.bl_listbox.pack(fill="both", expand=True, pady=(0, 6))
        self._refresh_bl_list()

        bl_btn_frm = ttk.Frame(bl_card, style="Card.TFrame")
        bl_btn_frm.pack(fill="x")
        ttk.Button(bl_btn_frm, text="Refresh", style="Secondary.TButton",
                    command=self._refresh_bl_list).pack(side="left")
        ttk.Button(bl_btn_frm, text="View Sources (Open3D)",
                    style="Secondary.TButton",
                    command=self._view_sources_o3d).pack(side="right")

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _refresh_bl_list(self):
        self.bl_listbox.delete(0, "end")
        src = self.var_src_dir.get()
        src_abs = _resolve_path(src) if not os.path.isabs(src) else src
        bl_files = sorted(glob.glob(os.path.join(src_abs, "*.bl")))
        for f in bl_files:
            self.bl_listbox.insert("end", os.path.basename(f))
        if not bl_files:
            self.bl_listbox.insert("end", "(no .bl files found)")

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write particle keys into *d*."""
        d["PARTICLE_SOURCE_DIR"] = self.var_src_dir.get()
        d["NUM_PARTICLES_PER_BEAMLET"] = self.var_npb.get()
        d["BEAMLET_RADIUS_M"] = self.var_radius.get()
        d["PARTICLE_BATCH_SIZE"] = self.var_batch.get()

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        self.var_src_dir.set(c.get("PARTICLE_SOURCE_DIR", "BEAM_CONFIG_EXAMPLE"))
        self.var_npb.set(c.get("NUM_PARTICLES_PER_BEAMLET", 10001))
        self.var_radius.set(c.get("BEAMLET_RADIUS_M", 0.007))
        self.var_batch.set(c.get("PARTICLE_BATCH_SIZE", 2_500_000))
        self._refresh_bl_list()


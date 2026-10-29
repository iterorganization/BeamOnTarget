# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Fields tab — per-component EM field configuration.

Provides :class:`FieldsTab`, a ``ttk.Frame`` that can be embedded in the
main notebook.  It exposes ``collect(d)`` and ``refresh(c)`` so the host
window can serialise / deserialise the field config without reaching into
widget internals.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from beamontarget.gui.gui_widgets import make_card, _SCRIPT_DIR

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
_MODE_LABELS = {"zero": "None", "fixed": "Fixed value",
                "file": "Field file (.fld)",
                "profile_x": "X profile file (x,value)",
                "rid_ey": "ERID (simplified)"}
_MODE_MAP = {v: k for k, v in _MODE_LABELS.items()}   # reverse


def _legacy_to_components(ef):
    """Translate an old-style flat EXTERNAL_FIELD dict into a components dict."""
    ef_type = ef.get("type", "zero")
    comps = {}
    if ef_type == "zero":
        return comps
    bvec = ef.get("magnetic_field_t", [0.0, 0.0, 0.0])
    evec = ef.get("electric_field_vpm", [0.0, 0.0, 0.0])
    for i, key in enumerate(("Bx", "By", "Bz")):
        if bvec[i] != 0:
            comps[key] = {"mode": "fixed", "value": bvec[i]}
    if ef_type in ("rid_segment_y", "rid_piecewise"):
        comps["Ey"] = {"mode": "rid_ey",
                       "v_rid_v": ef.get("v_rid_v", 20e3),
                       "x_min_m": ef.get("x_min_m", 5.4),
                       "x_max_m": ef.get("x_max_m", 7.2)}
    else:
        for i, key in enumerate(("Ex", "Ey", "Ez")):
            if evec[i] != 0:
                comps[key] = {"mode": "fixed", "value": evec[i]}
    return comps


def _parse_components(ef):
    """Return the per-component dict from an EXTERNAL_FIELD config,
    handling both new (``components``) and legacy formats."""
    comps = dict(ef.get("components", {}))
    if not comps:
        comps = _legacy_to_components(ef)
    return comps


# ---------------------------------------------------------------------------
#  FieldsTab
# ---------------------------------------------------------------------------
class FieldsTab(ttk.Frame):
    """Per-component EM field configuration panel."""

    def __init__(self, parent, cfg, *, get_collect_fn=None, **kw):
        super().__init__(parent, **kw)
        self._comp_vars = {}     # key -> {mode, val, file, rid, scale}
        self._comp_widgets = {}  # key -> {val_frame, file_frame, rid_frame, scale_frame}
        self._get_collect = get_collect_fn

        ef = cfg.get("EXTERNAL_FIELD", {})
        comps = _parse_components(ef)

        body = self._build_scroll_body()

        # --- Magnetic Field card ---
        b_card = make_card(body, "Magnetic Field", pady=(12, 10))
        b_modes = ["None", "Fixed value", "Field file (.fld)", "X profile file (x,value)"]
        for i, (label, key) in enumerate([("Bx:", "Bx"), ("By:", "By"), ("Bz:", "Bz")]):
            self._make_comp_row(b_card, i, label, key, b_modes, comps.get(key, {}))
        b_card.columnconfigure(2, weight=1)

        # --- Electric Field card ---
        e_card = make_card(body, "Electric Field")
        e_modes = ["None", "Fixed value", "Field file (.fld)", "X profile file (x,value)"]
        ey_modes = ["None", "Fixed value", "Field file (.fld)", "X profile file (x,value)", "ERID (simplified)"]
        for i, (label, key, modes) in enumerate([
            ("Ex:", "Ex", e_modes), ("Ey:", "Ey", ey_modes), ("Ez:", "Ez", e_modes)
        ]):
            self._make_comp_row(e_card, i, label, key, modes, comps.get(key, {}))
        e_card.columnconfigure(2, weight=1)

        # --- Larmor Radius estimate card ---
        lr_card = make_card(body, "Larmor Radius Estimate")

        param_row = ttk.Frame(lr_card, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 4))
        ttk.Label(param_row, text="Species:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_lr_species = tk.StringVar(value="H")
        ttk.Combobox(param_row, textvariable=self.var_lr_species,
                      values=["H", "D"], state="readonly", width=4).pack(
            side="left", padx=(0, 12))
        ttk.Label(param_row, text="Energy [eV]:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_lr_energy = tk.DoubleVar(value=870e3)
        ttk.Entry(param_row, textvariable=self.var_lr_energy, width=12).pack(
            side="left")

        lr_row = ttk.Frame(lr_card, style="Card.TFrame")
        lr_row.pack(fill="x")
        ttk.Button(lr_row, text="Calculate", style="Secondary.TButton",
                    command=self._calc_larmor_radius).pack(side="left")
        self._var_larmor = tk.StringVar(value="—")
        ttk.Label(lr_row, textvariable=self._var_larmor,
                  style="Card.TLabel", font=("Segoe UI", 10)).pack(
            side="left", padx=(12, 0))

        # --- Field line probe plotter card ---
        probe_card = make_card(body, "Field Line Probe Plotter")

        content_row = ttk.Frame(probe_card, style="Card.TFrame")
        content_row.pack(fill="both", expand=True)

        controls_frm = ttk.Frame(content_row, style="Card.TFrame")
        controls_frm.pack(side="left", fill="y", padx=(0, 10))
        plot_frm = ttk.Frame(content_row, style="Card.TFrame")
        plot_frm.pack(side="left", fill="both", expand=True)

        start_row = ttk.Frame(controls_frm, style="Card.TFrame")
        start_row.pack(fill="x", pady=(0, 4))
        ttk.Label(start_row, text="Start (x0,y0,z0) [m]:", style="Card.TLabel").pack(side="left")
        self.var_probe_start = tk.StringVar(value="0.0,0.0,0.0")
        ttk.Entry(start_row, textvariable=self.var_probe_start, width=28).pack(side="left", padx=(6, 0))

        dir_row = ttk.Frame(controls_frm, style="Card.TFrame")
        dir_row.pack(fill="x", pady=(0, 4))
        ttk.Label(dir_row, text="Direction (dirx,diry,dirz):", style="Card.TLabel").pack(side="left")
        self.var_probe_dir = tk.StringVar(value="1.0,0.0,0.0")
        ttk.Entry(dir_row, textvariable=self.var_probe_dir, width=28).pack(side="left", padx=(6, 0))

        param_row = ttk.Frame(controls_frm, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 6))
        ttk.Label(param_row, text="Step [m]:", style="Card.TLabel").pack(side="left")
        self.var_probe_step = tk.DoubleVar(value=0.01)
        ttk.Entry(param_row, textvariable=self.var_probe_step, width=10).pack(side="left", padx=(6, 10))
        ttk.Label(param_row, text="#Points:", style="Card.TLabel").pack(side="left")
        self.var_probe_npts = tk.IntVar(value=200)
        ttk.Entry(param_row, textvariable=self.var_probe_npts, width=8).pack(side="left", padx=(6, 10))

        select_row = ttk.Frame(controls_frm, style="Card.TFrame")
        select_row.pack(fill="x", pady=(0, 6))
        ttk.Label(select_row, text="Components:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        self._probe_component_vars = {
            "Bx": tk.BooleanVar(value=True),
            "By": tk.BooleanVar(value=False),
            "Bz": tk.BooleanVar(value=False),
            "Ex": tk.BooleanVar(value=False),
            "Ey": tk.BooleanVar(value=False),
            "Ez": tk.BooleanVar(value=False),
        }
        for comp in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            ttk.Checkbutton(select_row, text=comp,
                            variable=self._probe_component_vars[comp]).pack(side="left", padx=(0, 6))

        action_row = ttk.Frame(controls_frm, style="Card.TFrame")
        action_row.pack(fill="x")
        ttk.Button(action_row, text="Plot Selected Components", style="Secondary.TButton",
                   command=self._plot_field_probe).pack(side="left")

        self._probe_fig = Figure(figsize=(6.8, 3.6), dpi=100)
        self._probe_ax = self._probe_fig.add_subplot(111)
        self._probe_canvas = FigureCanvasTkAgg(self._probe_fig, master=plot_frm)
        self._probe_toolbar = NavigationToolbar2Tk(self._probe_canvas, plot_frm, pack_toolbar=False)
        self._probe_toolbar.update()
        # Pack the toolbar before the canvas so it keeps its height when the
        # window is short; the canvas then shrinks into the remaining space.
        self._probe_toolbar.pack(side="bottom", fill="x")
        self._probe_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_scroll_body(self):
        """Return a frame inside a vertically scrollable canvas filling the tab.

        The cards are taller than small windows; without scrolling the probe
        plot and its controls were squeezed and cut off.
        """
        bg = ttk.Style(self).lookup("TFrame", "background")
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=bg)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")
        # Keep the cards as wide as the visible area.
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(body_id, width=e.width))
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_wheel(event):
            # Only scroll while the pointer is over this tab.
            w = self.winfo_containing(event.x_root, event.y_root)
            if w is None or not str(w).startswith(str(self)):
                return
            if event.num == 4:
                step = -1
            elif event.num == 5:
                step = 1
            else:
                step = -1 if event.delta > 0 else 1
            canvas.yview_scroll(step * 3, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, _on_wheel, add="+")
        return body

    def _plot_field_probe(self):
        """Plot selected E/B components sampled along a user-defined line."""
        if not self._get_collect:
            messagebox.showerror("Field Plotter", "No configuration callback available.")
            return

        try:
            from beamontarget.interactions.field_provider import create_field_provider

            p0 = np.asarray([float(v) for v in self.var_probe_start.get().split(",")], dtype=np.float64)
            direction = np.asarray([float(v) for v in self.var_probe_dir.get().split(",")], dtype=np.float64)
            if p0.shape != (3,) or direction.shape != (3,):
                raise ValueError("Start and direction must be 3 comma-separated values")

            step = float(self.var_probe_step.get())
            npts = int(self.var_probe_npts.get())
            selected_components = [
                c for c in ("Bx", "By", "Bz", "Ex", "Ey", "Ez")
                if self._probe_component_vars[c].get()
            ]
            if not selected_components:
                raise ValueError("Select at least one component to plot")

            if step <= 0:
                raise ValueError("Step length must be > 0")
            if npts < 2:
                raise ValueError("Number of points must be at least 2")
            dnorm = float(np.linalg.norm(direction))
            if dnorm <= 0.0:
                raise ValueError("Direction vector must be non-zero")

            unit_dir = direction / dnorm
            idx = np.arange(npts, dtype=np.float64)
            distances = idx * step
            points = p0[np.newaxis, :] + distances[:, np.newaxis] * unit_dir[np.newaxis, :]

            cfg = self._get_collect()
            ef_cfg = cfg.get("EXTERNAL_FIELD", {})
            fp = create_field_provider(ef_cfg)
            e_field, b_field = fp.sample(points, np.zeros(npts, dtype=np.float64))
            e_arr = np.asarray(e_field, dtype=np.float64)
            b_arr = np.asarray(b_field, dtype=np.float64)

            data_by_comp = {
                "Bx": b_arr[:, 0],
                "By": b_arr[:, 1],
                "Bz": b_arr[:, 2],
                "Ex": e_arr[:, 0],
                "Ey": e_arr[:, 1],
                "Ez": e_arr[:, 2],
            }
            units_by_comp = {
                "Bx": "T", "By": "T", "Bz": "T",
                "Ex": "V/m", "Ey": "V/m", "Ez": "V/m",
            }

            ax = self._probe_ax
            ax.clear()
            for comp in selected_components:
                ax.plot(distances, data_by_comp[comp], linewidth=1.8,
                        label=f"{comp} [{units_by_comp[comp]}]")
            ax.set_xlabel("Distance along probe line [m]")
            ax.set_ylabel("Field component value")
            ax.set_title(
                f"Components {', '.join(selected_components)} along line: "
                f"start=({p0[0]:.3g},{p0[1]:.3g},{p0[2]:.3g}), "
                f"dir=({unit_dir[0]:.3g},{unit_dir[1]:.3g},{unit_dir[2]:.3g}), "
                f"step={step:.3g} m, n={npts}"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            self._probe_fig.tight_layout()
            self._probe_canvas.draw_idle()

        except Exception as exc:
            messagebox.showerror("Field Plotter", f"Failed to plot field probe: {exc}")

    # ------------------------------------------------------------------
    #  Larmor radius estimate
    # ------------------------------------------------------------------
    def _calc_larmor_radius(self):
        """Estimate Larmor radius from current field config + user-specified species/energy."""
        if not self._get_collect:
            self._var_larmor.set("(no config callback)")
            return
        try:
            from beamontarget.constants import (ELEMENTARY_CHARGE_C,
                                   HYDROGEN_MASS_KG, DEUTERIUM_MASS_KG)
            from beamontarget.interactions.field_provider import create_field_provider

            species = self.var_lr_species.get()
            energy_ev = self.var_lr_energy.get()
            if energy_ev <= 0:
                self._var_larmor.set("Energy must be > 0")
                return

            mass = HYDROGEN_MASS_KG if species == "H" else DEUTERIUM_MASS_KG
            q_c = ELEMENTARY_CHARGE_C          # |charge| = 1 e
            speed = np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE_C / mass)

            # Sample B field on a coarse grid through the bbox
            cfg = self._get_collect()
            ef_cfg = cfg.get("EXTERNAL_FIELD", {})
            fp = create_field_provider(ef_cfg)
            pts = np.zeros((1, 3))             # sample at origin
            _, b_field = fp.sample(pts, np.zeros(len(pts)))
            b_norm = np.linalg.norm(np.asarray(b_field, dtype=np.float64), axis=1)
            max_b = float(np.max(b_norm)) if b_norm.size > 0 else 0.0

            if max_b <= 0:
                self._var_larmor.set("r_L = ∞  (|B| = 0)")
            else:
                r_l = mass * speed / (q_c * max_b)
                self._var_larmor.set(
                    f"r_L ≈ {r_l:.4e} m   |   v = {speed:.3e} m/s   |   |B| = {max_b:.3e} T"
                )
        except Exception as exc:
            self._var_larmor.set(f"Error: {exc}")

    # ------------------------------------------------------------------
    #  Build one component row
    # ------------------------------------------------------------------
    def _make_comp_row(self, card, row, label, key, modes, init_cfg):
        mode = init_cfg.get("mode", "zero")
        if mode == "line":
            mode = "profile_x"
        init_mode = _MODE_LABELS.get(mode, "None")

        ttk.Label(card, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=3)
        mode_var = tk.StringVar(value=init_mode)
        ttk.Combobox(card, textvariable=mode_var, values=modes,
                      state="readonly", width=18).grid(
            row=row, column=1, sticky="w", padx=(8, 0), pady=3)

        # Fixed value entry
        val_var = tk.StringVar(value=str(init_cfg.get("value", 0.0)))
        val_frm = ttk.Frame(card, style="Card.TFrame")
        val_frm.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=3)
        ttk.Entry(val_frm, textvariable=val_var, width=12).pack(side="left")

        # File path entry + browse
        file_var = tk.StringVar(value=init_cfg.get("file", ""))
        file_frm = ttk.Frame(card, style="Card.TFrame")
        file_frm.grid(row=row, column=2, sticky="we", padx=(8, 0), pady=3)
        ttk.Entry(file_frm, textvariable=file_var, width=28).pack(
            side="left", fill="x", expand=True)
        ttk.Button(file_frm, text="Browse…", style="Secondary.TButton",
                    command=lambda fv=file_var: self._browse_fld_file(fv)).pack(
                        side="left", padx=(4, 0))

        # RID params frame (only for Ey)
        rid_frm = None
        rid_vars = {}
        if "ERID (simplified)" in modes:
            rid_frm = ttk.Frame(card, style="Card.TFrame")
            rid_frm.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=3)
            for rlbl, rkey, rdef in [("V:", "v_rid_v", 20000.0),
                                      ("x_min:", "x_min_m", 5.4),
                                      ("x_max:", "x_max_m", 7.2)]:
                ttk.Label(rid_frm, text=rlbl, style="Card.TLabel").pack(side="left")
                rv = tk.StringVar(value=str(init_cfg.get(rkey, rdef)))
                ttk.Entry(rid_frm, textvariable=rv, width=8).pack(
                    side="left", padx=(2, 8))
                rid_vars[rkey] = rv

        # Scale factor entry (shown for all non-None modes)
        scale_var = tk.StringVar(value=str(init_cfg.get("scale", 1.0)))
        scale_frm = ttk.Frame(card, style="Card.TFrame")
        scale_frm.grid(row=row, column=3, sticky="e", padx=(8, 0), pady=3)
        ttk.Label(scale_frm, text="×", style="Card.TLabel").pack(side="left")
        ttk.Entry(scale_frm, textvariable=scale_var, width=6).pack(
            side="left", padx=(2, 0))

        self._comp_vars[key] = {"mode": mode_var, "val": val_var,
                                 "file": file_var, "rid": rid_vars,
                                 "scale": scale_var}
        self._comp_widgets[key] = {"val_frame": val_frm, "file_frame": file_frm,
                                    "rid_frame": rid_frm, "scale_frame": scale_frm}

        def _on_mode_change(*_, k=key):
            m = self._comp_vars[k]["mode"].get()
            w = self._comp_widgets[k]
            w["val_frame"].grid() if m == "Fixed value" else w["val_frame"].grid_remove()
            w["file_frame"].grid() if m in ("Field file (.fld)", "X profile file (x,value)") else w["file_frame"].grid_remove()
            if w["rid_frame"]:
                w["rid_frame"].grid() if m == "ERID (simplified)" else w["rid_frame"].grid_remove()
            w["scale_frame"].grid() if m != "None" else w["scale_frame"].grid_remove()

        mode_var.trace_add("write", _on_mode_change)
        _on_mode_change()

    # ------------------------------------------------------------------
    #  Browse for a .fld file
    # ------------------------------------------------------------------
    def _browse_fld_file(self, var):
        p = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Select field file",
            filetypes=[("Field files", "*.fld *.csv *.txt"), ("All files", "*")])
        if p:
            try:
                rel = os.path.relpath(p, _SCRIPT_DIR)
                var.set(rel)
            except ValueError:
                var.set(p)

    # ------------------------------------------------------------------
    #  Collect field config into dict *d*
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write the EXTERNAL_FIELD key into *d*."""
        components = {}
        for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            cv = self._comp_vars[key]
            mode = _MODE_MAP.get(cv["mode"].get(), "zero")
            if mode == "zero":
                continue
            comp = {"mode": mode}
            scale = float(cv["scale"].get())
            if scale != 1.0:
                comp["scale"] = scale
            if mode == "fixed":
                comp["value"] = float(cv["val"].get())
            elif mode in ("file", "profile_x"):
                comp["file"] = cv["file"].get()
            elif mode == "rid_ey":
                for rk in ("v_rid_v", "x_min_m", "x_max_m"):
                    comp[rk] = float(cv["rid"][rk].get())
            components[key] = comp

        if components:
            d["EXTERNAL_FIELD"] = {"type": "composite", "components": components}
        else:
            d["EXTERNAL_FIELD"] = {"type": "zero"}

    # ------------------------------------------------------------------
    #  Refresh widgets from config dict *c*
    # ------------------------------------------------------------------
    def refresh(self, c):
        """Push values from config dict *c* into our widgets."""
        ef = c.get("EXTERNAL_FIELD", {})
        comps = _parse_components(ef)
        for key in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"):
            cv = self._comp_vars[key]
            cfg_comp = comps.get(key, {})
            mode = cfg_comp.get("mode", "zero")
            if mode == "line":
                mode = "profile_x"
            cv["mode"].set(_MODE_LABELS.get(mode, "None"))
            cv["val"].set(str(cfg_comp.get("value", 0.0)))
            cv["file"].set(cfg_comp.get("file", ""))
            cv["scale"].set(str(cfg_comp.get("scale", 1.0)))
            for rk in ("v_rid_v", "x_min_m", "x_max_m"):
                if rk in cv["rid"]:
                    cv["rid"][rk].set(str(cfg_comp.get(rk, {"v_rid_v": 20000.0,
                                                              "x_min_m": 5.4,
                                                              "x_max_m": 7.2}[rk])))


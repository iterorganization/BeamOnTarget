# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Reactions tab — background gas density, cross-section settings, diagnostic plots.

Provides :class:`ReactionsTab`, a ``ttk.Frame`` that the main window embeds
in the notebook.  Exposes ``collect(d, is_em)`` and ``refresh(c)`` so the
host can serialise / deserialise the reaction config without reaching into
widget internals.

The diagnostic plots (cross sections, gas density, species evolution) need
bounding-box and EM-step values that live in the General / Fields tabs.
Those are supplied via the *get_bbox_min*, *get_bbox_max* and *get_em_step*
callables passed at construction time.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from beamontarget.gui.gui_widgets import make_card, parse_vec3, _SCRIPT_DIR

# ---------------------------------------------------------------------------
#  ReactionsTab
# ---------------------------------------------------------------------------
class ReactionsTab(ttk.Frame):
    """Background gas & cross-section configuration plus diagnostic plots."""

    def __init__(self, parent, cfg, colours, *,
                 get_bbox_min, get_bbox_max, get_em_step,
                 get_collect_fn=None, **kw):
        super().__init__(parent, **kw)
        self._get_bbox_min = get_bbox_min
        self._get_bbox_max = get_bbox_max
        self._get_em_step = get_em_step
        self._get_collect = get_collect_fn

        top_pw = ttk.PanedWindow(self, orient="horizontal")
        top_pw.pack(fill="both", expand=True)

        # ===========================================================
        # LEFT side — scrollable cards
        # ===========================================================
        left_wrapper = ttk.Frame(top_pw)
        top_pw.add(left_wrapper, weight=1)

        rxn_canvas = tk.Canvas(left_wrapper, borderwidth=0,
                                highlightthickness=0, bg=colours["bg"])
        rxn_vscroll = ttk.Scrollbar(left_wrapper, orient="vertical",
                                     command=rxn_canvas.yview)
        outer = ttk.Frame(rxn_canvas)
        outer.bind("<Configure>",
                   lambda e: rxn_canvas.configure(
                       scrollregion=rxn_canvas.bbox("all")))
        rxn_canvas.create_window((0, 0), window=outer, anchor="nw")
        rxn_canvas.configure(yscrollcommand=rxn_vscroll.set)
        rxn_canvas.pack(side="left", fill="both", expand=True)
        rxn_vscroll.pack(side="right", fill="y")

        rm = cfg.get("REACTION_MODEL", {})

        # --- Background Gas Density card ---
        dens_card = make_card(outer, "Background Gas Density", pady=(12, 10))

        has_profile = bool(rm.get("density_profile_file", ""))
        dens_init = "Density profile file (.dens)" if has_profile else "Uniform density"

        row = 0
        ttk.Label(dens_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_density_mode = tk.StringVar(value=dens_init)
        ttk.Combobox(dens_card, textvariable=self.var_density_mode,
                      values=["Uniform density", "Density profile file (.dens)"],
                      state="readonly", width=28).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        # -- Uniform density widgets --
        row += 1
        self._dens_uniform_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_uniform_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self._dens_uniform_frame, text="Density (m⁻³):",
                  style="Card.TLabel").pack(side="left")
        self.var_bg_density = tk.DoubleVar(value=rm.get("background_density_m3", 0.0))
        ttk.Entry(self._dens_uniform_frame, textvariable=self.var_bg_density,
                  width=14).pack(side="left", padx=(8, 0))

        # -- Profile file widgets --
        row += 1
        self._dens_profile_frame = ttk.Frame(dens_card, style="Card.TFrame")
        self._dens_profile_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(self._dens_profile_frame, text="File:",
                  style="Card.TLabel").pack(side="left")
        self.var_density_file = tk.StringVar(value=rm.get("density_profile_file", ""))
        ttk.Entry(self._dens_profile_frame, textvariable=self.var_density_file,
                  width=30).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(self._dens_profile_frame, text="Browse…",
                    style="Secondary.TButton",
                    command=self._browse_density_file).pack(side="left")

        row += 1
        dens_scale_frame = ttk.Frame(dens_card, style="Card.TFrame")
        dens_scale_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(dens_scale_frame, text="Profile multiplier:",
                  style="Card.TLabel").pack(side="left")
        self.var_density_profile_scale = tk.DoubleVar(
            value=rm.get("density_profile_scale", 1.0)
        )
        ttk.Entry(dens_scale_frame, textvariable=self.var_density_profile_scale,
                  width=12).pack(side="left", padx=(8, 0))

        row += 1
        dens_dir_frame = ttk.Frame(dens_card, style="Card.TFrame")
        dens_dir_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=2)
        ttk.Label(dens_dir_frame, text="Profile direction:",
                  style="Card.TLabel").pack(side="left")
        dd = rm.get("density_profile_direction",
                     cfg.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir = tk.StringVar(value=f"{dd[0]}, {dd[1]}, {dd[2]}")
        ttk.Entry(dens_dir_frame, textvariable=self.var_density_dir,
                  width=24).pack(side="left", padx=(8, 0))

        def _on_density_mode(*_):
            if self.var_density_mode.get() == "Uniform density":
                self._dens_uniform_frame.grid()
                self._dens_profile_frame.grid_remove()
            else:
                self._dens_uniform_frame.grid_remove()
                self._dens_profile_frame.grid()

        self.var_density_mode.trace_add("write", _on_density_mode)
        _on_density_mode()
        dens_card.columnconfigure(1, weight=1)

        # --- Reaction Definitions (Cross Sections) card ---
        cs_card = make_card(outer, "Reaction Definitions (Cross Sections)")

        self._cs_channels = [
            ("H⁻/D⁻ → H⁰/D⁰", "Single stripping",  "single_strip_neg_to_neutral"),
            ("H⁻/D⁻ → H⁺/D⁺", "Double stripping",   "double_strip_neg_to_positive"),
            ("H⁰/D⁰ → H⁺/D⁺", "Neutral stripping",  "strip_neutral_to_positive"),
            ("H⁺/D⁺ → H⁰/D⁰", "Charge exchange",    "charge_exchange_pos_to_neutral"),
        ]
        manual_cs = rm.get("manual_cross_sections") or {}
        has_manual = bool(manual_cs)
        cs_mode_init = "Manual (m²)" if has_manual else "Built-in polynomial fit"

        row = 0
        ttk.Label(cs_card, text="Source:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        self.var_cs_mode = tk.StringVar(value=cs_mode_init)
        ttk.Combobox(cs_card, textvariable=self.var_cs_mode,
                      values=["Built-in polynomial fit", "Manual (m²)"],
                      state="readonly", width=24).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=(8, 0))

        row += 1
        self.var_fixed_cs = tk.BooleanVar(value=rm.get("fixed_cs", False))
        self._cs_freeze_check = ttk.Checkbutton(
            cs_card, text="Freeze at initial energy",
            variable=self.var_fixed_cs, style="Card.TCheckbutton")
        self._cs_freeze_check.grid(row=row, column=0, columnspan=3,
                                    sticky="w", pady=(2, 6))

        self._cs_manual_vars = {}
        self._cs_source_labels = {}
        self._cs_entry_widgets = {}
        self._cs_unit_labels = {}

        for i, (label, desc, key) in enumerate(self._cs_channels):
            r = row + 1 + i
            ttk.Label(cs_card, text=label, style="Card.TLabel").grid(
                row=r, column=0, sticky="w", pady=2)
            ttk.Label(cs_card, text=desc, style="Card.TLabel").grid(
                row=r, column=1, sticky="w", padx=(8, 0), pady=2)

            lbl = ttk.Label(cs_card, text="(built-in polynomial fit)",
                            style="Card.TLabel")
            lbl.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_source_labels[key] = lbl

            var = tk.StringVar(value=f"{manual_cs.get(key, 0.0):.3e}")
            self._cs_manual_vars[key] = var
            ent = ttk.Entry(cs_card, textvariable=var, width=12)
            ent.grid(row=r, column=2, sticky="w", padx=(8, 0), pady=2)
            self._cs_entry_widgets[key] = ent
            unit = ttk.Label(cs_card, text="m²", style="Card.TLabel")
            unit.grid(row=r, column=3, sticky="w", padx=(4, 0), pady=2)
            self._cs_unit_labels[key] = unit

        def _on_cs_mode(*_):
            is_manual = self.var_cs_mode.get() == "Manual (m²)"
            for key in self._cs_manual_vars:
                if is_manual:
                    self._cs_source_labels[key].grid_remove()
                    self._cs_entry_widgets[key].grid()
                    self._cs_unit_labels[key].grid()
                else:
                    self._cs_entry_widgets[key].grid_remove()
                    self._cs_unit_labels[key].grid_remove()
                    self._cs_source_labels[key].grid()
            if is_manual:
                self._cs_freeze_check.grid_remove()
            else:
                self._cs_freeze_check.grid()

        self.var_cs_mode.trace_add("write", _on_cs_mode)
        _on_cs_mode()
        cs_card.columnconfigure(1, weight=0)

        # ===========================================================
        # RIGHT side — embedded diagnostic plots
        # ===========================================================
        right_frm = ttk.Frame(top_pw, style="Card.TFrame", padding=8)
        top_pw.add(right_frm, weight=1)

        ttk.Label(right_frm, text="Diagnostic Plots",
                  style="CardHeader.TLabel").pack(anchor="w", pady=(0, 4))

        param_row = ttk.Frame(right_frm, style="Card.TFrame")
        param_row.pack(fill="x", pady=(0, 4))
        ttk.Label(param_row, text="Species:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_species = tk.StringVar(value="H")
        ttk.Combobox(param_row, textvariable=self.var_plot_species,
                      values=["H", "D"], state="readonly", width=4).pack(
            side="left", padx=(0, 12))
        ttk.Label(param_row, text="Energy [eV]:",
                  style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.var_plot_energy = tk.DoubleVar(value=870e3)
        ttk.Entry(param_row, textvariable=self.var_plot_energy, width=12).pack(
            side="left")

        btn_row = ttk.Frame(right_frm, style="Card.TFrame")
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Cross Sections", style="Secondary.TButton",
                    command=self._plot_cross_sections).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Gas Density", style="Secondary.TButton",
                    command=self._plot_gas_density).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Species Evolution", style="Secondary.TButton",
                    command=self._plot_species_evolution).pack(side="left")

        # --- Mean Free Path estimate ---
        mfp_row = ttk.Frame(right_frm, style="Card.TFrame")
        mfp_row.pack(fill="x", pady=(4, 6))
        ttk.Button(mfp_row, text="Calc Mean Free Path", style="Secondary.TButton",
                    command=self._calc_mean_free_path).pack(side="left")
        self._var_mfp = tk.StringVar(value="—")
        ttk.Label(mfp_row, textvariable=self._var_mfp,
                  style="Card.TLabel", font=("Segoe UI", 9)).pack(
            side="left", padx=(10, 0))

        self._rxn_fig = Figure(figsize=(5.5, 4.5), dpi=100)
        self._rxn_canvas = FigureCanvasTkAgg(self._rxn_fig, master=right_frm)
        self._rxn_toolbar = NavigationToolbar2Tk(self._rxn_canvas, right_frm)
        self._rxn_toolbar.update()
        self._rxn_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._rxn_cursor_artists = []
        self._rxn_canvas.mpl_connect("motion_notify_event",
                                      self._on_rxn_mouse_move)

    # ------------------------------------------------------------------
    #  Collect reactions config into dict *d*
    # ------------------------------------------------------------------
    def collect(self, d, is_em):
        """Write REACTION_MODEL and DENSITY_DIRECTION keys into *d*."""
        if is_em:
            dd_vec = parse_vec3(self.var_density_dir.get())
            # The engine uses the profile whenever a file is set, so only
            # store it when the profile source is selected.
            use_profile = self.var_density_mode.get() != "Uniform density"
            rm_dict = {
                "type": "beam_gas_cross_sections",
                "background_density_m3": self.var_bg_density.get(),
                "density_profile_file": self.var_density_file.get() if use_profile else "",
                "density_profile_scale": self.var_density_profile_scale.get(),
                "density_profile_direction": dd_vec,
            }
            if self.var_cs_mode.get() == "Manual (m²)":
                rm_dict["manual_cross_sections"] = {
                    k: float(v.get()) for k, v in self._cs_manual_vars.items()
                }
            else:
                rm_dict["fixed_cs"] = self.var_fixed_cs.get()
            d["REACTION_MODEL"] = rm_dict
            d["DENSITY_DIRECTION"] = dd_vec
        else:
            d["REACTION_MODEL"] = {"type": "none"}

    # ------------------------------------------------------------------
    #  Refresh widgets from config dict *c*
    # ------------------------------------------------------------------
    def refresh(self, c):
        """Push values from config dict *c* into our widgets."""
        rm = c.get("REACTION_MODEL", {})
        self.var_bg_density.set(rm.get("background_density_m3", 0.0))
        self.var_density_file.set(rm.get("density_profile_file", ""))
        self.var_density_mode.set(
            "Density profile file (.dens)" if rm.get("density_profile_file")
            else "Uniform density")
        self.var_density_profile_scale.set(rm.get("density_profile_scale", 1.0))
        dd = rm.get("density_profile_direction",
                     c.get("DENSITY_DIRECTION", [1.0, 0.0, 0.0]))
        self.var_density_dir.set(f"{dd[0]}, {dd[1]}, {dd[2]}")
        manual_cs = rm.get("manual_cross_sections") or {}
        if manual_cs:
            self.var_cs_mode.set("Manual (m²)")
            for key, var in self._cs_manual_vars.items():
                var.set(f"{manual_cs.get(key, 0.0):.3e}")
        else:
            self.var_cs_mode.set("Built-in polynomial fit")
            self.var_fixed_cs.set(rm.get("fixed_cs", False))

    # ------------------------------------------------------------------
    #  Interactive cursor
    # ------------------------------------------------------------------
    def _on_rxn_mouse_move(self, event):
        for a in self._rxn_cursor_artists:
            try:
                a.remove()
            except (NotImplementedError, ValueError):
                pass
        self._rxn_cursor_artists.clear()

        if not self._rxn_fig.axes:
            return
        ax = self._rxn_fig.axes[0]
        if event.inaxes != ax or event.xdata is None:
            self._rxn_canvas.draw_idle()
            return

        x = event.xdata
        is_logx = ax.get_xscale() == "log"
        is_logy = ax.get_yscale() == "log"

        vl = ax.axvline(x, color="#888888", ls=":", lw=0.8, alpha=0.5)
        self._rxn_cursor_artists.append(vl)

        items = []
        for line in ax.get_lines():
            lbl = line.get_label()
            if not lbl or lbl.startswith("_"):
                continue
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            if len(xd) <= 2:
                continue
            if x < xd.min() or x > xd.max():
                continue
            if is_logx and is_logy:
                yi = 10 ** np.interp(np.log10(x),
                                     np.log10(xd),
                                     np.log10(np.maximum(yd, 1e-300)))
            elif is_logx:
                yi = np.interp(np.log10(x), np.log10(xd), yd)
            else:
                yi = np.interp(x, xd, yd)
            items.append((yi, lbl, line.get_color()))

        for idx, (yi, lbl, col) in enumerate(items):
            dot, = ax.plot(x, yi, "o", color=col, ms=5, zorder=10)
            self._rxn_cursor_artists.append(dot)
            y_off = 6 * (1 if idx % 2 == 0 else -1)
            txt = ax.annotate(
                f"{lbl}: {yi:.3e}", xy=(x, yi),
                xytext=(8, y_off), textcoords="offset points",
                fontsize=7.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col,
                          alpha=0.85, lw=0.6),
                zorder=11,
            )
            self._rxn_cursor_artists.append(txt)

        self._rxn_canvas.draw_idle()

    # ------------------------------------------------------------------
    #  Diagnostic plots
    # ------------------------------------------------------------------
    def _plot_cross_sections(self):
        from beamontarget.interactions import cross_sections as cs
        iso = self.var_plot_species.get()
        energy = np.logspace(1, 7, 500)
        sigma_ss = np.maximum(cs.cs_hm_single_strip(energy, isotope=iso), 1e-25)
        sigma_ds = np.maximum(cs.cs_hm_double_strip(energy, isotope=iso), 1e-25)
        sigma_ns = np.maximum(cs.cs_proj_ionization_h0(energy, isotope=iso), 1e-25)
        sigma_cx = np.maximum(cs.cs_cx_hp(energy, isotope=iso), 1e-25)

        neg = "H⁻" if iso == "H" else "D⁻"
        neu = "H⁰" if iso == "H" else "D⁰"
        pos = "H⁺" if iso == "H" else "D⁺"

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.loglog(energy, sigma_ss, label=f"{neg}→{neu} (single strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ds, label=f"{neg}→{pos} (double strip)", linewidth=1.8)
        ax.loglog(energy, sigma_ns, label=f"{neu}→{pos} (neutral strip)", linewidth=1.8)
        ax.loglog(energy, sigma_cx, label=f"{pos}→{neu} (charge exchange)", linewidth=1.8)
        ax.set_xlabel("Particle energy [eV]")
        ax.set_ylabel("Cross section [m²]")
        ax.set_ylim(bottom=1e-25)
        ax.set_title(f"Beam–Gas Cross Sections ({iso})")
        ax.axvline(self.var_plot_energy.get(), color="gray", linestyle="--",
                    linewidth=1.0, alpha=0.7,
                    label=f"E = {self.var_plot_energy.get():.0f} eV")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_gas_density(self):
        from beamontarget.interactions.reactions import BeamCrossSectionReaction

        density_dir = parse_vec3(self.var_density_dir.get())
        bbox_min = self._get_bbox_min()
        bbox_max = self._get_bbox_max()

        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Gas Density", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from beamontarget.workflows.prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Gas Density",
                                 "Density direction does not traverse the bounding box.")
            return

        is_uniform = self.var_density_mode.get() == "Uniform density"
        density_file = None if is_uniform else self.var_density_file.get()
        bg_density = self.var_bg_density.get()

        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_scale=self.var_density_profile_scale.get(),
            density_profile_direction=density_dir,
            verbose=False,
        )

        steps = max(2, int(np.ceil(line_len / 0.01)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]
        dens = np.maximum(np.asarray(model._density_at_positions(positions),
                                      dtype=np.float64), 0.0)

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        if is_uniform:
            lbl = f"Uniform: {bg_density:.3e} m⁻³"
        else:
            lbl = os.path.basename(density_file) if density_file else "profile"
        ax.plot(distance, dens, color="tab:green", linewidth=2.0, label=lbl)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Gas density [m⁻³]")
        ax.set_title("Background Gas Density Profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    def _plot_species_evolution(self):
        from beamontarget.interactions import cross_sections as cs
        from beamontarget.interactions.reactions import BeamCrossSectionReaction

        bbox_min = self._get_bbox_min()
        bbox_max = self._get_bbox_max()
        step_len = self._get_em_step()
        density_dir = parse_vec3(self.var_density_dir.get())

        density_file = self.var_density_file.get() \
            if self.var_density_mode.get() != "Uniform density" else None
        bg_density = self.var_bg_density.get()
        model = BeamCrossSectionReaction(
            background_density_m3=bg_density,
            density_profile_file=density_file,
            density_profile_scale=self.var_density_profile_scale.get(),
            density_profile_direction=density_dir,
            verbose=False,
        )

        dir_arr = np.array(density_dir, dtype=np.float64)
        dir_norm = np.linalg.norm(dir_arr)
        if dir_norm <= 0:
            messagebox.showerror("Species Evolution", "Invalid density direction.")
            return
        dir_arr /= dir_norm

        from beamontarget.workflows.prerun_analysis import _ray_exit_distance_from_box
        start = np.array(bbox_min, dtype=np.float64)
        line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
        if line_len is None or line_len <= 0:
            messagebox.showerror("Species Evolution",
                                 "Beam direction does not traverse the bounding box.")
            return

        steps = max(2, int(np.ceil(line_len / step_len)) + 1)
        distance = np.linspace(0.0, line_len, steps)
        positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]

        density_m3 = np.maximum(
            np.asarray(model._density_at_positions(positions), dtype=np.float64), 0.0)

        iso = self.var_plot_species.get()
        avg_energy_ev = self.var_plot_energy.get()
        from beamontarget.constants import HYDROGEN_MASS_KG, DEUTERIUM_MASS_KG, ELEMENTARY_CHARGE_C
        avg_mass_kg = HYDROGEN_MASS_KG if iso == "H" else DEUTERIUM_MASS_KG
        avg_speed = np.sqrt(2.0 * avg_energy_ev * ELEMENTARY_CHARGE_C / avg_mass_kg)

        if self.var_cs_mode.get() == "Manual (m²)":
            s_ss = float(self._cs_manual_vars["single_strip_neg_to_neutral"].get())
            s_ds = float(self._cs_manual_vars["double_strip_neg_to_positive"].get())
            s_ns = float(self._cs_manual_vars["strip_neutral_to_positive"].get())
            s_cx = float(self._cs_manual_vars["charge_exchange_pos_to_neutral"].get())
        else:
            sigma = cs.channel_cross_sections(avg_energy_ev, isotope=iso)
            s_ss = float(np.asarray(sigma[cs.CH_SINGLE_STRIP]))
            s_ds = float(np.asarray(sigma[cs.CH_DOUBLE_STRIP]))
            s_ns = float(np.asarray(sigma[cs.CH_NEUTRAL_STRIP]))
            s_cx = float(np.asarray(sigma[cs.CH_CHARGE_EXCHANGE]))

        fractions = np.zeros((steps, 3), dtype=np.float64)
        fractions[0] = [1.0, 0.0, 0.0]

        for i in range(steps - 1):
            ds = distance[i + 1] - distance[i]
            dt = ds / avg_speed
            f_neg, f_neu, f_pos = fractions[i]
            n = density_m3[i]
            dn = -(s_ss + s_ds) * n * avg_speed * f_neg
            d0 = (s_ss * n * avg_speed * f_neg
                   - s_ns * n * avg_speed * f_neu
                   + s_cx * n * avg_speed * f_pos)
            dp = (s_ds * n * avg_speed * f_neg
                   + s_ns * n * avg_speed * f_neu
                   - s_cx * n * avg_speed * f_pos)
            nxt = fractions[i] + dt * np.array([dn, d0, dp])
            nxt = np.maximum(nxt, 0.0)
            s = nxt.sum()
            if s > 0:
                nxt /= s
            fractions[i + 1] = nxt

        self._rxn_cursor_artists.clear()
        self._rxn_fig.clear()
        ax = self._rxn_fig.add_subplot(111)
        ax.plot(distance, fractions[:, 0], label="H⁻/D⁻", linewidth=2.0)
        ax.plot(distance, fractions[:, 1], label="H⁰/D⁰", linewidth=2.0)
        ax.plot(distance, fractions[:, 2], label="H⁺/D⁺", linewidth=2.0)
        ax.set_xlabel("Distance along beam direction [m]")
        ax.set_ylabel("Species fraction")
        ax.set_title(f"Analytical Species Evolution (E={avg_energy_ev/1e3:.0f} keV)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        self._rxn_fig.tight_layout()
        self._rxn_canvas.draw()

    # ------------------------------------------------------------------
    def _browse_density_file(self):
        p = filedialog.askopenfilename(
            initialdir=_SCRIPT_DIR,
            title="Select density profile file",
            filetypes=[("Density files", "*.dens"), ("All files", "*")])
        if p:
            try:
                rel = os.path.relpath(p, _SCRIPT_DIR)
                self.var_density_file.set(rel)
            except ValueError:
                self.var_density_file.set(p)

    # ------------------------------------------------------------------
    #  Mean free path estimate
    # ------------------------------------------------------------------
    def _calc_mean_free_path(self):
        """Estimate minimum mean free path from reaction config + user-specified species/energy."""
        if not self._get_collect:
            self._var_mfp.set("(no config callback)")
            return
        try:
            from beamontarget.interactions import cross_sections as cs
            from beamontarget.interactions.reactions import BeamCrossSectionReaction
            from beamontarget.workflows.prerun_analysis import _ray_exit_distance_from_box

            bbox_min = self._get_bbox_min()
            bbox_max = self._get_bbox_max()

            # Species and energy from user fields
            isotope = self.var_plot_species.get()
            avg_energy_ev = self.var_plot_energy.get()
            if avg_energy_ev <= 0:
                self._var_mfp.set("Energy must be > 0")
                return

            # Cross sections at specified energy
            sigma = cs.channel_cross_sections(avg_energy_ev, isotope=isotope)
            sigma_neg = float(np.asarray(sigma[cs.CH_SINGLE_STRIP])) + \
                        float(np.asarray(sigma[cs.CH_DOUBLE_STRIP]))
            sigma_neu = float(np.asarray(sigma[cs.CH_NEUTRAL_STRIP]))
            sigma_pos = float(np.asarray(sigma[cs.CH_CHARGE_EXCHANGE]))

            # Density along beam direction
            density_dir = parse_vec3(self.var_density_dir.get())
            dir_arr = np.array(density_dir, dtype=np.float64)
            dir_norm = np.linalg.norm(dir_arr)
            if dir_norm <= 0:
                self._var_mfp.set("Invalid density direction.")
                return
            dir_arr /= dir_norm

            start = np.array(bbox_min, dtype=np.float64)
            line_len = _ray_exit_distance_from_box(start, dir_arr, bbox_min, bbox_max)
            if line_len is None or line_len <= 0:
                self._var_mfp.set("Direction doesn't traverse bbox.")
                return

            is_uniform = self.var_density_mode.get() == "Uniform density"
            density_file = None if is_uniform else self.var_density_file.get()
            bg_density = self.var_bg_density.get()

            model = BeamCrossSectionReaction(
                background_density_m3=bg_density,
                density_profile_file=density_file,
                density_profile_scale=self.var_density_profile_scale.get(),
                density_profile_direction=density_dir,
                verbose=False,
            )

            steps = max(2, int(np.ceil(line_len / 0.01)) + 1)
            distance = np.linspace(0.0, line_len, steps)
            positions = start[np.newaxis, :] + distance[:, np.newaxis] * dir_arr[np.newaxis, :]
            density_m3 = np.maximum(
                np.asarray(model._density_at_positions(positions), dtype=np.float64), 0.0)

            # mfp = 1 / (n * sigma) for each species transition
            mfp_arrays = []
            for s_total in (sigma_neg, sigma_neu, sigma_pos):
                if s_total > 0:
                    denom = density_m3 * s_total
                    mfp = np.where(denom > 0, 1.0 / np.maximum(denom, 1e-300), np.inf)
                    mfp_arrays.append(mfp)

            if not mfp_arrays:
                self._var_mfp.set("λ = ∞ (all σ = 0)")
                return

            all_mfp = np.concatenate(mfp_arrays)
            finite = all_mfp[np.isfinite(all_mfp) & (all_mfp > 0)]
            if finite.size == 0:
                self._var_mfp.set("λ = ∞ (zero density)")
                return

            mfp_min = float(np.min(finite))
            mfp_max = float(np.max(finite))
            em_step = self._get_em_step()
            ratio = em_step / mfp_min if mfp_min > 0 else float("inf")
            self._var_mfp.set(
                f"λ_min ≈ {mfp_min:.3e} m  |  λ_max ≈ {mfp_max:.3e} m  |  step/λ_min = {ratio:.3e}"
            )
        except Exception as exc:
            self._var_mfp.set(f"Error: {exc}")




# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Results tab — visualisation, summary CSV reports, comparison bar charts.

Provides :class:`ResultsTab`, a ``ttk.Frame`` that the main window embeds
in the notebook.  External dependencies (output dir, ParaView, pick-dialog,
extract-dialog) are supplied via constructor callbacks and tkinter
StringVars.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import csv as csv_mod
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from beamontarget.gui.gui_widgets import make_card, resolve_path

# ---------------------------------------------------------------------------
#  ResultsTab
# ---------------------------------------------------------------------------
class ResultsTab(ttk.Frame):
    """Summary-CSV loading, treeview table, and side-by-side bar charts."""

    def __init__(self, parent, cfg, colours, *,
                 var_outdir, var_pv_path, var_pv_module,
                 view_results_fn, open_extract_fn,
                 view_results_o3d_fn, **kw):
        super().__init__(parent, **kw)
        self._colours = colours
        self._var_outdir = var_outdir
        self._var_pv_path = var_pv_path
        self._var_pv_module = var_pv_module
        self._view_results_fn = view_results_fn
        self._open_extract_fn = open_extract_fn
        self._view_results_o3d_fn = view_results_o3d_fn

        top_pw = ttk.PanedWindow(self, orient="horizontal")
        top_pw.pack(fill="both", expand=True)

        # ===========================================================
        # LEFT side — scrollable cards
        # ===========================================================
        left_wrapper = ttk.Frame(top_pw)
        top_pw.add(left_wrapper, weight=1)

        res_canvas = tk.Canvas(left_wrapper, borderwidth=0,
                               highlightthickness=0, bg=colours["bg"])
        res_vscroll = ttk.Scrollbar(left_wrapper, orient="vertical",
                                     command=res_canvas.yview)
        outer = ttk.Frame(res_canvas)
        outer.bind("<Configure>",
                   lambda e: res_canvas.configure(
                       scrollregion=res_canvas.bbox("all")))
        res_canvas.create_window((0, 0), window=outer, anchor="nw")
        res_canvas.configure(yscrollcommand=res_vscroll.set)
        res_canvas.pack(side="left", fill="both", expand=True)
        res_vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            if event.num == 4:
                res_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                res_canvas.yview_scroll(1, "units")
            else:
                res_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        res_canvas.bind("<Button-4>", _on_mousewheel)
        res_canvas.bind("<Button-5>", _on_mousewheel)
        outer.bind("<Button-4>", _on_mousewheel)
        outer.bind("<Button-5>", _on_mousewheel)
        res_canvas.bind("<MouseWheel>", _on_mousewheel)
        outer.bind("<MouseWheel>", _on_mousewheel)

        # --- Visualisation card ---
        vis_card = make_card(outer, "Visualisation", pady=(12, 10))

        self.var_ENABLE_VISUALIZATION = tk.BooleanVar(
            value=cfg.get("ENABLE_VISUALIZATION", False))
        ttk.Checkbutton(vis_card, text="Enable visualisation (master switch)",
                         variable=self.var_ENABLE_VISUALIZATION,
                         style="Card.TCheckbutton").pack(anchor="w", pady=2)

        btn_frm = ttk.Frame(vis_card, style="Card.TFrame")
        btn_frm.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_frm, text="Results (Open3D)", style="Secondary.TButton",
                    command=self._view_results_o3d_fn).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frm, text="Results (ParaView)", style="Secondary.TButton",
                    command=self._view_results_fn).pack(side="left")

        # --- Summary Reports card ---
        sum_card = make_card(outer, "Summary Reports")

        sel_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sel_frm.pack(fill="x", pady=(0, 6))

        self.var_csv_use_smoothed = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel_frm, text="Use smoothed if available",
                         variable=self.var_csv_use_smoothed,
                         style="Card.TCheckbutton").pack(side="left", padx=(0, 12))
        ttk.Button(sel_frm, text="Refresh", style="Secondary.TButton",
                    command=self.refresh_csv_result_sets).pack(side="left", padx=(0, 4))
        ttk.Button(sel_frm, text="Load Selected", style="Secondary.TButton",
                    command=self._load_summary_csv).pack(side="left", padx=4)

        # Simulation selector
        sim_lbl_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sim_lbl_frm.pack(fill="x")
        ttk.Label(sim_lbl_frm, text="Simulations:",
                  style="Card.TLabel").pack(side="left")
        ttk.Button(sim_lbl_frm, text="All", style="Secondary.TButton",
                    command=self._csv_select_all).pack(side="right", padx=2)
        ttk.Button(sim_lbl_frm, text="None", style="Secondary.TButton",
                    command=self._csv_select_none).pack(side="right", padx=2)

        sim_list_frm = ttk.Frame(sum_card, style="Card.TFrame")
        sim_list_frm.pack(fill="x", pady=(2, 6))
        self._csv_sim_vars = {}
        sim_canvas = tk.Canvas(sim_list_frm, height=80, borderwidth=0,
                               highlightthickness=0, bg="white")
        sim_sb = ttk.Scrollbar(sim_list_frm, orient="vertical",
                                command=sim_canvas.yview)
        self._csv_sim_inner = ttk.Frame(sim_canvas, style="Card.TFrame")
        self._csv_sim_inner.bind(
            "<Configure>",
            lambda e: sim_canvas.configure(
                scrollregion=sim_canvas.bbox("all")))
        sim_canvas.create_window((0, 0), window=self._csv_sim_inner, anchor="nw")
        sim_canvas.configure(yscrollcommand=sim_sb.set)
        sim_canvas.pack(side="left", fill="both", expand=True)
        sim_sb.pack(side="right", fill="y")

        # Treeview
        ttk.Label(sum_card, text="Table (first selected sim):",
                  style="Card.TLabel").pack(anchor="w", pady=(2, 2))
        csv_tree_frm = ttk.Frame(sum_card, style="Card.TFrame")
        csv_tree_frm.pack(fill="both", expand=True)

        csv_cols = ("object", "total_power", "peak_density", "source")
        self.csv_tree = ttk.Treeview(csv_tree_frm, columns=csv_cols,
                                      show="headings", height=8)
        self.csv_tree.heading("object", text="Object")
        self.csv_tree.heading("total_power", text="Total Power [W]")
        self.csv_tree.heading("peak_density", text="Peak Density [W/m²]")
        self.csv_tree.heading("source", text="Source")
        self.csv_tree.column("object", width=160, anchor="w")
        self.csv_tree.column("total_power", width=130, anchor="e")
        self.csv_tree.column("peak_density", width=130, anchor="e")
        self.csv_tree.column("source", width=70, anchor="center")

        csv_vsb = ttk.Scrollbar(csv_tree_frm, orient="vertical",
                                 command=self.csv_tree.yview)
        self.csv_tree.configure(yscrollcommand=csv_vsb.set)
        self.csv_tree.pack(side="left", fill="both", expand=True)
        csv_vsb.pack(side="right", fill="y")

        self.var_csv_total_power = tk.StringVar(value="")
        ttk.Label(sum_card, textvariable=self.var_csv_total_power,
                  style="Card.TLabel",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))

        self.var_csv_status = tk.StringVar(
            value="Refresh, select simulations, then click Load.")
        ttk.Label(sum_card, textvariable=self.var_csv_status,
                  style="Card.TLabel",
                  foreground=colours["dim"]).pack(anchor="w", pady=(4, 0))

        # ===========================================================
        # RIGHT side — component filter + matplotlib bar charts
        # ===========================================================
        right_frm = ttk.Frame(top_pw, style="Card.TFrame", padding=8)
        top_pw.add(right_frm, weight=1)

        ttk.Label(right_frm, text="Comparison Charts",
                  style="CardHeader.TLabel").pack(anchor="w", pady=(0, 4))

        chart_pw = ttk.PanedWindow(right_frm, orient="horizontal")
        chart_pw.pack(fill="both", expand=True)

        # ---- Component filter ----
        comp_frm = ttk.Frame(chart_pw, style="Card.TFrame")
        chart_pw.add(comp_frm, weight=0)

        ttk.Label(comp_frm, text="Components:",
                  style="Card.TLabel").pack(anchor="w")
        btn_row_comp = ttk.Frame(comp_frm, style="Card.TFrame")
        btn_row_comp.pack(fill="x", pady=(0, 2))
        ttk.Button(btn_row_comp, text="All", style="Secondary.TButton",
                    command=self._chart_comp_select_all).pack(
                        side="left", padx=(0, 2))
        ttk.Button(btn_row_comp, text="None", style="Secondary.TButton",
                    command=self._chart_comp_select_none).pack(
                        side="left", padx=2)
        ttk.Button(btn_row_comp, text="Plot", style="Secondary.TButton",
                    command=self.update_csv_bar_plots).pack(
                        side="left", padx=2)

        sp_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        sp_frm.pack(fill="x", pady=(2, 2))
        ttk.Label(sp_frm, text="Species:", style="Card.TLabel").pack(side="left")
        self.var_chart_species = tk.StringVar(value="Total")
        ttk.Combobox(sp_frm, textvariable=self.var_chart_species,
                      values=["Total", "H⁻/D⁻ (negative)", "H⁰/D⁰ (neutrals)",
                              "H⁺/D⁺ (positive)"],
                      state="readonly", width=18).pack(side="left", padx=(4, 0))

        log_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        log_frm.pack(fill="x", pady=(2, 2))
        self.var_chart_log_peak = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_frm, text="Log scale (peak)",
                         variable=self.var_chart_log_peak,
                         style="Card.TCheckbutton").pack(anchor="w")
        self.var_chart_log_power = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_frm, text="Log scale (power)",
                         variable=self.var_chart_log_power,
                         style="Card.TCheckbutton").pack(anchor="w")

        comp_list_frm = ttk.Frame(comp_frm, style="Card.TFrame")
        comp_list_frm.pack(fill="both", expand=True, pady=(2, 0))
        self._chart_comp_vars = {}
        comp_canvas = tk.Canvas(comp_list_frm, width=220, borderwidth=0,
                                highlightthickness=0, bg="white")
        comp_sb = ttk.Scrollbar(comp_list_frm, orient="vertical",
                                 command=comp_canvas.yview)
        self._chart_comp_inner = ttk.Frame(comp_canvas, style="Card.TFrame")
        self._chart_comp_inner.bind(
            "<Configure>",
            lambda e: comp_canvas.configure(
                scrollregion=comp_canvas.bbox("all")))
        comp_canvas.create_window((0, 0), window=self._chart_comp_inner,
                                   anchor="nw")
        comp_canvas.configure(yscrollcommand=comp_sb.set)
        comp_canvas.pack(side="left", fill="both", expand=True)
        comp_sb.pack(side="right", fill="y")

        # ---- Plots ----
        plot_frm = ttk.Frame(chart_pw, style="Card.TFrame")
        chart_pw.add(plot_frm, weight=1)

        # Constrained layout (unlike tight_layout) reserves room for the
        # figure-level legend placed above the charts.
        self._csv_fig = Figure(figsize=(5, 6), dpi=100,
                                facecolor="white", layout="constrained")
        self._csv_ax_peak = self._csv_fig.add_subplot(2, 1, 1)
        self._csv_ax_power = self._csv_fig.add_subplot(
            2, 1, 2, sharex=self._csv_ax_peak)
        self._csv_legend = None
        self._csv_canvas_mpl = FigureCanvasTkAgg(self._csv_fig, plot_frm)
        self._csv_canvas_mpl.get_tk_widget().pack(fill="both", expand=True)

        # Hover metadata/artist for interactive value labels on bars.
        self._csv_bar_hover_targets = {"peak": [], "power": []}
        self._csv_hover_annot = None
        self._csv_canvas_mpl.mpl_connect("motion_notify_event", self._on_csv_plot_hover)
        self._csv_canvas_mpl.mpl_connect("figure_leave_event", self._on_csv_plot_leave)

        self._csv_ax_peak.set_title("Peak Heat Load [W/m²]", fontsize=10)
        self._csv_ax_power.set_title("Total Power [W]", fontsize=10)
        self._csv_canvas_mpl.draw()

        self._csv_plot_data = {}

        self.refresh_csv_result_sets()

    # ------------------------------------------------------------------
    #  Simulation-set management
    # ------------------------------------------------------------------
    def refresh_csv_result_sets(self):
        outdir = self._var_outdir.get()
        outdir_abs = (resolve_path(outdir) if not os.path.isabs(outdir) else outdir)
        sets = []
        if os.path.isdir(outdir_abs):
            for d in sorted(os.listdir(outdir_abs)):
                sub = os.path.join(outdir_abs, d)
                if os.path.isdir(sub):
                    sets.append(d)
        for w in self._csv_sim_inner.winfo_children():
            w.destroy()
        self._csv_sim_vars.clear()
        for name in sets:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self._csv_sim_inner, text=name, variable=var,
                             style="Card.TCheckbutton").pack(
                anchor="w", padx=2, pady=1)
            self._csv_sim_vars[name] = var

    def _csv_select_all(self):
        for v in self._csv_sim_vars.values():
            v.set(True)

    def _csv_select_none(self):
        for v in self._csv_sim_vars.values():
            v.set(False)

    def _chart_comp_select_all(self):
        for d in self._chart_comp_vars.values():
            d["var"].set(True)

    def _chart_comp_select_none(self):
        for d in self._chart_comp_vars.values():
            d["var"].set(False)

    # ------------------------------------------------------------------
    #  Component checklist
    # ------------------------------------------------------------------
    def _clear_csv_hover(self):
        if self._csv_hover_annot is not None:
            try:
                self._csv_hover_annot.remove()
            except Exception:
                pass
            self._csv_hover_annot = None

    def _on_csv_plot_leave(self, _event):
        self._clear_csv_hover()
        self._csv_canvas_mpl.draw_idle()

    def _on_csv_plot_hover(self, event):
        ax_peak = self._csv_ax_peak
        ax_power = self._csv_ax_power

        if event.inaxes == ax_peak:
            targets = self._csv_bar_hover_targets.get("peak", [])
            unit = "W/m²"
        elif event.inaxes == ax_power:
            targets = self._csv_bar_hover_targets.get("power", [])
            unit = "W"
        else:
            if self._csv_hover_annot is not None:
                self._clear_csv_hover()
                self._csv_canvas_mpl.draw_idle()
            return

        hit = None
        for item in targets:
            contains, _ = item["patch"].contains(event)
            if contains:
                hit = item
                break

        if hit is None:
            if self._csv_hover_annot is not None:
                self._clear_csv_hover()
                self._csv_canvas_mpl.draw_idle()
            return

        patch = hit["patch"]
        x = patch.get_x() + patch.get_width() * 0.5
        y = patch.get_height()
        txt = f"{hit['obj']}\n{hit['sim']}\n{hit['value']:.3e} {unit}"

        if self._csv_hover_annot is None or self._csv_hover_annot.axes != event.inaxes:
            self._clear_csv_hover()
            self._csv_hover_annot = event.inaxes.annotate(
                txt,
                xy=(x, y),
                xytext=(10, 8),
                textcoords="offset points",
                fontsize=8,
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#64748b", alpha=0.9),
                zorder=20,
            )
        else:
            self._csv_hover_annot.xy = (x, y)
            self._csv_hover_annot.set_text(txt)

        self._csv_canvas_mpl.draw_idle()

    def _refresh_chart_comp_list(self):
        all_objects = []
        for entries in self._csv_plot_data.values():
            for e in entries:
                if e["name"] not in all_objects:
                    all_objects.append(e["name"])

        prev = {}
        for k, d in self._chart_comp_vars.items():
            prev[k] = {
                "checked": d["var"].get(),
                "label": d["label"].get(),
                "mult": d["mult"].get(),
            }

        for w in self._chart_comp_inner.winfo_children():
            w.destroy()
        self._chart_comp_vars.clear()

        hdr = ttk.Frame(self._chart_comp_inner, style="Card.TFrame")
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        ttk.Label(hdr, text="", style="Card.TLabel", width=2).pack(side="left")
        ttk.Label(hdr, text="Label", style="Card.TLabel", width=14).pack(
            side="left", padx=(2, 0))
        ttk.Label(hdr, text="Mult", style="Card.TLabel", width=5).pack(
            side="left", padx=(2, 0))

        for name in all_objects:
            p = prev.get(name, {})
            var = tk.BooleanVar(value=p.get("checked", True))
            label_var = tk.StringVar(value=p.get("label", name))
            mult_var = tk.StringVar(value=p.get("mult", "1.0"))

            row_frm = ttk.Frame(self._chart_comp_inner, style="Card.TFrame")
            row_frm.pack(fill="x", padx=2, pady=1)
            ttk.Checkbutton(row_frm, variable=var,
                             style="Card.TCheckbutton").pack(side="left")
            ttk.Entry(row_frm, textvariable=label_var, width=14,
                       font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))
            ttk.Entry(row_frm, textvariable=mult_var, width=5,
                       font=("Segoe UI", 9)).pack(side="left", padx=(2, 0))

            self._chart_comp_vars[name] = {
                "var": var, "label": label_var, "mult": mult_var,
            }

    # ------------------------------------------------------------------
    #  CSV loading
    # ------------------------------------------------------------------
    def _load_summary_csv(self):
        selected = [name for name, var in self._csv_sim_vars.items() if var.get()]
        if not selected:
            self.var_csv_status.set("No simulations selected.")
            return

        outdir = self._var_outdir.get()
        outdir_abs = resolve_path(outdir) if not os.path.isabs(outdir) else outdir

        def _find_csv(search_dir, extra_candidates=None):
            candidates = list(extra_candidates or [])
            if os.path.isdir(search_dir):
                for f in sorted(glob.glob(os.path.join(search_dir, "*.csv"))):
                    if f not in candidates:
                        candidates.append(f)
            for c in candidates:
                if os.path.isfile(c):
                    return c
            return None

        def _read_csv(path):
            with open(path, "r", newline="") as fh:
                return list(csv_mod.DictReader(fh))

        def _detect_keys(rows):
            name_key = power_key = density_key = None
            species_power = {}
            species_density = {}
            if not rows:
                return None, None, None, {}, {}
            keys = list(rows[0].keys())
            for k in keys:
                kl = k.lower()
                if "name" in kl or "file" in kl:
                    name_key = k
                elif "total" in kl and "power" in kl:
                    for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                        if k.endswith(suffix):
                            species_power[suffix[1:]] = k
                            break
                    else:
                        if power_key is None:
                            power_key = k
                elif "peak" in kl or "density" in kl:
                    for suffix in ("_H-", "_H0", "_H+", "_D-", "_D0", "_D+"):
                        if k.endswith(suffix):
                            species_density[suffix[1:]] = k
                            break
                    else:
                        if density_key is None:
                            density_key = k
            if name_key is None and len(keys) >= 1:
                name_key = keys[0]
            if power_key is None and len(keys) >= 2:
                power_key = keys[1]
            if density_key is None and len(keys) >= 3:
                density_key = keys[2]
            return name_key, power_key, density_key, species_power, species_density

        def _norm(name):
            base = os.path.splitext(os.path.basename(str(name)))[0]
            for prefix in ("results_", "smoothed_"):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            return base

        self._csv_plot_data.clear()
        first_display_order = None
        first_merged = None
        first_raw_path = None
        n_loaded = 0
        n_errors = 0

        for result_set in selected:
            base_dir = os.path.join(outdir_abs, result_set)
            raw_candidates = [
                os.path.join(base_dir, f"summary_{result_set}.csv"),
                os.path.join(base_dir, "summary_report.csv"),
                os.path.join(base_dir, "power_summary_by_object.csv"),
            ]
            raw_path = _find_csv(base_dir, raw_candidates)
            if raw_path is None:
                n_errors += 1
                continue

            try:
                raw_rows = _read_csv(raw_path)
            except Exception:
                n_errors += 1
                continue

            raw_nk, raw_pk, raw_dk, raw_sp, raw_sd = _detect_keys(raw_rows)

            merged = {}
            display_order = []
            for row in raw_rows:
                obj_raw = row.get(raw_nk, "?") if raw_nk else "?"
                obj = _norm(obj_raw)
                tp = row.get(raw_pk, "N/A") if raw_pk else "N/A"
                pd_ = row.get(raw_dk, "N/A") if raw_dk else "N/A"
                entry = {"name": obj, "tp": tp, "pd": pd_, "source": "raw"}
                for sp_key, col in raw_sp.items():
                    entry[f"tp_{sp_key}"] = row.get(col, "N/A")
                for sp_key, col in raw_sd.items():
                    entry[f"pd_{sp_key}"] = row.get(col, "N/A")
                merged[obj] = entry
                if obj not in display_order:
                    display_order.append(obj)

            if self.var_csv_use_smoothed.get():
                sm_dir = os.path.join(base_dir, "SMOOTHED")
                sm_candidates = [
                    os.path.join(sm_dir, "smoothed_summary.csv"),
                    os.path.join(sm_dir, "summary_report.csv"),
                ]
                smoothed_path = _find_csv(sm_dir, sm_candidates)
                if smoothed_path:
                    try:
                        sm_rows = _read_csv(smoothed_path)
                        sm_nk, sm_pk, sm_dk, sm_sp, sm_sd = _detect_keys(sm_rows)
                        for row in sm_rows:
                            obj_raw = row.get(sm_nk, "?") if sm_nk else "?"
                            obj = _norm(obj_raw)
                            tp = row.get(sm_pk, "N/A") if sm_pk else "N/A"
                            pd_ = row.get(sm_dk, "N/A") if sm_dk else "N/A"
                            prev_entry = merged.get(obj, {"name": obj})
                            entry = dict(prev_entry)
                            entry.update({"tp": tp, "pd": pd_, "source": "smoothed"})
                            for sp_key, col in sm_sp.items():
                                entry[f"tp_{sp_key}"] = row.get(col, "N/A")
                            for sp_key, col in sm_sd.items():
                                entry[f"pd_{sp_key}"] = row.get(col, "N/A")
                            merged[obj] = entry
                            if obj not in display_order:
                                display_order.append(obj)
                    except Exception:
                        pass

            entries = [merged[o] for o in display_order]
            self._csv_plot_data[result_set] = entries
            n_loaded += 1

            if first_merged is None:
                first_merged = merged
                first_display_order = display_order
                first_raw_path = raw_path

        # Populate treeview
        for item in self.csv_tree.get_children():
            self.csv_tree.delete(item)

        total_power_sum = 0.0
        peak_max = 0.0
        if first_merged and first_display_order:
            for obj in first_display_order:
                d = first_merged[obj]
                self.csv_tree.insert("", "end",
                                     values=(d["name"], d["tp"], d["pd"],
                                             d["source"]))
                try:
                    total_power_sum += float(d["tp"])
                except (ValueError, TypeError):
                    pass
                try:
                    peak_max = max(peak_max, float(d["pd"]))
                except (ValueError, TypeError):
                    pass

        if total_power_sum > 0:
            self.var_csv_total_power.set(
                f"Total deposited power:  {total_power_sum:.4e} W")
        else:
            self.var_csv_total_power.set("")

        info_parts = []
        if first_raw_path:
            info_parts.append(os.path.basename(first_raw_path))
        info_parts.append(f"{n_loaded} sim(s) loaded")
        if n_errors:
            info_parts.append(f"{n_errors} failed")
        if peak_max > 0:
            info_parts.append(f"Max peak = {peak_max:.4e} W/m²")
        self.var_csv_status.set("  |  ".join(info_parts))

        self._refresh_chart_comp_list()
        self.update_csv_bar_plots()

    # ------------------------------------------------------------------
    #  Bar-plot rendering
    # ------------------------------------------------------------------
    def update_csv_bar_plots(self):
        ax_peak = self._csv_ax_peak
        ax_power = self._csv_ax_power
        ax_peak.clear()
        ax_power.clear()
        if self._csv_legend is not None:
            self._csv_legend.remove()
            self._csv_legend = None
        self._csv_bar_hover_targets = {"peak": [], "power": []}
        self._clear_csv_hover()

        data = self._csv_plot_data
        if not data:
            ax_peak.set_title("Peak Heat Load [W/m²]", fontsize=10)
            ax_power.set_title("Total Power [W]", fontsize=10)
            self._csv_canvas_mpl.draw()
            return

        sp_sel = self.var_chart_species.get()
        _species_map = {
            "H⁻/D⁻ (negative)": "H-",
            "H⁰/D⁰ (neutrals)": "H0",
            "H⁺/D⁺ (positive)": "H+",
        }
        sp_suffix = _species_map.get(sp_sel)
        tp_key = f"tp_{sp_suffix}" if sp_suffix else "tp"
        pd_key = f"pd_{sp_suffix}" if sp_suffix else "pd"
        species_label = f" ({sp_suffix})" if sp_suffix else ""

        sim_names = list(data.keys())

        all_objects = []
        for sim in sim_names:
            for entry in data[sim]:
                if entry["name"] not in all_objects:
                    all_objects.append(entry["name"])
        if self._chart_comp_vars:
            all_objects = [o for o in all_objects
                           if self._chart_comp_vars.get(
                               o, {"var": tk.BooleanVar(value=True)}
                           )["var"].get()]

        n_objects = len(all_objects)
        n_sims = len(sim_names)

        if n_objects == 0:
            self._csv_canvas_mpl.draw()
            return

        cmap = plt.get_cmap("tab10" if n_sims <= 10 else "tab20")
        colours = [cmap(i % cmap.N) for i in range(n_sims)]

        obj_mult = {}
        for obj in all_objects:
            d = self._chart_comp_vars.get(obj)
            if d:
                try:
                    obj_mult[obj] = float(d["mult"].get())
                except (ValueError, TypeError):
                    obj_mult[obj] = 1.0
            else:
                obj_mult[obj] = 1.0

        x = np.arange(n_objects)
        total_bar_width = 0.8
        bar_width = total_bar_width / max(n_sims, 1)
        # A white edge separates wide bars; on thin bars it covers most of the
        # fill and makes them look washed out.
        bar_edge = "white" if n_objects * n_sims <= 80 else "none"

        for si, sim in enumerate(sim_names):
            lookup = {e["name"]: e for e in data[sim]}
            peaks = []
            powers = []
            for obj in all_objects:
                m = obj_mult[obj]
                e = lookup.get(obj)
                if e:
                    try:
                        peaks.append(float(e.get(pd_key, "N/A")) * m)
                    except (ValueError, TypeError):
                        peaks.append(0.0)
                    try:
                        powers.append(float(e.get(tp_key, "N/A")) * m)
                    except (ValueError, TypeError):
                        powers.append(0.0)
                else:
                    peaks.append(0.0)
                    powers.append(0.0)

            offset = (si - (n_sims - 1) / 2) * bar_width
            label = sim if len(sim) <= 30 else sim[:27] + "…"
            peak_bars = ax_peak.bar(x + offset, peaks, bar_width * 0.9,
                                    label=label, color=colours[si], edgecolor=bar_edge,
                                    linewidth=0.5)
            power_bars = ax_power.bar(x + offset, powers, bar_width * 0.9,
                                       label=label, color=colours[si], edgecolor=bar_edge,
                                       linewidth=0.5)

            for obj, val, patch in zip(all_objects, peaks, peak_bars.patches):
                self._csv_bar_hover_targets["peak"].append(
                    {"patch": patch, "value": float(val), "obj": obj, "sim": sim}
                )
            for obj, val, patch in zip(all_objects, powers, power_bars.patches):
                self._csv_bar_hover_targets["power"].append(
                    {"patch": patch, "value": float(val), "obj": obj, "sim": sim}
                )

        display_labels = []
        for obj in all_objects:
            d = self._chart_comp_vars.get(obj)
            if d:
                display_labels.append(d["label"].get())
            else:
                display_labels.append(obj)

        # The x axis is shared, so component names are drawn only under the
        # bottom chart. Many components get vertical, smaller labels so that
        # neighbouring names do not overlap.
        if n_objects <= 15:
            rotation, ha, label_fs = 45, "right", 8
        else:
            rotation, ha = 90, "center"
            label_fs = 7 if n_objects <= 40 else 6
        ax_power.set_xticks(x)
        ax_power.set_xticklabels(display_labels, rotation=rotation, ha=ha,
                                 rotation_mode="anchor" if rotation == 45 else "default",
                                 fontsize=label_fs)
        ax_power.set_xlim(-0.5, n_objects - 0.5)
        ax_peak.tick_params(axis="x", labelbottom=False)

        for ax, title, unit, log_var in [
            (ax_peak, "Peak Heat Load", "W/m²", self.var_chart_log_peak),
            (ax_power, "Total Deposited Power", "W", self.var_chart_log_power),
        ]:
            ax.set_title(f"{title}{species_label} [{unit}]",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel(unit, fontsize=9)
            if log_var.get():
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)

        # One shared legend above both charts instead of one inside each,
        # where it covered the bars.
        if n_sims > 1:
            handles, labels = ax_power.get_legend_handles_labels()
            self._csv_legend = self._csv_fig.legend(
                handles, labels, loc="outside upper center",
                ncol=min(n_sims, 4), fontsize=7, frameon=False)

        self._csv_canvas_mpl.draw()


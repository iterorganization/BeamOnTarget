# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Transform tab — export result files to another reference frame as CSV / VTP.

The transform itself lives in ``io/transform_results.py``; this tab selects
the files, shows which outputs will be written, and collects the transform,
and export settings.
"""
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext

from beamontarget.gui.gui_widgets import (
    make_card,
    resolve_path as _resolve_path,
    to_relative_path as _to_relative_path,
)
from beamontarget.io import transform_results as tr

_SOURCE_LABELS = {
    "raw": "Raw (results_*.vtp)",
    "smoothed": "Smoothed (SMOOTHED/smoothed_results_*.vtp)",
    "merged": "Merged (merged_results_*.vtp)",
}
_CUSTOM = "Custom"
_FORMATS = {"csv": ("csv",), "vtp": ("vtp",), "both": ("csv", "vtp")}


class TransformTab(ttk.Frame):
    """Coordinate transform and CSV / VTP export of result files."""

    def __init__(self, parent, cfg, colours, *, get_output_dir):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._get_output_dir = get_output_dir
        self._presets = tr.presets()
        self._included = set()       # scenario folders ticked for export
        self._scen_items = {}        # {tree item: scenario folder}
        self._last_clicked = None
        self._jobs = []
        self._plan_job = None
        self._worker = None
        self._stop_event = threading.Event()
        self._log_queue = queue.Queue()
        self._setting_preset = False
        self._build(cfg.get("TRANSFORM", {}) or {})

    @staticmethod
    def _abs(path):
        return os.path.normpath(path if os.path.isabs(path) else _resolve_path(path))

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self, tc):
        dim = self._colours["dim"]

        # --- Inputs ---
        in_card = make_card(self, "Inputs", pady=(12, 8))
        ttk.Label(in_card, text="Result folders:", style="Card.TLabel").grid(
            row=0, column=0, sticky="nw", pady=2)
        ff = ttk.Frame(in_card, style="Card.TFrame")
        ff.grid(row=0, column=1, columnspan=3, sticky="we", padx=(8, 0))
        self._folders_list = tk.Listbox(ff, height=2, activestyle="none",
                                        selectmode="extended", bg="white",
                                        highlightthickness=1, bd=0)
        self._folders_list.pack(side="left", fill="x", expand=True)
        fb = ttk.Frame(ff, style="Card.TFrame")
        fb.pack(side="left", padx=(6, 0), anchor="n")
        ttk.Button(fb, text="Add...", style="Secondary.TButton",
                   command=self._add_folder).pack(fill="x")
        ttk.Button(fb, text="Remove", style="Secondary.TButton",
                   command=self._remove_folders).pack(fill="x", pady=(2, 0))
        self._set_folders(self.cfg)

        ttk.Label(in_card, text="Source files:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 2))
        sf = ttk.Frame(in_card, style="Card.TFrame")
        sf.grid(row=1, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(6, 2))
        self.var_source = tk.StringVar(value=tc.get("source", "raw"))
        for value, label in _SOURCE_LABELS.items():
            ttk.Radiobutton(sf, text=label, variable=self.var_source, value=value,
                            style="Card.TRadiobutton",
                            command=self.load_scenarios).pack(side="left", padx=(0, 16))

        ttk.Label(in_card, text="Name filter:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=2)
        self.var_filter = tk.StringVar(value=tc.get("name_filter", ""))
        ttk.Entry(in_card, textvariable=self.var_filter, width=40).grid(
            row=2, column=1, sticky="w", padx=(8, 0))
        ttk.Label(in_card, text="comma-separated; a file is used if its name contains any term",
                  style="Card.TLabel", foreground=dim).grid(row=2, column=2, sticky="w", padx=(8, 0))
        self.var_filter.trace_add("write", lambda *_: self._schedule_plan())

        lf = ttk.Frame(in_card, style="Card.TFrame")
        lf.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Button(lf, text="Load Scenarios", style="Accent.TButton",
                   command=self.load_scenarios).pack(side="left")
        self.var_load_status = tk.StringVar(
            value="Click to list the scenarios. For merged files, add the MERGED folder.")
        ttk.Label(lf, textvariable=self.var_load_status, style="Card.TLabel",
                  foreground=dim).pack(side="left", padx=(10, 0))
        in_card.columnconfigure(2, weight=1)

        # --- Settings + run: fixed at the bottom, always fully visible ---
        bottom_wrap = ttk.Frame(self, style="Card.TFrame", padding=14)
        bottom_wrap.pack(side="bottom", fill="x", padx=12, pady=(0, 10))
        settings = ttk.Frame(bottom_wrap, style="Card.TFrame")
        settings.pack(fill="x")
        self._build_transform_settings(settings, tc)
        self._build_output_settings(settings, tc)
        for col in range(3):
            settings.columnconfigure(col, weight=1)

        rb = ttk.Frame(bottom_wrap, style="Card.TFrame")
        rb.pack(fill="x", pady=(10, 0))
        self._run_btn = ttk.Button(rb, text="Run Transform", style="Accent.TButton",
                                   command=self._run)
        self._run_btn.pack(side="left")
        self._stop_btn = ttk.Button(rb, text="Stop", style="Danger.TButton",
                                    command=self._stop_event.set, state="disabled")
        self._stop_btn.pack(side="left", padx=(6, 0))
        self.var_run_status = tk.StringVar(value="")
        ttk.Label(rb, textvariable=self.var_run_status, style="Card.TLabel",
                  foreground=dim).pack(side="left", padx=(10, 0))

        # --- Scenarios | export plan, and the log, share the remaining height ---
        vpw = ttk.PanedWindow(self, orient="vertical")
        vpw.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        mid = ttk.Frame(vpw, style="Card.TFrame", padding=14)
        pw = ttk.PanedWindow(mid, orient="horizontal")
        pw.pack(fill="both", expand=True)
        pw.add(self._build_scenario_pane(pw), weight=1)
        pw.add(self._build_plan_pane(pw), weight=2)
        log_wrap = ttk.Frame(vpw, style="Card.TFrame", padding=(14, 8))
        self._log = scrolledtext.ScrolledText(log_wrap, height=3, wrap="none",
                                              font=("Consolas", 9), state="disabled")
        self._log.pack(fill="both", expand=True)
        vpw.add(mid, weight=4)
        vpw.add(log_wrap, weight=1)
        self._vpw = vpw

    def _build_scenario_pane(self, parent):
        outer = ttk.Frame(parent, style="Card.TFrame")
        hdr = ttk.Frame(outer, style="Card.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Scenarios  (click to include; Shift+click: range)",
                  style="Card.TLabel").pack(side="left")
        ttk.Button(hdr, text="None", style="Secondary.TButton",
                   command=lambda: self._set_all(False)).pack(side="right")
        ttk.Button(hdr, text="All", style="Secondary.TButton",
                   command=lambda: self._set_all(True)).pack(side="right", padx=(0, 3))
        box = ttk.Frame(outer, style="Card.TFrame")
        box.pack(fill="both", expand=True, pady=(2, 0))
        self._scen_tree = ttk.Treeview(box, columns=("use", "files"), show="tree headings",
                                       selectmode="none")
        self._scen_tree.heading("#0", text="Scenario")
        self._scen_tree.heading("use", text="Export")
        self._scen_tree.heading("files", text="Files")
        self._scen_tree.column("#0", width=220, stretch=True)
        self._scen_tree.column("use", width=60, anchor="center", stretch=False)
        self._scen_tree.column("files", width=50, anchor="center", stretch=False)
        self._scen_tree.tag_configure("on", foreground="#111827", background="#e8f0fe")
        self._scen_tree.tag_configure("off", foreground="#6b7280")
        self._scen_tree.tag_configure("empty", foreground="#9ca3af")
        self._scen_tree.tag_configure("root", foreground=self._colours["accent"],
                                      font=("Segoe UI", 9, "bold"))
        vsb = ttk.Scrollbar(box, orient="vertical", command=self._scen_tree.yview)
        self._scen_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._scen_tree.pack(side="left", fill="both", expand=True)
        self._scen_tree.bind("<Button-1>", self._on_scen_click)
        self._scen_tree.bind("<Shift-Button-1>", self._on_scen_shift_click)
        return outer

    def _build_plan_pane(self, parent):
        outer = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(outer, text="Export plan  (input file -> files written)",
                  style="Card.TLabel").pack(anchor="w")
        self.var_plan_status = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.var_plan_status, style="Card.TLabel",
                  foreground=self._colours["dim"], wraplength=700,
                  justify="left").pack(side="bottom", anchor="w", pady=(2, 0))
        box = ttk.Frame(outer, style="Card.TFrame")
        box.pack(fill="both", expand=True, pady=(2, 0))
        self._plan_tree = ttk.Treeview(box, columns=("path",), show="tree headings",
                                       selectmode="browse")
        self._plan_tree.heading("#0", text="Scenario / input file")
        self._plan_tree.heading("path", text="Output (relative to the output folder)")
        self._plan_tree.column("#0", width=240, stretch=True)
        self._plan_tree.column("path", width=360, stretch=True)
        vsb = ttk.Scrollbar(box, orient="vertical", command=self._plan_tree.yview)
        self._plan_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._plan_tree.pack(side="left", fill="both", expand=True)
        return outer

    def _build_transform_settings(self, parent, tc):
        frm = ttk.Frame(parent, style="Card.TFrame")
        frm.grid(row=0, column=0, sticky="nw")
        ttk.Label(frm, text="Transform", style="CardHeader.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

        ttk.Label(frm, text="Preset:", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.var_preset = tk.StringVar(value=tc.get("preset", "No transformation"))
        ttk.Combobox(frm, textvariable=self.var_preset, state="readonly", width=24,
                     values=list(self._presets) + [_CUSTOM]).grid(
            row=1, column=1, columnspan=3, sticky="w", padx=(8, 0))
        self.var_preset.trace_add("write", lambda *_: self._apply_preset())

        ttk.Label(frm, text="Rotation about Z (deg):", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=(4, 0))
        self.var_angle = tk.StringVar()
        ttk.Entry(frm, textvariable=self.var_angle, width=12).grid(
            row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        ttk.Label(frm, text="Translation X, Y, Z (m):", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=(4, 0))
        self.var_dx, self.var_dy, self.var_dz = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for i, v in enumerate((self.var_dx, self.var_dy, self.var_dz)):
            ttk.Entry(frm, textvariable=v, width=12).grid(
                row=3, column=1 + i, sticky="w", padx=(8 if i == 0 else 4, 0), pady=(4, 0))

        ttk.Label(frm, text="p' = R p + t: rotate about Z, then translate. "
                            "All coordinates in m, as in the simulation.",
                  style="Card.TLabel", foreground=self._colours["dim"]).grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Restore saved values; editing a value switches the preset to Custom.
        saved = tc.get("transform")
        if saved and self.var_preset.get() == _CUSTOM:
            self._fill_transform(tr.FrameTransform(
                float(saved.get("angle_deg", 0.0)), tuple(saved.get("translation_m", (0, 0, 0)))))
        else:
            self._apply_preset()
        for v in (self.var_angle, self.var_dx, self.var_dy, self.var_dz):
            v.trace_add("write", lambda *_: self._on_transform_edited())

    def _build_output_settings(self, parent, tc):
        frm = ttk.Frame(parent, style="Card.TFrame")
        frm.grid(row=0, column=1, sticky="nw", padx=(24, 0))
        ttk.Label(frm, text="Output", style="CardHeader.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        ttk.Label(frm, text="Format:", style="Card.TLabel").grid(row=1, column=0, sticky="w")
        self.var_format = tk.StringVar(value=tc.get("format", "csv"))
        fmt = ttk.Frame(frm, style="Card.TFrame")
        fmt.grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 0))
        for value, label in (("csv", "CSV"), ("vtp", "VTP"), ("both", "Both")):
            ttk.Radiobutton(fmt, text=label, variable=self.var_format, value=value,
                            style="Card.TRadiobutton",
                            command=self._schedule_plan).pack(side="left", padx=(0, 12))

        ttk.Label(frm, text="CSV columns:", style="Card.TLabel").grid(
            row=2, column=0, sticky="nw", pady=(4, 0))
        cf = ttk.Frame(frm, style="Card.TFrame")
        cf.grid(row=2, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(4, 0))
        cols = tc.get("csv_columns", {})
        self._col_vars = {}
        for key, label in (("geometry", "X, Y, Z (cell centres)"),
                           ("area", "Area (m²)"),
                           ("power", "Deposited_Power_W"),
                           ("density", "Power_Density_W_m2"),
                           ("total_power", "total_deposited_power_W")):
            v = tk.BooleanVar(value=cols.get(key, True))
            ttk.Checkbutton(cf, text=label, variable=v, style="Card.TCheckbutton").pack(anchor="w")
            self._col_vars[key] = v

        frm = ttk.Frame(parent, style="Card.TFrame")
        frm.grid(row=0, column=2, sticky="nw", padx=(24, 0))
        ttk.Label(frm, text="Values", style="CardHeader.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(frm, text="Power factor:", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        self.var_factor = tk.StringVar(value=str(tc.get("power_factor", 1.0)))
        ttk.Entry(frm, textvariable=self.var_factor, width=12).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(frm, text="applied to all power arrays in CSV and VTP\n(e.g. 1e-6 for MW, MW/m²)",
                  style="Card.TLabel", foreground=self._colours["dim"], justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.var_ignore_zeros = tk.BooleanVar(value=tc.get("ignore_zeros", False))
        ttk.Checkbutton(frm, text="Ignore zero-valued rows (cells without power)",
                        variable=self.var_ignore_zeros, style="Card.TCheckbutton").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="Output folder:", style="Card.TLabel").grid(
            row=6, column=0, sticky="w", pady=(6, 0))
        of = ttk.Frame(frm, style="Card.TFrame")
        of.grid(row=6, column=1, columnspan=2, sticky="we", padx=(8, 0), pady=(6, 0))
        self.var_outdir = tk.StringVar(value=tc.get("output_dir", "TRANSFORMED"))
        ttk.Entry(of, textvariable=self.var_outdir, width=24).pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="Browse...", style="Secondary.TButton",
                   command=self._browse_outdir).pack(side="left", padx=(4, 0))
        self.var_outdir.trace_add("write", lambda *_: self._schedule_plan())

    # ------------------------------------------------------------------
    #  Transform values
    # ------------------------------------------------------------------
    def _fill_transform(self, t):
        self._setting_preset = True
        try:
            # repr keeps every digit: "%g" would round 11.410436 to 11.4104.
            self.var_angle.set(repr(float(t.angle_deg)))
            for v, x in zip((self.var_dx, self.var_dy, self.var_dz), t.translation_m):
                v.set(repr(float(x)))
        finally:
            self._setting_preset = False

    def _apply_preset(self):
        t = self._presets.get(self.var_preset.get())
        if t is not None:
            self._fill_transform(t)

    def _on_transform_edited(self):
        if self._setting_preset:
            return
        t = self._presets.get(self.var_preset.get())
        if t is None:
            return
        try:
            typed = tr.FrameTransform(
                float(self.var_angle.get()),
                (float(self.var_dx.get()), float(self.var_dy.get()), float(self.var_dz.get())))
            if typed == t:
                return
        except ValueError:
            pass
        # Any edit that changes a value leaves the preset.
        self.var_preset.set(_CUSTOM)

    def current_transform(self):
        """The selected preset exactly, or else the entry fields (ValueError if invalid)."""
        preset = self._presets.get(self.var_preset.get())
        if preset is not None:
            return preset
        return tr.FrameTransform(
            float(self.var_angle.get()),
            (float(self.var_dx.get()), float(self.var_dy.get()), float(self.var_dz.get())))

    # ------------------------------------------------------------------
    #  Folders and scenarios
    # ------------------------------------------------------------------
    def _folders(self):
        return [self._folders_list.get(i) for i in range(self._folders_list.size())]

    def _show_folders(self, folders):
        self._folders_list.delete(0, "end")
        for f in folders:
            self._folders_list.insert("end", f)

    def _set_folders(self, cfg):
        """Saved folder list, or else the config's output folder (followed live)."""
        saved = (cfg.get("TRANSFORM") or {}).get("folders")
        self._folders_custom = bool(saved)
        self._show_folders(saved or [cfg.get("DETAILED_OUTPUT_DIR", "OUTPUT")])

    def _add_folder(self):
        d = filedialog.askdirectory(title="Select a result or MERGED folder", parent=self)
        if d:
            self._folders_list.insert("end", _to_relative_path(d))
            self._folders_custom = True

    def _remove_folders(self):
        for i in reversed(self._folders_list.curselection()):
            self._folders_list.delete(i)
            self._folders_custom = True

    def _browse_outdir(self):
        d = filedialog.askdirectory(title="Select output folder", parent=self)
        if d:
            self.var_outdir.set(_to_relative_path(d))

    def load_scenarios(self):
        """List every folder holding files of the selected source."""
        if not self._folders_custom:
            self._show_folders([self._get_output_dir()])
        source = self.var_source.get()
        self._scen_tree.delete(*self._scen_tree.get_children())
        self._scen_items.clear()
        n = 0
        for folder in self._folders():
            root = self._abs(folder)
            dirs = tr.find_result_dirs(root, source)
            r_id = self._scen_tree.insert("", "end", text=folder, open=True, tags=("root",),
                                          values=("", len(dirs) or "none"))
            for d in dirs:
                rel = os.path.relpath(d, root)
                item = self._scen_tree.insert(r_id, "end", text=rel if rel != "." else os.path.basename(d))
                self._scen_items[item] = d
                n += 1
        self._included &= set(self._scen_items.values())
        self.var_load_status.set(f"{n} folder(s) with {source} files.")
        self._refresh_plan()

    def _toggle(self, item, value=None):
        d = self._scen_items.get(item)
        if d is None:
            return
        on = (d not in self._included) if value is None else value
        (self._included.add if on else self._included.discard)(d)

    def _on_scen_click(self, event):
        item = self._scen_tree.identify_row(event.y)
        if item in self._scen_items:
            self._toggle(item)
            self._last_clicked = item
            self._refresh_plan()
        elif item:
            # A root row toggles all its scenarios.
            kids = self._scen_tree.get_children(item)
            on = not all(self._scen_items[k] in self._included for k in kids)
            for k in kids:
                self._toggle(k, on)
            self._refresh_plan()
        return "break" if item in self._scen_items else None

    def _on_scen_shift_click(self, event):
        item = self._scen_tree.identify_row(event.y)
        order = list(self._scen_items)
        if item in self._scen_items and self._last_clicked in order:
            lo, hi = sorted((order.index(self._last_clicked), order.index(item)))
            for k in order[lo:hi + 1]:
                self._toggle(k, True)
            self._last_clicked = item
            self._refresh_plan()
        return "break"

    def _set_all(self, on):
        for item in self._scen_items:
            self._toggle(item, on)
        self._refresh_plan()

    # ------------------------------------------------------------------
    #  Export plan
    # ------------------------------------------------------------------
    def _schedule_plan(self):
        if self._plan_job is not None:
            self.after_cancel(self._plan_job)
        self._plan_job = self.after(250, self._refresh_plan)

    def _output_root(self):
        return self._abs(self.var_outdir.get().strip() or "TRANSFORMED")

    def _refresh_plan(self):
        self._plan_job = None
        source = self.var_source.get()
        terms = tr.filter_terms(self.var_filter.get())
        formats = _FORMATS[self.var_format.get()]
        out_root = self._output_root()

        self._plan_tree.delete(*self._plan_tree.get_children())
        self._jobs = []
        for item, d in self._scen_items.items():
            files = [f for f in tr.source_files(d, source) if tr.matches_filter(f, terms)]
            used = d in self._included
            self._scen_tree.item(item, values=("Yes" if used else "", len(files)),
                                 tags=("empty",) if not files else ("on",) if used else ("off",))
            if not used or not files:
                continue
            s_id = self._plan_tree.insert("", "end", text=self._scen_tree.item(item, "text"),
                                          open=True, values=(f"{len(files)} file(s)",))
            for f in files:
                paths = tr.output_paths(f, d, out_root)
                outs = [os.path.relpath(paths[k], out_root) for k in formats]
                self._plan_tree.insert(s_id, "end", text=f.name, values=("   |   ".join(outs),))
                self._jobs.append((f, paths))

        if not self._scen_items:
            self.var_plan_status.set("Load scenarios, then click them to include them.")
        elif not self._jobs:
            self.var_plan_status.set("No file selected: include scenarios, or check the name filter.")
        else:
            self.var_plan_status.set(
                f"{len(self._jobs)} input file(s) -> {len(self._jobs) * len(formats)} output file(s) "
                f"in {out_root}")

    # ------------------------------------------------------------------
    #  Run
    # ------------------------------------------------------------------
    def _append_log(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _run(self):
        if self._worker is not None:
            return
        self._refresh_plan()
        if not self._jobs:
            messagebox.showwarning("Transform", self.var_plan_status.get(), parent=self)
            return
        try:
            transform = self.current_transform()
        except ValueError:
            messagebox.showerror("Transform", "Rotation and translation must be numbers.", parent=self)
            return
        try:
            factor = float(self.var_factor.get())
        except ValueError:
            messagebox.showerror("Transform", "The power factor must be a number.", parent=self)
            return
        formats = _FORMATS[self.var_format.get()]
        cols = {k: v.get() for k, v in self._col_vars.items()}
        if "csv" in formats and not any(cols.values()):
            messagebox.showwarning("Transform", "Select at least one CSV column.", parent=self)
            return
        existing = [p for _, paths in self._jobs for k, p in paths.items() if k in formats and p.exists()]
        if existing and not messagebox.askyesno(
                "Overwrite?", f"{len(existing)} output file(s) already exist and will be "
                f"overwritten, e.g.\n{existing[0]}\n\nContinue?", parent=self):
            return

        jobs = list(self._jobs)
        kwargs = dict(power_factor=factor, ignore_zeros=self.var_ignore_zeros.get(),
                      formats=formats, csv_columns=cols)
        self._stop_event.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self.var_run_status.set("Transforming...")
        # Open the log to about five lines (it starts small to leave room for the plan).
        h = self._vpw.winfo_height()
        if h > 200 and h - self._vpw.sashpos(0) < 100:
            self._vpw.sashpos(0, h - 110)
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._log_queue.put(f"Preset: {self.var_preset.get()}")

        def _worker():
            try:
                tr.run_transform(jobs, transform, log=self._log_queue.put,
                                 stop_event=self._stop_event, **kwargs)
            except Exception as exc:
                self._log_queue.put(f"ERROR: {exc}")
            finally:
                self._log_queue.put(None)

        self._worker = threading.Thread(target=_worker, daemon=True)
        self._worker.start()
        self.after(100, self._poll_log)

    def _poll_log(self):
        done = False
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg is None:
                    done = True
                    break
                self._append_log(msg)
        except queue.Empty:
            pass
        if not done:
            self.after(100, self._poll_log)
            return
        self._worker = None
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self.var_run_status.set("Stopped." if self._stop_event.is_set() else "Finished.")

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write the TRANSFORM settings into *d*."""
        try:
            t = self.current_transform()
            transform = {"angle_deg": t.angle_deg, "translation_m": list(t.translation_m)}
        except ValueError:
            transform = (self.cfg.get("TRANSFORM") or {}).get("transform")
        try:
            factor = float(self.var_factor.get())
        except ValueError:
            factor = 1.0
        d["TRANSFORM"] = {
            "folders": self._folders() if self._folders_custom else [],
            "source": self.var_source.get(),
            "name_filter": self.var_filter.get(),
            "preset": self.var_preset.get(),
            "transform": transform,
            "format": self.var_format.get(),
            "csv_columns": {k: v.get() for k, v in self._col_vars.items()},
            "power_factor": factor,
            "ignore_zeros": self.var_ignore_zeros.get(),
            "output_dir": self.var_outdir.get(),
        }

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        tc = c.get("TRANSFORM", {}) or {}
        self._set_folders(c)
        self.var_source.set(tc.get("source", "raw"))
        self.var_filter.set(tc.get("name_filter", ""))
        self.var_preset.set(tc.get("preset", "No transformation"))
        saved = tc.get("transform")
        if saved and self.var_preset.get() == _CUSTOM:
            self._fill_transform(tr.FrameTransform(
                float(saved.get("angle_deg", 0.0)), tuple(saved.get("translation_m", (0, 0, 0)))))
        self.var_format.set(tc.get("format", "csv"))
        for k, v in self._col_vars.items():
            v.set((tc.get("csv_columns") or {}).get(k, True))
        self.var_factor.set(str(tc.get("power_factor", 1.0)))
        self.var_ignore_zeros.set(tc.get("ignore_zeros", False))
        self.var_outdir.set(tc.get("output_dir", "TRANSFORMED"))
        self._included.clear()
        self._scen_tree.delete(*self._scen_tree.get_children())
        self._scen_items.clear()
        self.var_load_status.set("Click to list the scenarios. For merged files, add the MERGED folder.")
        self._refresh_plan()

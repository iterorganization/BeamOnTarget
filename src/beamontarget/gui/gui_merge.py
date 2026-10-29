# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Merge tab — cell-wise worst-case merge of scenarios grouped by colour.

Scenarios are assigned to colour groups by clicking them.  For each group the
result files of the same component are merged cell by cell (see
``workflows/merge_results.py``).  The "Merge plan" pane lists exactly which
files go into which output, and is what the merge executes.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext

from beamontarget.gui.gui_widgets import (
    make_card,
    resolve_path as _resolve_path,
    to_relative_path as _to_relative_path,
)
from beamontarget.workflows import merge_results as mr

# (key, colour); the key is the default label and is stored in the config.
MERGE_GROUPS = [
    ("blue", "#2563eb"),
    ("red", "#dc2626"),
    ("green", "#16a34a"),
    ("orange", "#ea580c"),
    ("purple", "#9333ea"),
]
_GROUP_COLOURS = dict(MERGE_GROUPS)
_EMPTY_IND = "#e5e7eb"
_RANK_LABELS = {
    mr.DENSITY_ARRAY: "Power density (Power_Density_W_m2)",
    mr.POWER_ARRAY: "Deposited power (Deposited_Power_W)",
}


class MergeTab(ttk.Frame):
    """Colour-group worst-case merge of result files."""

    def __init__(self, parent, cfg, colours, *, get_output_dir, open_in_paraview_fn):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._get_output_dir = get_output_dir
        self._open_in_paraview = open_in_paraview_fn

        mc = cfg.get("MERGE", {}) or {}
        self._group_labels = {k: (mc.get("group_labels") or {}).get(k, k)
                              for k, _ in MERGE_GROUPS}
        # {absolute scenario path: group key}; restored when the scenario is loaded
        self._saved_assignments = {self._abs(p): g
                                   for p, g in (mc.get("assignments") or {}).items()}
        self._assignment = {}         # {scenario path: group key}, loaded scenarios only
        self._scenarios = []          # [(result folder, scenario path)] in display order
        self._rows = {}               # {scenario path: (indicator, name label, count label)}
        self._last_clicked = None
        self._tree_files = {}         # {tree item id: [file paths]}
        self._plan = []
        self._plan_job = None
        self._worker = None
        self._stop_event = threading.Event()
        self._log_queue = queue.Queue()

        self._build(mc)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _abs(path):
        return os.path.normpath(path if os.path.isabs(path) else _resolve_path(path))

    def _group_label(self, key):
        return self._group_labels.get(key, key)

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self, mc):
        # --- Inputs card ---
        in_card = make_card(self, "Inputs", pady=(12, 8))

        ttk.Label(in_card, text="Result folders:", style="Card.TLabel").grid(
            row=0, column=0, sticky="nw", pady=2)
        folders_frm = ttk.Frame(in_card, style="Card.TFrame")
        folders_frm.grid(row=0, column=1, columnspan=3, sticky="we", padx=(8, 0))
        self._folders_list = tk.Listbox(folders_frm, height=2, activestyle="none",
                                        selectmode="extended", bg="white",
                                        highlightthickness=1, bd=0)
        self._folders_list.pack(side="left", fill="x", expand=True)
        fbtns = ttk.Frame(folders_frm, style="Card.TFrame")
        fbtns.pack(side="left", padx=(6, 0), anchor="n")
        ttk.Button(fbtns, text="Add...", style="Secondary.TButton",
                   command=self._add_folder).pack(fill="x")
        ttk.Button(fbtns, text="Remove", style="Secondary.TButton",
                   command=self._remove_folders).pack(fill="x", pady=(2, 0))
        self._set_folders(self.cfg)

        ttk.Label(in_card, text="Source files:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 2))
        src_frm = ttk.Frame(in_card, style="Card.TFrame")
        src_frm.grid(row=1, column=1, columnspan=3, sticky="w", padx=(8, 0), pady=(8, 2))
        self.var_source = tk.StringVar(value=mc.get("source", "raw"))
        ttk.Radiobutton(src_frm, text="Raw (results_*.vtp)",
                        variable=self.var_source, value="raw", style="Card.TRadiobutton",
                        command=self._schedule_refresh).pack(side="left")
        ttk.Radiobutton(src_frm, text="Smoothed (SMOOTHED/smoothed_results_*.vtp)",
                        variable=self.var_source, value="smoothed", style="Card.TRadiobutton",
                        command=self._schedule_refresh).pack(side="left", padx=(16, 0))

        ttk.Label(in_card, text="Name filter:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=2)
        self.var_filter = tk.StringVar(value=mc.get("name_filter", ""))
        ttk.Entry(in_card, textvariable=self.var_filter, width=40).grid(
            row=2, column=1, sticky="w", padx=(8, 0))
        ttk.Label(in_card, text="comma-separated; a file is used if its name contains any term",
                  style="Card.TLabel", foreground=self._colours["dim"]).grid(
            row=2, column=2, sticky="w", padx=(8, 0))
        self.var_filter.trace_add("write", lambda *_: self._schedule_refresh())

        load_frm = ttk.Frame(in_card, style="Card.TFrame")
        load_frm.grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(load_frm, text="Load Scenarios", style="Accent.TButton",
                   command=self.load_scenarios).pack(side="left")
        self.var_load_status = tk.StringVar(value="Click to list the scenarios in the result folders.")
        ttk.Label(load_frm, textvariable=self.var_load_status, style="Card.TLabel",
                  foreground=self._colours["dim"]).pack(side="left", padx=(10, 0))
        in_card.columnconfigure(2, weight=1)

        # Scenarios/plan and run/log share a vertical splitter, so the plan can
        # be given more room by dragging the border.
        vpw = ttk.PanedWindow(self, orient="vertical")
        vpw.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # --- Run card ---
        run_wrap = ttk.Frame(vpw, style="Card.TFrame", padding=14)
        run_card = ttk.Frame(run_wrap, style="Card.TFrame")
        run_card.pack(fill="both", expand=True)
        ttk.Label(run_card, text="Rank cells by:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w")
        self.var_rank = tk.StringVar(value=mc.get("rank_array", mr.DENSITY_ARRAY))
        rk = ttk.Frame(run_card, style="Card.TFrame")
        rk.grid(row=0, column=1, columnspan=3, sticky="w", padx=(8, 0))
        for value, label in _RANK_LABELS.items():
            ttk.Radiobutton(rk, text=label, variable=self.var_rank,
                            value=value, style="Card.TRadiobutton").pack(side="left", padx=(0, 16))

        ttk.Label(run_card, text="Output folder:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        self.var_outdir = tk.StringVar(value=mc.get("output_dir", "MERGED"))
        ttk.Entry(run_card, textvariable=self.var_outdir, width=40).grid(
            row=1, column=1, sticky="we", padx=(8, 4), pady=(6, 0))
        ttk.Button(run_card, text="Browse...", style="Secondary.TButton",
                   command=self._browse_outdir).grid(row=1, column=2, pady=(6, 0))
        self.var_outdir.trace_add("write", lambda *_: self._schedule_refresh())

        btns = ttk.Frame(run_card, style="Card.TFrame")
        btns.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 4))
        self._run_btn = ttk.Button(btns, text="Run Merge", style="Accent.TButton",
                                   command=self._run_merge)
        self._run_btn.pack(side="left")
        self._stop_btn = ttk.Button(btns, text="Stop", style="Danger.TButton",
                                    command=self._stop_event.set, state="disabled")
        self._stop_btn.pack(side="left", padx=(6, 0))
        self.var_run_status = tk.StringVar(value="")
        ttk.Label(btns, textvariable=self.var_run_status, style="Card.TLabel",
                  foreground=self._colours["dim"]).pack(side="left", padx=(10, 0))

        self._log = scrolledtext.ScrolledText(run_card, height=5, wrap="none",
                                              font=("Consolas", 9), state="disabled")
        self._log.grid(row=3, column=0, columnspan=4, sticky="nsew")
        run_card.columnconfigure(1, weight=1)
        run_card.rowconfigure(3, weight=1)

        # --- Middle: scenarios | merge plan ---
        mid_wrap = ttk.Frame(vpw, style="Card.TFrame", padding=14)
        self._build_group_toolbar(mid_wrap)

        pw = ttk.PanedWindow(mid_wrap, orient="horizontal")
        pw.pack(fill="both", expand=True, pady=(6, 0))
        pw.add(self._build_scenario_pane(pw), weight=1)
        pw.add(self._build_plan_pane(pw), weight=2)

        vpw.add(mid_wrap, weight=3)
        vpw.add(run_wrap, weight=1)

    def _build_group_toolbar(self, parent):
        bar = ttk.Frame(parent, style="Card.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text="Assign to:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        self._active_group = MERGE_GROUPS[0][0]
        self._tool_btns = {}
        for key, colour in MERGE_GROUPS:
            b = tk.Button(bar, text=self._group_label(key), bg=colour, fg="white",
                          activebackground=colour, activeforeground="white",
                          font=("Segoe UI", 9, "bold"), relief="raised", bd=1,
                          padx=8, pady=1, cursor="hand2",
                          command=lambda k=key: self._select_group(k))
            b.pack(side="left", padx=(0, 3))
            b.bind("<Double-Button-1>", lambda _e, k=key: self._rename_group(k))
            self._tool_btns[key] = b
        ttk.Button(bar, text="Assign all", style="Secondary.TButton",
                   command=self._assign_all).pack(side="left", padx=(12, 3))
        ttk.Button(bar, text="Clear all", style="Secondary.TButton",
                   command=self._clear_all).pack(side="left")
        ttk.Label(bar, text="Click: assign / unassign   Shift+click: range   "
                            "Double-click a colour: rename",
                  style="Card.TLabel", foreground=self._colours["dim"]).pack(
            side="left", padx=(12, 0))
        self._select_group(self._active_group)

    def _build_scenario_pane(self, parent):
        outer = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(outer, text="Scenarios  (number of matching files)",
                  style="Card.TLabel").pack(anchor="w")
        box = ttk.Frame(outer, style="Card.TFrame")
        box.pack(fill="both", expand=True, pady=(2, 0))
        self._scen_canvas = tk.Canvas(box, bg="white", highlightthickness=1,
                                      highlightbackground=self._colours["border"])
        vsb = ttk.Scrollbar(box, orient="vertical", command=self._scen_canvas.yview)
        self._scen_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._scen_canvas.pack(side="left", fill="both", expand=True)
        self._scen_inner = tk.Frame(self._scen_canvas, bg="white")
        win = self._scen_canvas.create_window((0, 0), window=self._scen_inner, anchor="nw")
        self._scen_inner.bind("<Configure>", lambda e: self._scen_canvas.configure(
            scrollregion=self._scen_canvas.bbox("all")))
        self._scen_canvas.bind("<Configure>", lambda e: self._scen_canvas.itemconfigure(
            win, width=e.width))

        def _on_wheel(event):
            w = self.winfo_containing(event.x_root, event.y_root)
            if w is None or not str(w).startswith(str(self._scen_canvas)):
                return
            step = -1 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 1
            self._scen_canvas.yview_scroll(step * 3, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, _on_wheel, add="+")
        tk.Label(self._scen_inner, text="(no scenarios loaded)", bg="white",
                 fg="#9ca3af").pack(anchor="w", padx=6, pady=4)
        return outer

    def _build_plan_pane(self, parent):
        outer = ttk.Frame(parent, style="Card.TFrame")
        hdr = ttk.Frame(outer, style="Card.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Merge plan  (exactly these files are merged)",
                  style="Card.TLabel").pack(side="left")
        ttk.Button(hdr, text="Open in ParaView", style="Secondary.TButton",
                   command=self._open_selected).pack(side="right")
        # Packed before the tree so it keeps its line when space is short.
        self.var_plan_status = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.var_plan_status, style="Card.TLabel",
                  foreground=self._colours["dim"], wraplength=700,
                  justify="left").pack(side="bottom", anchor="w", pady=(2, 0))
        box = ttk.Frame(outer, style="Card.TFrame")
        box.pack(fill="both", expand=True, pady=(2, 0))
        self._tree = ttk.Treeview(box, columns=("files", "path"), show="tree headings",
                                  selectmode="extended")
        self._tree.heading("#0", text="Group / output file / source scenario")
        self._tree.heading("files", text="Files")
        self._tree.heading("path", text="File")
        self._tree.column("#0", width=260, stretch=True)
        self._tree.column("files", width=60, anchor="center", stretch=False)
        self._tree.column("path", width=320, stretch=True)
        vsb = ttk.Scrollbar(box, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(side="left", fill="both", expand=True)
        for key, colour in MERGE_GROUPS:
            self._tree.tag_configure(key, foreground=colour, font=("Segoe UI", 9, "bold"))
        self._tree.tag_configure("single", foreground="#b45309")
        self._tree.bind("<Double-Button-1>", lambda e: self._open_selected())
        return outer

    # ------------------------------------------------------------------
    #  Result folders
    # ------------------------------------------------------------------
    def _folders(self):
        return [self._folders_list.get(i) for i in range(self._folders_list.size())]

    def _show_folders(self, folders):
        self._folders_list.delete(0, "end")
        for f in folders:
            self._folders_list.insert("end", f)

    def _set_folders(self, cfg):
        """Show the saved folder list, or else the config's output folder.

        Until the user adds or removes a folder, the list follows the output
        folder of the loaded config (and the Output tab), and is not saved.
        """
        saved = (cfg.get("MERGE") or {}).get("folders")
        self._folders_custom = bool(saved)
        self._show_folders(saved or [cfg.get("DETAILED_OUTPUT_DIR", "OUTPUT")])

    def _add_folder(self):
        d = filedialog.askdirectory(title="Select a result folder", parent=self)
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

    # ------------------------------------------------------------------
    #  Groups
    # ------------------------------------------------------------------
    def _select_group(self, key):
        self._active_group = key
        for k, b in self._tool_btns.items():
            b.configure(relief="sunken" if k == key else "raised",
                        bd=3 if k == key else 1, bg=_GROUP_COLOURS[k])

    def _rename_group(self, key):
        new = simpledialog.askstring(
            "Rename group",
            f"New name for the '{self._group_label(key)}' group\n"
            "(used as the output folder name):",
            initialvalue=self._group_label(key), parent=self)
        if not new or not new.strip():
            return
        if any(v == new.strip() for k, v in self._group_labels.items() if k != key):
            messagebox.showwarning("Rename group",
                                   f"Another group is already called '{new.strip()}'.",
                                   parent=self)
            return
        self._group_labels[key] = new.strip()
        self._tool_btns[key].configure(text=new.strip())
        for sp in self._rows:
            self._refresh_row(sp)
        self._select_group(self._active_group)
        self._schedule_refresh()

    def _assign(self, sp):
        if self._assignment.get(sp) == self._active_group:
            self._assignment.pop(sp, None)
        else:
            self._assignment[sp] = self._active_group
        self._last_clicked = sp
        self._refresh_row(sp)
        self._schedule_refresh()

    def _assign_range(self, sp):
        order = [p for _, p in self._scenarios]
        if self._last_clicked not in order:
            self._assign(sp)
            return
        lo, hi = sorted((order.index(self._last_clicked), order.index(sp)))
        for p in order[lo:hi + 1]:
            self._assignment[p] = self._active_group
            self._refresh_row(p)
        self._last_clicked = sp
        self._schedule_refresh()

    def _assign_all(self):
        for _, sp in self._scenarios:
            self._assignment[sp] = self._active_group
            self._refresh_row(sp)
        self._schedule_refresh()

    def _clear_all(self):
        self._assignment.clear()
        for sp in self._rows:
            self._refresh_row(sp)
        self._schedule_refresh()

    # ------------------------------------------------------------------
    #  Scenario list
    # ------------------------------------------------------------------
    def load_scenarios(self):
        """List the scenarios of every result folder."""
        if not self._folders_custom:
            self._show_folders([self._get_output_dir()])
        # Keep assignments of scenarios that are already loaded.
        self._saved_assignments.update(self._assignment)
        self._scenarios = []
        missing = []
        for folder in self._folders():
            folder_abs = self._abs(folder)
            found = mr.find_scenarios(folder_abs)
            if not found:
                missing.append(folder)
            self._scenarios.extend((folder, sp) for sp in found)
        self._assignment = {sp: self._saved_assignments[sp]
                            for _, sp in self._scenarios if sp in self._saved_assignments}

        for w in self._scen_inner.winfo_children():
            w.destroy()
        self._rows.clear()
        if not self._scenarios:
            tk.Label(self._scen_inner, text="(no scenarios found)", bg="white",
                     fg="#9ca3af").pack(anchor="w", padx=6, pady=4)
        current = None
        for folder, sp in self._scenarios:
            if folder != current:
                current = folder
                tk.Label(self._scen_inner, text=folder, bg="#e8f0fe",
                         fg=self._colours["accent"], font=("Segoe UI", 9, "bold"),
                         anchor="w").pack(fill="x", pady=(6, 0), ipady=2, ipadx=4)
            row = tk.Frame(self._scen_inner, bg="white", cursor="hand2")
            row.pack(fill="x", padx=2, pady=1)
            ind = tk.Label(row, width=2, font=("Segoe UI", 8, "bold"), fg="white",
                           bg=_EMPTY_IND)
            ind.pack(side="left", padx=(4, 6))
            name = tk.Label(row, text=os.path.basename(sp), anchor="w", bg="white",
                            font=("Segoe UI", 9))
            name.pack(side="left", fill="x", expand=True)
            count = tk.Label(row, text="", anchor="e", bg="white", fg="#6b7280",
                             font=("Segoe UI", 8))
            count.pack(side="right", padx=(4, 6))
            for w in (row, ind, name, count):
                w.bind("<Button-1>", lambda _e, p=sp: self._assign(p))
                w.bind("<Shift-Button-1>", lambda _e, p=sp: self._assign_range(p))
            self._rows[sp] = (ind, name, count)
            self._refresh_row(sp)

        msg = f"{len(self._scenarios)} scenario(s) in {len(self._folders()) - len(missing)} folder(s)."
        if missing:
            msg += f"  No results in: {', '.join(missing)}"
        self.var_load_status.set(msg)
        self._refresh_counts()
        self._refresh_plan()

    def _refresh_row(self, sp):
        ind, _, _ = self._rows.get(sp, (None, None, None))
        if ind is None:
            return
        grp = self._assignment.get(sp)
        if grp:
            ind.configure(bg=_GROUP_COLOURS[grp], text=self._group_label(grp)[:1].upper())
        else:
            ind.configure(bg=_EMPTY_IND, text="")

    def _refresh_counts(self):
        terms = mr.filter_terms(self.var_filter.get())
        source = self.var_source.get()
        for sp, (_, name, count) in self._rows.items():
            n = sum(1 for f in mr.scenario_files(sp, source) if mr.matches_filter(f, terms))
            count.configure(text=str(n))
            name.configure(fg="#111827" if n else "#9ca3af")

    # ------------------------------------------------------------------
    #  Merge plan
    # ------------------------------------------------------------------
    def _groups_for_plan(self):
        groups = {}
        for _, sp in self._scenarios:
            key = self._assignment.get(sp)
            if key:
                groups.setdefault(key, []).append(sp)
        return groups

    def _schedule_refresh(self):
        if self._plan_job is not None:
            self.after_cancel(self._plan_job)
        self._plan_job = self.after(250, self._refresh_all)

    def _refresh_all(self):
        self._plan_job = None
        self._refresh_counts()
        self._refresh_plan()

    def _output_root(self):
        out = self.var_outdir.get().strip() or "MERGED"
        return self._abs(out)

    def _refresh_plan(self):
        groups = self._groups_for_plan()
        # plan_merge works with labels; keep the key for colouring the tree.
        label_groups = {self._group_label(k): v for k, v in groups.items()}
        key_of = {self._group_label(k): k for k in groups}
        self._plan = mr.plan_merge(label_groups, self.var_source.get(),
                                   self.var_filter.get(), self._output_root())

        self._tree.delete(*self._tree.get_children())
        self._tree_files.clear()
        group_items = {}
        n_single = 0
        for slot in self._plan:
            key = key_of[slot.group]
            g_id = group_items.get(slot.group)
            if g_id is None:
                n_scen = len(groups[key])
                g_id = self._tree.insert("", "end", open=True, tags=(key,),
                                         text=f"{slot.group}  ({n_scen} scenario(s))",
                                         values=("", ""))
                group_items[slot.group] = g_id
                self._tree_files[g_id] = []
            n = len(slot.sources)
            tags = ("single",) if n == 1 else ()
            n_single += n == 1
            rel_out = os.path.relpath(slot.output_path, self._output_root())
            s_id = self._tree.insert(
                g_id, "end", open=False, tags=tags,
                text=f"{slot.component}" + ("   (only 1 file: copied unchanged)" if n == 1 else ""),
                values=(n, rel_out))
            files = [str(fp) for _, fp in slot.sources]
            self._tree_files[s_id] = files + ([str(slot.output_path)]
                                              if slot.output_path.exists() else [])
            self._tree_files[g_id].extend(files)
            for scenario, fp in slot.sources:
                f_id = self._tree.insert(s_id, "end", text=scenario, values=("", fp.name))
                self._tree_files[f_id] = [str(fp)]

        if not self._scenarios:
            self.var_plan_status.set("Load scenarios, then click them to assign a colour group.")
        elif not groups:
            self.var_plan_status.set("No scenario assigned to a group yet.")
        elif not self._plan:
            self.var_plan_status.set("No files match the assigned scenarios and the name filter.")
        else:
            n_files = sum(len(s.sources) for s in self._plan)
            msg = (f"{len(self._plan)} output file(s) from {n_files} source file(s) "
                   f"in {len(group_items)} group(s). Output: {self._output_root()}")
            if n_single:
                msg += f"   {n_single} output(s) have only one source file."
            self.var_plan_status.set(msg)

    def _open_selected(self):
        files = []
        for item in self._tree.selection():
            files.extend(f for f in self._tree_files.get(item, []) if f not in files)
        if not files:
            messagebox.showinfo("Open in ParaView",
                                "Select a group, output file or source file in the merge plan.",
                                parent=self)
            return
        self._open_in_paraview(files)

    # ------------------------------------------------------------------
    #  Run
    # ------------------------------------------------------------------
    def _append_log(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _run_merge(self):
        if self._worker is not None:
            return
        self._refresh_all()
        if not self._plan:
            messagebox.showwarning("Merge", self.var_plan_status.get() or "Nothing to merge.",
                                   parent=self)
            return
        existing = [s.output_path for s in self._plan if s.output_path.exists()]
        if existing and not messagebox.askyesno(
                "Overwrite?",
                f"{len(existing)} of {len(self._plan)} merged file(s) already exist and "
                f"will be overwritten, e.g.\n{existing[0]}\n\nContinue?", parent=self):
            return

        plan = list(self._plan)
        rank = self.var_rank.get()
        out_root = self._output_root()
        self._stop_event.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self.var_run_status.set("Merging...")
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

        def _worker():
            try:
                mr.run_merge(plan, rank_array=rank, output_root=out_root,
                             log=self._log_queue.put, stop_event=self._stop_event)
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
        self._refresh_plan()   # merged outputs can now be opened from the plan

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write the MERGE settings into *d*."""
        assignments = dict(self._saved_assignments)
        for _, sp in self._scenarios:
            assignments.pop(sp, None)
        assignments.update(self._assignment)
        d["MERGE"] = {
            # Empty means "the output folder of the config".
            "folders": self._folders() if self._folders_custom else [],
            "source": self.var_source.get(),
            "name_filter": self.var_filter.get(),
            "rank_array": self.var_rank.get(),
            "output_dir": self.var_outdir.get(),
            "group_labels": {k: v for k, v in self._group_labels.items() if v != k},
            "assignments": {_to_relative_path(p): g for p, g in sorted(assignments.items())},
        }

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        mc = c.get("MERGE", {}) or {}
        self._set_folders(c)
        self.var_source.set(mc.get("source", "raw"))
        self.var_filter.set(mc.get("name_filter", ""))
        self.var_rank.set(mc.get("rank_array", mr.DENSITY_ARRAY))
        self.var_outdir.set(mc.get("output_dir", "MERGED"))
        self._group_labels = {k: (mc.get("group_labels") or {}).get(k, k)
                              for k, _ in MERGE_GROUPS}
        for k, b in self._tool_btns.items():
            b.configure(text=self._group_label(k))
        self._saved_assignments = {self._abs(p): g
                                   for p, g in (mc.get("assignments") or {}).items()}
        self._assignment = {}
        self._scenarios = []
        for w in self._scen_inner.winfo_children():
            w.destroy()
        self._rows.clear()
        tk.Label(self._scen_inner, text="(no scenarios loaded)", bg="white",
                 fg="#9ca3af").pack(anchor="w", padx=6, pady=4)
        self.var_load_status.set("Click to list the scenarios in the result folders.")
        self._refresh_plan()

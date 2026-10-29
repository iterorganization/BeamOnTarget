# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Geometry tab — editable folder table with viewer buttons and 2-D bounding-box plot."""
import glob
import os
import re
import struct
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from beamontarget.gui.gui_widgets import (
    make_card,
    get_project_folder as _get_project_folder,
    to_relative_path as _to_relative_path,
    resolve_path as _resolve_path,
)


# Per-file STL bounding boxes, unscaled: {(path, size, mtime_ns): (min, max)}.
# Shared across tabs and config loads; entries are invalidated by size/mtime.
_STL_BBOX_CACHE = {}

_STL_CHUNK_TRIANGLES = 2_000_000
_STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attr", "<u2"),
])
_ASCII_VERTEX_RE = re.compile(
    rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)")


def _read_stl_bbox(filepath, file_size):
    """Read the unscaled (min_xyz, max_xyz) of a binary or ASCII STL file."""
    with open(filepath, "rb") as fh:
        fh.seek(80)
        n_tri_bytes = fh.read(4)
    if len(n_tri_bytes) < 4:
        return None
    n_tri = struct.unpack("<I", n_tri_bytes)[0]
    if file_size == 84 + n_tri * 50 and n_tri > 0:
        # Binary STL: memory-map and reduce in chunks, so a 300 MB file is
        # not copied into memory in one piece.
        data = np.memmap(filepath, dtype=_STL_DTYPE, mode="r", offset=84, shape=(n_tri,))
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for start in range(0, n_tri, _STL_CHUNK_TRIANGLES):
            verts = data["vertices"][start:start + _STL_CHUNK_TRIANGLES].reshape(-1, 3)
            lo = np.minimum(lo, verts.min(axis=0))
            hi = np.maximum(hi, verts.max(axis=0))
        del data
        return lo, hi

    # ASCII STL
    with open(filepath, "rb") as fh:
        matches = _ASCII_VERTEX_RE.findall(fh.read())
    if not matches:
        return None
    try:
        arr = np.array(matches, dtype=np.float64)
    except ValueError:
        return None
    return arr.min(axis=0), arr.max(axis=0)


class GeometryTab(ttk.Frame):
    """Treeview-based geometry folder manager."""

    def __init__(self, parent, cfg, colours, *,
                 view_geometry_fn, view_geometry_o3d_fn,
                 get_em_bbox=None, is_em_mode=None):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._view_geometry = view_geometry_fn
        self._view_geometry_o3d = view_geometry_o3d_fn
        # get_em_bbox() -> (min_xyz, max_xyz), or None if the entries are invalid.
        self._get_em_bbox = get_em_bbox or (lambda: None)
        self._is_em_mode = is_em_mode or (lambda: False)
        # {folder: (min_xyz, max_xyz)} once all STL files have been read;
        # None while unknown. Projection, toggle and box edits redraw from it.
        self._folder_bbox_cache = None
        self._loading = False
        self._load_generation = 0
        self._plot_drawn = False
        self._redraw_job = None
        self._build()

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self):
        card = make_card(self, "Geometry Folders", pady=(12, 10))

        # Treeview
        cols = ("folder", "scale", "target_length", "save_details",
                "is_diagnostic", "save_impact_data", "max_impact_records")
        tree_frame = ttk.Frame(card, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)

        self.geo_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        headers = {"folder": "Folder", "scale": "Scale", "target_length": "Target Len",
                   "save_details": "Details", "is_diagnostic": "Diagnostic",
                   "save_impact_data": "Impacts", "max_impact_records": "Max Records"}
        widths = {"folder": 140, "scale": 60, "target_length": 90, "save_details": 65,
                  "is_diagnostic": 80, "save_impact_data": 65, "max_impact_records": 100}
        for c in cols:
            self.geo_tree.heading(c, text=headers[c])
            self.geo_tree.column(c, width=widths[c], anchor="center")
        self.geo_tree.column("folder", anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.geo_tree.yview)
        self.geo_tree.configure(yscrollcommand=vsb.set)
        self.geo_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._populate_geo_tree()

        # Buttons
        btn_frm = ttk.Frame(card, style="Card.TFrame")
        btn_frm.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frm, text="Add Folder...", style="Secondary.TButton",
                    command=self._add_geo_folder).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frm, text="Edit Selected...", style="Secondary.TButton",
                    command=self._edit_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="Remove", style="Secondary.TButton",
                    command=self._remove_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="ParaView", style="Secondary.TButton",
                    command=self._view_geometry).pack(side="right", padx=(4, 0))
        ttk.Button(btn_frm, text="Open3D", style="Secondary.TButton",
                    command=self._view_geometry_o3d).pack(side="right", padx=4)

        # --- 2-D Bounding-box plot ---
        plot_card = make_card(self, "Geometry Overview Plot")

        ctrl_frm = ttk.Frame(plot_card, style="Card.TFrame")
        ctrl_frm.pack(fill="x", pady=(0, 6))

        ttk.Label(ctrl_frm, text="Projection:", style="Card.TLabel").pack(side="left")
        self._var_projection = tk.StringVar(value="X–Z")
        proj_combo = ttk.Combobox(ctrl_frm, textvariable=self._var_projection,
                                   values=["X–Y", "X–Z", "Y–Z"],
                                   state="readonly", width=8)
        proj_combo.pack(side="left", padx=(6, 12))

        self._refresh_btn = ttk.Button(ctrl_frm, text="Refresh Plot",
                                       style="Secondary.TButton",
                                       command=self._refresh_plot)
        self._refresh_btn.pack(side="left")
        proj_combo.bind("<<ComboboxSelected>>", lambda e: self._update_plot())

        self._var_show_em_box = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frm, text="Show EM bounding box",
                        variable=self._var_show_em_box,
                        style="Card.TCheckbutton",
                        command=self._update_plot).pack(side="left", padx=(12, 0))

        self._var_plot_status = tk.StringVar(value="")
        ttk.Label(ctrl_frm, textvariable=self._var_plot_status,
                  style="Card.TLabel", foreground="#64748b").pack(
                      side="left", padx=(12, 0))

        self._var_bbox_warning = tk.StringVar(value="")
        self._bbox_warning_lbl = ttk.Label(
            plot_card, textvariable=self._var_bbox_warning,
            style="Card.TLabel", foreground="#b91c1c",
            justify="left", wraplength=900)
        self._bbox_warning_lbl.pack(side="bottom", anchor="w", pady=(6, 0))

        self._fig = Figure(figsize=(6, 3.2), dpi=100)
        self._fig.patch.set_facecolor("#ffffff")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_card)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    #  Treeview helpers
    # ------------------------------------------------------------------
    def _populate_geo_tree(self):
        # Folders may have changed; re-read the STL files on the next redraw.
        self._folder_bbox_cache = None
        self.schedule_bbox_redraw()
        for item in self.geo_tree.get_children():
            self.geo_tree.delete(item)
        for folder, s in self.cfg.get("GEOMETRY_FOLDERS", {}).items():
            self.geo_tree.insert("", "end", values=(
                folder, s.get("scale", 1), s.get("target_length", 1.0),
                "Yes" if s.get("save_details") else "",
                "Yes" if s.get("is_diagnostic") else "",
                "Yes" if s.get("save_impact_data") else "",
                s.get("max_impact_records") if s.get("max_impact_records") is not None else "—"))

    def _add_geo_folder(self):
        self._geo_dialog(None)

    def _edit_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = self.geo_tree.item(sel[0])["values"][0]
        self._geo_dialog(str(folder))

    def _remove_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = str(self.geo_tree.item(sel[0])["values"][0])
        if messagebox.askyesno("Remove", f"Remove geometry folder '{folder}'?"):
            self.cfg["GEOMETRY_FOLDERS"].pop(folder, None)
            self._populate_geo_tree()

    def _geo_dialog(self, existing_folder):
        """Pop up a dialog to add/edit a geometry folder entry."""
        dlg = tk.Toplevel(self)
        dlg.title("Edit Geometry Folder" if existing_folder else "Add Geometry Folder")
        dlg.geometry("400x360")
        dlg.transient(self)
        dlg.grab_set()

        s = self.cfg.get("GEOMETRY_FOLDERS", {}).get(existing_folder, {}) if existing_folder else {}

        entries = {}
        row = 0

        ttk.Label(dlg, text="Folder path:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        v_folder = tk.StringVar(value=existing_folder or "")
        folder_frm = ttk.Frame(dlg)
        folder_frm.grid(row=row, column=1, sticky="we", padx=8)
        e = ttk.Entry(folder_frm, textvariable=v_folder, width=24)
        e.pack(side="left", fill="x", expand=True)

        def _browse_geo_folder():
            base_dir = _get_project_folder()
            d = filedialog.askdirectory(
                initialdir=base_dir,
                title="Select geometry folder",
                parent=dlg,
            )
            if d:
                rel = _to_relative_path(d, base_dir)
                if existing_folder:
                    e.config(state="normal")
                v_folder.set(rel)
                if existing_folder:
                    e.config(state="disabled")

        ttk.Button(folder_frm, text="Browse…", command=_browse_geo_folder).pack(
            side="left", padx=(4, 0)
        )
        if existing_folder:
            e.config(state="disabled")
        entries["folder"] = v_folder

        fields = [
            ("scale", "Scale:", float, 1),
            ("target_length", "Target length:", float, 1.0),
        ]
        for key, label, typ, default in fields:
            row += 1
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            v = tk.StringVar(value=str(s.get(key, default)))
            ttk.Entry(dlg, textvariable=v, width=15).grid(row=row, column=1, sticky="w", padx=8)
            entries[key] = (v, typ)

        bools = [
            ("save_details", "Save detailed reports (VTP files)", False),
            ("is_diagnostic", "Is diagnostic (transparent)", False),
            ("save_impact_data", "Save impact data", False),
            ("show_in_plot", "Show in plot", False),
        ]
        for key, label, default in bools:
            row += 1
            v = tk.BooleanVar(value=s.get(key, default))
            ttk.Checkbutton(dlg, text=label, variable=v).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            entries[key] = (v, bool)

        row += 1
        ttk.Label(dlg, text="Max impact records:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        mir = s.get("max_impact_records")
        v_mir = tk.StringVar(value=str(mir) if mir is not None else "")
        ttk.Entry(dlg, textvariable=v_mir, width=15).grid(row=row, column=1, sticky="w", padx=8)
        entries["max_impact_records"] = v_mir

        def _ok():
            folder_name = entries["folder"].get().strip()
            if not folder_name:
                messagebox.showwarning("Missing", "Folder path is required."); return
            new_s = {}
            for key, (var, typ) in [(k, v) for k, v in entries.items() if k not in ("folder", "max_impact_records")]:
                try:
                    new_s[key] = typ(var.get())
                except (ValueError, tk.TclError):
                    new_s[key] = var.get()
            mir_val = entries["max_impact_records"].get().strip()
            new_s["max_impact_records"] = int(mir_val) if mir_val else None
            if "GEOMETRY_FOLDERS" not in self.cfg:
                self.cfg["GEOMETRY_FOLDERS"] = {}
            # If the folder was renamed via Browse, remove the old key
            if existing_folder and folder_name != existing_folder:
                self.cfg["GEOMETRY_FOLDERS"].pop(existing_folder, None)
            self.cfg["GEOMETRY_FOLDERS"][folder_name] = new_s
            self._populate_geo_tree()
            dlg.destroy()

        row += 1
        ttk.Button(dlg, text="OK", command=_ok).grid(row=row, column=0, columnspan=2, pady=10)
        dlg.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    #  2-D bounding-box plot
    # ------------------------------------------------------------------
    _PLOT_COLOURS = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706",
        "#7c3aed", "#db2777", "#0891b2", "#65a30d",
    ]

    @staticmethod
    def _stl_bbox(filepath, scale=1.0):
        """Return (min_xyz, max_xyz) arrays for a binary or ASCII STL file.

        Results are cached per file (keyed by path, size and modification
        time), so redrawing only re-reads files that changed.
        """
        try:
            st = os.stat(filepath)
        except OSError:
            return None
        key = (os.path.normcase(os.path.abspath(filepath)), st.st_size, st.st_mtime_ns)
        bb = _STL_BBOX_CACHE.get(key)
        if bb is None:
            bb = _read_stl_bbox(filepath, st.st_size)
            if bb is None:
                return None
            _STL_BBOX_CACHE[key] = bb
        lo, hi = bb[0] * scale, bb[1] * scale
        # A negative scale mirrors the part, swapping min and max.
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def _folder_stl_files(self):
        """Return [(folder, scale, [stl paths])] for every configured folder."""
        jobs = []
        for folder, settings in self.cfg.get("GEOMETRY_FOLDERS", {}).items():
            folder_abs = _resolve_path(folder) if not os.path.isabs(folder) else folder
            if not os.path.isdir(folder_abs):
                continue
            # dict.fromkeys de-duplicates on case-insensitive file systems.
            stl_files = sorted(dict.fromkeys(
                glob.glob(os.path.join(folder_abs, "*.stl"))
                + glob.glob(os.path.join(folder_abs, "*.STL"))))
            if stl_files:
                jobs.append((folder, settings.get("scale", 1.0), stl_files))
        return jobs

    @classmethod
    def _compute_folder_bboxes(cls, jobs, progress_fn=None):
        """Return {folder: (min_xyz, max_xyz)} for the given folder jobs."""
        result = {}
        done = 0
        n_files = sum(len(files) for _, _, files in jobs)
        for folder, scale, stl_files in jobs:
            g_min = np.full(3, np.inf)
            g_max = np.full(3, -np.inf)
            for stl in stl_files:
                bb = cls._stl_bbox(stl, scale)
                done += 1
                if progress_fn is not None:
                    progress_fn(done, n_files)
                if bb is None:
                    continue
                g_min = np.minimum(g_min, bb[0])
                g_max = np.maximum(g_max, bb[1])
            if np.all(np.isfinite(g_min)):
                result[folder] = (g_min, g_max)
        return result

    def _refresh_plot(self):
        """Re-check the STL files (re-reading only changed ones), then redraw."""
        self._folder_bbox_cache = None
        self._update_plot()

    def _start_loading(self):
        """Read the STL bounding boxes in a background thread.

        Reading large geometries (GBs of STL) takes many seconds; doing it on
        the UI thread froze the window. Nothing is drawn until all folders
        are known, so the plot never shows a partial picture.
        """
        self._load_generation += 1
        gen = self._load_generation
        self._loading = True
        jobs = self._folder_stl_files()
        self._refresh_btn.configure(state="disabled")
        self._var_plot_status.set("Reading geometry...")

        ax = self._ax
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Reading geometry files...", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#64748b")
        self._canvas.draw_idle()

        # The worker never touches Tk; it only fills *state*, which the UI
        # thread polls.
        state = {"done": 0, "total": 0, "result": None}

        def _progress(done, total):
            state["done"], state["total"] = done, total

        def _worker():
            try:
                bboxes = self._compute_folder_bboxes(jobs, _progress)
                error = None
            except Exception as e:  # keep the GUI usable if a file is unreadable
                bboxes, error = {}, e
            state["result"] = (bboxes, error)

        def _poll():
            if gen != self._load_generation:
                return  # superseded by a newer load
            if state["result"] is None:
                if state["total"]:
                    self._var_plot_status.set(
                        f"Reading geometry... {state['done']}/{state['total']} files")
                self.after(100, _poll)
                return
            self._on_bboxes_loaded(*state["result"])

        threading.Thread(target=_worker, daemon=True).start()
        self.after(100, _poll)

    def _on_bboxes_loaded(self, bboxes, error):
        self._loading = False
        self._refresh_btn.configure(state="normal")
        self._var_plot_status.set(f"Error reading geometry: {error}" if error else "")
        self._folder_bbox_cache = bboxes
        self._update_plot()

    def schedule_bbox_redraw(self):
        """Redraw shortly after the EM bounding box or tracking mode changes.

        Only redraws once the plot has been drawn, so typing in the General
        tab does not trigger a slow first read of all STL files.
        """
        if not self._plot_drawn or self._loading:
            return
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(400, self._update_plot)

    def _update_plot(self):
        """Redraw the 2-D bounding-box overview."""
        self._redraw_job = None
        if self._loading:
            return  # the finished load redraws with the current settings
        if self._folder_bbox_cache is None:
            self._start_loading()
            return
        bboxes = self._folder_bbox_cache

        ax = self._ax
        ax.clear()
        ax.set_axis_on()

        proj = self._var_projection.get()
        idx_map = {"X–Y": (0, 1), "X–Z": (0, 2), "Y–Z": (1, 2)}
        axis_labels = {"X–Y": ("X (m)", "Y (m)"),
                       "X–Z": ("X (m)", "Z (m)"),
                       "Y–Z": ("Y (m)", "Z (m)")}
        ix, iy = idx_map.get(proj, (0, 2))
        xlabel, ylabel = axis_labels.get(proj, ("", ""))

        em_bbox = self._get_em_bbox() if self._var_show_em_box.get() else None
        if em_bbox is not None:
            # The tracker accepts the corners in either order.
            em_lo = np.minimum(np.asarray(em_bbox[0], float), np.asarray(em_bbox[1], float))
            em_hi = np.maximum(np.asarray(em_bbox[0], float), np.asarray(em_bbox[1], float))

        outside = []
        if not bboxes:
            ax.text(0.5, 0.5, "No geometry folders with STL files found",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#64748b")
        else:
            for ci, (folder, (lo, hi)) in enumerate(bboxes.items()):
                colour = self._PLOT_COLOURS[ci % len(self._PLOT_COLOURS)]
                x0, y0 = lo[ix], lo[iy]
                w, h = hi[ix] - lo[ix], hi[iy] - lo[iy]
                rect = Rectangle((x0, y0), w, h,
                                  linewidth=1.8, edgecolor=colour,
                                  facecolor=colour, alpha=0.18)
                ax.add_patch(rect)
                # Label at top-left corner
                lbl = os.path.basename(folder.rstrip("/\\")) or folder
                ax.text(x0 + w * 0.02, y0 + h * 0.96, lbl,
                        fontsize=8, color=colour, fontweight="bold",
                        va="top", clip_on=True)
                if em_bbox is not None and (np.any(lo < em_lo) or np.any(hi > em_hi)):
                    outside.append(lbl)

        is_em = self._is_em_mode()
        if em_bbox is not None:
            # Drawn last and on top so it stands out against the geometry.
            label = "EM bounding box" if is_em else "EM bounding box (not used by ray tracing)"
            ax.add_patch(Rectangle(
                (em_lo[ix], em_lo[iy]), em_hi[ix] - em_lo[ix], em_hi[iy] - em_lo[iy],
                linewidth=2.4, linestyle="--", edgecolor="#111827",
                facecolor="none", zorder=5))
            # A plain dashed line reads better in the legend than a patch box.
            ax.legend(handles=[Line2D([], [], linestyle="--", linewidth=2.0,
                                      color="#111827", label=label)],
                      fontsize=7, loc="best", framealpha=0.85)

        if bboxes or em_bbox is not None:
            ax.set_aspect("equal", adjustable="datalim")
            ax.autoscale_view()
            # Small margin
            ax.margins(0.06)

        names = ", ".join(outside)
        if em_bbox is None:
            msg, colour = "", "#b91c1c"
        elif is_em:
            msg = (f"Warning: {names} extend(s) outside the EM bounding box. EM tracking "
                   "stops each particle when it leaves the box, so no power is "
                   "deposited on those parts outside it.") if outside else ""
            colour = "#b91c1c"
        else:
            msg = "Ray tracing does not use the EM bounding box; it only limits EM tracing."
            if outside:
                msg += f" With EM tracing, {names} would extend outside it."
            colour = "#64748b"
        self._var_bbox_warning.set(msg)
        self._bbox_warning_lbl.configure(foreground=colour)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Bounding Boxes — {proj}", fontsize=10, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._plot_drawn = True

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write geometry keys into *d*."""
        d["GEOMETRY_FOLDERS"] = self.cfg.get("GEOMETRY_FOLDERS", {})

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        self._populate_geo_tree()


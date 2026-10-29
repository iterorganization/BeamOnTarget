#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# viewer.py
"""
Built-in 3D viewer for BeamOnTarget.

Renders the 3D scene with Open3D (visible=False, GPU-accelerated) and
displays the result inside the tkinter selection dialog — no separate
window needed.  Mouse drag rotates the view; scroll-wheel zooms.

Entry points called from sim_gui.py:
  - ``view_geometry(parent, script_dir, geometry_folders)``
  - ``view_results(parent, script_dir, output_dir, geometry_folders)``
  - ``view_all(parent, script_dir, output_dir, geometry_folders)``
"""
import os
import glob
import threading
import math

import numpy as np
import pandas as pd
import open3d as o3d
import pyvista as pv

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw


def _resolve_viewer_path(script_dir, relative_path):
    """Resolve a simulation-file path relative to the main application folder."""
    return os.path.join(script_dir, relative_path)


# ---------------------------------------------------------------------------
#  Colour palette for geometry folders
# ---------------------------------------------------------------------------
_FOLDER_COLORS = [
    (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
    (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
    (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
]

# Jet-like colour-map for power density
_CMAP_STOPS = np.array([
    [0.0, 0.0, 0.5],   # dark blue
    [0.0, 0.0, 1.0],   # blue
    [0.0, 0.5, 1.0],
    [0.0, 1.0, 1.0],   # cyan
    [0.5, 1.0, 0.5],
    [1.0, 1.0, 0.0],   # yellow
    [1.0, 0.5, 0.0],
    [1.0, 0.0, 0.0],   # red
    [0.5, 0.0, 0.0],   # dark red
])


def _apply_jet_colormap(values):
    """Map *values* (0-1 normalised) to RGB via a jet-like colour map."""
    n_stops = len(_CMAP_STOPS)
    t = np.clip(values, 0.0, 1.0) * (n_stops - 1)
    idx = np.clip(t.astype(int), 0, n_stops - 2)
    frac = (t - idx)[:, None]
    return _CMAP_STOPS[idx] * (1 - frac) + _CMAP_STOPS[idx + 1] * frac


# ===================================================================
#  Mesh loading helpers
# ===================================================================

def _load_stl(stl_path, scale, color):
    """Read an STL, optionally scale, paint uniform colour."""
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh.is_empty():
        return None
    if scale != 1:
        mesh.scale(scale, center=(0, 0, 0))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _read_vtp_as_o3d(vtp_path):
    """Read a VTP via pyvista, convert to Open3D TriangleMesh + power array."""
    pv_mesh = pv.read(vtp_path)
    if pv_mesh.n_cells == 0:
        return None, None

    verts = np.asarray(pv_mesh.points, dtype=np.float64)
    if hasattr(pv_mesh, "faces"):
        faces_raw = np.asarray(pv_mesh.faces)
    else:
        faces_raw = np.asarray(pv_mesh.cells)
    try:
        faces = faces_raw.reshape(-1, 4)[:, 1:4].astype(np.int32)
    except ValueError:
        return None, None

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    power = None
    if "Power_Density_W_m2" in pv_mesh.cell_data:
        power = np.asarray(pv_mesh.cell_data["Power_Density_W_m2"], dtype=np.float64)
    elif "Deposited_Power_W" in pv_mesh.cell_data:
        power = np.asarray(pv_mesh.cell_data["Deposited_Power_W"], dtype=np.float64)
    return o3d_mesh, power


def _colour_mesh_by_power(mesh, power, global_vmax=None):
    """Apply a jet colour-map to *mesh* based on per-cell *power*.

    If *global_vmax* is given the normalisation uses that value so that
    several meshes share the same colour scale.
    """
    triangles = np.asarray(mesh.triangles)
    n_verts = len(mesh.vertices)

    if power is None or len(power) == 0:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return 0.0

    vmax = power.max()
    if vmax <= 0:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return 0.0

    norm_max = global_vmax if (global_vmax and global_vmax > 0) else vmax
    normed = np.clip(power / norm_max, 0, 1)
    cell_colors = _apply_jet_colormap(normed)

    vert_colors = np.zeros((n_verts, 3), dtype=np.float64)
    vert_counts = np.zeros(n_verts, dtype=np.float64)
    for vi in range(3):
        np.add.at(vert_colors, triangles[:, vi], cell_colors)
        np.add.at(vert_counts, triangles[:, vi], 1)
    mask = vert_counts > 0
    vert_colors[mask] /= vert_counts[mask, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
    return vmax


# ===================================================================
#  Colorbar helper — draws into an existing Tk Canvas widget
# ===================================================================

def _draw_colorbar_on_canvas(canvas, vmin, vmax, scale_factor,
                              unit_label="W/m²"):
    """Draw a vertical colour bar with tick labels onto *canvas*.

    Clears any previous content first.  Can be called repeatedly
    (e.g. after each View click with a new vmax / scale).
    """
    canvas.delete("all")

    cw = int(canvas.cget("width"))
    ch = int(canvas.cget("height"))

    bar_width = 24
    margin_top = 30
    margin_bot = 10
    margin_left = 6
    bar_height = ch - margin_top - margin_bot

    if bar_height < 20 or vmax <= 0:
        canvas.create_text(cw // 2, ch // 2, text="No data",
                           fill="#888", font=("Segoe UI", 9))
        return

    display_vmin = vmin * scale_factor
    display_vmax = vmax * scale_factor

    # Title
    if scale_factor != 1.0:
        title = f"[× {scale_factor:.1e}] {unit_label}"
    else:
        title = unit_label
    canvas.create_text(cw // 2, 12, text=title, fill="#1e293b",
                       font=("Segoe UI", 8, "bold"), anchor="center")

    # Gradient
    n_stops = len(_CMAP_STOPS)
    for i in range(bar_height):
        t = 1.0 - i / max(bar_height - 1, 1)
        pos = t * (n_stops - 1)
        idx = min(int(pos), n_stops - 2)
        frac = pos - idx
        r, g, b = (_CMAP_STOPS[idx] * (1 - frac) + _CMAP_STOPS[idx + 1] * frac)
        hex_c = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        y = margin_top + i
        canvas.create_line(margin_left, y, margin_left + bar_width, y,
                           fill=hex_c, width=1)

    # Border
    canvas.create_rectangle(margin_left, margin_top,
                            margin_left + bar_width,
                            margin_top + bar_height - 1,
                            outline="#333", width=1)

    # Ticks
    tick_count = 6
    x0 = margin_left + bar_width
    for j in range(tick_count + 1):
        frac_j = j / tick_count
        y = margin_top + int((1.0 - frac_j) * (bar_height - 1))
        val = display_vmin + frac_j * (display_vmax - display_vmin)
        canvas.create_line(x0, y, x0 + 4, y, fill="#333", width=1)
        if abs(val) >= 1e4 or (abs(val) > 0 and abs(val) < 0.01):
            lbl = f"{val:.1e}"
        else:
            lbl = f"{val:.2f}"
        canvas.create_text(x0 + 6, y, text=lbl, anchor="w",
                           font=("Segoe UI", 7), fill="#1e293b")


def _show_in_open3d(geometries, title="BeamOnTarget Viewer"):
    """Open an Open3D visualisation window (blocking).  Legacy fallback."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.15, 0.15, 0.18])
    opt.mesh_show_back_face = True
    opt.light_on = True
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


def _scan_vtp_vmax(vtp_paths):
    """Return the global max power-density across *vtp_paths* (lightweight)."""
    vmax = 0.0
    for vtp in vtp_paths:
        try:
            pv_mesh = pv.read(vtp)
        except Exception:
            continue
        for key in ("Power_Density_W_m2", "Deposited_Power_W"):
            if key in pv_mesh.cell_data:
                arr = np.asarray(pv_mesh.cell_data[key])
                if len(arr) > 0:
                    vmax = max(vmax, float(arr.max()))
                break
    return vmax


# ===================================================================
#  Embedded 3D viewport — renders with Open3D, displays in Tk Canvas
# ===================================================================

class _EmbeddedViewer:
    """Renders meshes off-screen with Open3D (visible=False) and shows
    the image in a Tk Canvas, with mouse-drag rotation and scroll-wheel
    zoom.

    The Visualizer is created once and kept alive — only the camera is
    updated on each interaction, giving GPU-accelerated re-renders.
    """

    _BG = np.array([0.15, 0.15, 0.18])
    _RENDER_W = 720
    _RENDER_H = 520

    def __init__(self, parent_frame):
        self._frame = parent_frame
        self._canvas = tk.Canvas(parent_frame, bg="#262630",
                                 highlightthickness=0,
                                 width=self._RENDER_W,
                                 height=self._RENDER_H)
        self._canvas.pack(fill="both", expand=True)
        self._photo = None          # keep reference to avoid GC
        self._vis = None            # persistent Open3D Visualizer
        self._resize_after_id = None  # debounce id for resize
        self._render_pending = None   # throttle id for rendering
        self._rendering = False       # True while _render is executing

        # Camera spherical coordinates (azimuth, elevation, distance)
        self._azimuth = 45.0
        self._elevation = 30.0
        self._distance = None       # auto-computed on first render
        self._focus = np.zeros(3)   # look-at point

        # Mouse state
        self._drag_start = None
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>", self._on_scroll)        # Windows/macOS
        self._canvas.bind("<Button-4>", self._on_scroll_up)       # Linux
        self._canvas.bind("<Button-5>", self._on_scroll_down)     # Linux
        self._canvas.bind("<ButtonPress-3>", self._on_press_right)
        self._canvas.bind("<B3-Motion>", self._on_drag_right)
        self._canvas.bind("<ButtonRelease-3>", self._on_release_right)
        self._drag_right_start = None

        # Re-render on canvas resize (debounced)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # Placeholder text
        self._canvas.create_text(
            self._RENDER_W // 2, self._RENDER_H // 2,
            text="Click View to render",
            fill="#666", font=("Segoe UI", 12))

    # ------ public API ------

    def set_meshes(self, o3d_meshes):
        """Build a persistent Open3D Visualizer with the given meshes."""
        # Cancel any pending throttled render / resize callbacks
        if self._render_pending is not None:
            self._canvas.after_cancel(self._render_pending)
            self._render_pending = None
        if self._resize_after_id is not None:
            self._canvas.after_cancel(self._resize_after_id)
            self._resize_after_id = None

        # Close any previous visualizer
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None

        # Use current canvas size for the render target
        self._canvas.update_idletasks()
        w = max(self._canvas.winfo_width(), 64)
        h = max(self._canvas.winfo_height(), 64)
        self._RENDER_W = w
        self._RENDER_H = h

        # Keep meshes so we can rebuild on resize
        self._o3d_meshes = list(o3d_meshes)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False,
                          width=self._RENDER_W, height=self._RENDER_H)

        for mesh in o3d_meshes:
            vis.add_geometry(mesh)

        opt = vis.get_render_option()
        opt.background_color = self._BG
        opt.mesh_show_back_face = True
        opt.light_on = True
        opt.point_size = 5.0  # visible size for PointCloud sources

        # Compute scene bounds for camera setup
        all_pts = []
        for geom in o3d_meshes:
            if isinstance(geom, o3d.geometry.PointCloud):
                pts = np.asarray(geom.points)
            else:
                pts = np.asarray(geom.vertices)
            if len(pts) > 0:
                all_pts.append(pts)
        if not all_pts:
            return
        all_pts = np.vstack(all_pts)
        bmin = all_pts.min(axis=0)
        bmax = all_pts.max(axis=0)
        self._focus = (bmin + bmax) / 2.0
        diag = np.linalg.norm(bmax - bmin)
        self._distance = diag * 1.5

        self._vis = vis
        self._render()

    def _request_render(self):
        """Schedule a render if one is not already pending (throttled)."""
        if self._vis is None:
            return
        if self._render_pending is not None:
            return  # already scheduled
        if self._rendering:
            # A render is in progress — schedule one for after it finishes
            self._render_pending = self._canvas.after(30, self._do_throttled_render)
            return
        self._render_pending = self._canvas.after(16, self._do_throttled_render)

    def _do_throttled_render(self):
        """Execute the throttled render."""
        self._render_pending = None
        self._render()

    def _render(self):
        """Update camera, render, and display the frame in the Tk canvas."""
        if self._vis is None or self._rendering:
            return
        self._rendering = True
        try:
            self._render_impl()
        finally:
            self._rendering = False

    def _render_impl(self):
        """Internal: compute camera, render, blit to canvas."""
        vis = self._vis
        if vis is None:
            return

        # Compute camera position from spherical coordinates
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        d = self._distance or 1.0
        eye = np.array([
            self._focus[0] + d * math.cos(el) * math.cos(az),
            self._focus[1] + d * math.cos(el) * math.sin(az),
            self._focus[2] + d * math.sin(el),
        ])

        # Set the camera via ViewControl
        try:
            ctr = vis.get_view_control()
            param = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            return  # visualizer was destroyed between check and use

        # Build extrinsic matrix (world-to-camera)
        forward = self._focus - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            rn = np.linalg.norm(right)
        right /= rn
        up = np.cross(right, forward)

        # Open3D extrinsic: [R|t] where R columns are right, -up, forward
        # and t = -R @ eye
        R = np.array([right, -up, forward])  # 3x3
        t = -R @ eye
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        param.extrinsic = extrinsic
        try:
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

            # Render and capture
            vis.poll_events()
            vis.update_renderer()
            img_buf = vis.capture_screen_float_buffer(do_render=True)
        except Exception:
            return  # visualizer was destroyed during rendering
        img_arr = (np.asarray(img_buf) * 255).astype(np.uint8)

        # Display in Tk canvas
        pil_img = Image.fromarray(img_arr)

        # Draw orientation axes overlay (bottom-left, ParaView-style)
        self._draw_axis_overlay(pil_img, R)

        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ------ axis overlay ------

    def _draw_axis_overlay(self, pil_img, R):
        """Draw an orientation axis widget in the bottom-left corner.

        *R* is the 3×3 camera rotation matrix (world-to-camera) used for
        the current frame.  The axes are projected with the same
        rotation but drawn at a fixed screen position so they behave
        like ParaView's orientation widget.
        """
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        # Centre of the axis widget (bottom-left with some margin)
        cx, cy = 52, h - 52
        axis_len = 40  # pixels

        # World axes projected to screen via camera rotation R
        # R rows are: right, -up, forward  →  screen x = right, y = up
        axes = [
            (np.array([1.0, 0.0, 0.0]), (220, 60, 60),   "X"),
            (np.array([0.0, 1.0, 0.0]), (60, 180, 60),   "Y"),
            (np.array([0.0, 0.0, 1.0]), (70, 120, 220),  "Z"),
        ]

        # Sort by depth (forward component) so farther axes draw first
        projected = []
        for world_dir, color, label in axes:
            cam = R @ world_dir        # [right, down, forward]
            sx = cam[0] * axis_len     # screen x (right)
            sy = cam[1] * axis_len     # screen y (down — matches PIL coords)
            depth = cam[2]             # forward (into screen)
            projected.append((depth, sx, sy, color, label))
        projected.sort(key=lambda t: t[0])  # draw far-to-near

        a_size = 7   # arrowhead length in pixels
        a_half = 3   # arrowhead half-width

        for depth, sx, sy, color, label in projected:
            tip_len = math.hypot(sx, sy)
            if tip_len < 1e-3:
                continue

            # Unit direction from centre to tip
            dx = sx / tip_len
            dy = sy / tip_len
            # Perpendicular
            px = -dy
            py = dx

            # End-point of the shaft (stop short of arrowhead)
            shaft_ex = cx + sx - dx * a_size
            shaft_ey = cy + sy - dy * a_size
            tip_x = cx + sx
            tip_y = cy + sy

            # Draw shaft line (from centre to base of arrowhead)
            draw.line([(cx, cy), (shaft_ex, shaft_ey)],
                      fill=color, width=2)

            # Arrowhead — filled triangle
            base1 = (shaft_ex + px * a_half,
                     shaft_ey + py * a_half)
            base2 = (shaft_ex - px * a_half,
                     shaft_ey - py * a_half)
            draw.polygon([(tip_x, tip_y), base1, base2], fill=color)

            # Label — offset outward from the tip
            lx = cx + sx + dx * 10
            ly = cy + sy + dy * 10
            # Centre the text on (lx, ly); approximate glyph size ~6×10 px
            draw.text((lx - 4, ly - 6), label, fill=color)

    # ------ mouse interaction ------

    def _on_press(self, event):
        self._drag_start = (event.x, event.y, self._azimuth, self._elevation)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        x0, y0, az0, el0 = self._drag_start
        dx = event.x - x0
        dy = event.y - y0
        self._azimuth = az0 - dx * 0.4
        self._elevation = max(-89, min(89, el0 + dy * 0.4))
        self._request_render()

    def _on_release(self, event):
        self._drag_start = None

    def _on_press_right(self, event):
        self._drag_right_start = (event.x, event.y,
                                  self._focus.copy())

    def _on_drag_right(self, event):
        if self._drag_right_start is None:
            return
        x0, y0, foc0 = self._drag_right_start
        dx = event.x - x0
        dy = event.y - y0
        scale = (self._distance or 1.0) * 0.002
        az = math.radians(self._azimuth)
        rx, ry = math.sin(az), -math.cos(az)
        self._focus = foc0.copy()
        self._focus[0] += dx * scale * rx
        self._focus[1] += dx * scale * ry
        self._focus[2] -= dy * scale
        self._request_render()

    def _on_release_right(self, event):
        self._drag_right_start = None

    def _on_scroll(self, event):
        factor = 0.9 if event.delta > 0 else 1.1
        if self._distance is not None:
            self._distance *= factor
            self._request_render()

    def _on_scroll_up(self, event):
        if self._distance is not None:
            self._distance *= 0.9
            self._request_render()

    def _on_scroll_down(self, event):
        if self._distance is not None:
            self._distance *= 1.1
            self._request_render()

    def _on_canvas_resize(self, event):
        """Debounced handler: rebuild the Open3D window at the new size."""
        new_w = max(event.width, 64)
        new_h = max(event.height, 64)
        if new_w == self._RENDER_W and new_h == self._RENDER_H:
            return
        # Debounce — only act 200 ms after the last resize event
        if self._resize_after_id is not None:
            self._canvas.after_cancel(self._resize_after_id)
        self._resize_after_id = self._canvas.after(
            200, lambda: self._do_resize(new_w, new_h))

    def _do_resize(self, new_w, new_h):
        """Recreate the Open3D visualizer at the new canvas size."""
        self._resize_after_id = None
        if self._vis is None:
            return  # nothing rendered yet
        if self._rendering:
            # A render is in progress — retry after a short delay
            self._resize_after_id = self._canvas.after(
                100, lambda: self._do_resize(new_w, new_h))
            return
        self._RENDER_W = new_w
        self._RENDER_H = new_h
        # Rebuild with the stored meshes
        meshes = getattr(self, '_o3d_meshes', None)
        if meshes:
            self.set_meshes(meshes)


# ===================================================================
#  Unified tkinter selection dialog
# ===================================================================

def _pick_items_dialog(parent, geometry_folders, results_dir,
                       geo_checked=True, res_checked=True,
                       load_and_show_fn=None, source_dir=None):
    """Persistent selection dialog with geometry folders, result files,
    and optionally particle-source (.bl) files.

    The dialog stays open so the user can change the selection and click
    **View** repeatedly.  Each click launches a new Open3D window via
    *load_and_show_fn(sel_geo, sel_res, sf, sel_bl, show_dir, arrow_len)*
    in a daemon thread.
    The dialog closes only when the user clicks **Close** or the ✕.

    If *load_and_show_fn* is ``None`` the dialog falls back to the old
    one-shot behaviour (returns selection and closes).

    Returns ``(None, None)`` — the callback-based flow makes the return
    value unused when *load_and_show_fn* is supplied.
    """
    from tkinter import ttk

    dlg = tk.Toplevel(parent)
    dlg.title("Select items to display")
    dlg.geometry("1100x650")
    dlg.minsize(700, 400)
    dlg.result_geo = None
    dlg.result_res = None

    # Use update + deiconify to avoid flicker
    dlg.withdraw()
    dlg.update_idletasks()
    dlg.deiconify()

    # ====== BOTTOM BAR (scale + status + buttons) — packed FIRST so it
    #        never gets pushed off-screen by the expanding main pane. ======
    bottom_frm = ttk.Frame(dlg)
    bottom_frm.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    # --- Power density scale factor ---
    scale_frm = ttk.Frame(bottom_frm)
    scale_frm.pack(fill="x", pady=(4, 0))
    ttk.Label(scale_frm, text="Power density scale factor:",
              font=("", 10)).pack(side="left")
    scale_var = tk.StringVar(value="1.0")
    scale_entry = ttk.Entry(scale_frm, textvariable=scale_var, width=14)
    scale_entry.pack(side="left", padx=(6, 0))
    ttk.Label(scale_frm, text="(e.g. 1e-6 for MW/m²)",
              foreground="grey", font=("", 9)).pack(side="left", padx=(6, 0))

    # --- Source options row (only shown when source_dir is supplied) ---
    show_dir_var = tk.BooleanVar(value=True)
    arrow_len_var = tk.StringVar(value="0")
    src_abs = source_dir  # already absolute when passed from entry points
    if src_abs is not None and os.path.isdir(src_abs):
        src_opts_frm = ttk.Frame(bottom_frm)
        src_opts_frm.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(src_opts_frm, text="Plot source direction",
                         variable=show_dir_var).pack(side="left")
        ttk.Label(src_opts_frm, text="    Arrow length (m):",
                  font=("", 10)).pack(side="left", padx=(12, 0))
        ttk.Entry(src_opts_frm, textvariable=arrow_len_var,
                  width=8).pack(side="left", padx=(6, 0))
        ttk.Label(src_opts_frm, text="(0 = auto)",
                  foreground="grey", font=("", 9)).pack(side="left", padx=(4, 0))

    # --- status label ---
    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(bottom_frm, textvariable=status_var,
                           foreground="grey")
    status_lbl.pack(fill="x", pady=(2, 0))

    # --- buttons ---
    btn_frm = ttk.Frame(bottom_frm)
    btn_frm.pack(fill="x", pady=(4, 0))

    # ====== MAIN AREA — horizontal PanedWindow ======
    main_pane = tk.PanedWindow(dlg, orient="horizontal", sashwidth=6,
                                sashrelief="raised", bg="#cccccc")
    main_pane.pack(fill="both", expand=True, padx=4, pady=4)

    # -- LEFT: scrollable checkbox list (resizable via sash) --
    left_frm = ttk.Frame(main_pane)

    # --- scrollable frame with BOTH vertical and horizontal scrollbars ---
    canvas = tk.Canvas(left_frm, borderwidth=0, highlightthickness=0)
    v_scroll = ttk.Scrollbar(left_frm, orient="vertical",
                              command=canvas.yview)
    h_scroll = ttk.Scrollbar(left_frm, orient="horizontal",
                              command=canvas.xview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=v_scroll.set,
                     xscrollcommand=h_scroll.set)

    # Grid layout: canvas fills, scrollbars on edges
    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")
    left_frm.rowconfigure(0, weight=1)
    left_frm.columnconfigure(0, weight=1)

    main_pane.add(left_frm, minsize=180, width=260)

    # -- CENTRE: embedded 3D viewport --
    view_frm = ttk.LabelFrame(main_pane, text="3D View")
    viewer = _EmbeddedViewer(view_frm)
    main_pane.add(view_frm, minsize=300)

    # -- RIGHT: colorbar --
    cbar_frm = ttk.LabelFrame(main_pane, text="Colour Bar")
    cbar_canvas = tk.Canvas(cbar_frm, width=120, height=500,
                            bg="#f0f2f5", highlightthickness=0)
    cbar_canvas.pack(fill="both", expand=True, padx=4, pady=4)
    cbar_canvas.create_text(60, 250, text="No data yet",
                            fill="#999", font=("Segoe UI", 9))
    main_pane.add(cbar_frm, minsize=130, width=140)

    dlg._cbar_canvas = cbar_canvas
    dlg._viewer = viewer

    all_geo_vars = {}   # folder_name → BooleanVar
    all_res_vars = {}   # vtp_abs_path → BooleanVar

    # --- Geometry folders ---
    if geometry_folders:
        ttk.Label(inner, text="Geometry",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        ci = 0
        for folder in geometry_folders:
            color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
            ci += 1
            var = tk.BooleanVar(value=geo_checked)
            frm = ttk.Frame(inner)
            frm.pack(fill="x", pady=1, padx=8)
            sw = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
            hex_c = "#%02x%02x%02x" % tuple(int(c * 255) for c in color)
            sw.create_rectangle(0, 0, 14, 14, fill=hex_c, outline=hex_c)
            sw.pack(side="left", padx=(0, 6))
            ttk.Checkbutton(frm, text=folder, variable=var).pack(side="left")
            all_geo_vars[folder] = var

    # --- Result files ---
    results_abs = results_dir or ""
    result_sets = {}  # subdir_name → [vtp_abs_paths]
    if os.path.isdir(results_abs):
        for d in sorted(os.listdir(results_abs)):
            sub = os.path.join(results_abs, d)
            if not os.path.isdir(sub):
                continue
            vtps = sorted(glob.glob(os.path.join(sub, "**", "*.vtp"),
                                    recursive=True))
            if vtps:
                result_sets[d] = vtps

    if result_sets:
        ttk.Label(inner, text="Results",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
        for set_name, vtps in result_sets.items():
            ttk.Label(inner, text=f"  {set_name}",
                      font=("", 10, "italic")).pack(anchor="w", padx=8, pady=(4, 0))
            for vtp in vtps:
                bn = os.path.splitext(os.path.basename(vtp))[0]
                var = tk.BooleanVar(value=res_checked)
                ttk.Checkbutton(inner, text=f"    {bn}",
                                variable=var).pack(anchor="w", padx=12, pady=0)
                all_res_vars[vtp] = var

    # --- Source (.bl) files ---
    all_bl_vars = {}   # bl_abs_path → BooleanVar
    if src_abs and os.path.isdir(src_abs):
        bl_files = sorted(glob.glob(os.path.join(src_abs, "*.bl")))
        if bl_files:
            ttk.Label(inner, text="Sources",
                      font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
            for bl in bl_files:
                bn = os.path.splitext(os.path.basename(bl))[0]
                var = tk.BooleanVar(value=False)
                ttk.Checkbutton(inner, text=f"  {bn}",
                                variable=var).pack(anchor="w", padx=12, pady=1)
                all_bl_vars[bl] = var

    # --- buttons (in the bottom bar) ---
    def _all():
        for v in (list(all_geo_vars.values()) + list(all_res_vars.values())
                  + list(all_bl_vars.values())):
            v.set(True)

    def _none():
        for v in (list(all_geo_vars.values()) + list(all_res_vars.values())
                  + list(all_bl_vars.values())):
            v.set(False)

    ttk.Button(btn_frm, text="All", width=6, command=_all).pack(side="left", padx=2)
    ttk.Button(btn_frm, text="None", width=6, command=_none).pack(side="left", padx=2)

    def _get_selection():
        sel_g = [f for f, v in all_geo_vars.items() if v.get()]
        sel_r = [p for p, v in all_res_vars.items() if v.get()]
        sel_bl = [p for p, v in all_bl_vars.items() if v.get()]
        try:
            sf = float(scale_var.get())
        except (ValueError, TypeError):
            sf = 1.0
        sd = show_dir_var.get()
        try:
            al = float(arrow_len_var.get())
        except (ValueError, TypeError):
            al = 0.0
        return sel_g, sel_r, sf, sel_bl, sd, al

    if load_and_show_fn is not None:
        # ---- Persistent mode: View button renders into embedded viewer ----
        _busy = threading.Lock()

        def _view():
            if not _busy.acquire(blocking=False):
                status_var.set("Still loading… please wait.")
                return
            sel_g, sel_r, sf, sel_bl, sd, al = _get_selection()
            if not sel_g and not sel_r and not sel_bl:
                status_var.set("Nothing selected.")
                _busy.release()
                return
            n = len(sel_g) + len(sel_r) + len(sel_bl)
            status_var.set(f"Loading {n} item(s)…")
            view_btn.configure(state="disabled")

            # Draw colour-bar immediately (fast VTP scan)
            if sel_r:
                try:
                    vmax_quick = _scan_vtp_vmax(sel_r)
                except Exception:
                    vmax_quick = 0.0
                if vmax_quick > 0 and hasattr(dlg, '_cbar_canvas'):
                    _draw_colorbar_on_canvas(dlg._cbar_canvas,
                                             0.0, vmax_quick, sf)
            dlg.update_idletasks()

            # Load meshes in worker thread, then render
            threading.Thread(
                target=lambda: _view_worker(sel_g, sel_r, sf,
                                            sel_bl, sd, al),
                daemon=True,
            ).start()

        def _view_worker(sel_g, sel_r, sf, sel_bl, sd, al):
            try:
                geoms, vmax, title = load_and_show_fn(
                    sel_g, sel_r, sf, sel_bl, sd, al)
            except Exception as exc:
                try:
                    dlg.after(0, lambda: status_var.set(f"Error: {exc}"))
                except tk.TclError:
                    pass
                return
            finally:
                _busy.release()
                try:
                    dlg.after(0, lambda: view_btn.configure(state="normal"))
                except tk.TclError:
                    pass
            if not geoms:
                try:
                    dlg.after(0, lambda: status_var.set("No meshes to display."))
                except tk.TclError:
                    pass
                return
            # Schedule the render on the main Tk thread
            def _do_render():
                try:
                    dlg._viewer.set_meshes(geoms)
                    info = title
                    info += "  —  drag to rotate, scroll to zoom"
                    status_var.set(info)
                except Exception as exc:
                    status_var.set(f"Render error: {exc}")
            try:
                dlg.after(0, _do_render)
            except tk.TclError:
                pass

        def _close():
            # Destroy the Open3D visualizer before closing
            if hasattr(dlg, '_viewer') and dlg._viewer._vis is not None:
                try:
                    dlg._viewer._vis.destroy_window()
                    dlg._viewer._vis = None
                except Exception:
                    pass
            dlg.destroy()

        view_btn = ttk.Button(btn_frm, text="View", command=_view)
        view_btn.pack(side="right", padx=2)
        ttk.Button(btn_frm, text="Close", command=_close).pack(side="right", padx=2)

        # Don't block with wait_window — the dialog is non-modal while
        # the viewer is open.  The caller returns immediately.
    else:
        # ---- One-shot fallback (legacy) ----
        def _ok():
            dlg.result_geo, dlg.result_res, _, _, _, _ = _get_selection()
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        ttk.Button(btn_frm, text="View", command=_ok).pack(side="right", padx=2)
        ttk.Button(btn_frm, text="Cancel", command=_cancel).pack(side="right", padx=2)

        parent.wait_window(dlg)

    return dlg.result_geo, dlg.result_res, dlg


# ===================================================================
#  Loaders
# ===================================================================

def _load_selected_geometry(script_dir, geometry_folders, selected_folders):
    """Load STL meshes for *selected_folders*."""
    meshes = []
    ci = 0
    for folder in geometry_folders:
        color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
        ci += 1
        if folder not in selected_folders:
            continue
        settings = geometry_folders[folder]
        scale = settings.get("scale", 1)
        folder_abs = _resolve_viewer_path(script_dir, folder)
        if not os.path.isdir(folder_abs):
            continue
        for stl_path in sorted(
            glob.glob(os.path.join(folder_abs, "*.stl"))
            + glob.glob(os.path.join(folder_abs, "*.STL"))
        ):
            mesh = _load_stl(stl_path, scale, color)
            if mesh is not None:
                meshes.append(mesh)
    return meshes


def _load_selected_results(selected_vtp_paths, scale_factor=1.0):
    """Load VTP result files and return coloured meshes + max power.

    Power density values are multiplied by *scale_factor* before
    colouring so that the user can express them in convenient units
    (e.g. 1e-6 → MW/m²).  The returned *vmax_all* is the raw
    (un-scaled) maximum so the colour-bar can display scaled ticks.
    """
    meshes = []
    vmax_all = 0.0
    # First pass: find global vmax across all VTPs
    all_data = []
    for vtp in selected_vtp_paths:
        mesh, power = _read_vtp_as_o3d(vtp)
        if mesh is None:
            continue
        if power is not None and len(power) > 0:
            vmax_all = max(vmax_all, power.max())
        all_data.append((mesh, power))

    # Second pass: colour with consistent scale
    for mesh, power in all_data:
        _colour_mesh_by_power(mesh, power, global_vmax=vmax_all if vmax_all > 0 else None)
        meshes.append(mesh)

    return meshes, vmax_all


# ===================================================================
#  Particle source helpers
# ===================================================================

def _load_bl_file(bl_path):
    """Read a .bl file and return a DataFrame with source data."""
    try:
        df = pd.read_csv(bl_path, comment='#', sep=r'\s+')
    except Exception:
        return None
    return df


def _build_source_geometries(bl_paths, arrow_length=0.0,
                              show_direction=False):
    """Build Open3D geometries for particle sources.

    Source positions are rendered as a **PointCloud** (zero triangle
    overhead — just coloured dots).  Direction arrows, when requested,
    are built as a single merged TriangleMesh (cylinder + cone per
    source).

    Each source is coloured by its ``CurrentDensity_A_m2``.

    Returns
    -------
    geoms : list[o3d.geometry.Geometry]
        A PointCloud and, optionally, a merged arrow TriangleMesh.
    vmax : float
        Maximum current density across all sources.
    """
    # Collect all sources across files (vectorised — no iterrows)
    dfs = []
    for bl in bl_paths:
        df = _load_bl_file(bl)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return [], 0.0

    combined = pd.concat(dfs, ignore_index=True)

    # Extract positions, directions, current densities as numpy arrays
    positions = combined[['CenterX', 'CenterY', 'CenterZ']].to_numpy(
        dtype=np.float64)
    directions = combined[['DirX', 'DirY', 'DirZ']].to_numpy(
        dtype=np.float64)
    currents = combined['CurrentDensity_A_m2'].to_numpy(dtype=np.float64)

    # Normalise for colouring
    vmax = currents.max() if len(currents) > 0 else 0.0
    if vmax > 0:
        normed = np.clip(currents / vmax, 0, 1)
    else:
        normed = np.zeros(len(currents))

    colors = _apply_jet_colormap(normed)

    # --- Source dots as PointCloud (zero triangles) ---
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(positions)
    pc.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pc]

    # --- Direction arrows (only when requested) ---
    if show_direction:
        # Auto-scale arrow length from scene size
        if len(positions) > 1:
            extent = positions.max(axis=0) - positions.min(axis=0)
            scene_size = np.linalg.norm(extent)
        else:
            scene_size = 1.0
        if arrow_length <= 0:
            arrow_length = scene_size * 0.08
        shaft_r = scene_size * 0.003
        shaft_len = arrow_length * 0.75
        cone_len = arrow_length * 0.25
        cone_r = shaft_r * 2.5

        unit_cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=shaft_r, height=shaft_len, resolution=6)
        unit_cyl.translate([0, 0, shaft_len / 2])  # base at origin
        unit_cyl.compute_vertex_normals()
        cyl_verts_np = np.asarray(unit_cyl.vertices)
        cyl_tris_np = np.asarray(unit_cyl.triangles)
        n_cv = len(cyl_verts_np)

        unit_cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_r, height=cone_len, resolution=6)
        unit_cone.translate([0, 0, shaft_len])  # base at top of shaft
        unit_cone.compute_vertex_normals()
        cone_verts_np = np.asarray(unit_cone.vertices)
        cone_tris_np = np.asarray(unit_cone.triangles)
        n_kov = len(cone_verts_np)

        all_verts = []
        all_tris = []
        all_colors_v = []
        vert_offset = 0
        z_axis = np.array([0.0, 0.0, 1.0])

        for i in range(len(positions)):
            pos = positions[i]
            col = colors[i]
            d = directions[i]
            d_norm = d / (np.linalg.norm(d) + 1e-12)

            # Compute rotation matrix from Z to direction
            rot_axis = np.cross(z_axis, d_norm)
            rot_norm = np.linalg.norm(rot_axis)
            if rot_norm > 1e-6:
                rot_axis /= rot_norm
                angle = np.arccos(np.clip(np.dot(z_axis, d_norm), -1, 1))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rot_axis * angle)
            elif np.dot(z_axis, d_norm) < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1.0, 0.0, 0.0]) * np.pi)
            else:
                R = np.eye(3)

            # Cylinder
            cv = (cyl_verts_np @ R.T) + pos
            all_verts.append(cv)
            all_tris.append(cyl_tris_np + vert_offset)
            all_colors_v.append(np.tile(col, (n_cv, 1)))
            vert_offset += n_cv

            # Cone
            kov = (cone_verts_np @ R.T) + pos
            all_verts.append(kov)
            all_tris.append(cone_tris_np + vert_offset)
            all_colors_v.append(np.tile(col, (n_kov, 1)))
            vert_offset += n_kov

        # Build single merged arrow mesh
        arrow_mesh = o3d.geometry.TriangleMesh()
        arrow_mesh.vertices = o3d.utility.Vector3dVector(
            np.vstack(all_verts))
        arrow_mesh.triangles = o3d.utility.Vector3iVector(
            np.vstack(all_tris))
        arrow_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.vstack(all_colors_v))
        arrow_mesh.compute_vertex_normals()
        geoms.append(arrow_mesh)

    return geoms, vmax


# ===================================================================
#  Source-viewer dialog
# ===================================================================

def _pick_sources_dialog(parent, script_dir, source_dir, geometry_folders):
    """Selection dialog for particle sources with embedded 3D viewer.

    Shows .bl files as checkboxes, plus options for direction arrows and
    arrow length.  Geometry folders can optionally be overlaid.
    """
    from tkinter import ttk

    src_abs = (_resolve_viewer_path(script_dir, source_dir)
               if not os.path.isabs(source_dir) else source_dir)

    dlg = tk.Toplevel(parent)
    dlg.title("View Particle Sources")
    dlg.geometry("1100x650")
    dlg.minsize(700, 400)

    dlg.withdraw()
    dlg.update_idletasks()
    dlg.deiconify()

    # ====== BOTTOM BAR ======
    bottom_frm = ttk.Frame(dlg)
    bottom_frm.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

    # --- Options row ---
    opts_frm = ttk.Frame(bottom_frm)
    opts_frm.pack(fill="x", pady=(4, 0))

    show_dir_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(opts_frm, text="Plot source direction",
                     variable=show_dir_var).pack(side="left")

    ttk.Label(opts_frm, text="    Arrow length (m):",
              font=("", 10)).pack(side="left", padx=(12, 0))
    arrow_len_var = tk.StringVar(value="0")
    ttk.Entry(opts_frm, textvariable=arrow_len_var, width=8).pack(
        side="left", padx=(6, 0))
    ttk.Label(opts_frm, text="(0 = auto)",
              foreground="grey", font=("", 9)).pack(side="left", padx=(4, 0))

    # --- Status ---
    status_var = tk.StringVar(value="")
    ttk.Label(bottom_frm, textvariable=status_var,
              foreground="grey").pack(fill="x", pady=(2, 0))

    # --- Buttons ---
    btn_frm = ttk.Frame(bottom_frm)
    btn_frm.pack(fill="x", pady=(4, 0))

    # ====== MAIN AREA ======
    main_pane = tk.PanedWindow(dlg, orient="horizontal", sashwidth=6,
                                sashrelief="raised", bg="#cccccc")
    main_pane.pack(fill="both", expand=True, padx=4, pady=4)

    # -- LEFT: checkboxes --
    left_frm = ttk.Frame(main_pane)
    canvas = tk.Canvas(left_frm, borderwidth=0, highlightthickness=0)
    v_scroll = ttk.Scrollbar(left_frm, orient="vertical",
                              command=canvas.yview)
    h_scroll = ttk.Scrollbar(left_frm, orient="horizontal",
                              command=canvas.xview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=v_scroll.set,
                     xscrollcommand=h_scroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")
    left_frm.rowconfigure(0, weight=1)
    left_frm.columnconfigure(0, weight=1)
    main_pane.add(left_frm, minsize=180, width=260)

    # -- CENTRE: 3D viewport --
    view_frm = ttk.LabelFrame(main_pane, text="3D View")
    viewer = _EmbeddedViewer(view_frm)
    main_pane.add(view_frm, minsize=300)

    # -- RIGHT: colorbar --
    cbar_frm = ttk.LabelFrame(main_pane, text="Current Density")
    cbar_canvas = tk.Canvas(cbar_frm, width=120, height=500,
                            bg="#f0f2f5", highlightthickness=0)
    cbar_canvas.pack(fill="both", expand=True, padx=4, pady=4)
    cbar_canvas.create_text(60, 250, text="No data yet",
                            fill="#999", font=("Segoe UI", 9))
    main_pane.add(cbar_frm, minsize=130, width=140)

    dlg._cbar_canvas = cbar_canvas
    dlg._viewer = viewer

    # --- Populate checkboxes ---
    all_bl_vars = {}   # bl_abs_path → BooleanVar
    all_geo_vars = {}  # folder_name → BooleanVar

    # Source files
    bl_files = sorted(glob.glob(os.path.join(src_abs, "*.bl")))
    if bl_files:
        ttk.Label(inner, text="Source Files",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        for bl in bl_files:
            bn = os.path.splitext(os.path.basename(bl))[0]
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(inner, text=f"  {bn}",
                            variable=var).pack(anchor="w", padx=12, pady=1)
            all_bl_vars[bl] = var
    else:
        ttk.Label(inner, text="(no .bl files found)",
                  foreground="grey").pack(padx=8, pady=8)

    # Geometry overlays
    if geometry_folders:
        ttk.Label(inner, text="Geometry Overlay",
                  font=("", 11, "bold")).pack(anchor="w", padx=4, pady=(12, 2))
        ci = 0
        for folder in geometry_folders:
            color = _FOLDER_COLORS[ci % len(_FOLDER_COLORS)]
            ci += 1
            var = tk.BooleanVar(value=False)
            frm = ttk.Frame(inner)
            frm.pack(fill="x", pady=1, padx=8)
            sw = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
            hex_c = "#%02x%02x%02x" % tuple(int(c * 255) for c in color)
            sw.create_rectangle(0, 0, 14, 14, fill=hex_c, outline=hex_c)
            sw.pack(side="left", padx=(0, 6))
            ttk.Checkbutton(frm, text=folder, variable=var).pack(side="left")
            all_geo_vars[folder] = var

    # --- Button actions ---
    def _all():
        for v in list(all_bl_vars.values()) + list(all_geo_vars.values()):
            v.set(True)

    def _none():
        for v in list(all_bl_vars.values()) + list(all_geo_vars.values()):
            v.set(False)

    ttk.Button(btn_frm, text="All", width=6, command=_all).pack(
        side="left", padx=2)
    ttk.Button(btn_frm, text="None", width=6, command=_none).pack(
        side="left", padx=2)

    _busy = threading.Lock()

    def _view():
        if not _busy.acquire(blocking=False):
            status_var.set("Still loading… please wait.")
            return
        sel_bl = [p for p, v in all_bl_vars.items() if v.get()]
        sel_geo = [f for f, v in all_geo_vars.items() if v.get()]
        if not sel_bl and not sel_geo:
            status_var.set("Nothing selected.")
            _busy.release()
            return

        try:
            al = float(arrow_len_var.get())
        except (ValueError, TypeError):
            al = 0.0
        sd = show_dir_var.get()

        n = len(sel_bl) + len(sel_geo)
        status_var.set(f"Loading {n} item(s)…")
        view_btn.configure(state="disabled")
        dlg.update_idletasks()

        threading.Thread(
            target=lambda: _view_worker(sel_bl, sel_geo, al, sd),
            daemon=True).start()

    def _view_worker(sel_bl, sel_geo, al, sd):
        try:
            geoms = []
            vmax = 0.0

            # Load sources
            if sel_bl:
                src_geoms, vmax = _build_source_geometries(
                    sel_bl, arrow_length=al, show_direction=sd)
                geoms += src_geoms

            # Load geometry overlays
            if sel_geo:
                geoms += _load_selected_geometry(script_dir, geometry_folders,
                                                 sel_geo)
        except Exception as exc:
            try:
                dlg.after(0, lambda: status_var.set(f"Error: {exc}"))
            except tk.TclError:
                pass
            return
        finally:
            _busy.release()
            try:
                dlg.after(0, lambda: view_btn.configure(state="normal"))
            except tk.TclError:
                pass

        if not geoms:
            try:
                dlg.after(0, lambda: status_var.set("No meshes to display."))
            except tk.TclError:
                pass
            return

        def _do_render():
            try:
                dlg._viewer.set_meshes(geoms)
                if vmax > 0:
                    _draw_colorbar_on_canvas(
                        dlg._cbar_canvas, 0.0, vmax, 1.0,
                        unit_label="A/m²")
                info = f"Sources ({len(geoms)} objects)"
                if vmax > 0:
                    info += f"  —  max j = {vmax:.2e} A/m²"
                info += "  —  drag to rotate, scroll to zoom"
                status_var.set(info)
            except Exception as exc:
                status_var.set(f"Render error: {exc}")
        try:
            dlg.after(0, _do_render)
        except tk.TclError:
            pass

    def _close():
        if hasattr(dlg, '_viewer') and dlg._viewer._vis is not None:
            try:
                dlg._viewer._vis.destroy_window()
                dlg._viewer._vis = None
            except Exception:
                pass
        dlg.destroy()

    view_btn = ttk.Button(btn_frm, text="View", command=_view)
    view_btn.pack(side="right", padx=2)
    ttk.Button(btn_frm, text="Close", command=_close).pack(
        side="right", padx=2)


# ===================================================================
#  High-level entry points (called from sim_gui.py)
# ===================================================================

class _MeshCache:
    """Caches loaded geometry / results / sources across View clicks.

    Only the items whose selection or parameters changed are reloaded;
    everything else is reused from the previous call.
    """

    def __init__(self, script_dir, geometry_folders):
        self._script_dir = script_dir
        self._geometry_folders = geometry_folders
        # geometry:  folder_name → list[mesh]
        self._geo_cache = {}
        # results:   vtp_path → (mesh, power_array)
        self._res_cache = {}
        self._res_vmax = 0.0
        self._prev_res_sel = []
        # sources:   separate caches for dots (PointCloud) and arrows
        self._src_files_key = None   # tuple(sorted bl paths)
        self._src_pc = None          # PointCloud (dots)
        self._src_vmax = 0.0
        self._src_arrow_key = None   # (files_key, arrow_len, show_dir)
        self._src_arrow_mesh = None  # merged arrow TriangleMesh or None

    # -- geometry --
    def get_geometry(self, sel_folders):
        """Return meshes for *sel_folders*, loading only new ones."""
        meshes = []
        for folder in sel_folders:
            if folder not in self._geo_cache:
                self._geo_cache[folder] = _load_selected_geometry(
                    self._script_dir, self._geometry_folders, [folder])
            meshes += self._geo_cache[folder]
        return meshes

    # -- results --
    def get_results(self, sel_vtps, sf):
        """Return coloured meshes + vmax, reloading only new VTPs.

        If the selection changed we must re-colour everything because
        the global vmax may have changed.
        """
        sel_set = tuple(sorted(sel_vtps))
        if sel_set == tuple(sorted(self._prev_res_sel)):
            # Exact same selection — return cached
            return [m for m, _ in
                    [self._res_cache[v] for v in sel_vtps
                     if v in self._res_cache]], self._res_vmax

        # Load any VTPs we haven't seen
        for vtp in sel_vtps:
            if vtp not in self._res_cache:
                mesh, power = _read_vtp_as_o3d(vtp)
                if mesh is not None:
                    self._res_cache[vtp] = (mesh, power)

        # Compute global vmax across selection
        vmax_all = 0.0
        pairs = []
        for vtp in sel_vtps:
            if vtp not in self._res_cache:
                continue
            mesh, power = self._res_cache[vtp]
            if power is not None and len(power) > 0:
                vmax_all = max(vmax_all, float(power.max()))
            pairs.append((mesh, power))

        # Re-colour with consistent scale
        for mesh, power in pairs:
            _colour_mesh_by_power(mesh, power,
                                  global_vmax=vmax_all if vmax_all > 0 else None)

        self._res_vmax = vmax_all
        self._prev_res_sel = list(sel_vtps)
        return [m for m, _ in pairs], vmax_all

    # -- sources --
    def get_sources(self, sel_bl, arrow_len, show_dir):
        """Return source geometries + vmax.

        The PointCloud (dots) is cached by file selection alone —
        changing arrow length or direction toggle does NOT rebuild it.
        The arrow mesh is cached separately by (files, arrow_len,
        show_dir) so only the arrows are rebuilt when those change.
        """
        files_key = tuple(sorted(sel_bl))

        # Rebuild dots only when file selection changes
        if files_key != self._src_files_key:
            # Need full rebuild (dots + arrows)
            all_geoms, vmax = _build_source_geometries(
                sel_bl, arrow_length=arrow_len, show_direction=show_dir)
            # First element is always the PointCloud
            self._src_pc = all_geoms[0] if all_geoms else None
            self._src_vmax = vmax
            self._src_files_key = files_key
            arrow_key = (files_key, arrow_len, show_dir)
            self._src_arrow_key = arrow_key
            self._src_arrow_mesh = all_geoms[1] if len(all_geoms) > 1 else None
            return list(all_geoms), vmax

        # Dots are cached — check if arrows need rebuilding
        arrow_key = (files_key, arrow_len, show_dir)
        if arrow_key != self._src_arrow_key:
            # Rebuild only arrows (dots will be identical, discard them)
            all_geoms, _ = _build_source_geometries(
                sel_bl, arrow_length=arrow_len, show_direction=show_dir)
            self._src_arrow_key = arrow_key
            self._src_arrow_mesh = all_geoms[1] if len(all_geoms) > 1 else None

        # Assemble result
        geoms = []
        if self._src_pc is not None:
            geoms.append(self._src_pc)
        if self._src_arrow_mesh is not None:
            geoms.append(self._src_arrow_mesh)
        return geoms, self._src_vmax


def _status_msg(parent, msg):
    if hasattr(parent, "_log"):
        parent._log(f"{msg}\n")


def view_geometry(parent, script_dir, geometry_folders, source_dir=None):
    """Geometry viewer — persistent selection dialog → Open3D window."""
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, f"Geometry ({len(geoms)} meshes)"

    _pick_items_dialog(
        parent, geometry_folders, results_dir=None,
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_results(parent, script_dir, output_dir, geometry_folders=None,
                 source_dir=None):
    """Results viewer — persistent selection dialog → Open3D heatmap."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)
    geometry_folders = geometry_folders or {}
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        title = f"Results ({len(geoms)} meshes"
        if vmax > 0:
            title += f", max {vmax * sf:.2e}"
        title += ")"
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, title

    _pick_items_dialog(
        parent, geometry_folders, outdir_abs,
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_all(parent, script_dir, output_dir, geometry_folders,
             source_dir=None):
    """Show everything — persistent selection dialog → Open3D window."""
    outdir_abs = (os.path.join(script_dir, output_dir)
                  if not os.path.isabs(output_dir) else output_dir)
    src_abs = None
    if source_dir:
        src_abs = (_resolve_viewer_path(script_dir, source_dir)
                   if not os.path.isabs(source_dir) else source_dir)
    cache = _MeshCache(script_dir, geometry_folders)

    def _load_and_show(sel_geo, sel_res, sf, sel_bl, sd, al):
        geoms = cache.get_geometry(sel_geo) if sel_geo else []
        vmax = 0.0
        if sel_res:
            res_meshes, vmax = cache.get_results(sel_res, sf)
            geoms += res_meshes
        if sel_bl:
            src_geoms, _ = cache.get_sources(sel_bl, al, sd)
            geoms += src_geoms
        if not geoms:
            parent.after(0, lambda: _status_msg(parent, "No meshes found."))
            return [], 0.0, ""
        title = f"Viewer ({len(geoms)} meshes)"
        parent.after(0, lambda: _status_msg(parent,
                     f"Loading {len(geoms)} meshes…"))
        return geoms, vmax, title

    _pick_items_dialog(
        parent, geometry_folders, outdir_abs,
        geo_checked=False, res_checked=False,
        load_and_show_fn=_load_and_show,
        source_dir=src_abs,
    )


def view_sources(parent, script_dir, source_dir, geometry_folders=None):
    """Particle-source viewer — shows beamlet locations + optional arrows.

    Colour encodes current density (A/m²).
    """
    geometry_folders = geometry_folders or {}
    _pick_sources_dialog(parent, script_dir, source_dir, geometry_folders)


# ===================================================================
#  Standalone test
# ===================================================================
if __name__ == "__main__":
    import json
    from beamontarget.paths import get_project_root

    script_dir = str(get_project_root())
    cfg_path = os.path.join(script_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    gf = cfg.get("GEOMETRY_FOLDERS", {})
    folders = list(gf.keys())
    print(f"Available folders: {folders}")

    items = _load_selected_geometry(script_dir, gf, folders)
    print(f"Loaded {len(items)} geometry meshes")
    if items:
        _show_in_open3d(items, "Geometry Test")

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# sim_gui.py
"""
Tkinter GUI for managing the BeamOnTarget simulation.

Reads / writes config.json through the config module.
Launches run_simulation.py as a subprocess (preserving CLI compatibility).
Launches ParaView externally for geometry and results viewing.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
import ctypes
import os
import sys
import subprocess
import glob
import threading

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; sub-tabs import their own
from PIL import Image, ImageTk

from beamontarget.visualization import viewer  # built-in Open3D viewer
from beamontarget.config import load_config, save_config
from beamontarget.paths import get_project_root
from beamontarget.gui.gui_widgets import (
    make_card,
    parse_vec3,
    resolve_path as _resolve_path,
    set_project_folder as _set_project_folder,
    get_project_folder as _get_project_folder,
    to_relative_path as _to_relative_path,
    choose_font_family,
)
from beamontarget.gui.gui_fields import FieldsTab
from beamontarget.gui.gui_reactions import ReactionsTab
from beamontarget.gui.gui_results import ResultsTab
from beamontarget.gui.gui_merge import MergeTab
from beamontarget.gui.gui_transform import TransformTab
from beamontarget.gui.gui_geometry import GeometryTab
from beamontarget.gui.gui_particles import ParticlesTab
from beamontarget.gui.gui_output import OutputTab
from beamontarget.gui.gui_run import RunTab

# ---------------------------------------------------------------------------
# Resolve paths  (canonical copies live in gui_widgets; keep local for
# the handful of module-level constants that still need _SCRIPT_DIR).
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)  # True when running from PyInstaller exe
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else str(get_project_root()))
_CONFIG_JSON = os.path.join(_SCRIPT_DIR, "config.json")
_SPLASH_LOGO = os.path.join(_SCRIPT_DIR, "BOT_logo.png")
_APP_ICON_ICO = os.path.join(_SCRIPT_DIR, "BOT_icon.ico")


# ===================================================================
#  ParaView launcher helpers
# ===================================================================

def _pv_geometry_script(cfg):
    """Return a Python script string that ParaView will execute to show STLs."""
    folders = cfg.get("GEOMETRY_FOLDERS", {})
    colors = [
        (0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.2, 0.8, 0.3),
        (0.9, 0.7, 0.1), (0.6, 0.3, 0.7), (0.9, 0.4, 0.1),
        (0.4, 0.8, 0.8), (0.8, 0.4, 0.6),
    ]
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
        "rv.ResetCamera()",
    ]
    ci = 0
    for folder, settings in folders.items():
        scale = settings.get("scale", 1)
        folder_abs = _resolve_path(folder)
        if not os.path.isdir(folder_abs):
            continue
        stl_files = sorted(glob.glob(os.path.join(folder_abs, "*.stl")) +
                           glob.glob(os.path.join(folder_abs, "*.STL")))
        r, g, b = colors[ci % len(colors)]
        ci += 1
        for stl in stl_files:
            name = os.path.splitext(os.path.basename(stl))[0]
            lines.append(f"reader = STLReader(FileNames=[r'{stl}'])")
            if scale != 1:
                lines.append(f"t = Transform(Input=reader)")
                lines.append(f"t.Transform.Scale = [{scale}, {scale}, {scale}]")
                lines.append(f"dp = Show(t, rv)")
            else:
                lines.append(f"dp = Show(reader, rv)")
            lines.append(f"dp.DiffuseColor = [{r}, {g}, {b}]")
            lines.append(f"dp.Opacity = 0.85")
            lines.append(f"RenameSource('{folder}/{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    return "\n".join(lines)


def _pv_results_script(results_dir):
    """Return a Python script that opens all .vtp files in a results dir."""
    vtp_files = sorted(glob.glob(os.path.join(results_dir, "**", "*.vtp"), recursive=True))
    if not vtp_files:
        return None
    return _pv_files_script(vtp_files)


def _pv_files_script(vtp_files):
    """Return a Python script that opens the given .vtp files, coloured by power density."""
    lines = [
        "from paraview.simple import *",
        "paraview.simple._DisableFirstRenderCameraReset()",
        "rv = GetActiveViewOrCreate('RenderView')",
    ]
    stems = [os.path.splitext(os.path.basename(v))[0] for v in vtp_files]
    for vtp, stem in zip(vtp_files, stems):
        # Prefix the folder when the same file name occurs in several scenarios.
        name = (f"{os.path.basename(os.path.dirname(vtp))}/{stem}"
                if stems.count(stem) > 1 else stem)
        lines.append(f"reader = XMLPolyDataReader(FileName=[r'{vtp}'])")
        lines.append(f"dp = Show(reader, rv)")
        lines.append(f"dp.SetRepresentationType('Surface')")
        lines.append(f"ColorBy(dp, ('CELLS', 'Power_Density_W_m2'))")
        lines.append(f"dp.RescaleTransferFunctionToDataRange(True, False)")
        lines.append(f"RenameSource('{name}', reader)")
    lines.append("rv.ResetCamera()")
    lines.append("Render()")
    lines.append("# Show color bar")
    lines.append("dp.SetScalarBarVisibility(rv, True)")
    return "\n".join(lines)


def launch_paraview(script_content, pv_path, pv_module="ParaView"):
    """Write a temp script and launch ParaView.

    On Linux the EasyBuild module *pv_module* is loaded first (via
    ``bash -lc 'ml …'``) so that LD_LIBRARY_PATH etc. are set.

    On Windows ParaView is invoked directly - no module system needed.
    """
    tmp_script = os.path.join(_SCRIPT_DIR, ".pv_temp_script.py")
    with open(tmp_script, "w") as f:
        f.write(script_content)

    try:
        if sys.platform == "win32":
            # Windows: call ParaView executable directly
            proc = subprocess.Popen(
                [pv_path, "--script=" + tmp_script],
                cwd=_SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)
        else:
            # Linux / macOS: load EasyBuild module, then launch
            shell_cmd = (f'ml {pv_module} 2>/dev/null; '
                         f'"{pv_path}" --script="{tmp_script}"')
            proc = subprocess.Popen(
                ["bash", "-l", "-c", shell_cmd],
                cwd=_SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True)

        # Start a thread to watch for early failure (give it a few seconds)
        def _watch():
            try:
                proc.wait(timeout=5)
                if proc.returncode and proc.returncode != 0:
                    err = proc.stderr.read().strip() if proc.stderr else ""
                    msg = f"ParaView exited with code {proc.returncode}."
                    if err:
                        msg += f"\n\n{err[:600]}"
                    try:
                        messagebox.showerror("ParaView Error", msg)
                    except Exception:
                        pass
            except subprocess.TimeoutExpired:
                pass  # still running - good
        threading.Thread(target=_watch, daemon=True).start()
    except FileNotFoundError:
        messagebox.showerror("ParaView not found",
                             f"Cannot find ParaView at:\n{pv_path}\n\n"
                             "Update the path in the General tab.")


# ===================================================================
#  Main GUI Application
# ===================================================================

class SimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self._icon_photo = None
        self._splash_logo_photo = None
        self._splash = None
        self._active_config_path = _CONFIG_JSON
        self._queued_config_paths = []
        self._stop_requested = False
        self._font_family = choose_font_family()

        self.withdraw()
        self._set_app_icon()
        self._show_startup_logo()

        self.title("BeamOnTarget - Simulation Manager")
        self.geometry("1000x760")
        self.minsize(860, 640)
        self.cfg = load_config(self._active_config_path)
        self._sync_project_folder(default_to_active=True)
        self._apply_theme()
        self._build_statusbar()
        self._build_ui()

        self.deiconify()
        self.lift()
        self.after(250, self._close_startup_logo)

    def _set_app_icon(self):
        """Set the runtime app icon for title bar/taskbar on Windows."""
        try:
            if sys.platform == "win32":
                # Ensure Windows taskbar groups and displays this app as BeamOnTarget.
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BeamOnTarget.App")

            if os.path.exists(_APP_ICON_ICO):
                if sys.platform == "win32":
                    # Prefer ICO for Windows taskbar/title bar behavior.
                    self.iconbitmap(_APP_ICON_ICO)
                self._icon_photo = ImageTk.PhotoImage(Image.open(_APP_ICON_ICO))
                self.iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _show_startup_logo(self):
        """Show a centered splash logo while the GUI is initializing."""
        if not os.path.exists(_SPLASH_LOGO):
            return
        try:
            img = Image.open(_SPLASH_LOGO)
            self._splash_logo_photo = ImageTk.PhotoImage(img)
            splash = tk.Toplevel(self)
            splash.overrideredirect(True)
            splash.attributes("-topmost", True)

            label = tk.Label(splash, image=self._splash_logo_photo, borderwidth=0, highlightthickness=0)
            label.pack()

            splash.update_idletasks()
            width = img.width
            height = img.height
            x = (self.winfo_screenwidth() - width) // 2
            y = (self.winfo_screenheight() - height) // 2
            splash.geometry(f"{width}x{height}+{x}+{y}")
            splash.update()
            self._splash = splash
        except Exception:
            self._splash = None

    def _close_startup_logo(self):
        if self._splash is not None and self._splash.winfo_exists():
            self._splash.destroy()
            self._splash = None

    # ------------------------------------------------------------------
    #  Modern theme & styling
    # ------------------------------------------------------------------
    def _apply_theme(self):
        style = ttk.Style(self)
        # Use clam as the base - it supports most colour overrides
        style.theme_use("clam")

        try:
            for name in ("TkDefaultFont", "TkMenuFont", "TkHeadingFont", "TkTextFont"):
                font = tkfont.nametofont(name)
                if font:
                    font.configure(family=self._font_family)
        except Exception:
            pass
        self.option_add("*Font", f"{{{self._font_family}}} 10")

        # --- Colour palette ---
        BG       = "#f0f2f5"   # main background
        CARD_BG  = "#ffffff"   # card / frame background
        ACCENT   = "#2563eb"   # blue accent (buttons, active tab)
        ACCENT2  = "#1d4ed8"   # darker accent (pressed)
        FG       = "#1e293b"   # primary text
        FG_DIM   = "#64748b"   # secondary text
        BORDER   = "#cbd5e1"   # subtle borders
        SUCCESS  = "#16a34a"
        DANGER   = "#dc2626"

        self.configure(bg=BG)

        # Notebook (tabs)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                         background=BG, foreground=FG, padding=[14, 6],
                         font=("DejaVu Sans", 10))
        style.map("TNotebook.Tab",
                   background=[("selected", CARD_BG)],
                   foreground=[("selected", ACCENT)],
                   expand=[("selected", [0, 0, 0, 2])])

        # Frames
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat")

        # Labels
        style.configure("TLabel", background=BG, foreground=FG,
                         font=("DejaVu Sans", 10))
        style.configure("Card.TLabel", background=CARD_BG, foreground=FG,
                         font=("DejaVu Sans", 10))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT,
                         font=("DejaVu Sans", 12, "bold"))
        style.configure("CardHeader.TLabel", background=CARD_BG,
                         foreground=ACCENT, font=("DejaVu Sans", 11, "bold"))
        style.configure("Dim.TLabel", background=BG, foreground=FG_DIM,
                         font=("DejaVu Sans", 9))
        style.configure("Status.TLabel", background=BORDER, foreground=FG,
                         font=("DejaVu Sans", 9), padding=[8, 4])

        # Buttons
        style.configure("TButton", font=("DejaVu Sans", 10), padding=[10, 5],
                         background=ACCENT, foreground="white", borderwidth=0)
        style.map("TButton",
                   background=[("active", ACCENT2), ("pressed", ACCENT2)],
                   foreground=[("disabled", FG_DIM)])

        style.configure("Accent.TButton", font=("DejaVu Sans", 10, "bold"),
                         padding=[14, 6], background=ACCENT, foreground="white")
        style.map("Accent.TButton",
                   background=[("active", ACCENT2)])

        style.configure("Danger.TButton", font=("DejaVu Sans", 10),
                         padding=[10, 5], background=DANGER, foreground="white")
        style.map("Danger.TButton",
                   background=[("active", "#b91c1c")])

        style.configure("Success.TButton", font=("DejaVu Sans", 10),
                         padding=[10, 5], background=SUCCESS, foreground="white")
        style.map("Success.TButton",
                   background=[("active", "#15803d")])

        style.configure("Secondary.TButton", font=("DejaVu Sans", 10),
                         padding=[10, 5], background="#e2e8f0", foreground=FG,
                         borderwidth=0)
        style.map("Secondary.TButton",
                   background=[("active", "#cbd5e1")])

        # Entries
        style.configure("TEntry", fieldbackground="white", foreground=FG,
                         borderwidth=1, padding=[6, 4],
                         font=("DejaVu Sans", 10))

        # Spinbox
        style.configure("TSpinbox", fieldbackground="white", foreground=FG,
                         padding=[6, 4], font=("DejaVu Sans", 10))

        # Checkbutton
        style.configure("TCheckbutton", background=BG, foreground=FG,
                         font=("DejaVu Sans", 10))
        style.configure("Card.TCheckbutton", background=CARD_BG,
                         foreground=FG, font=("DejaVu Sans", 10))
        style.configure("Card.TRadiobutton", background=CARD_BG,
                         foreground=FG, font=("DejaVu Sans", 10))
        style.map("Card.TRadiobutton", background=[("active", CARD_BG)])

        # Treeview
        style.configure("Treeview", background="white", foreground=FG,
                         fieldbackground="white", rowheight=26,
                         font=("DejaVu Sans", 10), borderwidth=0)
        style.configure("Treeview.Heading", background=BG, foreground=FG,
                         font=("DejaVu Sans", 10, "bold"), padding=[4, 4])
        style.map("Treeview",
                   background=[("selected", "#dbeafe")],
                   foreground=[("selected", ACCENT)])

        # Separator
        style.configure("TSeparator", background=BORDER)

        # Scrollbar
        style.configure("Vertical.TScrollbar", background=BG,
                         troughcolor=CARD_BG, borderwidth=0)

        # Store colours for use in widgets that don't support style
        self._colours = {
            "bg": BG, "card": CARD_BG, "accent": ACCENT, "fg": FG,
            "dim": FG_DIM, "border": BORDER, "success": SUCCESS,
            "danger": DANGER,
        }

    # ------------------------------------------------------------------
    #  Helper: card-like frame with optional title
    # ------------------------------------------------------------------
    def _make_card(self, parent, title=None, padx=12, pady=(0, 10)):
        """Delegate to the shared module-level ``make_card``."""
        return make_card(parent, title=title, padx=padx, pady=pady)

    # ------------------------------------------------------------------
    #  Status bar
    # ------------------------------------------------------------------
    def _build_statusbar(self):
        bottom = ttk.Frame(self, style="TFrame")
        bottom.pack(side="bottom", fill="x")

        cfg_row = ttk.Frame(bottom, style="TFrame")
        cfg_row.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(cfg_row, text="Configuration file:", style="TLabel").pack(side="left")
        self.var_config_file = tk.StringVar(value=self._active_config_path)
        ttk.Entry(cfg_row, textvariable=self.var_config_file, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(cfg_row, text="Browse...", style="Secondary.TButton",
                   command=self._browse_active_config).pack(side="left")

        self._statusbar = ttk.Label(bottom, text="  Ready", style="Status.TLabel")
        self._statusbar.pack(fill="x", pady=(4, 0))

    def _set_status(self, text):
        self._statusbar.config(text=f"  {text}")

    def _set_active_config_path(self, path):
        self._active_config_path = os.path.abspath(path)
        self.var_config_file.set(self._active_config_path)

    def _sync_project_folder(self, default_to_active=False):
        """Update global project-folder resolver from current cfg/active config."""
        project_folder = (self.cfg.get("PROJECT_FOLDER") if isinstance(self.cfg, dict) else None) or ""
        if hasattr(self, "var_project_folder"):
            widget_value = self.var_project_folder.get().strip()
            if widget_value:
                project_folder = widget_value
        project_folder = project_folder.strip() if isinstance(project_folder, str) else ""
        if not project_folder and default_to_active:
            project_folder = os.path.dirname(self._active_config_path)
            self.cfg["PROJECT_FOLDER"] = project_folder
        elif project_folder and not os.path.isabs(project_folder):
            # Relative PROJECT_FOLDER is interpreted from the active config location.
            project_folder = os.path.abspath(os.path.join(os.path.dirname(self._active_config_path), project_folder))
        _set_project_folder(project_folder or os.path.dirname(self._active_config_path))

    def _browse_active_config(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Select configuration file",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if path:
            self._load_config_from_path(path)

    # ------------------------------------------------------------------
    #  Build the tabbed interface
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menubar()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._build_general_tab(notebook)
        self._build_geometry_tab(notebook)
        self._build_particles_tab(notebook)
        self._build_fields_tab(notebook)
        self._build_reactions_tab(notebook)
        self._build_output_tab(notebook)
        self._build_run_tab(notebook)
        self._build_results_tab(notebook)
        self._build_merge_tab(notebook)
        self._build_transform_tab(notebook)

    # ------------------------------------------------------------------
    #  Menu bar
    # ------------------------------------------------------------------
    def _build_menubar(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Config", accelerator="Ctrl+S",
                              command=self._save)
        file_menu.add_command(label="Save Config As…", accelerator="Ctrl+Shift+S",
                              command=self._save_as)
        file_menu.add_command(label="Load Config…", accelerator="Ctrl+O",
                              command=self._load_config_file)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="View All (Open3D)…",
                              command=self._view_all_o3d)
        view_menu.add_command(label="View Geometry (Open3D)…",
                              command=self._view_geometry_o3d)
        view_menu.add_command(label="View Results (Open3D)…",
                              command=self._view_results_o3d)
        view_menu.add_command(label="View Sources (Open3D)…",
                              command=self._view_sources_o3d)
        view_menu.add_separator()
        view_menu.add_command(label="View Geometry (ParaView)…",
                              command=self._view_geometry)
        view_menu.add_command(label="View Results (ParaView)…",
                              command=self._view_results)
        menubar.add_cascade(label="View", menu=view_menu)

        self.bind_all("<Control-s>", lambda e: self._save())
        self.bind_all("<Control-Shift-S>", lambda e: self._save_as())
        self.bind_all("<Control-o>", lambda e: self._load_config_file())

    # ------------------------------------------------------------------
    #  GENERAL tab
    # ------------------------------------------------------------------
    def _build_general_tab(self, nb):
        outer = ttk.Frame(nb)
        nb.add(outer, text="  General  ")

        # --- Tracking Method card ---
        method_card = self._make_card(outer, "Tracking Method", pady=(12, 10))

        row = 0
        ttk.Label(method_card, text="Method:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        raw_mode = self.cfg.get("TRACKING_MODE", "ray")
        initial_mode = "EM Tracing" if raw_mode == "em_track_then_bvh" else "Ray Tracing"
        self.var_tracking_mode = tk.StringVar(value=initial_mode)
        mode_frame = ttk.Frame(method_card, style="Card.TFrame")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.var_tracking_mode,
                                   values=["Ray Tracing", "EM Tracing"],
                                   state="readonly", width=20)
        mode_combo.pack(side="left")

        rm = self.cfg.get("REACTION_MODEL", {})
        reactions_on = rm.get("type", "none") not in ("none", "off", "null")
        self.var_reactions_enabled = tk.BooleanVar(value=reactions_on and raw_mode == "em_track_then_bvh")
        self._reactions_check = ttk.Checkbutton(
            mode_frame, text="Enable reactions",
            variable=self.var_reactions_enabled, style="Card.TCheckbutton")
        self._reactions_check.pack(side="left", padx=(16, 0))

        mode_frame.grid(row=row, column=1, sticky="w", padx=(8, 0))

        def _on_mode_change(*_):
            is_em = self.var_tracking_mode.get() == "EM Tracing"
            state = "normal" if is_em else "disabled"
            self._reactions_check.configure(state=state)
            if not is_em:
                self.var_reactions_enabled.set(False)
            for child in self._em_card.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass

        self.var_tracking_mode.trace_add("write", _on_mode_change)

        row += 1
        ttk.Label(method_card, text="Bounding box min (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        bbox_min = self.cfg.get("EM_BOUNDING_BOX_MIN_CORNER_M") or [0.0, -0.5, -1.3]
        self.var_bbox_min = tk.StringVar(value=f"{bbox_min[0]}, {bbox_min[1]}, {bbox_min[2]}")
        ttk.Entry(method_card, textvariable=self.var_bbox_min, width=24).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(method_card, text="Bounding box max (m):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=4)
        bbox_max = self.cfg.get("EM_BOUNDING_BOX_MAX_CORNER_M") or [13.0, 0.5, 0.8]
        self.var_bbox_max = tk.StringVar(value=f"{bbox_max[0]}, {bbox_max[1]}, {bbox_max[2]}")
        ttk.Entry(method_card, textvariable=self.var_bbox_max, width=24).grid(
            row=row, column=1, sticky="w", padx=(8, 0))

        method_card.columnconfigure(1, weight=1)

        # --- EM Tracking card (near mode selector) ---
        self._em_card = self._make_card(outer, "EM Tracking", pady=(12, 10))
        em_card = self._em_card

        r = 0
        ttk.Label(em_card, text="Step length (m):", style="Card.TLabel").grid(
            row=r, column=0, sticky="w", pady=4)
        self.var_em_step = tk.DoubleVar(value=self.cfg.get("EM_STEP_LENGTH_M", 0.02))
        ttk.Entry(em_card, textvariable=self.var_em_step, width=12).grid(
            row=r, column=1, sticky="w", padx=(8, 0))

        ttk.Label(em_card, text="Max steps:", style="Card.TLabel").grid(
            row=r, column=2, sticky="w", padx=(24, 0), pady=4)
        self.var_em_max_steps = tk.IntVar(value=self.cfg.get("EM_MAX_STEPS", 500))
        ttk.Entry(em_card, textvariable=self.var_em_max_steps, width=10).grid(
            row=r, column=3, sticky="w", padx=(8, 0))

        r += 1
        ttk.Label(em_card, text="Min energy (eV):", style="Card.TLabel").grid(
            row=r, column=0, sticky="w", pady=4)
        val = self.cfg.get("EM_MIN_ENERGY_EV")
        self.var_em_min_energy = tk.StringVar(value=str(val) if val is not None else "")
        ttk.Entry(em_card, textvariable=self.var_em_min_energy, width=12).grid(
            row=r, column=1, sticky="w", padx=(8, 0))

        ttk.Label(em_card, text="BVH checkpoint (m):", style="Card.TLabel").grid(
            row=r, column=2, sticky="w", padx=(24, 0), pady=4)
        self.var_em_checkpoint = tk.DoubleVar(value=self.cfg.get("EM_BVH_CHECKPOINT_DISTANCE_M", 1.0))
        ttk.Entry(em_card, textvariable=self.var_em_checkpoint, width=12).grid(
            row=r, column=3, sticky="w", padx=(8, 0))

        em_card.columnconfigure(1, weight=1)
        em_card.columnconfigure(3, weight=1)

        _on_mode_change()  # apply initial state

        # --- Project card ---
        card = self._make_card(outer, "Project Settings", pady=(12, 10))
        card.master.pack_configure(before=method_card.master)

        row = 0
        ttk.Label(card, text="CPU cores (-1 = all):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cpu = tk.IntVar(value=self.cfg.get("NUM_CPU_CORES", 1))
        ttk.Spinbox(card, from_=-1, to=256, textvariable=self.var_cpu,
                     width=8).grid(row=row, column=1, sticky="w", padx=(8, 0))

        row += 1
        ttk.Label(card, text="Project folder (if blank resolves to the .json file folder):", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_project_folder = tk.StringVar(value=self.cfg.get("PROJECT_FOLDER", os.path.dirname(self._active_config_path)))
        ttk.Entry(card, textvariable=self.var_project_folder, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 4))

        def _on_project_folder_change(*_):
            self.cfg["PROJECT_FOLDER"] = self.var_project_folder.get().strip()
            self._sync_project_folder(default_to_active=True)

        self.var_project_folder.trace_add("write", _on_project_folder_change)
        ttk.Button(card, text="Browse…", style="Secondary.TButton",
                    command=self._browse_project_folder).grid(row=row, column=2)

        row += 1
        ttk.Label(card, text="Geometry cache dir:", style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=5)
        self.var_cache = tk.StringVar(value=self.cfg.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        ttk.Entry(card, textvariable=self.var_cache, width=30).grid(
            row=row, column=1, sticky="we", padx=(8, 0))

        card.columnconfigure(1, weight=1)

        # --- ParaView card ---
        pv_card = self._make_card(outer, "ParaView Integration")

        ttk.Label(pv_card, text="ParaView path:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=5)
        self.var_pv_path = tk.StringVar(value=self.cfg.get("PARAVIEW_PATH", "paraview"))
        ttk.Entry(pv_card, textvariable=self.var_pv_path, width=50).grid(
            row=0, column=1, sticky="we", padx=(8, 4))
        ttk.Button(pv_card, text="Browse…", style="Secondary.TButton",
                    command=self._browse_pv).grid(row=0, column=2)

        # ParaView module (EasyBuild) - only shown on Linux
        self.var_pv_module = tk.StringVar(value=self.cfg.get("PARAVIEW_MODULE", "ParaView"))
        if sys.platform != "win32":
            ttk.Label(pv_card, text="ParaView module (ml):", style="Card.TLabel").grid(
                row=1, column=0, sticky="w", pady=5)
            ttk.Entry(pv_card, textvariable=self.var_pv_module, width=30).grid(
                row=1, column=1, sticky="w", padx=(8, 0))

        pv_card.columnconfigure(1, weight=1)

    def _browse_pv(self):
        if sys.platform == "win32":
            ftypes = [("Executable files", "*.exe"), ("All files", "*")]
        else:
            ftypes = [("All files", "*")]
        p = filedialog.askopenfilename(title="Select ParaView executable",
                                       filetypes=ftypes)
        if p:
            self.var_pv_path.set(p)

    def _browse_project_folder(self):
        initial = _get_project_folder()
        p = filedialog.askdirectory(
            title="Select project folder",
            initialdir=initial)
        if p:
            self.var_project_folder.set(_to_relative_path(p, os.path.dirname(self._active_config_path)))
            self.cfg["PROJECT_FOLDER"] = self.var_project_folder.get().strip()
            self._sync_project_folder(default_to_active=True)

    # ------------------------------------------------------------------
    #  GEOMETRY tab (delegated to gui_geometry.py)
    # ------------------------------------------------------------------
    def _build_geometry_tab(self, nb):
        self._geometry_tab = GeometryTab(
            nb, self.cfg, self._colours,
            view_geometry_fn=self._view_geometry,
            view_geometry_o3d_fn=self._view_geometry_o3d,
            get_em_bbox=self._get_em_bbox,
            is_em_mode=lambda: self.var_tracking_mode.get() == "EM Tracing",
        )
        nb.add(self._geometry_tab, text="  Geometry  ")
        for var in (self.var_bbox_min, self.var_bbox_max, self.var_tracking_mode):
            var.trace_add("write",
                          lambda *_: self._geometry_tab.schedule_bbox_redraw())

    def _get_em_bbox(self):
        """Return the EM bounding box corners, or None if the entries are invalid."""
        try:
            return (parse_vec3(self.var_bbox_min.get()),
                    parse_vec3(self.var_bbox_max.get()))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    #  PARTICLES tab (delegated to gui_particles.py)
    # ------------------------------------------------------------------
    def _build_particles_tab(self, nb):
        self._particles_tab = ParticlesTab(
            nb, self.cfg, self._colours,
            view_sources_o3d_fn=self._view_sources_o3d,
        )
        nb.add(self._particles_tab, text="  Particles  ")

    @property
    def var_src_dir(self):
        return self._particles_tab.var_src_dir

    # ------------------------------------------------------------------
    #  FIELDS tab (delegated to gui_fields.py)
    # ------------------------------------------------------------------
    def _build_fields_tab(self, nb):
        self._fields_tab = FieldsTab(nb, self.cfg,
                                     get_collect_fn=self._collect)
        nb.add(self._fields_tab, text="  Fields  ")

    # ------------------------------------------------------------------
    #  REACTIONS tab (delegated to gui_reactions.py)
    # ------------------------------------------------------------------
    def _build_reactions_tab(self, nb):
        self._reactions_tab = ReactionsTab(
            nb, self.cfg, self._colours,
            get_bbox_min=lambda: self._parse_vec3(self.var_bbox_min.get()),
            get_bbox_max=lambda: self._parse_vec3(self.var_bbox_max.get()),
            get_em_step=lambda: self.var_em_step.get(),
            get_collect_fn=self._collect,
        )
        nb.add(self._reactions_tab, text="  Reactions  ")

    # ------------------------------------------------------------------
    #  OUTPUT tab (delegated to gui_output.py)
    # ------------------------------------------------------------------
    def _build_output_tab(self, nb):
        self._output_tab = OutputTab(
            nb, self.cfg, self._colours,
            open_extract_fn=self._open_extract_dialog,
        )
        nb.add(self._output_tab, text="  Output  ")

    @property
    def var_outdir(self):
        return self._output_tab.var_outdir

    def _view_results(self):
        outdir = self.var_outdir.get()
        outdir_abs = _resolve_path(outdir) if not os.path.isabs(outdir) else outdir
        if not os.path.isdir(outdir_abs):
            messagebox.showwarning("No results", f"Output directory not found:\n{outdir_abs}")
            return
        subdirs = sorted([d for d in os.listdir(outdir_abs)
                          if os.path.isdir(os.path.join(outdir_abs, d))])
        if not subdirs:
            messagebox.showinfo("No results", "No result subfolders found.")
            return
        pick = _PickDialog(self, "Select result set", subdirs)
        self.wait_window(pick)
        if pick.result is None:
            return
        chosen_dir = os.path.join(outdir_abs, pick.result)
        script = _pv_results_script(chosen_dir)
        if script is None:
            messagebox.showinfo("No VTP", f"No .vtp files in {chosen_dir}")
            return
        launch_paraview(script, self.var_pv_path.get(), self.var_pv_module.get())

    def _open_extract_dialog(self):
        outdir = self.var_outdir.get()
        outdir_abs = (_resolve_path(outdir)
                      if not os.path.isabs(outdir) else outdir)
        dlg = _ExtractDialog(self, outdir_abs)
        self.wait_window(dlg)

    # ------------------------------------------------------------------
    #  Viewer helpers (shared by menu bar + tab buttons)
    # ------------------------------------------------------------------
    def _view_geometry(self):
        script = _pv_geometry_script(self.cfg)
        pv = self.var_pv_path.get()
        pv_mod = self.var_pv_module.get()
        launch_paraview(script, pv, pv_mod)

    def _view_geometry_o3d(self):
        src_dir = self.var_src_dir.get()
        viewer.view_geometry(self, _get_project_folder(),
                             self.cfg.get("GEOMETRY_FOLDERS", {}),
                             source_dir=src_dir)

    def _view_results_o3d(self):
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_results(self, _get_project_folder(), outdir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}),
                            source_dir=src_dir)

    def _view_all_o3d(self):
        outdir = self.var_outdir.get()
        src_dir = self.var_src_dir.get()
        viewer.view_all(self, _get_project_folder(), outdir,
                        self.cfg.get("GEOMETRY_FOLDERS", {}),
                        source_dir=src_dir)

    def _view_sources_o3d(self):
        src_dir = self.var_src_dir.get()
        viewer.view_sources(self, _get_project_folder(), src_dir,
                            geometry_folders=self.cfg.get("GEOMETRY_FOLDERS", {}))

    # ------------------------------------------------------------------
    #  RESULTS tab
    # ------------------------------------------------------------------
    def _build_results_tab(self, nb):
        self._results_tab = ResultsTab(
            nb, self.cfg, self._colours,
            var_outdir=self.var_outdir,
            var_pv_path=self.var_pv_path,
            var_pv_module=self.var_pv_module,
            view_results_fn=self._view_results,
            open_extract_fn=self._open_extract_dialog,
            view_results_o3d_fn=self._view_results_o3d,
        )
        nb.add(self._results_tab, text="  Results  ")

    # ------------------------------------------------------------------
    #  MERGE tab (delegated to gui_merge.py)
    # ------------------------------------------------------------------
    def _build_merge_tab(self, nb):
        self._merge_tab = MergeTab(
            nb, self.cfg, self._colours,
            get_output_dir=lambda: self.var_outdir.get(),
            open_in_paraview_fn=self._open_files_in_paraview,
        )
        nb.add(self._merge_tab, text="  Merge  ")

    # ------------------------------------------------------------------
    #  TRANSFORM tab (delegated to gui_transform.py)
    # ------------------------------------------------------------------
    def _build_transform_tab(self, nb):
        self._transform_tab = TransformTab(
            nb, self.cfg, self._colours,
            get_output_dir=lambda: self.var_outdir.get(),
        )
        nb.add(self._transform_tab, text="  Transform  ")

    def _open_files_in_paraview(self, files):
        launch_paraview(_pv_files_script(files), self.var_pv_path.get(),
                        self.var_pv_module.get())


    # ------------------------------------------------------------------
    #  RUN tab (delegated to gui_run.py)
    # ------------------------------------------------------------------
    def _build_run_tab(self, nb):
        self._run_tab = RunTab(
            nb, self.cfg, self._colours,
            save_fn=self._save,
            save_as_fn=self._save_as,
            load_config_fn=self._load_config_file,
            set_status_fn=self._set_status,
            get_active_config_path=lambda: self._active_config_path,
            set_active_config_path=self._set_active_config_path,
            get_smoothing_params=lambda: {
                "method": self._output_tab.smoothing_method(),
                "radius": self._output_tab.var_sm_radius.get(),
                "max_cell_area": self._output_tab.var_sm_mca.get().strip(),
                "normal_threshold_deg": self._output_tab.var_sm_angle.get(),
                "target_rel_error": self._output_tab.var_sm_target_err.get(),
                "max_bandwidth_m": self._output_tab.var_sm_max_bw.get(),
            },
        )
        nb.add(self._run_tab, text="  Run  ")

    # ------------------------------------------------------------------
    #  Shared helpers
    # ------------------------------------------------------------------
    _parse_vec3 = staticmethod(parse_vec3)

    def _load_config_from_path(self, path):
        try:
            path = os.path.abspath(path)
            new_cfg = load_config(path)
            if not str(new_cfg.get("PROJECT_FOLDER", "")).strip():
                new_cfg["PROJECT_FOLDER"] = os.path.dirname(path)
            self.cfg = new_cfg
            self._set_active_config_path(path)
            self._sync_project_folder(default_to_active=True)
            self._refresh_all_from_cfg()
            self._log(f"Configuration loaded from {os.path.basename(path)}\n")
            self._set_status(f"Loaded config: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _sync_cfg_to_tabs(self):
        """Ensure every tab references the same cfg dict."""
        for tab in (self._geometry_tab, self._particles_tab,
                    self._output_tab, self._run_tab):
            tab.cfg = self.cfg

    def _log(self, text):
        """Delegate logging to the Run tab."""
        self._run_tab._log(text)

    def _collect(self):
        d = dict(self.cfg)  # start from current (preserves unknown keys)
        # Tracking method
        is_em = self.var_tracking_mode.get() == "EM Tracing"
        d["TRACKING_MODE"] = "em_track_then_bvh" if is_em else "ray"
        # EM settings
        d["EM_STEP_LENGTH_M"] = self.var_em_step.get()
        d["EM_MAX_STEPS"] = self.var_em_max_steps.get()
        v = self.var_em_min_energy.get().strip()
        d["EM_MIN_ENERGY_EV"] = float(v) if v else None
        d["EM_BVH_CHECKPOINT_DISTANCE_M"] = self.var_em_checkpoint.get()
        d["EM_BOUNDING_BOX_MIN_CORNER_M"] = self._parse_vec3(self.var_bbox_min.get())
        d["EM_BOUNDING_BOX_MAX_CORNER_M"] = self._parse_vec3(self.var_bbox_max.get())
        # External field - delegate to FieldsTab
        self._fields_tab.collect(d)
        # Reaction model - delegate to ReactionsTab
        self._reactions_tab.collect(d, is_em and self.var_reactions_enabled.get())
        project_folder = self.var_project_folder.get().strip() or os.path.dirname(self._active_config_path)
        d["PROJECT_FOLDER"] = project_folder
        d["NUM_CPU_CORES"] = self.var_cpu.get()
        d["GEOMETRY_CACHE_DIR"] = self.var_cache.get()
        d["PARAVIEW_PATH"] = self.var_pv_path.get()
        d["PARAVIEW_MODULE"] = self.var_pv_module.get()
        # Geometry / Particles / Output - delegated
        self._geometry_tab.collect(d)
        self._particles_tab.collect(d)
        self._output_tab.collect(d)
        self._merge_tab.collect(d)
        self._transform_tab.collect(d)
        d["ENABLE_VISUALIZATION"] = self._results_tab.var_ENABLE_VISUALIZATION.get()
        return d

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    def _save(self):
        try:
            self.cfg = self._collect()
            self._sync_project_folder(default_to_active=True)
            self._sync_cfg_to_tabs()
            save_config(self.cfg, self._active_config_path)
            self._log(f"Configuration saved to {os.path.basename(self._active_config_path)}\n")
            self._set_status("Configuration saved")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _save_as(self):
        """Save the current config to a user-chosen JSON file."""
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Save Config As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            self.cfg = self._collect()
            self._sync_cfg_to_tabs()
            path = os.path.abspath(path)
            if not str(self.cfg.get("PROJECT_FOLDER", "")).strip():
                self.cfg["PROJECT_FOLDER"] = os.path.dirname(path)
            save_config(self.cfg, path)
            self._set_active_config_path(path)
            self._sync_project_folder(default_to_active=True)
            self._log(f"Configuration saved to {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_config_file(self):
        """Load a config from a user-chosen JSON file and refresh the GUI."""
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._active_config_path),
            title="Load Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        self._load_config_from_path(path)

    def _refresh_all_from_cfg(self):
        """Push self.cfg values back into every GUI widget."""
        c = self.cfg
        # General - tracking method
        raw_mode = c.get("TRACKING_MODE", "ray")
        self.var_tracking_mode.set("EM Tracing" if raw_mode == "em_track_then_bvh" else "Ray Tracing")
        rm = c.get("REACTION_MODEL", {})
        reactions_on = rm.get("type", "none") not in ("none", "off", "null")
        self.var_reactions_enabled.set(reactions_on and raw_mode == "em_track_then_bvh")
        # General - engine
        self.var_project_folder.set(c.get("PROJECT_FOLDER", os.path.dirname(self._active_config_path)))
        self.var_cpu.set(c.get("NUM_CPU_CORES", 1))
        self.var_cache.set(c.get("GEOMETRY_CACHE_DIR", "geometry_cache"))
        self.var_pv_path.set(c.get("PARAVIEW_PATH", "paraview"))
        self.var_pv_module.set(c.get("PARAVIEW_MODULE", "ParaView"))
        self._sync_project_folder(default_to_active=True)
        # Geometry / Particles / Output - delegated
        self._geometry_tab.refresh(c)
        self._particles_tab.refresh(c)
        self._output_tab.refresh(c)
        self._merge_tab.refresh(c)
        self._transform_tab.refresh(c)
        self._results_tab.var_ENABLE_VISUALIZATION.set(c.get("ENABLE_VISUALIZATION", True))
        # Fields - EM settings
        self.var_em_step.set(c.get("EM_STEP_LENGTH_M", 0.02))
        self.var_em_max_steps.set(c.get("EM_MAX_STEPS", 500))
        v = c.get("EM_MIN_ENERGY_EV")
        self.var_em_min_energy.set(str(v) if v is not None else "")
        self.var_em_checkpoint.set(c.get("EM_BVH_CHECKPOINT_DISTANCE_M", 1.0))
        bbox_min = c.get("EM_BOUNDING_BOX_MIN_CORNER_M") or [0.0, -0.5, -1.3]
        self.var_bbox_min.set(f"{bbox_min[0]}, {bbox_min[1]}, {bbox_min[2]}")
        bbox_max = c.get("EM_BOUNDING_BOX_MAX_CORNER_M") or [13.0, 0.5, 0.8]
        self.var_bbox_max.set(f"{bbox_max[0]}, {bbox_max[1]}, {bbox_max[2]}")
        # Fields - per-component (delegated)
        self._fields_tab.refresh(c)
        # Reactions (delegated)
        self._reactions_tab.refresh(c)


# ===================================================================
#  VTP data extraction dialog
# ===================================================================
class _ExtractDialog(tk.Toplevel):
    """Dialog for extracting mesh cell data from a VTP file to CSV."""

    def __init__(self, parent, initial_dir):
        super().__init__(parent)
        self.title("Extract Results Data")
        self.geometry("560x440")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self._parent = parent

        pad = dict(padx=12, pady=4)

        # --- Input VTP file ---
        ttk.Label(self, text="Input VTP file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=0, column=0, sticky="w", **pad)

        self.var_input = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_input, width=48).grid(
            row=0, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=lambda: self._browse_vtp(initial_dir)).grid(
            row=0, column=2, padx=(0, 12), pady=4)

        # --- Output CSV file ---
        ttk.Label(self, text="Output CSV file:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").grid(
            row=1, column=0, sticky="w", **pad)

        self.var_output = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_output, width=48).grid(
            row=1, column=1, sticky="we", padx=(0, 4), pady=4)
        ttk.Button(self, text="Browse…",
                    command=self._browse_output).grid(
            row=1, column=2, padx=(0, 12), pady=4)

        # --- Properties to export ---
        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        prop_lbl = ttk.Label(self, text="Properties to export:",
                              font=("Segoe UI", 10, "bold"),
                              background="#f0f2f5")
        prop_lbl.grid(row=3, column=0, sticky="w", **pad)

        prop_frm = ttk.Frame(self)
        prop_frm.grid(row=4, column=0, columnspan=3, sticky="w", padx=16)

        self.var_geometry = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export geometry (X, Y, Z)",
                         variable=self.var_geometry).pack(anchor="w", pady=1)

        self.var_area = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export cell area",
                         variable=self.var_area).pack(anchor="w", pady=1)

        self.var_power = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power (Deposited_Power_W)",
                         variable=self.var_power).pack(anchor="w", pady=1)

        self.var_powerload = tk.BooleanVar(value=True)
        ttk.Checkbutton(prop_frm, text="Export power load (Power_Density_W_m2)",
                         variable=self.var_powerload).pack(anchor="w", pady=1)

        # --- Multiplier ---
        ttk.Separator(self, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="we", padx=12, pady=8)

        mult_frm = ttk.Frame(self)
        mult_frm.grid(row=6, column=0, columnspan=3, sticky="w", padx=12)

        ttk.Label(mult_frm, text="Multiplication factor:",
                  font=("Segoe UI", 10, "bold"),
                  background="#f0f2f5").pack(side="left")
        self.var_mult = tk.StringVar(value="1.0")
        ttk.Entry(mult_frm, textvariable=self.var_mult, width=10).pack(
            side="left", padx=(8, 0))
        ttk.Label(mult_frm, text="(applied to power & power load)",
                  background="#f0f2f5",
                  foreground="#64748b").pack(side="left", padx=(8, 0))

        # --- Ignore zeros ---
        self.var_ignore_zeros = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Ignore zero-valued rows",
                         variable=self.var_ignore_zeros).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=16, pady=(8, 4))

        # --- Status label ---
        self.var_status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.var_status,
                  background="#f0f2f5", foreground="#64748b",
                  font=("Segoe UI", 9)).grid(
            row=8, column=0, columnspan=3, sticky="w", padx=12, pady=(4, 0))

        # --- Buttons ---
        btn_frm = ttk.Frame(self)
        btn_frm.grid(row=9, column=0, columnspan=3, pady=12)

        ttk.Button(btn_frm, text="Save CSV",
                    command=self._do_extract).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="Cancel",
                    command=self.destroy).pack(side="left", padx=4)

        self.columnconfigure(1, weight=1)

    def _browse_vtp(self, initial_dir):
        p = filedialog.askopenfilename(
            parent=self,
            initialdir=initial_dir,
            title="Select VTP file",
            filetypes=[("VTP files", "*.vtp"), ("All files", "*")])
        if p:
            self.var_input.set(p)
            # Auto-fill output name
            if not self.var_output.get():
                base = os.path.splitext(p)[0]
                self.var_output.set(base + "_extracted.csv")

    def _browse_output(self):
        init_dir = os.path.dirname(self.var_input.get()) or _SCRIPT_DIR
        p = filedialog.asksaveasfilename(
            parent=self,
            initialdir=init_dir,
            title="Save CSV As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if p:
            self.var_output.set(p)

    def _do_extract(self):
        """Run the extraction in a background thread."""
        input_vtp = self.var_input.get().strip()
        output_csv = self.var_output.get().strip()
        if not input_vtp:
            messagebox.showwarning("Missing", "Please select an input VTP file.",
                                    parent=self)
            return
        if not output_csv:
            messagebox.showwarning("Missing", "Please specify an output file.",
                                    parent=self)
            return
        if not os.path.isfile(input_vtp):
            messagebox.showerror("Not found",
                                  f"Input file not found:\n{input_vtp}",
                                  parent=self)
            return

        try:
            mult = float(self.var_mult.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Multiplication factor must be a number.",
                                    parent=self)
            return

        export_geom = self.var_geometry.get()
        export_area = self.var_area.get()
        export_power = self.var_power.get()
        export_pload = self.var_powerload.get()
        ignore_zeros = self.var_ignore_zeros.get()

        if not any([export_geom, export_area, export_power, export_pload]):
            messagebox.showwarning("Nothing selected",
                                    "Select at least one property to export.",
                                    parent=self)
            return

        self.var_status.set("Extracting...")
        self.update_idletasks()

        def _worker():
            try:
                import pyvista as pv
                import pandas as pd
                import numpy as np

                dataset = pv.read(input_vtp)
                if isinstance(dataset, pv.MultiBlock):
                    if dataset.n_blocks == 0:
                        self.after(0, lambda: self.var_status.set(
                            "ERROR: MultiBlock dataset is empty."))
                        return
                    mesh = dataset.combine()
                elif isinstance(dataset, pv.PolyData):
                    mesh = dataset
                else:
                    self.after(0, lambda: self.var_status.set(
                        f"ERROR: Unsupported type {type(dataset)}"))
                    return

                mesh.clean(inplace=True)
                if mesh.n_cells == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: Mesh has 0 cells."))
                    return

                face_centers = mesh.cell_centers().points
                valid = np.isfinite(face_centers).all(axis=1)
                n_valid = int(np.sum(valid))
                if n_valid == 0:
                    self.after(0, lambda: self.var_status.set(
                        "ERROR: No cells with finite coordinates."))
                    return

                # Build DataFrame
                if export_geom:
                    df = pd.DataFrame(face_centers[valid],
                                       columns=["X", "Y", "Z"])
                else:
                    df = pd.DataFrame(index=range(n_valid))

                if export_area:
                    try:
                        mesh = mesh.compute_cell_sizes()
                        df["Area"] = mesh.cell_data["Area"][valid]
                    except Exception:
                        pass

                power_cols = []
                if export_power:
                    key = "Deposited_Power_W"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                if export_pload:
                    key = "Power_Density_W_m2"
                    if key in mesh.cell_data:
                        df[key] = mesh.cell_data[key][valid] * mult
                        power_cols.append(key)

                # Filter zeros
                if ignore_zeros and power_cols:
                    mask = pd.Series([True] * len(df))
                    for col in power_cols:
                        if col in df.columns:
                            mask &= (np.abs(df[col]) > 1e-12)
                    df = df[mask].reset_index(drop=True)

                n_rows = len(df)
                df.to_csv(output_csv, index=False, float_format="%.6e")

                self.after(0, lambda: self.var_status.set(
                    f"Saved {n_rows} rows to {os.path.basename(output_csv)}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Done",
                    f"Extracted {n_rows} rows to:\n{output_csv}",
                    parent=self))

            except Exception as e:
                self.after(0, lambda: self.var_status.set(f"ERROR: {e}"))

        threading.Thread(target=_worker, daemon=True).start()


# ===================================================================
#  Small pick-list dialog
# ===================================================================
class _PickDialog(tk.Toplevel):
    def __init__(self, parent, title, items):
        super().__init__(parent)
        self.title(title)
        self.geometry("380x340")
        self.transient(parent)
        self.grab_set()
        self.configure(bg="#f0f2f5")
        self.result = None

        ttk.Label(self, text="Select a result set:",
                   font=("Segoe UI", 11, "bold"),
                   background="#f0f2f5", foreground="#2563eb").pack(
            anchor="w", padx=12, pady=(12, 6))

        lb = tk.Listbox(self, height=12, font=("Segoe UI", 10),
                          bg="white", fg="#1e293b",
                          selectbackground="#2563eb",
                          selectforeground="white",
                          highlightthickness=0, bd=1, relief="solid")
        lb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        for item in items:
            lb.insert("end", item)
        lb.selection_set(0)

        def _ok():
            sel = lb.curselection()
            if sel:
                self.result = lb.get(sel[0])
            self.destroy()

        ttk.Button(self, text="Open in ParaView", command=_ok).pack(pady=(0, 12))
        lb.bind("<Double-1>", lambda e: _ok())


# ===================================================================
#  Entry point
# ===================================================================
def main():
    app = SimGUI()
    app.mainloop()


if __name__ == "__main__":
    main()




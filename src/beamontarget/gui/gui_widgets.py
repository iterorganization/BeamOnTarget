# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Shared GUI helpers, constants, and the ``make_card`` factory.

All tab modules import from here so that styling is consistent.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
import os, sys
from beamontarget.paths import get_project_root

# ---------------------------------------------------------------------------
#  Path helpers
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)
_SCRIPT_DIR = (os.path.dirname(sys.executable) if _IS_FROZEN
               else str(get_project_root()))
_PROJECT_FOLDER = _SCRIPT_DIR
_UI_FONT_FAMILIES = (
    "Segoe UI",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Helvetica",
)


def choose_font_family():
    """Return a plain UI font family that is available in the current Tk installation."""
    available = set(tkfont.families())
    for family in _UI_FONT_FAMILIES:
        if family in available:
            return family
    return tkfont.nametofont("TkDefaultFont").actual("family")


def set_project_folder(path):
    """Set the active project folder used for relative GUI paths."""
    global _PROJECT_FOLDER
    base = (path or "").strip()
    if not base:
        _PROJECT_FOLDER = _SCRIPT_DIR
        return
    if not os.path.isabs(base):
        base = os.path.abspath(os.path.join(_SCRIPT_DIR, base))
    _PROJECT_FOLDER = base


def get_project_folder():
    """Return the active project folder used for relative GUI paths."""
    return _PROJECT_FOLDER or _SCRIPT_DIR


def to_relative_path(path, base_dir=None):
    """Return *path* relative to *base_dir* when possible, else unchanged."""
    if not path:
        return path
    base = base_dir or get_project_folder()
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


def resolve_path(relative_path):
    """Resolve a simulation-file path relative to the active project folder."""
    return os.path.join(get_project_folder(), relative_path)


def parse_vec3(text):
    """Parse a comma-separated string into a list of 3 floats."""
    parts = [s.strip() for s in text.split(",")]
    return [float(parts[i]) if i < len(parts) else 0.0 for i in range(3)]


# ---------------------------------------------------------------------------
#  Card factory
# ---------------------------------------------------------------------------
def make_card(parent, title=None, padx=12, pady=(0, 10)):
    """Return a content Frame inside a white card with rounded-look padding."""
    wrapper = ttk.Frame(parent, style="Card.TFrame", padding=14)
    wrapper.pack(fill="x", padx=padx, pady=pady)
    if title:
        ttk.Label(wrapper, text=title, style="CardHeader.TLabel").pack(
            anchor="w", pady=(0, 8))
    content = ttk.Frame(wrapper, style="Card.TFrame")
    content.pack(fill="both", expand=True)
    return content


# ---------------------------------------------------------------------------
#  Common browsing helpers
# ---------------------------------------------------------------------------
def browse_directory(var, title="Select directory"):
    """Ask user for a directory and store it relative to the project folder."""
    from tkinter import filedialog
    initial = get_project_folder()
    d = filedialog.askdirectory(initialdir=initial, title=title)
    if d:
        var.set(to_relative_path(d, initial))

# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""Run tab - save/load config, execute simulation, and stream console logs."""
import datetime
import os
import sys
import subprocess
import shlex
import importlib
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading

from beamontarget.gui.gui_widgets import (
    make_card,
    _SCRIPT_DIR,
)

# ---------------------------------------------------------------------------
#  Module-level constants (mirrors sim_gui.py)
# ---------------------------------------------------------------------------
_IS_FROZEN = getattr(sys, 'frozen', False)
_PYTHON = sys.executable
_RUN_SIMULATION_MODULE = "beamontarget.workflows.run_simulation"
_RUN_SMOOTHING_MODULE = "beamontarget.io.smooth_results"


def _run_module_frozen(module_name, argv=None, log_fn=None):
    """Run a bundled module main(argv) directly when frozen."""
    argv = argv or []

    class _LiveStream:
        def write(self, text):
            if text and log_fn:
                log_fn(text)
        def flush(self):
            pass

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_cwd = os.getcwd()

    sys.stdout = _LiveStream()
    sys.stderr = _LiveStream()

    try:
        os.chdir(_SCRIPT_DIR)
        module = importlib.import_module(module_name)
        module.main(argv)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.chdir(old_cwd)


class RunTab(ttk.Frame):
    """Save/load config, run simulation queue, console output."""

    def __init__(self, parent, cfg, colours, *,
                 save_fn, save_as_fn, load_config_fn,
                 set_status_fn,
                 get_active_config_path,
                 set_active_config_path,
                 get_smoothing_params):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._save_ext = save_fn
        self._save_as_ext = save_as_fn
        self._load_config_ext = load_config_fn
        self._set_status = set_status_fn
        self._get_active_config_path = get_active_config_path
        self._set_active_config_path = set_active_config_path
        self._get_smoothing_params = get_smoothing_params
        self._queued_config_paths = []
        self._sim_process = None
        self._stop_requested = False
        self._build()

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self):
        # --- Action buttons ---
        btn_card = make_card(self, pady=(12, 6))
        btn_row = ttk.Frame(btn_card, style="Card.TFrame")
        btn_row.pack(fill="x")

        ttk.Button(btn_row, text="Save Config", style="Secondary.TButton",
                    command=self._save_ext).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Save As...", style="Secondary.TButton",
                    command=self._save_as_ext).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Load Config...", style="Secondary.TButton",
                    command=self._load_config_ext).pack(side="left", padx=4)

        ttk.Separator(btn_row, orient="vertical").pack(side="left", fill="y",
                                                         padx=12, pady=2)

        ttk.Button(btn_row, text="Run Simulation", style="Accent.TButton",
                    command=self._run_sim).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Run Smoothing Only", style="Secondary.TButton",
                command=self._run_smoothing_only).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Stop", style="Danger.TButton",
                    command=self._stop_sim).pack(side="left", padx=4)

        # SLURM submission checkbox (Linux/HPC only — hidden on Windows)
        self.var_slurm = tk.BooleanVar(value=False)
        if sys.platform != "win32":
            ttk.Checkbutton(btn_card, text="Run on SLURM server (srun --exclusive)",
                             variable=self.var_slurm,
                             style="Card.TCheckbutton").pack(anchor="w", pady=(8, 0))

        queue_card = make_card(self, "Configuration Queue", pady=(6, 6))
        self.run_cfg_listbox = tk.Listbox(
            queue_card, height=6, font=("Segoe UI", 10),
            bg="white", fg=self._colours["fg"],
            selectbackground=self._colours["accent"],
            selectforeground="white", highlightthickness=0, bd=1, relief="solid")
        self.run_cfg_listbox.pack(fill="both", expand=False, pady=(0, 6))

        queue_btns = ttk.Frame(queue_card, style="Card.TFrame")
        queue_btns.pack(fill="x")
        ttk.Button(queue_btns, text="Add...", style="Secondary.TButton",
                    command=self._add_run_config).pack(side="left")
        ttk.Button(queue_btns, text="Delete", style="Secondary.TButton",
                    command=self._remove_run_config).pack(side="left", padx=(4, 0))
        ttk.Button(queue_btns, text="Use Active Config", style="Secondary.TButton",
                    command=self._add_active_config_to_queue).pack(side="left", padx=(8, 0))

        # --- Log output ---
        log_card = make_card(self, "Console Output", pady=(6, 10))

        self.log_text = scrolledtext.ScrolledText(
            log_card, height=24, state="disabled",
            font=("Consolas", 10), wrap="word",
            bg="#1e293b", fg="#e2e8f0", insertbackground="#e2e8f0",
            selectbackground=self._colours["accent"],
            highlightthickness=0, bd=0, padx=10, pady=8)
        self.log_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    #  Config queue management
    # ------------------------------------------------------------------
    def _add_run_config(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self._get_active_config_path()),
            title="Add configuration file",
            filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if not path:
            return
        path = os.path.abspath(path)
        if path not in self._queued_config_paths:
            self._queued_config_paths.append(path)
            self._refresh_run_config_listbox()

    def _add_active_config_to_queue(self):
        path = os.path.abspath(self._get_active_config_path())
        if path not in self._queued_config_paths:
            self._queued_config_paths.append(path)
            self._refresh_run_config_listbox()

    def _remove_run_config(self):
        sel = list(self.run_cfg_listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            del self._queued_config_paths[idx]
        self._refresh_run_config_listbox()

    def _refresh_run_config_listbox(self):
        self.run_cfg_listbox.delete(0, "end")
        for p in self._queued_config_paths:
            self.run_cfg_listbox.insert("end", p)

    def _get_execution_config_paths(self):
        if self._queued_config_paths:
            return list(self._queued_config_paths)
        return [os.path.abspath(self._get_active_config_path())]

    # ------------------------------------------------------------------
    #  Simulation execution
    # ------------------------------------------------------------------
    def _run_sim(self):
        self._run_jobs(mode="simulation")

    def _run_smoothing_only(self):
        self._run_jobs(mode="smoothing")

    def _run_jobs(self, mode="simulation"):
        if self._sim_process and self._sim_process.poll() is None:
            messagebox.showinfo("Running", "A simulation is already running.")
            return

        # Save first
        self._save_ext()
        self._stop_requested = False
        cfg_paths = self._get_execution_config_paths()

        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        mode_label = "simulation" if mode == "simulation" else "smoothing"
        self._log(f"Starting {mode_label} queue with {len(cfg_paths)} configuration file(s)…\n\n")
        self._set_status(f"{mode_label.capitalize()} queue running…")

        def _worker():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(_SCRIPT_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{mode_label}_{timestamp}.log")

            with open(log_path, "w", encoding="utf-8") as _lf:
                def _tee(text):
                    self._log(text)
                    _lf.write(text)
                    _lf.flush()

                _tee(f"Log file: {log_path}\n\n")
                successes = 0
                failures = 0
                try:
                    for idx, cfg_path in enumerate(cfg_paths, 1):
                        if self._stop_requested:
                            break
                        _tee(f"\n=== [{idx}/{len(cfg_paths)}] {mode_label.capitalize()} for {cfg_path} ===\n")
                        rc = self._run_single_job(mode, cfg_path, _tee)
                        if rc == 0:
                            successes += 1
                            _tee("Completed successfully.\n")
                        else:
                            failures += 1
                            _tee(f"FAILED with return code {rc}.\n")
                except Exception as e:
                    _tee(f"\nERROR: {e}\n")
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} error"))
                    return

                if self._stop_requested:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue stopped"))
                    _tee("\nQueue stopped by user.\n")
                elif failures == 0:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue completed successfully"))
                    _tee(f"\nQueue complete: {successes} succeeded, {failures} failed.\n")
                else:
                    self.after(0, lambda: self._set_status(f"{mode_label.capitalize()} queue completed with failures"))
                    _tee(f"\nQueue complete with FAILURES: {successes} succeeded, {failures} failed.\n")

        threading.Thread(target=_worker, daemon=True).start()

    def _run_single_job(self, mode, cfg_path, log_fn):
        cfg_path = os.path.abspath(cfg_path)
        sm = self._get_smoothing_params()

        if mode == "simulation":
            module_name = _RUN_SIMULATION_MODULE
            argv = ["-i", cfg_path]
        else:
            module_name = _RUN_SMOOTHING_MODULE
            argv = ["-i", cfg_path, "-m", sm["method"]]
            if sm["method"] == "physical":
                argv += ["--target-rel-error", str(sm["target_rel_error"]),
                         "--max-bandwidth", str(sm["max_bandwidth_m"])]
            else:
                argv += ["-r", str(sm["radius"]), "-n", str(sm["normal_threshold_deg"])]
                if sm["max_cell_area"]:
                    argv += ["-a", sm["max_cell_area"]]

        if _IS_FROZEN:
            _run_module_frozen(module_name, argv=argv, log_fn=log_fn)
            self._sim_process = None
            return 0

        if mode == "simulation" and sys.platform != "win32" and self.var_slurm.get():
            py_cmd = f"{shlex.quote(_PYTHON)} -m {shlex.quote(module_name)} " + " ".join(shlex.quote(a) for a in argv)
            shell_cmd = f'srun --exclusive --pty /bin/bash -c "{py_cmd}"'
            cmd = ["bash", "-l", "-c", shell_cmd]
        else:
            cmd = [_PYTHON, "-m", module_name] + argv

        self._sim_process = subprocess.Popen(
            cmd,
            cwd=_SCRIPT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1)
        for line in self._sim_process.stdout:
            log_fn(line)
            if self._stop_requested:
                break
        if self._stop_requested and self._sim_process.poll() is None:
            self._sim_process.terminate()
        self._sim_process.wait()
        rc = self._sim_process.returncode
        self._sim_process = None
        return rc

    def _stop_sim(self):
        self._stop_requested = True
        if self._sim_process and self._sim_process.poll() is None:
            self._sim_process.terminate()
            self._log("\nSimulation terminated by user.\n")
            self._set_status("Simulation stopped")

    def _log(self, text):
        """Thread-safe append to the log widget."""
        def _append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", text)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(0, _append)


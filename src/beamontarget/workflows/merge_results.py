# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# merge_results.py
"""
Cell-wise worst-case merge of result files across scenarios.

Scenarios (one result subfolder per beam configuration) are assigned to named
groups.  For every group, the result files of the same component are merged
cell by cell: each cell takes *all* its cell arrays from the scenario with the
highest ranking value there (``Power_Density_W_m2`` by default).  Taking every
array from the same winner keeps a cell's species split, error arrays and
total consistent with each other.  Different cells can come from different
scenarios, so the result is an envelope that matches no single scenario; its
summed power is not a physical power.

Layout
------
Input, as written by ``run_simulation`` and the batch smoother::

    <result folder>/<scenario>/results_<component>.vtp                  (raw)
    <result folder>/<scenario>/SMOOTHED/smoothed_results_<component>.vtp

Output::

    <output root>/<result folder>/<group>/merged_results_<component>.vtp
    <output root>/<result folder>/<group>/merged_results_<component>__smoothed.vtp
    <output root>/<result folder>/<group>/merge_sources.txt
    <output root>/merge_log_<timestamp>.csv

Each merged file also lists its source files in the ``merged_from`` field
array.

Use :func:`plan_merge` to see which files will be merged into which output,
then :func:`run_merge` to execute that plan.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

POWER_ARRAY = "Deposited_Power_W"
DENSITY_ARRAY = "Power_Density_W_m2"
RANK_ARRAYS = (DENSITY_ARRAY, POWER_ARRAY)

SOURCES = ("raw", "smoothed")
"""``raw``: ``results_*.vtp`` in the scenario folder;
``smoothed``: ``SMOOTHED/smoothed_results_*.vtp``."""

_SOURCE_GLOBS = {
    "raw": ("", "results_*.vtp"),
    "smoothed": ("SMOOTHED", "smoothed_results_*.vtp"),
}


@dataclass
class MergeSlot:
    """One merged output file and the source files that go into it."""

    group: str
    output_name: str
    component: str
    output_path: Path
    sources: list[tuple[str, Path]] = field(default_factory=list)
    """(scenario name, file) pairs, in merge order."""


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def _has_results(folder: str) -> bool:
    return bool(glob.glob(os.path.join(folder, "*.vtp"))
                or glob.glob(os.path.join(folder, "SMOOTHED", "*.vtp")))


def find_scenarios(result_folder: str) -> list[str]:
    """Return the scenario folders inside *result_folder*.

    A folder that holds result files itself is a single scenario; otherwise
    each subfolder (except ``SMOOTHED``) with result files is one.
    """
    if not os.path.isdir(result_folder):
        return []
    if _has_results(result_folder):
        return [os.path.normpath(result_folder)]
    found = []
    for entry in sorted(os.listdir(result_folder)):
        sub = os.path.join(result_folder, entry)
        if os.path.isdir(sub) and entry.upper() != "SMOOTHED" and _has_results(sub):
            found.append(os.path.normpath(sub))
    return found


def scenario_files(scenario_dir: str, source: str) -> list[Path]:
    """Result files of *source* (``raw`` or ``smoothed``) in a scenario folder."""
    sub, pattern = _SOURCE_GLOBS[source]
    return sorted(Path(scenario_dir, sub).glob(pattern))


def component_name(path: Path) -> str:
    """``smoothed_results_NEU.vtp`` / ``results_NEU.vtp`` -> ``NEU``."""
    stem = path.stem
    for prefix in ("smoothed_", "results_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def filter_terms(name_filter: str) -> list[str]:
    """Split a comma-separated name filter into lower-case terms."""
    return [t.strip().lower() for t in (name_filter or "").split(",") if t.strip()]


def matches_filter(path: Path, terms: list[str]) -> bool:
    """True if the file name contains any of *terms* (or there are none)."""
    return not terms or any(t in path.stem.lower() for t in terms)


def output_name_for(scenario_dir: str) -> str:
    """Name of the result folder a scenario belongs to (its parent folder)."""
    return Path(scenario_dir).parent.name or Path(scenario_dir).name


def safe_name(name: str) -> str:
    """Make a group label safe to use as a folder name."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "group"


def plan_merge(
        groups: dict[str, list[str]],
        source: str,
        name_filter: str,
        output_root: str,
) -> list[MergeSlot]:
    """Work out which files are merged into which output.

    Parameters
    ----------
    groups : dict
        ``{group label: [scenario folder, ...]}``.
    source : str
        ``"raw"`` or ``"smoothed"``.
    name_filter : str
        Comma-separated terms; a file is used if its name contains any term.
    output_root : str
        Root folder for the merged files.

    Returns
    -------
    list of MergeSlot
        Sorted by group, result folder and component.  Files of the same
        component are merged only within the same result folder.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    terms = filter_terms(name_filter)
    suffix = "__smoothed" if source == "smoothed" else ""
    slots: dict[tuple[str, str, str], MergeSlot] = {}
    for group, scenario_dirs in groups.items():
        for sd in scenario_dirs:
            out_name = output_name_for(sd)
            for fp in scenario_files(sd, source):
                if not matches_filter(fp, terms):
                    continue
                comp = component_name(fp)
                key = (group, out_name, comp)
                if key not in slots:
                    slots[key] = MergeSlot(
                        group=group, output_name=out_name, component=comp,
                        output_path=Path(output_root, out_name, safe_name(group),
                                         f"merged_results_{comp}{suffix}.vtp"))
                slots[key].sources.append((Path(sd).name, fp))
    return [slots[k] for k in sorted(slots)]


# ---------------------------------------------------------------------------
#  Merge
# ---------------------------------------------------------------------------

def merge_slot(slot: MergeSlot, rank_array: str = DENSITY_ARRAY,
               log=print, stop_event=None) -> dict | None:
    """Merge the source files of one slot and write the result.

    Returns a summary dict, or None if no source file could be used.
    """
    running: dict[str, np.ndarray] = {}
    geom = None
    used: list[tuple[str, Path]] = []
    for scenario, fp in slot.sources:
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            mesh = pv.read(str(fp))
        except Exception as exc:
            log(f"    [ERROR] {fp}: {exc}")
            continue
        if rank_array not in mesh.cell_data:
            log(f"    [SKIP] {fp.name}: '{rank_array}' not found.")
            continue
        if geom is None:
            geom = mesh
            running = {k: np.array(mesh.cell_data[k]) for k in mesh.cell_data.keys()}
        else:
            if mesh.n_cells != geom.n_cells or mesh.n_points != geom.n_points:
                log(f"    [SKIP] {scenario}/{fp.name}: {mesh.n_cells:,} cells vs "
                    f"{geom.n_cells:,} in the first file - cannot merge cell-wise.")
                continue
            win = np.asarray(mesh.cell_data[rank_array]) > running[rank_array]
            for name in list(running):
                new = np.asarray(mesh.cell_data[name]) if name in mesh.cell_data else None
                if new is None or new.shape != running[name].shape:
                    # A partial worst case would be misleading, so drop the array.
                    log(f"    [NOTE] {scenario}/{fp.name}: '{name}' missing or of a "
                        "different shape - dropped from the merged output.")
                    del running[name]
                    continue
                running[name][win] = new[win]
        used.append((scenario, fp))

    if geom is None:
        log("    No usable source file - skipped.")
        return None

    merged = geom.copy(deep=True)
    for name in list(merged.cell_data.keys()):
        if name not in running:
            del merged.cell_data[name]
    for name, values in running.items():
        merged.cell_data[name] = values
    if DENSITY_ARRAY in running:
        merged.cell_data.active_scalars_name = DENSITY_ARRAY
    merged.field_data["merged_from"] = [str(fp) for _, fp in used]
    merged.field_data["merge_rank_array"] = [rank_array]

    slot.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(slot.output_path), binary=True)

    density = running.get(DENSITY_ARRAY)
    power = running.get(POWER_ARRAY)
    return {
        "n_used": len(used),
        "used": used,
        "peak_density_W_m2": float(density.max()) if density is not None and density.size else None,
        "envelope_power_W": float(power.sum()) if power is not None else None,
    }


def run_merge(
        slots: list[MergeSlot],
        rank_array: str = DENSITY_ARRAY,
        output_root: str | None = None,
        log=print,
        stop_event=None,
) -> str | None:
    """Merge every slot of a plan and write the logs.

    Parameters
    ----------
    slots : list of MergeSlot
        From :func:`plan_merge`.
    rank_array : str
        Cell array that decides the worst case per cell.
    output_root : str, optional
        Where the run CSV is written; defaults to the common root of the slots.
    log : callable
        Receives one message string per call.
    stop_event : threading.Event, optional
        Stops the run between files when set.

    Returns
    -------
    str or None
        Path of the run CSV, or None if there was nothing to merge.
    """
    if rank_array not in RANK_ARRAYS:
        raise ValueError(f"rank_array must be one of {RANK_ARRAYS}")
    if not slots:
        log("Nothing to merge: no files match the assigned scenarios and the name filter.")
        return None
    if output_root is None:
        output_root = str(slots[0].output_path.parents[2])
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, f"merge_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    log(f"Merging {len(slots)} output file(s); worst case per cell ranked by {rank_array}.")
    log("Every cell array is taken from the winning scenario of that cell.")
    t_start = time.perf_counter()
    sources_txt: dict[Path, list[str]] = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "result_folder", "component", "n_files_merged",
                         "source_files", "peak_power_density_W_m2",
                         "envelope_power_W", "output_file"])
        for i, slot in enumerate(slots, 1):
            if stop_event is not None and stop_event.is_set():
                log("Stopped by user.")
                break
            t0 = time.perf_counter()
            log(f"[{i}/{len(slots)}] [{slot.group}] {slot.output_name} / {slot.component} "
                f"({len(slot.sources)} file(s)):")
            for scenario, fp in slot.sources:
                log(f"    + {scenario}  {fp.name}")
            res = merge_slot(slot, rank_array, log=log, stop_event=stop_event)
            if res is None:
                continue
            peak = res["peak_density_W_m2"]
            env = res["envelope_power_W"]
            log(f"    -> {slot.output_path}  ({res['n_used']} merged, "
                f"peak {peak:.4e} W/m2, {time.perf_counter() - t0:.1f} s)"
                if peak is not None else f"    -> {slot.output_path}")
            writer.writerow([
                slot.group, slot.output_name, slot.component, res["n_used"],
                "; ".join(str(fp) for _, fp in res["used"]),
                f"{peak:.6e}" if peak is not None else "",
                f"{env:.6e}" if env is not None else "",
                str(slot.output_path),
            ])
            lines = sources_txt.setdefault(slot.output_path.parent, [])
            lines.append(f"{slot.output_path.name}  (ranked by {rank_array})")
            lines.extend(f"    {fp}" for _, fp in res["used"])
            lines.append("")

    for folder, lines in sources_txt.items():
        with open(folder / "merge_sources.txt", "w", encoding="utf-8") as fh:
            fh.write(f"# Cell-wise worst-case merge, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("# Each merged file and the source files it was built from.\n\n")
            fh.write("\n".join(lines))

    log(f"Done in {time.perf_counter() - t_start:.1f} s. Log: {csv_path}")
    return csv_path

# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# transform_results.py
"""
Transform result files into another reference frame and export them as CSV
(per-cell data) and/or VTP (for visualisation).

Transform
---------
From a beam-line frame (HNB1-3, DNB) to the Tokamak frame: a rotation about
Z by ``angle_deg``, then a translation ``t``::

    p' = R(angle) p + t

The beam frame's +X axis (the beam direction) is rotated into the Tokamak
frame, then the frame origin is moved to the beam source.  Only this
direction is provided: BeamOnTarget always works in the beam-line frame.
(Its inverse would be ``p = R(-angle) (p' - t)``, i.e. the negated values
applied as translate-then-rotate; negating the values but keeping the order
is wrong by tens of metres.)

Units
-----
Everything is in metres, as in the simulation: coordinates are read and
written in m, the translation is in m, and the exported ``Area`` is in m^2.
A geometry that is not in metres must be scaled in the geometry setup.

Power factor
------------
``power_factor`` multiplies every array that is linear in power: deposited
power and power density (total and per species) and their error arrays
(sigma, bias, p95, standard errors); sums of squares are multiplied by the
factor squared.  Dimensionless or geometric arrays (relative errors, N_eff,
bandwidth, normals, area) are left unchanged.  The CSV and the VTP of the
same file therefore carry identical values.  Vector arrays called
``Normals`` are rotated with the mesh.

Ignore zero-valued rows
-----------------------
Drops cells whose selected power values (deposited power and/or power
density) are all zero in the *input* file, i.e. before the factor is
applied, so a small factor cannot make real values count as zero.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

POWER_ARRAY = "Deposited_Power_W"
DENSITY_ARRAY = "Power_Density_W_m2"
TOTAL_POWER_COLUMN = "total_deposited_power_W"
ZERO_TOL = 1e-12

# Beam frame -> Tokamak frame: rotation about Z [deg], then translation [m].
FRAME_TO_TOKAMAK = {
    "HNB1": (-79.543, (-0.57723, 32.385248, 1.453462)),
    "HNB2": (-59.543, (-11.618826, 30.234754, 1.453462)),
    "HNB3": (-39.543, (-21.25902, 24.437503, 1.453462)),
    "DNB": (-116.0, (11.410436, 26.617882, 0.920)),
}

SOURCES = {
    # source: (subfolder, glob)
    "raw": ("", "results_*.vtp"),
    "smoothed": ("SMOOTHED", "smoothed_results_*.vtp"),
    "merged": ("", "merged_results_*.vtp"),
}


@dataclass(frozen=True)
class FrameTransform:
    """Rotation about Z, then translation:  p' = R p + t  (metres)."""

    angle_deg: float = 0.0
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def rotation(self) -> np.ndarray:
        a = np.radians(self.angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def apply_m(self, xyz_m: np.ndarray) -> np.ndarray:
        """Transform (N, 3) coordinates in metres; returns metres."""
        p = np.asarray(xyz_m, dtype=np.float64)
        t = np.asarray(self.translation_m, dtype=np.float64)
        return p @ self.rotation.T + t

    def rotate_vectors(self, v: np.ndarray) -> np.ndarray:
        """Rotate (N, 3) direction vectors (no translation)."""
        return np.asarray(v, dtype=np.float64) @ self.rotation.T

    @property
    def is_identity(self) -> bool:
        return self.angle_deg % 360.0 == 0.0 and not any(self.translation_m)


def presets() -> dict[str, FrameTransform]:
    """Named transforms: each beam-line frame to Tokamak, plus none."""
    out = {"No transformation": FrameTransform()}
    for frame, (angle, t) in FRAME_TO_TOKAMAK.items():
        out[f"{frame} → Tokamak"] = FrameTransform(angle, t)
    return out


# ---------------------------------------------------------------------------
#  Power-factor rules
# ---------------------------------------------------------------------------

def power_scale_exponent(name: str) -> int:
    """Return k so that array *name* is multiplied by ``power_factor ** k``.

    1 for arrays linear in power (deposited power, power density and their
    species, sigma, bias, p95 and standard-error variants), 2 for sums of
    squares, 0 for everything else.
    """
    if name.startswith("Deposited_Power_SumSq"):
        return 2
    if name.startswith(("Deposited_Power_", "Power_Density_")):
        return 1
    return 0


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def source_files(folder: str, source: str) -> list[Path]:
    """Result files of *source* belonging to the scenario *folder*."""
    sub, pattern = SOURCES[source]
    return sorted(Path(folder, sub).glob(pattern))


def find_result_dirs(root: str, source: str, max_depth: int = 3) -> list[str]:
    """Folders below *root* (inclusive) that hold files of *source*.

    Works for simulation output (``<root>/<scenario>/``) as well as merged
    output (``MERGED/<result folder>/<group>/``).
    """
    found = []
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        return found
    base_depth = root.count(os.sep)
    for d, subdirs, _ in os.walk(root):
        subdirs[:] = sorted(s for s in subdirs if s.upper() != "SMOOTHED")
        if d.count(os.sep) - base_depth >= max_depth:
            subdirs[:] = []
        if source_files(d, source):
            found.append(os.path.normpath(d))
    return found


def filter_terms(name_filter: str) -> list[str]:
    return [t.strip().lower() for t in (name_filter or "").split(",") if t.strip()]


def matches_filter(path: Path, terms: list[str]) -> bool:
    return not terms or any(t in path.stem.lower() for t in terms)


def output_paths(input_file: Path, scenario_dir: str, output_root: str) -> dict[str, Path]:
    """``{"csv": ..., "vtp": ...}`` for one input file.

    Layout: ``<output root>/<parent>/<scenario>/<scenario>__<stem>.<ext>``, so
    the folders mirror the input and every file name is unique on its own.
    """
    scen = Path(scenario_dir)
    folder = Path(output_root, scen.parent.name, scen.name)
    stem = f"{scen.name}__{input_file.stem}"
    return {"csv": folder / f"{stem}.csv", "vtp": folder / f"{stem}.vtp"}


# ---------------------------------------------------------------------------
#  Transform one mesh
# ---------------------------------------------------------------------------

def _read_polydata(path: Path) -> pv.PolyData:
    data = pv.read(str(path))
    if isinstance(data, pv.MultiBlock):
        data = data.combine().extract_surface()
    if not isinstance(data, pv.PolyData):
        data = data.extract_surface()
    return data


def transform_mesh(
        mesh: pv.PolyData,
        transform: FrameTransform,
        power_factor: float = 1.0,
        ignore_zeros: bool = False,
        zero_check_power: bool = True,
        zero_check_density: bool = True,
) -> tuple[pv.PolyData, np.ndarray, dict]:
    """Return the transformed copy of *mesh*, the per-cell areas [m^2] and info.

    Areas are computed before the transform (they are invariant under it).  The power factor and the zero filter follow the module rules.
    The returned areas match the cells of the returned mesh.
    """
    out = mesh.copy(deep=True)
    n_in = out.n_cells

    areas_m2 = np.asarray(out.compute_cell_sizes(length=False, area=True, volume=False)
                          .cell_data["Area"], dtype=np.float64)

    # Density from power / area when the file has none (as the CSV export did).
    if DENSITY_ARRAY not in out.cell_data and POWER_ARRAY in out.cell_data:
        p = np.asarray(out.cell_data[POWER_ARRAY], dtype=np.float64)
        out.cell_data[DENSITY_ARRAY] = np.where(areas_m2 > 0, p / np.where(areas_m2 > 0, areas_m2, 1.0), 0.0)

    total_power = (float(np.asarray(out.cell_data[POWER_ARRAY], dtype=np.float64).sum())
                   * power_factor if POWER_ARRAY in out.cell_data else None)

    keep = np.ones(n_in, dtype=bool)
    if ignore_zeros:
        checks = [name for name, on in ((POWER_ARRAY, zero_check_power),
                                        (DENSITY_ARRAY, zero_check_density))
                  if on and name in out.cell_data]
        if checks:
            zero = np.ones(n_in, dtype=bool)
            for name in checks:
                zero &= np.abs(np.asarray(out.cell_data[name], dtype=np.float64)) <= ZERO_TOL
            keep = ~zero

    scaled = []
    if power_factor != 1.0:
        for name in list(out.cell_data.keys()):
            k = power_scale_exponent(name)
            if k:
                arr = out.cell_data[name]
                out.cell_data[name] = (np.asarray(arr, dtype=np.float64)
                                       * power_factor ** k).astype(arr.dtype)
                scaled.append(name)

    out.points = transform.apply_m(np.asarray(out.points, dtype=np.float64))
    for data in (out.cell_data, out.point_data):
        if "Normals" in data and np.asarray(data["Normals"]).ndim == 2:
            n = data["Normals"]
            data["Normals"] = transform.rotate_vectors(n).astype(np.asarray(n).dtype)

    if not keep.all():
        field = {k: out.field_data[k] for k in out.field_data.keys()}
        out = out.remove_cells(~keep, inplace=False)
        for k, v in field.items():
            if k not in out.field_data:
                out.field_data[k] = v
        areas_m2 = areas_m2[keep]

    if total_power is not None:
        out.field_data[TOTAL_POWER_COLUMN] = np.array([total_power])
    out.field_data["transform_power_factor"] = np.array([float(power_factor)])
    out.field_data["transform_angle_deg"] = np.array([float(transform.angle_deg)])
    out.field_data["transform_translation_m"] = np.array(transform.translation_m, dtype=np.float64)
    out.field_data["coordinate_unit"] = ["m"]

    info = {"n_cells_in": n_in, "n_cells_out": out.n_cells, "total_power_W": total_power,
            "scaled_arrays": scaled}
    return out, areas_m2, info


def write_csv(mesh: pv.PolyData, areas_m2: np.ndarray, path: Path, *,
              geometry: bool = True, area: bool = True, power: bool = True,
              density: bool = True, total_power: bool = True) -> int:
    """Write one row per cell: X, Y, Z (cell centre), Area [m^2], power columns.

    Returns the number of rows written.
    """
    cols, data = [], []
    if geometry:
        c = np.asarray(mesh.cell_centers().points, dtype=np.float64)
        cols += ["X", "Y", "Z"]
        data += [c[:, 0], c[:, 1], c[:, 2]]
    if area:
        cols.append("Area")
        data.append(areas_m2)
    for on, name in ((power, POWER_ARRAY), (density, DENSITY_ARRAY)):
        if on and name in mesh.cell_data:
            cols.append(name)
            data.append(np.asarray(mesh.cell_data[name], dtype=np.float64))
    if total_power and TOTAL_POWER_COLUMN in mesh.field_data:
        cols.append(TOTAL_POWER_COLUMN)
        data.append(np.full(mesh.n_cells, float(mesh.field_data[TOTAL_POWER_COLUMN][0])))
    if not cols:
        raise ValueError("No CSV columns selected.")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(data) if data else np.empty((mesh.n_cells, 0))
    np.savetxt(path, table, delimiter=",", header=",".join(cols), comments="", fmt="%.9g")
    return mesh.n_cells


def transform_file(
        input_file: Path,
        out_paths: dict[str, Path],
        transform: FrameTransform,
        *,
        power_factor: float = 1.0,
        ignore_zeros: bool = False,
        formats: tuple[str, ...] = ("csv",),
        csv_columns: dict | None = None,
) -> dict:
    """Transform one result file and write the requested formats.

    *csv_columns* holds the flags of :func:`write_csv`; deposited power and
    power density also decide which values the zero filter checks.
    """
    cols = {"geometry": True, "area": True, "power": True, "density": True,
            "total_power": True, **(csv_columns or {})}
    mesh = _read_polydata(input_file)
    out, areas, info = transform_mesh(
        mesh, transform, power_factor, ignore_zeros,
        zero_check_power=cols["power"] or not cols["density"],
        zero_check_density=cols["density"] or not cols["power"])
    written = {}
    if "csv" in formats:
        write_csv(out, areas, out_paths["csv"], **cols)
        written["csv"] = out_paths["csv"]
    if "vtp" in formats:
        out_paths["vtp"].parent.mkdir(parents=True, exist_ok=True)
        out.save(str(out_paths["vtp"]), binary=True)
        written["vtp"] = out_paths["vtp"]
    info["written"] = written
    return info


def run_transform(jobs, transform: FrameTransform, *, log=print, stop_event=None, **kwargs) -> int:
    """Transform a list of ``(input_file, out_paths)`` jobs.  Returns files done."""
    t0 = time.perf_counter()
    log(f"Transform: rotate {transform.angle_deg!r} deg about Z, then translate by "
        f"{tuple(transform.translation_m)} m.")
    log(f"Power factor {kwargs.get('power_factor', 1.0):g}, "
        f"ignore zeros {bool(kwargs.get('ignore_zeros', False))}, "
        f"formats {', '.join(kwargs.get('formats', ('csv',)))}.")
    done = 0
    for i, (fp, paths) in enumerate(jobs, 1):
        if stop_event is not None and stop_event.is_set():
            log("Stopped by user.")
            break
        try:
            info = transform_file(Path(fp), paths, transform, **kwargs)
        except Exception as exc:
            log(f"[{i}/{len(jobs)}] ERROR {fp}: {exc}")
            continue
        done += 1
        dropped = info["n_cells_in"] - info["n_cells_out"]
        log(f"[{i}/{len(jobs)}] {Path(fp).name}: {info['n_cells_out']:,} cells"
            + (f" ({dropped:,} zero cells dropped)" if dropped else "")
            + "".join(f"\n    -> {p}" for p in info["written"].values()))
    log(f"Done: {done}/{len(jobs)} file(s) in {time.perf_counter() - t0:.1f} s.")
    return done

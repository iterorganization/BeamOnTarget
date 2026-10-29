# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# smooth_results.py
"""
This file now acts as a "library" containing the core logic for smoothing.
The `apply_smoothing` function can be imported by other scripts.
It can also be run standalone to process the specific tasks defined below.

Smoothing method: Area-weighted moving average.
For each cell, we find all neighbours whose centroid lies within a given
radius (using a fast KDTree spatial lookup).  The smoothed power density
for that cell is then computed as:

    Power_Density_smoothed = Sum(Deposited_Power) / Sum(Area)

over the cell itself and all its neighbours within the radius.  This approach
is physically consistent: it preserves the total deposited power and computes
a meaningful averaged power density.
"""
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os
import time
import glob
import argparse

from beamontarget import config


def _build_noop_stats(mesh, deposited_power, areas, smoothed_density, radius, max_cell_area, elapsed_s, species_stats=None):
    species_stats = species_stats or {}
    return {
        "n_cells": mesh.n_cells,
        "n_smoothed": 0,
        "radius": radius,
        "max_cell_area": max_cell_area,
        "elapsed_s": round(elapsed_s, 2),
        "total_power_W": float(np.sum(deposited_power)),
        "peak_density_before": float(smoothed_density.max()) if smoothed_density.size else 0.0,
        "peak_density_after": float(smoothed_density.max()) if smoothed_density.size else 0.0,
        "min_area_m2": float(areas.min()) if areas.size else 0.0,
        "max_area_m2": float(areas.max()) if areas.size else 0.0,
        "species": species_stats,
    }


def _populate_species_density_noop(mesh, areas):
    species_stats = {}
    n_cells = mesh.n_cells
    for key in list(mesh.cell_data.keys()):
        if key.startswith('Deposited_Power_W_') and key != 'Deposited_Power_W':
            suffix = key[len('Deposited_Power_W_'):]
            sp_power = np.array(mesh.cell_data[key], dtype=np.float64)
            density_key = f'Power_Density_W_m2_{suffix}'
            sp_density = np.zeros(n_cells, dtype=np.float64)
            np.divide(sp_power, areas, out=sp_density, where=areas > 0)
            mesh.cell_data[density_key] = sp_density
            peak_density = float(sp_density.max()) if sp_density.size else 0.0
            species_stats[suffix] = {
                "total_power_W": float(np.sum(sp_power)),
                "peak_density_before": peak_density,
                "peak_density_after": peak_density,
            }
    return species_stats

# --- Core reusable function ---
def apply_smoothing(mesh, radius=None, n_iter=None, max_cell_area=None, normal_threshold_deg=7.0):
    """
    Applies area-weighted moving average smoothing to a PyVista mesh.
    Only neighbours with a normal within 'normal_threshold_deg' are considered.
    """
    if 'Deposited_Power_W' not in mesh.cell_data:
        return mesh, None
    if radius is None or radius <= 0:
        return mesh, None

    t0 = time.time()
    deposited_power = np.array(mesh.cell_data['Deposited_Power_W'], dtype=np.float64)
    n_cells = mesh.n_cells

    # --- Compute Geometry Data ---
    cell_geom = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.array(cell_geom.cell_data['Area'], dtype=np.float64)
    centroids = mesh.cell_centers().points

    # --- Compute Normals for internal geometry separation ---
    if 'Normals' not in mesh.cell_data:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
    cell_normals = np.array(mesh.cell_data['Normals'], dtype=np.float64)
    
    # Dot product threshold (cos(theta))
    cos_threshold = np.cos(np.deg2rad(normal_threshold_deg))

    # --- Setup Smoothing Target ---
    smoothed_density = np.divide(deposited_power, areas, out=np.zeros(n_cells), where=areas > 0)
    
    # FIX 1: Defined exactly as named in the stats dictionary
    peak_density_before = float(smoothed_density.max())

    if max_cell_area is not None:
        small_mask = (areas < max_cell_area) & (deposited_power > 0)
    else:
        small_mask = deposited_power > 0
    
    small_indices = np.where(small_mask)[0]
    
    # FIX 2: Defined exactly as used in the loops
    n_small = len(small_indices)

    if n_small == 0:
        mesh.cell_data['Power_Density_W_m2'] = smoothed_density
        species_stats = _populate_species_density_noop(mesh, areas)
        mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
        elapsed = time.time() - t0
        print("  - No cells found matching smoothing criteria; preserving original values.")
        return mesh, _build_noop_stats(
            mesh,
            deposited_power,
            areas,
            smoothed_density,
            radius,
            max_cell_area,
            elapsed,
            species_stats=species_stats,
        )

    # --- Spatial Query ---
    tree = cKDTree(centroids)
    small_tree = cKDTree(centroids[small_indices])
    
    # FIX 3: Defined consistently for both loops
    neighbours_list = small_tree.query_ball_tree(tree, r=radius)

    # --- Apply Filtered Average (Main Array) ---
    print(f"  - Smoothing {n_small} cells using Normal Threshold: {normal_threshold_deg} deg...")
    for j in range(n_small):
        target_idx = small_indices[j]
        # Get the list of spatial neighbor indices for this target cell
        current_neighbours = np.array(neighbours_list[j])
        
        if current_neighbours.size == 0: 
            continue
        
        # Filter neighbours by Normal Angle using dot product
        dots = np.dot(cell_normals[current_neighbours], cell_normals[target_idx])
        valid_mask = dots >= cos_threshold
        valid_neighs = current_neighbours[valid_mask]

        if valid_neighs.size > 0:
            smoothed_density[target_idx] = np.sum(deposited_power[valid_neighs]) / np.sum(areas[valid_neighs])

    mesh.cell_data['Power_Density_W_m2'] = smoothed_density

    # --- Smooth per-species arrays (using same logic) ---
    species_stats = {}
    for key in list(mesh.cell_data.keys()):
        if key.startswith('Deposited_Power_W_') and key != 'Deposited_Power_W':
            suffix = key[len('Deposited_Power_W_'):]
            sp_power = np.array(mesh.cell_data[key], dtype=np.float64)
            density_key = f'Power_Density_W_m2_{suffix}'
            
            sp_density = np.zeros(n_cells, dtype=np.float64)
            np.divide(sp_power, areas, out=sp_density, where=areas > 0)
            sp_peak_before = float(sp_density.max())

            for j in range(n_small):
                target_idx = small_indices[j]
                current_neighbours = np.array(neighbours_list[j])
                if current_neighbours.size == 0: 
                    continue
                
                dots = np.dot(cell_normals[current_neighbours], cell_normals[target_idx])
                valid_neighs = current_neighbours[dots >= cos_threshold]

                if valid_neighs.size > 0:
                    sp_density[target_idx] = np.sum(sp_power[valid_neighs]) / np.sum(areas[valid_neighs])
            
            mesh.cell_data[density_key] = sp_density
            species_stats[suffix] = {
                "total_power_W": float(np.sum(sp_power)),
                "peak_density_before": sp_peak_before,
                "peak_density_after": float(sp_density.max()),
            }

    mesh.cell_data.active_scalars_name = 'Power_Density_W_m2'
    elapsed = time.time() - t0
    
    stats = {
        "n_cells": n_cells,
        "n_smoothed": n_small,
        "radius": radius,
        "max_cell_area": max_cell_area,
        "elapsed_s": round(elapsed, 2),
        "total_power_W": float(np.sum(deposited_power)),
        "peak_density_before": peak_density_before,
        "peak_density_after": float(smoothed_density.max()),
        "min_area_m2": float(areas.min()),
        "max_area_m2": float(areas.max()),
        "species": species_stats,
    }
    return mesh, stats


def _resolve_config_relative_paths(config_path):
    """Resolve path-like config entries relative to PROJECT_FOLDER/config file."""
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_folder = str(getattr(config, "PROJECT_FOLDER", "") or "").strip()
    if not project_folder:
        project_folder = config_dir
    elif not os.path.isabs(project_folder):
        project_folder = os.path.abspath(os.path.join(config_dir, project_folder))
    config.PROJECT_FOLDER = project_folder

    def _abs_if_relative(path_value):
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(project_folder, path_value))

    config.DETAILED_OUTPUT_DIR = _abs_if_relative(config.DETAILED_OUTPUT_DIR)


def _find_subdirs_with_results(parent_dir):
    subdirs = []
    for entry in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.upper() == "SMOOTHED":
            continue
        has_results = (glob.glob(os.path.join(full_path, "*.vtp")) or
                       glob.glob(os.path.join(full_path, "*.vtm")))
        if has_results:
            subdirs.append(full_path)
    return subdirs


def main(argv=None):
    """Smooth existing simulation outputs using a selected JSON config."""
    from beamontarget.io import batch_smoother
    from beamontarget.io import generate_report

    parser = argparse.ArgumentParser(description="Apply smoothing to completed simulation outputs.")
    
    # --- 1. Add ALL arguments BEFORE calling parse_args ---
    parser.add_argument(
        '-i', '--input-config',
        default=None,
        help="Path to a JSON configuration file. Defaults to config.json.")
    parser.add_argument(
        '-r', '--radius',
        type=float,
        default=None,
        help="Override smoothing radius in metres.")
    parser.add_argument(
        '-a', '--max-cell-area',
        type=float,
        default=None,
        help="Override maximum smoothed cell area in m^2. Use 0 to smooth all cells.")
    parser.add_argument(
        '-n', '--normal-threshold',
        type=float,
        default=None,
        help="Angle threshold in degrees for surface normal filtering.")
    parser.add_argument(
        '-m', '--method',
        choices=batch_smoother.SMOOTHING_METHODS,
        default=None,
        help="Override the smoothing method: 'radius' or 'physical'.")
    parser.add_argument(
        '--target-rel-error',
        type=float,
        default=None,
        help="Physical method: override the ceiling on the 1-sigma statistical error.")
    parser.add_argument(
        '--max-bandwidth',
        type=float,
        default=None,
        help="Physical method: override the maximum smoothing length in metres.")

    args = parser.parse_args(argv)

    # --- 2. Load Config first so we have defaults ---
    if args.input_config:
        cfg_path = os.path.abspath(args.input_config)
        if not os.path.isfile(cfg_path):
            print(f"FATAL ERROR: Config file not found: '{cfg_path}'")
            return
        config.apply_config(path=cfg_path)
        _resolve_config_relative_paths(cfg_path)
        print(f"Using configuration file: {cfg_path}")

    # --- 3. Resolve Variables (CLI overrides Config) ---
    radius = args.radius if args.radius is not None else config.SMOOTHING_RADIUS
    
    if args.max_cell_area is None:
        max_cell_area = config.SMOOTHING_MAX_CELL_AREA
    else:
        max_cell_area = args.max_cell_area if args.max_cell_area > 0 else None

    normal_threshold = args.normal_threshold if args.normal_threshold is not None else \
                       config.SMOOTHING_NORMAL_THRESHOLD_DEG
    method = args.method or config.SMOOTHING_METHOD
    target_rel_error = args.target_rel_error if args.target_rel_error is not None else \
                       config.SMOOTHING_TARGET_REL_ERROR
    max_bandwidth_m = args.max_bandwidth if args.max_bandwidth is not None else \
                      config.SMOOTHING_MAX_BANDWIDTH_M

    output_root = config.DETAILED_OUTPUT_DIR
    if not os.path.isdir(output_root):
        print(f"FATAL ERROR: Output directory '{output_root}' not found.")
        return

    # --- 4. Find directories to process ---
    has_direct_results = (glob.glob(os.path.join(output_root, "*.vtp")) or
                          glob.glob(os.path.join(output_root, "*.vtm")))
    subdirs = _find_subdirs_with_results(output_root)

    if has_direct_results and not subdirs:
        dirs_to_process = [output_root]
    elif subdirs:
        dirs_to_process = subdirs
    else:
        print(f"No .vtp/.vtm files found in '{output_root}'.")
        return

    print(f"\n=== Smoothing Configuration ===")
    print(f"  Method           : {method}")
    if method == "physical":
        print(f"  Target rel. error: {target_rel_error}")
        print(f"  Max bandwidth    : {max_bandwidth_m} m")
    else:
        print(f"  Radius           : {radius} m")
        print(f"  Max cell area    : {max_cell_area} m2")
        print(f"  Normal Threshold : {normal_threshold} deg")
    print(f"  Directories      : {len(dirs_to_process)}")
    print(f"===============================")

    for idx, result_dir in enumerate(dirs_to_process, 1):
        print(f"\n[{idx}/{len(dirs_to_process)}] Processing: {result_dir}")
        try:
            batch_smoother.batch_process_directory(
                result_dir,
                radius=radius,
                max_cell_area=max_cell_area,
                normal_threshold_deg=normal_threshold,
                method=method,
                target_rel_error=target_rel_error,
                max_bandwidth_m=max_bandwidth_m)
        except Exception as e:
            print(f"  ERROR while smoothing '{result_dir}': {e}")
            continue

        smoothed_dir = os.path.join(result_dir, "SMOOTHED")
        if os.path.isdir(smoothed_dir):
            try:
                generate_report.generate_summary_csv(smoothed_dir)
            except Exception as e:
                print(f"  ERROR during smoothed report generation: {e}")

    print("\n=== Smoothing Complete ===")


if __name__ == "__main__":
    main()




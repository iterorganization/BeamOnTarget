# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# post_smooth.py
"""
Standalone post-processing script to apply smoothing to simulation results
that have already been produced.

Walks all subfolders inside an output directory (or a single specified
subfolder) and runs the same area-weighted smoothing used by
run_simulation.py.  Smoothed files are saved into a 'SMOOTHED' subfolder
within each processed directory.

Usage examples
--------------
  # Smooth ALL subfolders in the default OUTPUT directory, using config defaults:
  python post_smooth.py

  # Smooth a specific output directory:
  python post_smooth.py -i /path/to/OUTPUT/DNB_10mrad

  # Override smoothing parameters:
  python post_smooth.py -i OUTPUT -r 0.03 -a 2e-6

  # Smooth all cells (disable area filter):
  python post_smooth.py -i OUTPUT -a 0
"""

import os
import glob
import argparse

from beamontarget import config
from beamontarget.io import batch_smoother
from beamontarget.io import generate_report


def find_subdirs_with_results(parent_dir):
    """
    Return a sorted list of subdirectories inside *parent_dir* that contain
    at least one .vtp or .vtm file.
    """
    subdirs = []
    for entry in sorted(os.listdir(parent_dir)):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue
        # Skip the SMOOTHED output folders themselves
        if entry.upper() == "SMOOTHED":
            continue
        has_results = (glob.glob(os.path.join(full_path, "*.vtp")) or
                       glob.glob(os.path.join(full_path, "*.vtm")))
        if has_results:
            subdirs.append(full_path)
    return subdirs


def main():
    parser = argparse.ArgumentParser(
        description="Apply smoothing to already-completed simulation results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default=config.DETAILED_OUTPUT_DIR,
        help=(
            "Path to the output directory.  If this directory contains .vtp/.vtm\n"
            "files directly, it will be smoothed as-is.  If it contains subfolders\n"
            "with results (the typical batch layout), each subfolder will be\n"
            f"processed.  Defaults to '{config.DETAILED_OUTPUT_DIR}'."
        ),
    )
    parser.add_argument(
        "-r", "--radius",
        type=float,
        default=config.SMOOTHING_RADIUS,
        help=f"Smoothing radius in metres.  Defaults to {config.SMOOTHING_RADIUS}.",
    )
    parser.add_argument(
        "-a", "--max_cell_area",
        type=float,
        default=config.SMOOTHING_MAX_CELL_AREA if config.SMOOTHING_MAX_CELL_AREA else 0,
        help=(
            f"Only smooth cells with area below this threshold (m²).\n"
            f"Defaults to {config.SMOOTHING_MAX_CELL_AREA}.  Set to 0 to smooth all cells with power."
        ),
    )
    parser.add_argument(
        "-n", "--normal_threshold",
        type=float,
        default=config.SMOOTHING_NORMAL_THRESHOLD_DEG,
        help=(
            f"Normal angle threshold in degrees for neighbour filtering.\n"
            f"Defaults to {config.SMOOTHING_NORMAL_THRESHOLD_DEG}."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    radius = args.radius
    max_cell_area = args.max_cell_area if args.max_cell_area > 0 else None
    normal_threshold = args.normal_threshold

    if not os.path.isdir(input_dir):
        print(f"FATAL ERROR: Input directory '{input_dir}' not found.")
        return

    # Decide whether the directory itself contains results or has subfolders
    has_direct_results = (glob.glob(os.path.join(input_dir, "*.vtp")) or
                          glob.glob(os.path.join(input_dir, "*.vtm")))
    subdirs = find_subdirs_with_results(input_dir)

    if has_direct_results and not subdirs:
        # Single flat directory (e.g. user pointed directly at a subfolder)
        dirs_to_process = [input_dir]
    elif subdirs:
        # Batch layout with subfolders
        dirs_to_process = subdirs
        print(f"Found {len(dirs_to_process)} result subfolder(s) in '{input_dir}':")
        for d in dirs_to_process:
            print(f"  - {os.path.basename(d)}")
    else:
        print(f"No .vtp/.vtm files or result subfolders found in '{input_dir}'. Nothing to do.")
        return

    print(f"\n=== Post-Smoothing Configuration ===")
    print(f"  Smoothing radius : {radius} m")
    print(f"  Max cell area    : {max_cell_area}")
    print(f"  Normal threshold : {normal_threshold} deg")
    print(f"  Directories      : {len(dirs_to_process)}")
    print(f"====================================\n")

    for i, result_dir in enumerate(dirs_to_process, 1):
        print(f"\n[{i}/{len(dirs_to_process)}] Processing: {result_dir}")
        try:
            batch_smoother.batch_process_directory(
                result_dir,
                radius=radius,
                max_cell_area=max_cell_area,
                normal_threshold_deg=normal_threshold,
            )
        except Exception as e:
            print(f"  ERROR while processing '{result_dir}': {e}")

        # Generate CSV report for the smoothed results
        smoothed_dir = os.path.join(result_dir, "SMOOTHED")
        if os.path.isdir(smoothed_dir):
            print(f"  Generating summary report for SMOOTHED results ...")
            try:
                generate_report.generate_summary_csv(smoothed_dir)
            except Exception as e:
                print(f"  ERROR during report generation: {e}")

    print("\n=== Post-Smoothing Complete ===")


if __name__ == "__main__":
    main()




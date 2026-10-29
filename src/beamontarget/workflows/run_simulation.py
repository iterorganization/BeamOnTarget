# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# run_simulation.py
"""
Main entry point for the particle-mesh interaction simulation.

This script can run a batch of simulations, one for each beamlet
configuration file (.bl) found in the specified directory. Each run's
results are saved to a dedicated, named subfolder.

Can also be run in a setup preview mode:
  --view-setup          (Shows geometry and particle sources from the first .bl file)
  --view-setup geo      (Shows geometry only)
"""
import argparse
import trimesh
import numpy as np
import os
import glob
from beamontarget import config
from beamontarget.geometry import geometry
from beamontarget.particles import particles
from beamontarget.engine import engine
from beamontarget.io import output
from beamontarget.workflows.prerun_analysis import run_em_prerun_analysis
from beamontarget.io import batch_smoother # <-- NEW: Import the batch smoother script
from beamontarget.io import generate_report


def _resolve_config_relative_paths(config_path):
    """Resolve path-like config entries relative to the selected config file."""
    if not config_path:
        return

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

    config.GEOMETRY_CACHE_DIR = _abs_if_relative(config.GEOMETRY_CACHE_DIR)
    config.PARTICLE_SOURCE_DIR = _abs_if_relative(config.PARTICLE_SOURCE_DIR)
    config.DETAILED_OUTPUT_DIR = _abs_if_relative(config.DETAILED_OUTPUT_DIR)

    resolved_folders = {}
    for folder, settings in config.GEOMETRY_FOLDERS.items():
        resolved_folders[_abs_if_relative(folder)] = settings
    config.GEOMETRY_FOLDERS = resolved_folders


def run_full_simulation(grouped_meshes, particle_source_file, output_subfolder):
    """
    The main simulation workflow for a SINGLE run.
    It takes a specific particle file and an output subfolder as arguments.
    """
    if particle_source_file:
        print(f"\n--- Starting Simulation for Beam Config: '{os.path.basename(particle_source_file)}' ---")
    else:
        print(f"\n--- Starting Simulation using fallback particle sources ---")
    
    # --- Flatten the grouped geometry for the engine ---
    original_meshes, object_names, save_details_flags = [], [], []
    is_diagnostic_flags_per_mesh = []
    save_impact_flags_per_mesh = []
    max_impact_records_per_mesh = []
    for folder_path, mesh_list in grouped_meshes.items():
        settings = config.GEOMETRY_FOLDERS.get(folder_path, {})
        save_flag = settings.get("save_details", False)
        is_diagnostic = settings.get("is_diagnostic", False)
        save_impact = settings.get("save_impact_data", False)
        max_impacts = settings.get("max_impact_records", None)
        for mesh in mesh_list:
            original_meshes.append(mesh)
            object_names.append(mesh.metadata['name'])
            save_details_flags.append(save_flag)
            is_diagnostic_flags_per_mesh.append(is_diagnostic)
            save_impact_flags_per_mesh.append(save_impact)
            max_impact_records_per_mesh.append(max_impacts)
            
    face_counts = [len(m.faces) for m in original_meshes]
    if original_meshes:
        scene_mesh = trimesh.util.concatenate(original_meshes)
    else:
        # trimesh.util.concatenate([]) returns a list, which the ray engines
        # cannot build an intersector from. Give them a valid empty mesh.
        scene_mesh = trimesh.Trimesh()
    face_offsets = np.cumsum([0] + face_counts[:-1])

    # Load Particle Sources from the SPECIFIED file
    if particle_source_file:
        particle_sources_list = particles.load_beamlets_from_file(
            filename=particle_source_file,
            num_particles_per_beamlet=config.NUM_PARTICLES_PER_BEAMLET,
            beamlet_area=config.BEAMLET_AREA_FOR_CURRENT)
    else: # Fallback
        particle_sources_list = config.PARTICLE_SOURCES

    if not particle_sources_list:
        print(f"Error: No particle sources loaded or defined for this run. Skipping.")
        return

    output_dir_for_run = os.path.join(config.DETAILED_OUTPUT_DIR, output_subfolder)

    # --- Engine Selection Logic ---
    tracking_mode = str(getattr(config, "TRACKING_MODE", "ray")).strip().lower()
    
    if tracking_mode == "ray":
        deposited_power, impact_data = engine.run_simulation_single_hit(
            scene_mesh,
            face_offsets,
            face_counts,
            particle_sources_list,
            config.get_deposition_fraction,
            particle_batch_size=config.PARTICLE_BATCH_SIZE,
            num_cpu_cores=config.NUM_CPU_CORES,
            save_impact_flags=save_impact_flags_per_mesh,
            max_impact_records=max_impact_records_per_mesh,
        )
        per_species_power = {}  # not available in ray mode
    elif tracking_mode == "em_track_then_bvh":
        reaction_model_cfg = dict(config.REACTION_MODEL or {})
        reaction_model_cfg.setdefault(
            "density_direction",
            getattr(config, "DENSITY_DIRECTION", getattr(config, "MAIN_BEAM_AXIS_DIRECTION", [1.0, 0.0, 0.0])),
        )
        run_em_prerun_analysis(
            particle_sources_list=particle_sources_list,
            external_field_cfg=config.EXTERNAL_FIELD,
            reaction_model_cfg=reaction_model_cfg,
            bbox_min_corner_m=config.EM_BOUNDING_BOX_MIN_CORNER_M,
            bbox_max_corner_m=config.EM_BOUNDING_BOX_MAX_CORNER_M,
            output_dir_for_run=output_dir_for_run,
            em_step_length_m=config.EM_STEP_LENGTH_M,
        )
        deposited_power, impact_data, per_species_power = engine.run_simulation_em_track_then_bvh(
            scene_mesh,
            face_offsets,
            face_counts,
            particle_sources_list,
            config.get_deposition_fraction,
            particle_batch_size=config.PARTICLE_BATCH_SIZE,
            num_cpu_cores=config.NUM_CPU_CORES,
            em_step_length_m=config.EM_STEP_LENGTH_M,
            em_max_steps=config.EM_MAX_STEPS,
            em_max_distance_m=config.EM_MAX_DISTANCE_M,
            em_min_energy_ev=config.EM_MIN_ENERGY_EV,
            external_field_cfg=config.EXTERNAL_FIELD,
            reaction_model_cfg=reaction_model_cfg,
            bounding_box_min_corner_m=config.EM_BOUNDING_BOX_MIN_CORNER_M,
            bounding_box_max_corner_m=config.EM_BOUNDING_BOX_MAX_CORNER_M,
            save_impact_flags=save_impact_flags_per_mesh,
            max_impact_records=max_impact_records_per_mesh,
            em_bvh_checkpoint_distance_m=config.EM_BVH_CHECKPOINT_DISTANCE_M,
            em_min_steps_per_gyro_orbit=config.EM_MIN_STEPS_PER_GYRO_ORBIT,
        )
    else:
        raise ValueError(
            f"Unknown TRACKING_MODE: '{tracking_mode}'. "
            "Supported modes: 'ray', 'em_track_then_bvh'"
        )

    # --- Handle Outputs, saving to the specified subfolder ---
    if config.SAVE_PARAVIEW_FILES and any(save_details_flags):
        output.save_paraview_reports(original_meshes, deposited_power, object_names, save_details_flags,
                                     output_dir_for_run, per_species_power=per_species_power or None)
    if (config.SAVE_BINARY_POWERLOADS or config.SAVE_CSV_REPORTS) and any(save_details_flags):
        output.save_detailed_reports(original_meshes, deposited_power, object_names, save_details_flags, output_dir_for_run,
            save_binary=config.SAVE_BINARY_POWERLOADS, save_csv=config.SAVE_CSV_REPORTS,
            per_species_power=per_species_power or None)
    if config.SUMMARY_CSV_FILENAME:
        # We can also put the summary in the subfolder to keep results together
        summary_filename = f"summary_{output_subfolder}.csv"
        summary_path = os.path.join(output_dir_for_run, summary_filename)
        output.save_summary_to_csv(original_meshes, deposited_power, object_names, summary_path,
                                    per_species_power=per_species_power or None)

    # --- Save per-particle impact data if requested ---
    if any(save_impact_flags_per_mesh):
        output.save_impact_data_csv(impact_data, object_names, save_impact_flags_per_mesh, output_dir_for_run)

    # Automatic visualization is not supported in this memory-safe workflow
    if config.RUN_VISUALIZATION_AFTER_SIM:
        print("\nWARNING: Automatic visualization is disabled in the memory-safe workflow.")
        print("         Please use post_process.py to view results after the run completes.")

    print(f"\n--- Finished Simulation for: '{os.path.basename(particle_source_file) if particle_source_file else 'fallback_run'}' ---")

    # --- NEW: Automatic call to the batch smoother ---
    if config.RUN_SMOOTHER_AFTER_SIM:
        print("\n--- Auto-running Batch Smoother ---")
        try:
            batch_smoother.batch_process_directory(
                output_dir_for_run,
                radius=config.SMOOTHING_RADIUS,
                max_cell_area=config.SMOOTHING_MAX_CELL_AREA,
                normal_threshold_deg=config.SMOOTHING_NORMAL_THRESHOLD_DEG,
                method=config.SMOOTHING_METHOD,
                target_rel_error=config.SMOOTHING_TARGET_REL_ERROR,
                max_bandwidth_m=config.SMOOTHING_MAX_BANDWIDTH_M,
            )
        except Exception as e:
            print(f"An error occurred during automatic batch smoothing: {e}")
        print("--- Batch Smoothing Finished ---")

        # Generate CSV report for the smoothed results
        smoothed_dir = os.path.join(output_dir_for_run, "SMOOTHED")
        if os.path.isdir(smoothed_dir):
            print("\n--- Generating Smoothed Summary Report ---")
            try:
                generate_report.generate_summary_csv(smoothed_dir)
            except Exception as e:
                print(f"An error occurred during report generation: {e}")
            print("--- Report Generation Finished ---")

def run_setup_preview(grouped_meshes, view_mode):
    """Shows a 3D plot of the setup based on the view_mode."""
    print(f"--- Running Setup Preview Mode (Mode: {view_mode}) ---")
    
    # For preview, we need to pick a beam file to show. Let's pick the first one found.
    particle_source_file_for_preview = None
    if config.PARTICLE_SOURCE_DIR:
        bl_files = sorted(glob.glob(os.path.join(config.PARTICLE_SOURCE_DIR, '*.bl')))
        if bl_files:
            particle_source_file_for_preview = bl_files[0]
            
    show_sources = (view_mode != 'geo')
    particle_sources_list = []
    if show_sources:
        if particle_source_file_for_preview:
            print(f"Showing setup preview using beam file: {os.path.basename(particle_source_file_for_preview)}")
            particle_sources_list = particles.load_beamlets_from_file(
                filename=particle_source_file_for_preview, num_particles_per_beamlet=config.NUM_PARTICLES_PER_BEAMLET,
                beamlet_area=config.BEAMLET_AREA_FOR_CURRENT)
        else:
            print("Warning: No .bl files found in PARTICLE_SOURCE_DIR to show in preview.")
            
    output.visualize_setup(
        grouped_meshes=grouped_meshes, particle_sources=particle_sources_list,
        geometry_folders_config=config.GEOMETRY_FOLDERS, show_sources=show_sources)
    print("\nSetup preview finished.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a particle-mesh interaction simulation.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input-config',
        default=None,
        help="Path to a JSON configuration file. Defaults to config.json.")
    parser.add_argument(
        '--view-setup', nargs='?', const='full', default=None,
        choices=['geo', 'full'],
        help="Display a 3D preview of the setup.\n"
             "  'geo': Show geometry only.\n"
             "  'full': Show geometry and particle sources.\n"
             "  (default if flag is used with no value: 'full')")
    args = parser.parse_args(argv)

    if args.input_config:
        cfg_path = os.path.abspath(args.input_config)
        if not os.path.isfile(cfg_path):
            print(f"FATAL ERROR: Config file not found: '{cfg_path}'")
            return
        config.apply_config(path=cfg_path)
        _resolve_config_relative_paths(cfg_path)
        print(f"Using configuration file: {cfg_path}")

    # Load geometry ONCE, as it's shared by all runs.
    print("--- Loading shared geometry for all simulation runs... ---")
    grouped_geometry = geometry.load_scene(
        geometry_folders=config.GEOMETRY_FOLDERS,
        cache_dir=config.GEOMETRY_CACHE_DIR
    )

    if args.view_setup:
        run_setup_preview(grouped_geometry, view_mode=args.view_setup)
    else:
        # --- Batch Simulation Loop ---
        if config.PARTICLE_SOURCE_DIR:
            search_path = os.path.join(config.PARTICLE_SOURCE_DIR, '*.bl')
            beam_config_files = sorted(glob.glob(search_path))

            if not beam_config_files:
                print(f"Error: No .bl files found in the specified directory: '{config.PARTICLE_SOURCE_DIR}'")
            else:
                print(f"\nFound {len(beam_config_files)} beam configurations to simulate.")
                for beam_file in beam_config_files:
                    subfolder_name = os.path.splitext(os.path.basename(beam_file))[0]
                    run_full_simulation(grouped_geometry, beam_file, subfolder_name)
        else:
            print("\nPARTICLE_SOURCE_DIR not specified. Attempting single fallback run...")
            if config.PARTICLE_SOURCES:
                run_full_simulation(grouped_geometry, None, "fallback_run")
            else:
                print("No particle sources defined for fallback run.")


if __name__ == "__main__":
    main()



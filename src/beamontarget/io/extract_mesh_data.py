# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# extract_mesh_data.py
"""
A command-line tool to extract mesh element data from a .vtp file and
save it to a structured text file (CSV).

This script robustly handles MultiBlock files and filters out any mesh
elements with non-finite (NaN/inf) coordinates to prevent errors.
"""
import pyvista as pv
import pandas as pd
import numpy as np  # Import numpy for NaN checking
import argparse
import os
import sys

def extract_data_to_file(input_vtp, output_txt, properties_to_extract, filter_zero_properties=None):
    """
    Reads a .vtp file, extracts specified data for each mesh cell,
    and writes the results to a text file.
    
    Args:
        input_vtp: Path to input VTP file
        output_txt: Path to output CSV file
        properties_to_extract: List of properties to extract
        filter_zero_properties: List of properties to filter (exclude rows where these are zero)
    """
    print(f"--- Starting Data Extraction ---")
    print(f"  > Input file: {input_vtp}")
    print(f"  > Output file: {output_txt}")
    print(f"  > Properties to extract: {', '.join(properties_to_extract)}")

    # --- 1. Robustly load the mesh ---
    try:
        dataset = pv.read(input_vtp)
        if isinstance(dataset, pv.MultiBlock):
            print("  - Input is a MultiBlock dataset. Combining into a single mesh...")
            if dataset.n_blocks == 0:
                print("ERROR: The MultiBlock dataset is empty. Aborting.", file=sys.stderr)
                return
            mesh = dataset.combine()
        elif isinstance(dataset, pv.PolyData):
            mesh = dataset
        else:
            print(f"ERROR: Unsupported data type: {type(dataset)}. Aborting.", file=sys.stderr)
            return
            
        # Clean the mesh to remove duplicate points, etc.
        mesh.clean(inplace=True)

    except FileNotFoundError:
        print(f"ERROR: Input file not found: '{input_vtp}'. Aborting.", file=sys.stderr)
        return
    except Exception as e:
        print(f"ERROR: Failed to read or process VTP file. {e}", file=sys.stderr)
        return

    # --- 2. Validate the mesh for cells ---
    if mesh.n_cells == 0:
        print(f"ERROR: The final mesh from '{input_vtp}' has 0 cells. Aborting.", file=sys.stderr)
        return

    # --- 3. Extract coordinates and filter for valid ones ---
    print("\nExtracting element center coordinates...")
    face_centers = mesh.cell_centers().points
    
    # [THE CRITICAL FIX] Create a mask to filter out NaN/inf coordinates
    valid_coords_mask = np.isfinite(face_centers).all(axis=1)
    
    num_original_cells = mesh.n_cells
    num_valid_cells = np.sum(valid_coords_mask)

    if num_valid_cells == 0:
        print(f"ERROR: Found {num_original_cells} cells, but none have valid (finite) coordinates. Aborting.", file=sys.stderr)
        return
    
    if num_valid_cells < num_original_cells:
         print(f"  - WARNING: Found {num_original_cells} cells, but filtering to {num_valid_cells} with valid coordinates.")
    else:
         print(f"  > Found and validated {num_valid_cells} mesh elements (cells).")
         
    # Create the DataFrame using ONLY the valid coordinates
    df = pd.DataFrame(face_centers[valid_coords_mask], columns=['X', 'Y', 'Z'])

    # --- 4. Extract each requested property, applying the same mask ---
    print("\nExtracting requested properties...")
    for prop_name in properties_to_extract:
        
        if prop_name.lower() == 'area':
            try:
                print("  - Computing and adding 'Area'...")
                mesh = mesh.compute_cell_sizes()
                # Apply the mask to the area data to keep it in sync
                df['Area'] = mesh.cell_data['Area'][valid_coords_mask]
            except Exception as e:
                print(f"  - WARNING: Could not compute cell area. Skipping. Error: {e}")
            continue

        if prop_name in mesh.cell_data:
            print(f"  - Extracting '{prop_name}'...")
            # Apply the mask to the property data to keep it in sync
            df[prop_name] = mesh.cell_data[prop_name][valid_coords_mask]
        else:
            print(f"  - WARNING: Property '{prop_name}' not found in VTP cell data. Skipping.")

    # --- 5. Filter out zero values if requested ---
    if filter_zero_properties:
        print(f"\nFiltering out rows with zero values in: {', '.join(filter_zero_properties)}")
        initial_count = len(df)
        
        # Create filter mask - exclude rows where ANY of the specified properties are zero
        filter_mask = pd.Series([True] * len(df))
        
        for filter_prop in filter_zero_properties:
            if filter_prop in df.columns:
                # Exclude rows where this property is zero (or very close to zero)
                filter_mask &= (np.abs(df[filter_prop]) > 1e-12)
                print(f"  - Filtering based on '{filter_prop}'...")
            else:
                print(f"  - WARNING: Filter property '{filter_prop}' not found in data. Skipping filter.")
        
        # Apply the filter
        df = df[filter_mask].reset_index(drop=True)
        
        filtered_count = len(df)
        removed_count = initial_count - filtered_count
        
        print(f"  - Removed {removed_count} rows with zero values")
        print(f"  - Remaining rows: {filtered_count}")
        
        if filtered_count == 0:
            print("WARNING: All rows were filtered out. Output file will be empty.")

    # --- 6. Save the final DataFrame ---
    print(f"\nWriting data to '{output_txt}'...")
    try:
        df.to_csv(output_txt, index=False, float_format='%.6e')
        print("--- Extraction Complete ---")
    except Exception as e:
        print(f"ERROR: Could not write to output file. {e}", file=sys.stderr)


if __name__ == "__main__":
    # The argparse section remains unchanged
    parser = argparse.ArgumentParser(
        description="Extract mesh cell data (centers, properties) from a VTP file to a text file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input_vtp', type=str, required=True, help="Path to the input .vtp file.")
    parser.add_argument('-o', '--output_txt', type=str, required=True, help="Path for the output text file (will be saved in CSV format).")
    parser.add_argument('-p', '--properties', nargs='+', required=True, help="""Space-separated list of properties to extract.
Available options include any data array present in the VTP file, plus 'area'.
Example: -p Power_Density_W_m2 Deposited_Power_W area""")
    parser.add_argument('--filter-zero', nargs='*', help="""Filter out rows where specified properties are zero.
If no properties are specified after --filter-zero, filters based on the first property in -p.
Example: --filter-zero Power_Density_W_m2
Example: --filter-zero (uses first property from -p)""")
    args = parser.parse_args()
    
    # Handle filter-zero logic
    filter_zero_properties = None
    if args.filter_zero is not None:
        if len(args.filter_zero) == 0:
            # No specific properties given, use the first property from -p
            filter_zero_properties = [args.properties[0]]
        else:
            filter_zero_properties = args.filter_zero
    
    extract_data_to_file(args.input_vtp, args.output_txt, args.properties, filter_zero_properties)
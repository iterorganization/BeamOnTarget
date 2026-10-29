# BeamOnTarget

A high-performance Python application for simulating the interaction of
particle beams with complex 3D geometries.  It calculates the power
deposited on the surfaces of mesh objects (from `.stl` files) and is
optimised for parallel processing, memory efficiency, and batch-run
capabilities.

The primary workflow involves defining geometry and particle sources,
running the simulation, and analysing the detailed, ParaView-compatible
output files — all from a graphical interface or the command line.

---

## Key Features

- **Graphical User Interface** — Tkinter-based GUI (`beamontarget.gui.sim_gui`) for
  managing configurations, launching simulations, viewing results in an
  embedded 3D viewer (Open3D), and comparing CSV summaries with
  interactive bar-charts.
- **Built-in 3D Viewer** — GPU-accelerated Open3D renderer embedded
  inside the GUI with mouse-drag rotation, scroll-wheel zoom,
  right-click pan, orientation axis overlay, and jet colour-map with
  colour bar.
- **High-Performance Engine** — Uses `trimesh` with the `pyembree`
  ray-tracing backend for extremely fast intersection calculations.
- **Parallel Processing** — Leverages `joblib` to distribute work
  across multiple CPU cores.
- **Memory-Safe Design** — Handles billions of particles without
  storing them all in memory simultaneously.
- **Folder-Centric Geometry** — Organise `.stl` files into folders;
  apply scaling and mesh refinement to entire groups.
- **Geometry Caching** — Saves processed (refined) meshes to a cache,
  speeding up subsequent runs.
- **Batch Simulation** — Automatically finds and runs a simulation for
  every `.bl` file in the source directory.
- **Advanced Particle Sources** — Multiple beam models including
  `GaussianTwissBeam`.
- **Professional Output** — Saves results as ParaView-compatible `.vtp`
  files with cell data (Deposited Power, Power Density).
- **Automated Post-Processing** — Optional smoothing for clearer heat
  maps: a radius average, or an energy-conserving "physical" method that
  also reports a per-cell uncertainty.
- **VTP Data Extraction** — Extract cell data from `.vtp` files to CSV
  directly from the GUI.
- **CSV Result Comparison** — Side-by-side bar-charts of peak power
  density and deposited power across multiple simulation runs, with
  per-component labels, multipliers, and log-scale toggles.
- **Cross-Platform** — Runs on Linux and Windows (Python ≥ 3.10).

---

## File Structure

| File | Description |
|---|---|
| `src/beamontarget/gui/sim_gui.py` | Tkinter GUI — the main graphical entry point (`beamontarget` command). |
| `src/beamontarget/visualization/viewer.py` | Built-in 3D viewer (Open3D off-screen rendering in Tk canvas). |
| `src/beamontarget/workflows/run_simulation.py` | CLI entry point — handles arguments and orchestrates simulation runs. |
| `src/beamontarget/config.py` | Central configuration module; reads/writes `config.json`. |
| `config.json` | Default configuration — the simple cube-and-sphere example. |
| `config_beamline_example.json` | Second configuration — the neutral beam injector example. |
| `src/beamontarget/engine/engine.py` | Core computational engine — power deposition calculation. |
| `src/beamontarget/geometry/geometry.py` | Loading, processing, caching, and grouping of `.stl` meshes. |
| `src/beamontarget/particles/particles.py` | Particle source classes and `.bl` file loading. |
| `src/beamontarget/io/trajectory_export.py` | Records a sample of particle trajectories to `.vtp`. |
| `src/beamontarget/io/output.py` | File output — `.vtp` saving and summary `.csv` reports. |
| `src/beamontarget/io/smooth_results.py` | Core smoothing logic (library) and the smoothing CLI. |
| `src/beamontarget/io/physical_smooth.py` | Energy-conserving "physical" smoothing with error estimate. |
| `src/beamontarget/io/batch_smoother.py` | Applies smoothing to result `.vtp` files in a directory. |
| `src/beamontarget/workflows/post_smooth.py` | Additional post-processing / smoothing utilities. |
| `src/beamontarget/workflows/merge_results.py` | Cell-wise worst-case merge of scenarios (Merge tab). |
| `src/beamontarget/io/transform_results.py` | Coordinate transform and CSV / VTP export (Transform tab). |
| `src/beamontarget/io/generate_report.py` | Automated report generation from results. |
| `src/beamontarget/io/extract_mesh_data.py` | CLI tool for extracting mesh cell data from `.vtp` to CSV. |
| `pyproject.toml` | Modern Python packaging metadata (PEP 621). |
| `requirements.txt` | Pinned dependency list (legacy, kept for reference). |
| `install.sh` | One-step installer for **Linux**. |
| `install.bat` | One-step installer for **Windows**. |

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python ≥ 3.10** | Any CPython 3.10–3.12 build. |
| **Tkinter** | Usually bundled with Python. On Debian/Ubuntu: `apt install python3-tk`. |
| **C compiler** (Linux only) | Required to build `pyembree` / `embreex` from source. |

> **Running on an HPC cluster with environment modules?**
> BeamOnTarget makes no assumption about your site. If your cluster uses Lmod or
> Environment Modules, load a suitable Python and Tk before installing, e.g.
> `module load Python/3.11.5-GCCcore-13.2.0 Tkinter/3.11.5-GCCcore-13.2.0` —
> substitute the module names your site actually provides. Likewise,
> `PARAVIEW_PATH` in `config.json` defaults to `paraview` (resolved on `PATH`);
> set it to an absolute path, or set `PARAVIEW_MODULE` to the module the GUI
> should load, if that is how ParaView is provided at your site.

### Option A — Quick Install (recommended)

**Linux / macOS:**

```bash
git clone https://github.com/iterorganization/BeamOnTarget.git
cd BeamOnTarget
chmod +x install.sh
./install.sh            # creates ./venv and installs everything
source venv/bin/activate
beamontarget            # launch the GUI
```

**Windows:**

```batch
git clone https://github.com/iterorganization/BeamOnTarget.git
cd BeamOnTarget
install.bat             &:: creates .\venv and installs everything
venv\Scripts\activate
beamontarget            &:: launch the GUI
```

Both scripts accept an optional argument to specify a custom venv path:

```bash
./install.sh /opt/beamontarget/env    # Linux
install.bat  C:\bot\env               :: Windows
```

### Option B — Manual Install

```bash
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .                      # editable install from pyproject.toml
```

### Option C — pip install from requirements.txt (legacy)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note on Windows:** `pyembree` and `embreex` may fail to compile.
> If so, install them via conda instead:
> ```
> conda install -c conda-forge pyembree embreex
> ```
> All other dependencies (Open3D, pyvista, matplotlib, …) have
> pre-built wheels for Windows.

---

## Quick Start

### 1. Launch the GUI

```bash
beamontarget              # if installed via pip install -e .
# or
python -m beamontarget.gui.sim_gui
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| **General** | ParaView path, number of CPU cores, particles per beamlet, energy. |
| **Geometry** | Add / edit / remove geometry folders and their STL files. |
| **Particles** | Select particle source directory and `.bl` files. |
| **Output** | Output directory, save options, CSV result comparison charts, VTP data extraction. |
| **Run** | Save configuration, launch simulation, view live log output. |

### 2. The two shipped examples

The repository ships **two complete examples**, so a fresh checkout runs without
any input of your own.

| Config | Geometry | Beam | Runtime |
|---|---|---|---|
| `config.json` (default) | `GEOMETRY_EXAMPLE/` — a cube and a sphere | `BEAM_CONFIG_EXAMPLE/example_simple.bl` — 50 random beamlets, 50 kW | seconds |
| `config_beamline_example.json` | `NEU/`, `RID/`, `CAL/` — neutral beam injector components | `BEAM_CONFIG_BEAMLINE/example_bl_config.bl` — 1280 beamlets at 870 keV | minutes |

```bash
beamontarget-run                                      # the simple example
beamontarget-run -i config_beamline_example.json      # the beamline example
```

Run both from the repository root. Relative paths in a config are resolved
against the project folder, so launching from elsewhere will not find the
geometry.

#### The simple example

`GEOMETRY_EXAMPLE/` holds a 1 m cube and a sphere of radius 0.75 m, side by side
in y with a gap between them. 50 beamlets are scattered at random across a source
plane upstream and travel along +x, so roughly a third strike the cube, a third
the sphere, and a third pass between. The beam carries exactly 1 kW per beamlet,
50 kW in total, of which about 37 kW lands on the two bodies.

It finishes in a few seconds and is the fastest way to confirm your installation
works. See `GEOMETRY_EXAMPLE/README.md` and `BEAM_CONFIG_EXAMPLE/README.md`.

#### The beamline example

| Folder | Contents | Description |
|---|---|---|
| `NEU/` | `NEU1.stl` - `NEU5.stl` | Five rectangular slabs forming a channel, x = 1.900-4.900 m |
| `RID/` | `RID1.stl` - `RID5.stl` | Five rectangular slabs, x = 5.400-7.200 m |
| `CAL/` | `BENCHMARK.stl` | A single flat plane at x = 7.750 m, 0.800 x 1.700 m |

These are generic blocks and planes, not a model of any particular machine. The
beam is a nominal unsteered configuration of 1280 parallel beamlets on a 20 x 64
grid at 870 keV, 3 mrad core divergence, inclined 49.22 mrad downward. Each
folder has a `README.md` with dimensions and tessellation; note in particular
that the deflecting field region in `config_beamline_example.json` is aligned to
the RID panel extent.

### 3. Using your own geometry and beam

- Create folders (e.g. `MY_GEOMETRY/`, `TARGETS/`) and place `.stl` files
  inside. In the **Geometry** tab, add each folder and configure its scale and
  mesh refinement settings.
- Create a directory with one or more `.bl` files — space-separated text with
  columns:

  ```
  # CenterX CenterY CenterZ DirX DirY DirZ Mass_kg Charge_e CurrentDensity_A_m2 SigmaY_m DeltaY_rad SigmaZ_m DeltaZ_rad HaloFraction DeltaHY_rad DeltaHZ_rad
  0 0 0 1 0 0 1.67e-27 1 1.0 0.005 0.005 0.005 0.005 0.1 0.01 0.01
  ```

  Select it in the **Particles** tab, or set `PARTICLE_SOURCE_DIR`. Every `.bl`
  file in that directory becomes a separate simulation run.

  `BEAM_CONFIG_BEAMLINE/README.md` documents every column.

### 4. Run the Simulation

Click **▶ Run** in the **Run** tab, or from the command line:

```bash
python -m beamontarget.workflows.run_simulation
```

Results are saved as `.vtp` files in the output directory (default
`OUTPUT/`), organised by `.bl` filename.

### 5. View Results

- **Built-in viewer:** Use the 3D viewer buttons in the GUI
  (Geometry / Results / All / Sources).  The viewer supports rotation
  (left-drag), zoom (scroll), and pan (right-drag).
- **ParaView:** Open the `.vtp` files directly in
  [ParaView](https://www.paraview.org/), or use the GUI's ParaView
  integration buttons.
- **CSV comparison:** In the **Output** tab, load summary CSV files
  and compare peak power density / deposited power across runs with
  interactive bar charts.
- **Data extraction:** Use the **Extract results data** button to
  export VTP cell data (coordinates, area, power, power load) to CSV.

---

## Command-Line Usage

```bash
# Run the default config.json (the simple cube-and-sphere example)
beamontarget-run

# Run a different configuration
beamontarget-run -i config_beamline_example.json

# Preview geometry and source positions (needs a display)
beamontarget-run --view-setup

# Preview geometry only
beamontarget-run --view-setup geo
```

`beamontarget-run` is equivalent to
`python -m beamontarget.workflows.run_simulation`. The only options are
`-i/--input-config` and `--view-setup`.

Run from the repository root: relative paths in a config are resolved against
the project folder, not the current directory.

Results are written to `<DETAILED_OUTPUT_DIR>/<beam file name>/`, so the two
shipped examples land in `OUTPUT/example_simple/` and
`OUTPUT/example_bl_config/` respectively.

### Running without geometry

Geometry is optional. If no `.stl` files are found, the run continues with an
empty scene and particles are tracked through free space -- nothing is struck, so
no power is deposited.

This is most useful with `"TRACKING_MODE": "em_track_then_bvh"`, where the
electromagnetic integration and the pre-run analysis (Larmor radius, reaction
evolution) are the point and targets are irrelevant. To study how a beam drifts
and disperses in a field before adding any hardware, point `GEOMETRY_FOLDERS` at
an empty folder and configure `EXTERNAL_FIELD` as usual.

### Adaptive EM step in strong magnetic fields

EM tracking uses a fixed step of `EM_STEP_LENGTH_M` (±10 % jitter). Where the
magnetic field is strong, one step can cover a large part of a gyro-orbit
(an 870 keV D⁺ in 2 T has a 9.5 cm Larmor radius, so a 5 cm step gives only
~12 steps per orbit) and the trajectory becomes inaccurate. Set
`EM_MIN_STEPS_PER_GYRO_ORBIT` (GUI: *Min steps per gyro-orbit*, e.g. 20) to
cap each particle's step so it turns by at most 360°/N about the local
field: `dt ≤ (2π/N) / (|q/m|·|B|)`. The step only shrinks where the field
requires it; neutrals and weak-field regions keep the full step. 0 (default)
turns it off. Shorter steps use up `EM_MAX_STEPS` sooner: the run reports how
many particles were still in flight when the step budget ran out.

### Trajectory export

With `TRAJECTORY_EXPORT_ENABLED` (Output tab: *Save particle trajectories*)
the tracks of `TRAJECTORY_EXPORT_MAX_PARTICLES` particles (default 200) are
written to `<run output>/TRAJECTORIES/trajectories.vtp`, one polyline per
particle, in both tracking modes. The sample is spread evenly over the beam
origins, so every beamlet is represented. In EM mode each polyline ends at
the particle's impact point (or where it left the bounding box); in ray mode
a track is a straight line to its hit, or past the scene if it misses.
Point arrays: `ParticleID`, `SourceIndex`, `ChargeState`, `KineticEnergy_eV`
(from the particle's velocity, so it follows acceleration in electric fields)
and `StepIndex`.

### Manual Smoothing

```bash
python -m beamontarget.io.batch_smoother -i OUTPUT/my_beam_run
python -m beamontarget.io.batch_smoother -i OUTPUT/my_beam_run -m physical
```

Two methods are available (`SMOOTHING_METHOD` in the config, or *Method* in
the Output tab):

- `radius` (default) — each cell gets `sum(P) / sum(A)` over the cells within
  `SMOOTHING_RADIUS` whose normals are within `SMOOTHING_NORMAL_THRESHOLD_DEG`.
  Only the density is changed; the total of `density × area` is not conserved.
- `physical` — estimates the density by conservative diffusion along the
  surface, so `sum(density × area)` equals the deposited power to solver
  precision. The bandwidth adapts per cell: it is widened only where the
  Monte Carlo counting error exceeds `SMOOTHING_TARGET_REL_ERROR` (default
  0.05), up to `SMOOTHING_MAX_BANDWIDTH_M` (default 0.04 m). Besides the
  smoothed `Power_Density_W_m2` and a consistent `Deposited_Power_W`, it writes
  `Power_Density_sigma_stat_W_m2` (1σ counting error, by bootstrap),
  `Power_Density_bias_W_m2` (smoothing error), `Power_Density_p95_W_m2`,
  `Smoothing_N_eff` and `Smoothing_bandwidth_m`. `smooth_log.txt` reports the
  conservation error and χ²/dof (≈ 1 means as much smoothing as the
  statistics support). It is slower than the radius method (minutes for a
  mesh with millions of cells) and needs triangle meshes.

### Worst-Case Merge (Merge tab)

The Merge tab combines several scenarios into a worst-case envelope. Add one
or more result folders (each holds one subfolder per scenario), choose raw or
smoothed files and, optionally, a comma-separated name filter, then click
*Load Scenarios*. Click a scenario to assign it to the active colour group;
Shift+click assigns a range, and double-clicking a colour renames the group
(the name is used as the output folder).

For every group, the files of the same component are merged cell by cell:
each cell takes all its arrays (totals, species, error arrays) from the
scenario with the highest `Power_Density_W_m2` there, or `Deposited_Power_W`
if selected. The *Merge plan* pane lists exactly which files go into which
output before anything is run; *Open in ParaView* opens the selection.
Results are written to

```
MERGED/<result folder>/<group>/merged_results_<component>[__smoothed].vtp
```

together with a `merge_sources.txt` per group folder, a `merge_log_<time>.csv`
per run, and a `merged_from` field array in every merged file. The summed
power of a merged file is an envelope, not a physical power.

### Coordinate Transform (Transform tab)

Exports raw, smoothed or merged result files from the beam-line frame to the
Tokamak frame, as CSV (one row per cell) and/or VTP. The transform rotates
about Z, then translates: `p' = R p + t`. The presets:

| Frame | Rotation about Z | Translation X, Y, Z |
|---|---|---|
| HNB1 → Tokamak | −79.543° | −0.57723, 32.385248, 1.453462 m |
| HNB2 → Tokamak | −59.543° | −11.618826, 30.234754, 1.453462 m |
| HNB3 → Tokamak | −39.543° | −21.25902, 24.437503, 1.453462 m |
| DNB → Tokamak | −116° | 11.410436, 26.617882, 0.920 m |

Editing a value switches the preset to *Custom*. Everything is in metres, as
in the simulation; a geometry in other units must be scaled in the geometry
setup. The CSV holds the cell centres `X, Y, Z` [m], `Area` [m²],
`Deposited_Power_W`, `Power_Density_W_m2` and `total_deposited_power_W` (sum
over all cells, before zero filtering). The *power factor* multiplies every
array that is linear in power, in the CSV and the VTP alike (totals,
species, sigma/bias/p95, standard errors; sums of squares by the factor
squared); ratios, `Smoothing_N_eff`, bandwidths and normals are unchanged,
and `Normals` are rotated with the mesh. *Ignore zero-valued rows* drops
cells whose selected power values are zero in the input file. Output goes to
`TRANSFORMED/<result folder>/<scenario>/<scenario>__<file>.csv|.vtp`, and
each VTP records the transform and factor in its field data.

### VTP Data Extraction (CLI)

```bash
python -m beamontarget.io.extract_mesh_data -i input.vtp -o output.csv -p Power_Density_W_m2 Deposited_Power_W area
```

---

## Dependencies

All dependencies are declared in `pyproject.toml` and installed
automatically.  Key packages:

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `trimesh` + `pyembree` / `embreex` | Mesh loading and ray tracing |
| `pyvista` | VTP file I/O |
| `open3d ≥ 0.17` | GPU-accelerated 3D rendering (off-screen) |
| `matplotlib` | Bar-chart plotting in GUI |
| `Pillow` | Image processing for viewer overlay |
| `pandas` | CSV / DataFrame handling |
| `scipy` | Smoothing algorithms |
| `joblib` | Parallel processing |
| `tqdm` | Progress bars |

---

## Authors and attribution

Copyright &copy; 2025-2026 ITER Organization,
Route de Vinon-sur-Verdon, CS 90 046, 13067 St Paul Lez Durance Cedex, France.

BeamOnTarget was developed at the ITER Organization by **Carlo Poggi**, with
contributions from **David Atteln** (HTWK Leipzig). See [AUTHORS](AUTHORS) for
the full list and a breakdown of contributions.

The views and opinions expressed herein do not necessarily reflect those of the
ITER Organization.

---

## Citing BeamOnTarget

If you use BeamOnTarget in published work, please cite it using the metadata in
[CITATION.cff](CITATION.cff). GitHub renders this as a **"Cite this repository"**
button in the sidebar.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening an issue or a pull request.

---

## License

BeamOnTarget is free software: you can redistribute it and/or modify it under
the terms of the **GNU Lesser General Public License, version 3**
(`LGPL-3.0-only`) as published by the Free Software Foundation.

BeamOnTarget is distributed in the hope that it will be useful, but **WITHOUT
ANY WARRANTY**; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
details.

- [LICENSE](LICENSE) &mdash; the full text of the GNU LGPL v3.
- [GPLv3.txt](GPLv3.txt) &mdash; the full text of the GNU GPL v3, whose terms
  the LGPL v3 incorporates by reference.

The LGPL was chosen, rather than the GPL, so that BeamOnTarget can be used as a
component of larger integrated modelling frameworks released under other
licences.

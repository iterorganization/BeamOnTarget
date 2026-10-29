# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# engine.py
"""
The definitive high-performance, parallel, memory-safe simulation engine.
Its ONLY job is to compute the final power deposition. It does not handle
visualization sampling to ensure minimal memory footprint.
"""
import trimesh
import numpy as np
import os
import math
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from beamontarget.interactions.field_provider import create_field_provider
from beamontarget.interactions.reactions import create_reaction_model
from beamontarget.interactions.em_tracker_v2 import trace_particle_batch_em_only
from beamontarget.engine.trajectory_intersector import intersect_trajectory_segments_bvh

def _empty_particle_batch():
    return {
        "origins": np.empty((0, 3), dtype=np.float64),
        "directions": np.empty((0, 3), dtype=np.float64),
        "powers": np.empty(0, dtype=np.float64),
        "energies_ev": np.empty(0, dtype=np.float64),
        "currents_a": np.empty(0, dtype=np.float64),
        "masses_kg": np.empty(0, dtype=np.float64),
        "charge_states": np.empty(0, dtype=np.int32),
        "source_indices": np.empty(0, dtype=np.int32),
    }


def _generate_source_particle_data(source, count):
    origins, dirs, powers, energies, currents, charge_states = source.generate(num_particles=count)
    count = len(origins)
    return {
        "origins": np.asarray(origins, dtype=np.float64),
        "directions": np.asarray(dirs, dtype=np.float64),
        "powers": np.asarray(powers, dtype=np.float64),
        "energies_ev": np.asarray(energies, dtype=np.float64),
        "currents_a": np.asarray(currents, dtype=np.float64),
        "masses_kg": np.full(count, source.mass, dtype=np.float64),
        "charge_states": np.asarray(charge_states, dtype=np.int32),
        "source_indices": np.full(count, source.source_index, dtype=np.int32),
    }


def _iter_particle_batches(particle_sources_list, particle_batch_size):
    particle_batch_size = max(1, int(particle_batch_size))

    batch_parts = {key: [] for key in _empty_particle_batch().keys()}
    batch_count = 0

    for source in particle_sources_list:
        remaining = int(source.num_particles)
        if remaining <= 0:
            continue

        while remaining > 0:
            space_left = particle_batch_size - batch_count
            take_count = min(space_left, remaining)
            generated = _generate_source_particle_data(source, take_count)
            if generated["origins"].size == 0:
                break
            for key, value in generated.items():
                batch_parts[key].append(value)
            produced = int(generated["origins"].shape[0])
            batch_count += produced
            remaining -= produced

            if batch_count >= particle_batch_size:
                yield {
                    key: np.concatenate(parts, axis=0)
                    for key, parts in batch_parts.items()
                }
                batch_parts = {key: [] for key in _empty_particle_batch().keys()}
                batch_count = 0

    if batch_count > 0:
        yield {
            key: np.concatenate(parts, axis=0)
            for key, parts in batch_parts.items()
        }


def _empty_impact_data(num_objects):
    return [{"total_hits": 0, "stored_hits": 0, "records": []} for _ in range(num_objects)]


def _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records):
    num_objects = len(impact_data)
    for obj_idx in range(num_objects):
        if not save_impact_flags[obj_idx]:
            continue
        cap = max_impact_records[obj_idx]
        new_records = chunk_impacts[obj_idx]["data"]
        reservoir = impact_data[obj_idx]["records"]
        total_seen = impact_data[obj_idx]["total_hits"]

        for record in new_records:
            total_seen += 1
            if cap is None or len(reservoir) < cap:
                reservoir.append(record)
            else:
                j = np.random.randint(0, total_seen)
                if j < cap:
                    reservoir[j] = record

        impact_data[obj_idx]["total_hits"] = total_seen
        impact_data[obj_idx]["stored_hits"] = len(reservoir)


def _process_particle_batch_ray(particle_batch, intersector, face_offsets, face_counts,
                                deposition_model, save_impact_flags=None, max_impact_records=None):
    """
        WORKER FUNCTION: Processes one particle batch and returns sparse updates per object.
    Returns: (sparse_updates, impact_records)
      sparse_updates: list of tuples (indices_array, values_array) per object.
      impact_records: list of dicts per object (only for objects with save_impact_flags=True).
        Each dict has keys: 'hit_count', 'data' (list of row-tuples, capped at max_impact_records).
        If an object has no hits in the batch, returns (None, None) for that object in sparse_updates,
    and {'hit_count': 0, 'data': []} in impact_records.
    """
    num_objects = len(face_counts)
    sparse_updates = [(None, None) for _ in face_counts]

    # Initialise per-object impact record containers
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects
    impact_records = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]
    
    if num_objects == 0:
        # Free-space run: no geometry, so no intersections to test. Particles
        # still travel, they simply have nothing to strike.
        return sparse_updates, impact_records

    if particle_batch["origins"].size == 0:
        return sparse_updates, impact_records  # No particles; no updates

    ray_origins = particle_batch["origins"]
    ray_directions = particle_batch["directions"]
    particle_powers = particle_batch["powers"]
    particle_energies_eV = particle_batch["energies_ev"]
    particle_masses = particle_batch["masses_kg"]
    particle_charges = particle_batch["charge_states"]
    particle_source_indices = particle_batch["source_indices"]
    particle_currents = particle_batch["currents_a"]

    # Sub-chunk the ray intersection to avoid memory allocation failures inside trimesh
    _RAY_SUB_CHUNK = 500_000
    n_rays = ray_origins.shape[0]
    if n_rays <= _RAY_SUB_CHUNK:
        locations, index_ray, index_tri_global = intersector.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    else:
        loc_parts, iray_parts, itri_parts = [], [], []
        for start in range(0, n_rays, _RAY_SUB_CHUNK):
            end = min(start + _RAY_SUB_CHUNK, n_rays)
            locs_c, iray_c, itri_c = intersector.intersects_location(
                ray_origins=ray_origins[start:end],
                ray_directions=ray_directions[start:end],
                multiple_hits=False)
            if len(locs_c) > 0:
                loc_parts.append(locs_c)
                iray_parts.append(iray_c + start)  # offset ray indices
                itri_parts.append(itri_c)
        if loc_parts:
            locations = np.concatenate(loc_parts, axis=0)
            index_ray = np.concatenate(iray_parts, axis=0)
            index_tri_global = np.concatenate(itri_parts, axis=0)
        else:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.intp)
            index_tri_global = np.empty(0, dtype=np.intp)
    
    if len(locations) > 0:
        colliding_particle_power = particle_powers[index_ray]
        colliding_particle_energy_eV = particle_energies_eV[index_ray]
        fraction = deposition_model(colliding_particle_energy_eV)
        power_to_deposit = colliding_particle_power * fraction

        object_indices = np.searchsorted(face_offsets, index_tri_global, side='right') - 1
        local_tri_indices = index_tri_global - face_offsets[object_indices]

        # Build sparse updates per object — fully vectorized
        for obj_idx in np.unique(object_indices):
            mask = (object_indices == obj_idx)
            idxs = local_tri_indices[mask]
            vals = power_to_deposit[mask]
            if idxs.size > 0:
                # Vectorized duplicate summation via bincount
                n_faces = face_counts[obj_idx]
                summed = np.bincount(idxs, weights=vals, minlength=n_faces)
                nonzero = summed.nonzero()[0]
                sparse_updates[obj_idx] = (nonzero.astype(np.int32),
                                           summed[nonzero].astype(np.float32))

            # --- Collect impact data for flagged objects (vectorized) ---
            if save_impact_flags[obj_idx]:
                hit_rays = index_ray[mask]
                hit_count = hit_rays.size
                impact_records[obj_idx]['hit_count'] += hit_count

                hit_locs = locations[mask]
                hit_dirs = ray_directions[hit_rays]
                hit_energies = particle_energies_eV[hit_rays]
                hit_masses = particle_masses[hit_rays]
                hit_charges = particle_charges[hit_rays]
                hit_src_indices = particle_source_indices[hit_rays]
                hit_currents = particle_currents[hit_rays]

                # Build records as a structured numpy array, then convert to list of tuples
                records_array = np.column_stack([
                    hit_src_indices.astype(np.float64),
                    hit_masses,
                    hit_charges.astype(np.float64),
                    hit_locs,
                    hit_dirs,
                    hit_energies,
                    hit_currents,
                ])
                impact_records[obj_idx]['data'].extend(
                    [tuple(row) for row in records_array])

    return sparse_updates, impact_records


def run_simulation_single_hit(scene_mesh, face_offsets, face_counts, particle_sources_list,
                              deposition_model, particle_batch_size, num_cpu_cores,
                              save_impact_flags=None, max_impact_records=None):
    """
    MANAGER FUNCTION: Dispatches chunks, combines power results and (optionally) impact data.
    Returns: (final_deposited_power, impact_data)
      impact_data: list of dicts per object. Each dict has:
        'total_hits': int — total number of particle impacts on this object
        'stored_hits': int — number of impact records actually stored (≤ max_impact_records)
        'records': list of tuples (source_index, mass_kg, charge_state, pos_x, pos_y, pos_z,
                                   dir_x, dir_y, dir_z, kinetic_energy_eV, current_A)
      Only populated for objects where save_impact_flags[i] is True.
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects

    # Resolve cores: -1 means all available
    available_cores = os.cpu_count() or 1
    n_jobs = available_cores if (num_cpu_cores == -1) else max(1, int(num_cpu_cores))
    print(f"\nInitializing FAST, Memory-Safe Parallel simulation engine...")
    print(f"  - Using {n_jobs} threads (available cores: {available_cores}).")
    print(f"  - Target particle batch size: {int(particle_batch_size)}")
    
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)

    # Count total batches for progress bar without materializing them all in memory
    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    # Progressive accumulation: avoid keeping all partial results in memory
    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]

    # Accumulators for impact data (reservoir sampling for unbiased selection)
    impact_data = _empty_impact_data(num_objects)

    print(f"Processing ~{total_batches} Particle Batches (sequential)...")
    batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)
    for batch in tqdm(batch_iter, total=total_batches, desc="Processing Particle Batches"):
        sparse_chunk, chunk_impacts = _process_particle_batch_ray(
            batch, intersector, face_offsets, face_counts,
            deposition_model, save_impact_flags, max_impact_records)
        # Apply sparse updates to the accumulator immediately
        for obj_idx, (idxs, vals) in enumerate(sparse_chunk):
            if idxs is None:
                continue
            np.add.at(final_deposited_power[obj_idx], idxs, vals)
        _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records)

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")

    # Print impact data summary
    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")
    
    return final_deposited_power, impact_data


def run_simulation_em_track_then_bvh(
    scene_mesh,
    face_offsets,
    face_counts,
    particle_sources_list,
    deposition_model,
    particle_batch_size,
    num_cpu_cores,
    em_step_length_m,
    em_max_steps,
    em_max_distance_m=None,
    em_min_energy_ev=None,
    external_field_cfg=None,
    reaction_model_cfg=None,
    bounding_box_min_corner_m=None,
    bounding_box_max_corner_m=None,
    save_impact_flags=None,
    max_impact_records=None,
    em_bvh_checkpoint_distance_m=None,
    em_min_steps_per_gyro_orbit=None,
):
    """
    Two-phase EM particle tracking:
    Phase 1: EM integration inside bounding box (no geometry checks).
    Phase 2: Deferred BVH intersection check on all trajectory segments.

    Optional intermediate BVH checkpoints (em_bvh_checkpoint_distance_m):
    If set, a BVH check is performed every time a particle has travelled that
    distance. Particles confirmed to have hit geometry are deactivated immediately
    and not traced further, reducing total computation for early-hitting particles.

    Optional adaptive step (em_min_steps_per_gyro_orbit): the step is shortened
    where the local B field would otherwise resolve a gyro-orbit in fewer than
    that many steps (see trace_particle_batch_em_only).
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    if max_impact_records is None:
        max_impact_records = [None] * num_objects

    perf_enabled = os.environ.get("BEAMONTARGET_PROFILE_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}

    available_cores = os.cpu_count() or 1
    n_jobs = available_cores if (num_cpu_cores == -1) else max(1, int(num_cpu_cores))
    print("\nInitializing Two-Phase EM BVH simulation engine...")
    print(f"  - Using {n_jobs} threads (available cores: {available_cores}).")
    print(f"  - Target particle batch size: {int(particle_batch_size)}")
    print(f"  - Step length: {em_step_length_m:.3e} m, max steps: {int(em_max_steps)}")
    if em_min_steps_per_gyro_orbit is not None and em_min_steps_per_gyro_orbit > 0:
        print(f"  - Adaptive step: shorter than {em_step_length_m:.3e} m wherever the local B "
              f"would resolve a gyro-orbit in fewer than {int(em_min_steps_per_gyro_orbit)} steps")
    if bounding_box_min_corner_m is not None and bounding_box_max_corner_m is not None:
        print(f"  - Bounding box min corner: {bounding_box_min_corner_m}")
        print(f"  - Bounding box max corner: {bounding_box_max_corner_m}")

    # Compute checkpoint steps (None = disabled)
    bvh_checkpoint_steps = None
    if em_bvh_checkpoint_distance_m is not None and em_bvh_checkpoint_distance_m > 0:
        # Convert user distance (m) into a fixed integer number of EM steps.
        bvh_checkpoint_steps = max(1, int(math.ceil(em_bvh_checkpoint_distance_m / em_step_length_m)))
        effective_checkpoint_distance_m = bvh_checkpoint_steps * em_step_length_m
        print(
            f"  - BVH checkpoint every {em_bvh_checkpoint_distance_m:.3g} m "
            f"(~{effective_checkpoint_distance_m:.3g} m, {bvh_checkpoint_steps} steps)"
        )

    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(scene_mesh)

    # Count total batches for progress bar without materializing all in memory
    total_particles = sum(int(s.num_particles) for s in particle_sources_list)
    total_batches = math.ceil(total_particles / particle_batch_size) if total_particles > 0 else 0

    final_deposited_power = [np.zeros(count, dtype=np.float32) for count in face_counts]
    per_species_power = {}  # {charge_state: [np.zeros per object]}
    impact_data = _empty_impact_data(num_objects)

    print(f"Processing ~{total_batches} particle batches (Phase 1 EM tracing + Phase 2 BVH)...")

    def _process_particle_batch_em(batch, seed):
        perf_stats = {
            "em_pure_s": 0.0,
            "reaction_apply_s": 0.0,
            "checkpoint_bvh_s": 0.0,
            "final_bvh_s": 0.0,
        }
        field_provider = create_field_provider(external_field_cfg)
        reaction_model = create_reaction_model(reaction_model_cfg)

        # Accumulated sparse updates and impact records across all checkpoint passes
        all_sparse = [(None, None) for _ in face_counts]
        all_impacts = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]
        # Per-species sparse updates: {charge_state: [(idxs, vals) per object]}
        all_sparse_by_species = {}

        def _merge_sparse_list(target, source):
            for obj_idx, (idxs, vals) in enumerate(source):
                if idxs is None:
                    continue
                prev_idxs, prev_vals = target[obj_idx]
                if prev_idxs is None:
                    target[obj_idx] = (idxs, vals)
                else:
                    target[obj_idx] = (
                        np.concatenate([prev_idxs, idxs]),
                        np.concatenate([prev_vals, vals]),
                    )

        def _merge_into_all(sparse_chunk, imp_chunk, species_chunk):
            _merge_sparse_list(all_sparse, sparse_chunk)
            for obj_idx in range(num_objects):
                c = imp_chunk[obj_idx]
                all_impacts[obj_idx]['hit_count'] += c['hit_count']
                all_impacts[obj_idx]['data'].extend(c['data'])
            for cs, sp_list in species_chunk.items():
                if cs not in all_sparse_by_species:
                    all_sparse_by_species[cs] = [(None, None) for _ in face_counts]
                _merge_sparse_list(all_sparse_by_species[cs], sp_list)

        def _bvh_checkpoint_callback(segments_chunk):
            """Run BVH on a checkpoint segment chunk; accumulate results; return hit PIDs."""
            sparse_chunk, imp_chunk, hit_pids, species_chunk = intersect_trajectory_segments_bvh(
                segments_chunk, intersector, face_offsets, face_counts, deposition_model,
                save_impact_flags=save_impact_flags,
            )
            _merge_into_all(sparse_chunk, imp_chunk, species_chunk)
            return hit_pids

        run_stats = {}
        trajectory_segments = trace_particle_batch_em_only(
            batch,
            field_provider,
            reaction_model,
            em_step_length_m,
            em_max_steps,
            em_max_distance_m,
            em_min_energy_ev,
            bounding_box_min_corner_m,
            bounding_box_max_corner_m,
            seed=seed,
            bvh_checkpoint_steps=bvh_checkpoint_steps,
            bvh_hit_callback=_bvh_checkpoint_callback if bvh_checkpoint_steps is not None else None,
            perf_stats=perf_stats if perf_enabled else None,
            min_steps_per_gyro_orbit=em_min_steps_per_gyro_orbit,
            run_stats=run_stats,
        )

        # Final BVH pass on the surviving segments (last checkpoint interval or all if no checkpoints)
        final_bvh_started_at = time.perf_counter()
        final_sparse, final_impacts, final_hit_pids, final_species = intersect_trajectory_segments_bvh(
            trajectory_segments, intersector, face_offsets, face_counts, deposition_model,
            save_impact_flags=save_impact_flags,
        )
        # Particles that ran out of steps without hitting anything are lost.
        still_flying = run_stats.get("active_at_end")
        n_out_of_steps = (len(set(still_flying.tolist()) - set(final_hit_pids))
                          if still_flying is not None else 0)
        perf_stats["out_of_steps"] = n_out_of_steps
        if perf_enabled:
            perf_stats["final_bvh_s"] += time.perf_counter() - final_bvh_started_at
        _merge_into_all(final_sparse, final_impacts, final_species)

        # Consolidate duplicate triangle indices within each object
        consolidated = []
        for obj_idx in range(num_objects):
            idxs, vals = all_sparse[obj_idx]
            if idxs is None or idxs.size == 0:
                consolidated.append((None, None))
                continue
            order = np.argsort(idxs)
            idxs_s = idxs[order]
            vals_s = vals[order]
            u_idxs, inv = np.unique(idxs_s, return_inverse=True)
            u_vals = np.zeros(u_idxs.size, dtype=np.float32)
            np.add.at(u_vals, inv, vals_s)
            consolidated.append((u_idxs, u_vals))

        # Consolidate per-species sparse updates
        consolidated_by_species = {}
        for cs, sp_list in all_sparse_by_species.items():
            cons = []
            for obj_idx in range(num_objects):
                idxs, vals = sp_list[obj_idx]
                if idxs is None or idxs.size == 0:
                    cons.append((None, None))
                    continue
                order = np.argsort(idxs)
                idxs_s = idxs[order]
                vals_s = vals[order]
                u_idxs, inv = np.unique(idxs_s, return_inverse=True)
                u_vals = np.zeros(u_idxs.size, dtype=np.float32)
                np.add.at(u_vals, inv, vals_s)
                cons.append((u_idxs, u_vals))
            consolidated_by_species[cs] = cons

        return consolidated, all_impacts, perf_stats, consolidated_by_species

    perf_totals = {
        "em_pure_s": 0.0,
        "reaction_apply_s": 0.0,
        "checkpoint_bvh_s": 0.0,
        "final_bvh_s": 0.0,
    }

    total_out_of_steps = 0

    # Cap concurrency: EM workers are memory-heavy (trajectory storage)
    effective_workers = min(n_jobs, max(total_batches, 1), 4)
    # Feed batches lazily: only keep 'effective_workers' batches in flight at once
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        batch_iter = _iter_particle_batches(particle_sources_list, particle_batch_size)
        batches_submitted = 0

        # Prime the pool with initial batches
        for batch in batch_iter:
            futures[executor.submit(_process_particle_batch_em, batch, batches_submitted)] = None
            batches_submitted += 1
            if batches_submitted >= effective_workers * 2:
                break

        with tqdm(total=total_batches, desc="Processing Particle Batches") as pbar:
            while futures:
                done_futures = []
                for future in as_completed(futures):
                    done_futures.append(future)
                    break  # process one at a time to keep memory bounded

                for future in done_futures:
                    del futures[future]
                    sparse_updates, chunk_impacts, chunk_perf, chunk_species = future.result()
                    for obj_idx, (idxs, vals) in enumerate(sparse_updates):
                        if idxs is None:
                            continue
                        np.add.at(final_deposited_power[obj_idx], idxs, vals)
                    # Accumulate per-species power
                    for cs, sp_list in chunk_species.items():
                        if cs not in per_species_power:
                            per_species_power[cs] = [np.zeros(count, dtype=np.float32) for count in face_counts]
                        for obj_idx, (idxs, vals) in enumerate(sp_list):
                            if idxs is None:
                                continue
                            np.add.at(per_species_power[cs][obj_idx], idxs, vals)
                    _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records)
                    total_out_of_steps += chunk_perf.get("out_of_steps", 0)
                    if perf_enabled:
                        for key in perf_totals:
                            perf_totals[key] += chunk_perf.get(key, 0.0)
                    pbar.update(1)

                    # Submit next batch from the generator to keep pipeline full
                    try:
                        next_batch = next(batch_iter)
                        futures[executor.submit(_process_particle_batch_em, next_batch, batches_submitted)] = None
                        batches_submitted += 1
                    except StopIteration:
                        pass

    total_deposited = sum(arr.sum() for arr in final_deposited_power)
    print(f"Total power deposited: {total_deposited:.2f} W")
    if total_out_of_steps:
        print(f"  WARNING: {total_out_of_steps:,} of {total_particles:,} particles were still in "
              f"flight when EM_MAX_STEPS ({int(em_max_steps)}) ran out and deposited nothing. "
              "Increase EM_MAX_STEPS (the adaptive step uses more steps in strong B), "
              "or set EM_MAX_DISTANCE_M / the bounding box to stop them on purpose.")
    if per_species_power:
        for cs in sorted(per_species_power.keys()):
            sp_total = sum(arr.sum() for arr in per_species_power[cs])
            label = {-1: "H-", 0: "H0", 1: "H+"}.get(cs, f"q={cs}")
            if total_deposited > 0:
                print(f"  Species {label}: {sp_total:.2f} W ({100*sp_total/total_deposited:.1f}%)")
            else:
                print(f"  Species {label}: {sp_total:.2f} W")

    for obj_idx in range(num_objects):
        if save_impact_flags[obj_idx] and impact_data[obj_idx]['total_hits'] > 0:
            d = impact_data[obj_idx]
            print(f"  Impact data for object {obj_idx}: {d['stored_hits']} records stored out of {d['total_hits']} total hits.")

    if perf_enabled:
        total_traced = (perf_totals['em_pure_s'] + perf_totals['reaction_apply_s']
                        + perf_totals['checkpoint_bvh_s'] + perf_totals['final_bvh_s'])
        print("Performance timing summary (aggregated across worker batches):")
        print(f"  - EM pure (Boris + bookkeeping): {perf_totals['em_pure_s']:.2f} s")
        print(f"  - Reaction apply: {perf_totals['reaction_apply_s']:.2f} s")
        print(f"  - BVH checkpoint callbacks: {perf_totals['checkpoint_bvh_s']:.2f} s")
        print(f"  - Final BVH passes: {perf_totals['final_bvh_s']:.2f} s")
        print(f"  - Total traced: {total_traced:.2f} s")

    return final_deposited_power, impact_data, per_species_power





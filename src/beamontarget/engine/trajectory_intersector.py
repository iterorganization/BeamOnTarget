# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""
Phase 2: Deferred trajectory-to-BVH intersection checking.
Takes trajectory segments from Phase 1 and finds first hits against geometry.
"""

import numpy as np


def _build_sparse_updates(object_indices, local_tri_indices, deposit, num_objects):
    """Build per-object sparse (face_index, power_value) arrays from hit data."""
    sparse = [(None, None) for _ in range(num_objects)]
    if len(deposit) == 0:
        return sparse
    for obj_idx in np.unique(object_indices):
        mask = object_indices == obj_idx
        idxs = local_tri_indices[mask]
        vals = deposit[mask]
        if idxs.size > 0:
            order = np.argsort(idxs)
            idxs_sorted = idxs[order]
            vals_sorted = vals[order]
            unique_idxs = []
            unique_vals = []
            prev = None
            acc = 0.0
            for i in range(idxs_sorted.size):
                ii = int(idxs_sorted[i])
                vv = float(vals_sorted[i])
                if prev is None:
                    prev = ii
                    acc = vv
                elif ii == prev:
                    acc += vv
                else:
                    unique_idxs.append(prev)
                    unique_vals.append(acc)
                    prev = ii
                    acc = vv
            if prev is not None:
                unique_idxs.append(prev)
                unique_vals.append(acc)
            sparse[obj_idx] = (
                np.asarray(unique_idxs, dtype=np.int32),
                np.asarray(unique_vals, dtype=np.float32),
            )
    return sparse


def intersect_trajectory_segments_bvh(
    trajectory_segments,
    intersector,
    face_offsets,
    face_counts,
    deposition_model,
    save_impact_flags=None,
):
    """
    Check trajectory segments against BVH and compute power deposition.
    
    Args:
        trajectory_segments: structured array from Phase 1 with fields:
            - start_pos, end_pos, velocity, mass_kg, charge_state, 
              kinetic_energy_ev, current_a, source_index
        intersector: trimesh BVH intersector
        face_offsets: cumulative face counts per object (for unmapping triangles to objects)
        face_counts: face count per object
        deposition_model: function(energy_eV) -> fraction_deposited
    
    Returns:
        (sparse_updates, impact_records, hit_info, sparse_by_species)
            sparse_updates: list of (index_array, value_array) tuples per object
            impact_records: list of dicts per object with 'hit_count' and 'data'
            hit_info: {particle_id: (step_index, position, charge_state,
                kinetic_energy_eV)} for every particle whose first impact is in
                these segments.  Iterating it yields the particle ids, so it can
                be used wherever a set of hit ids is expected; the details let
                the trajectory recorder end each polyline at its impact point.
            sparse_by_species: {charge_state: sparse_updates}
    """
    num_objects = len(face_counts)
    if save_impact_flags is None:
        save_impact_flags = [False] * num_objects
    sparse_updates = [(None, None) for _ in face_counts]
    impact_records = [{'hit_count': 0, 'data': []} for _ in range(num_objects)]

    if num_objects == 0 or len(trajectory_segments) == 0:
        # num_objects == 0 is a free-space run: EM integration still happens in
        # phase 1, there is simply no geometry for phase 2 to test against.
        return sparse_updates, impact_records, {}, {}

    # Keep only the earliest collision per particle after BVH testing.

    # Extract segments for BVH query
    ray_origins = trajectory_segments['start_pos'].astype(np.float64)
    ray_ends = trajectory_segments['end_pos'].astype(np.float64)
    segment_vectors = ray_ends - ray_origins
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    # Normalize to ray directions
    ray_directions = np.zeros_like(segment_vectors)
    valid_segs = segment_lengths > 1e-14
    ray_directions[valid_segs] = segment_vectors[valid_segs] / segment_lengths[valid_segs][:, np.newaxis]

    # BVH query — direct embree call with finite segment distances (dists).
    # Embree internally uses float32, which has limited precision for large
    # coordinates. Trimesh's _EmbreeWrap compensates by translating the
    # geometry so that min(vertices) maps to the origin, and then scaling
    # the result to maximise float32 precision. Ray origins and max-ray
    # distances (dists) must undergo the same transform so that they are
    # consistent with the stored BVH.
    #
    # By passing dists (= segment_length * scale), embree rejects triangles
    # beyond the segment during BVH traversal. This avoids shooting infinite
    # rays and post-filtering hits that fall outside the actual EM step.
    scene_wrap = intersector._scene
    scale = scene_wrap.scale
    scene_origin = scene_wrap.origin

    # Transform ray origins into embree's scaled coordinate system
    scaled_origins = (ray_origins - scene_origin) * scale
    # Transform segment lengths into the same scaled space; a small epsilon
    # is added because embree's tfar bound is exclusive (hit at exactly
    # tfar is not reported)
    scaled_dists = (segment_lengths + 1e-9) * scale

    # Query embree: returns triangle index per ray (-1 for miss)
    tri_ids = scene_wrap.scene.run(
        scaled_origins.astype(np.float32),
        ray_directions.astype(np.float32),
        dists=scaled_dists.astype(np.float32),
    )

    hit_mask = tri_ids != -1
    if not np.any(hit_mask):
        return sparse_updates, impact_records, {}, {}

    index_ray = np.where(hit_mask)[0]
    index_tri_global = tri_ids[hit_mask]

    # Compute hit locations via ray-plane intersection:
    # t = dot(v0 - origin, normal) / dot(direction, normal)
    # location = origin + t * direction
    mesh = intersector.mesh
    hit_origins = ray_origins[index_ray]
    hit_dirs = ray_directions[index_ray]
    tri_v0 = mesh.triangles[index_tri_global, 0, :]
    tri_normals = mesh.face_normals[index_tri_global]

    t_num = np.einsum('ij,ij->i', tri_v0 - hit_origins, tri_normals)
    t_den = np.einsum('ij,ij->i', hit_dirs, tri_normals)
    # Avoid division by zero for degenerate cases
    safe_den = np.where(np.abs(t_den) > 1e-30, t_den, 1e-30)
    t = t_num / safe_den
    locations = hit_origins + t[:, np.newaxis] * hit_dirs

    # Compute impact distances for earliest-hit selection
    impact_dist = np.abs(t)

    hit_particle_ids = trajectory_segments['particle_id'][index_ray]
    hit_step_indices = trajectory_segments['step_index'][index_ray]

    keep_mask = np.zeros(index_ray.shape[0], dtype=bool)
    earliest_hits = {}
    for idx, (particle_id, step_index, distance) in enumerate(zip(hit_particle_ids, hit_step_indices, impact_dist)):
        current = earliest_hits.get(int(particle_id))
        candidate = (int(step_index), float(distance), idx)
        if current is None or candidate[:2] < current[:2]:
            earliest_hits[int(particle_id)] = candidate
    for _, (_, _, idx) in earliest_hits.items():
        keep_mask[idx] = True

    index_ray = index_ray[keep_mask]
    index_tri_global = index_tri_global[keep_mask]
    hit_positions = locations[keep_mask]

    # Extract particle properties at hit
    hit_segments = trajectory_segments[index_ray]
    hit_velocities = hit_segments['velocity']
    hit_masses = hit_segments['mass_kg']
    hit_charges = hit_segments['charge_state']
    hit_kinetic_eV = hit_segments['kinetic_energy_ev']
    hit_currents = hit_segments['current_a']
    hit_source_indices = hit_segments['source_index']

    # Compute kinetic energy from velocity
    v_mag_sq = np.einsum("ij,ij->i", hit_velocities, hit_velocities)
    e_kinetic_eV = 0.5 * hit_masses * v_mag_sq / 1.602176634e-19

    # Deposition
    charge_div = np.maximum(np.abs(hit_charges), 1)
    power = e_kinetic_eV * hit_currents / charge_div
    frac = deposition_model(e_kinetic_eV)
    deposit = power * frac

    # Map to objects
    object_indices = np.searchsorted(face_offsets, index_tri_global, side='right') - 1
    local_tri_indices = index_tri_global - face_offsets[object_indices]

    # Build total sparse updates per object
    sparse_updates = _build_sparse_updates(object_indices, local_tri_indices, deposit, num_objects)

    # Build per-species sparse updates (keyed by charge_state)
    sparse_by_species = {}
    for cs in np.unique(hit_charges):
        cs_mask = hit_charges == cs
        sparse_by_species[int(cs)] = _build_sparse_updates(
            object_indices[cs_mask], local_tri_indices[cs_mask], deposit[cs_mask], num_objects,
        )

    # Collect impact records for flagged objects
    for obj_idx in np.unique(object_indices):
        mask = object_indices == obj_idx
        hit_count = int(mask.sum())
        impact_records[obj_idx]['hit_count'] += hit_count

        if not save_impact_flags[obj_idx]:
            continue

        local_hits = np.flatnonzero(mask)
        for h in local_hits:
            hv = hit_velocities[h]
            hv_norm = np.linalg.norm(hv)
            hv_dir = hv / hv_norm if hv_norm > 1e-12 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
            impact_records[obj_idx]['data'].append((
                int(hit_source_indices[h]),
                float(hit_masses[h]),
                int(hit_charges[h]),
                float(hit_positions[h, 0]),
                float(hit_positions[h, 1]),
                float(hit_positions[h, 2]),
                float(hv_dir[0]),
                float(hv_dir[1]),
                float(hv_dir[2]),
                float(e_kinetic_eV[h]),
                float(hit_currents[h]),
            ))

    # index_ray is already filtered to earliest-hit-per-particle at this point,
    # so there is one entry per particle: where and in which step it landed.
    hit_info = {
        int(pid): (int(step), position, int(charge), float(energy))
        for pid, step, position, charge, energy in zip(
            hit_segments['particle_id'], hit_segments['step_index'],
            hit_positions, hit_charges, e_kinetic_eV)
    }
    return sparse_updates, impact_records, hit_info, sparse_by_species

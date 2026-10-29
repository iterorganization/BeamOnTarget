# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# trajectory_export.py
"""
Record the trajectories of a sample of particles and write them as one
VTK PolyData (.vtp) file, one polyline per particle, for ParaView.

The sample is spread evenly over the beam origins (``source_index``) by
:class:`OriginQuotaTracker`, so every beamlet is represented rather than only
the ones processed first.  :class:`TrajectoryRecorder` collects the segments
of the selected particles from all batches (thread-safe) and, in EM mode,
cuts each polyline at the impact point found by the BVH pass.

Point arrays: ``ParticleID``, ``SourceIndex``, ``ChargeState``,
``KineticEnergy_eV`` (from the particle's velocity, so it follows
acceleration in electric fields) and ``StepIndex``.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pyvista as pv

_EV = 1.602176634e-19


class OriginQuotaTracker:
    """Per-origin quota of recorded trajectories, shared by all batches of a run.

    The total budget is split evenly over the distinct ``source_index``
    values, so each beam origin contributes about the same number of
    trajectories.  Thread-safe.
    """

    def __init__(self, source_indices, total_budget):
        unique = sorted(set(int(s) for s in source_indices))
        self.num_origins = max(1, len(unique))
        total_budget = int(total_budget) if total_budget else 0
        self.quota_per_origin = (max(1, total_budget // self.num_origins)
                                 if total_budget > 0 else 0)
        self._remaining = {src: self.quota_per_origin for src in unique}
        self._total_remaining = self.quota_per_origin * self.num_origins
        self._lock = threading.Lock()

    def reserve(self, source_indices_per_particle) -> set[int]:
        """Return the local particle ids of one batch that may be recorded.

        *source_indices_per_particle* is indexed by local particle id.  Each
        accepted particle uses up one unit of its origin's quota.
        """
        allowed = set()
        with self._lock:
            if self._total_remaining <= 0:
                return allowed
            for pid, src in enumerate(source_indices_per_particle):
                src = int(src)
                remaining = self._remaining.get(src, 0)
                if remaining > 0:
                    self._remaining[src] = remaining - 1
                    self._total_remaining -= 1
                    allowed.add(pid)
                    if self._total_remaining <= 0:
                        break
        return allowed


class TrajectoryRecorder:
    """Collects trajectory segments of the selected particles over a whole run.

    One instance is shared by all batches, so the run writes a single file.
    Particle ids are made globally unique with a per-batch offset (batches
    reuse local ids 0..batch_size-1).  All methods are thread-safe.
    """

    def __init__(self):
        # global pid -> list of (points (n,3), charge (n), energy_eV (n), step (n), source (n))
        self._chunks: dict[int, list] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _segment_energy_ev(seg):
        """Kinetic energy from the velocity at the end of each segment.

        Falls back to the stored ``kinetic_energy_ev`` where the velocity is
        not set (ray mode).
        """
        v2 = np.einsum("ij,ij->i", seg["velocity"].astype(np.float64),
                       seg["velocity"].astype(np.float64))
        e = 0.5 * seg["mass_kg"] * v2 / _EV
        return np.where(v2 > 0.0, e, seg["kinetic_energy_ev"])

    def add_segments(self, segments, allowed_local_pids, particle_id_offset=0):
        """Add the segments of the allowed particles from one batch.

        *segments* is a SEGMENT_DTYPE array; *allowed_local_pids* the local
        particle ids to record (see :class:`OriginQuotaTracker`).  May be
        called several times per batch, e.g. once per BVH checkpoint.
        """
        if segments is None or len(segments) == 0 or not allowed_local_pids:
            return
        mask = np.isin(segments["particle_id"],
                       np.fromiter(allowed_local_pids, dtype=np.int64))
        if not np.any(mask):
            return
        seg = segments[mask]
        order = np.lexsort((seg["step_index"], seg["particle_id"]))
        seg = seg[order]
        energy = self._segment_energy_ev(seg)
        pids, starts = np.unique(seg["particle_id"], return_index=True)
        bounds = list(starts[1:]) + [len(seg)]

        with self._lock:
            for pid_local, a, b in zip(pids, starts, bounds):
                s = seg[a:b]
                e = energy[a:b]
                pid = int(pid_local) + particle_id_offset
                chunks = self._chunks.setdefault(pid, [])
                if not chunks:
                    # The polyline starts at the start of the first segment,
                    # with the particle's initial energy.
                    chunks.append((s["start_pos"][:1].astype(np.float64),
                                   s["charge_state"][:1].astype(np.int32),
                                   s["kinetic_energy_ev"][:1].astype(np.float64),
                                   s["step_index"][:1].astype(np.int32),
                                   s["source_index"][:1].astype(np.int32)))
                # The end of segment k is tagged step k + 1.
                chunks.append((s["end_pos"].astype(np.float64),
                               s["charge_state"].astype(np.int32), e,
                               s["step_index"].astype(np.int32) + 1,
                               s["source_index"].astype(np.int32)))

    def _joined(self, pid):
        chunks = self._chunks[pid]
        return tuple(np.concatenate([c[k] for c in chunks]) for k in range(5))

    def truncate_at_impact(self, hit_info, particle_id_offset=0):
        """End each recorded trajectory at its impact point.

        Phase 1 has no geometry, so a particle is stepped on until a BVH
        checkpoint retires it, past the surface it struck.  *hit_info* is
        ``{local particle id: (step_index, position, charge, energy_eV)}`` from
        ``intersect_trajectory_segments_bvh``: every point after the start of
        the hit segment is dropped and the impact point is appended.
        Particles without a hit are left alone.
        """
        if not hit_info:
            return
        with self._lock:
            for local_pid, (step_index, position, charge, energy_ev) in hit_info.items():
                pid = int(local_pid) + particle_id_offset
                if pid not in self._chunks:
                    continue
                pts, cs, en, st, src = self._joined(pid)
                keep = int(np.count_nonzero(st <= step_index))
                if keep == 0:
                    continue
                self._chunks[pid] = [(
                    np.vstack([pts[:keep], np.asarray(position, dtype=np.float64)[None, :]]),
                    np.append(cs[:keep], int(charge)),
                    np.append(en[:keep], float(energy_ev)),
                    np.append(st[:keep], int(step_index) + 1),
                    np.append(src[:keep], src[0]),
                )]

    @property
    def n_particles_recorded(self) -> int:
        return len(self._chunks)

    def build_polydata(self) -> pv.PolyData | None:
        """One polyline per particle with at least two points; None if empty."""
        pts_all, lines, arrays = [], [], {k: [] for k in
                                          ("ParticleID", "SourceIndex", "ChargeState",
                                           "KineticEnergy_eV", "StepIndex")}
        offset = 0
        with self._lock:
            for pid in sorted(self._chunks):
                pts, cs, en, st, src = self._joined(pid)
                n = len(pts)
                if n < 2:
                    continue
                pts_all.append(pts)
                lines.append(np.concatenate([[n], np.arange(offset, offset + n)]))
                arrays["ParticleID"].append(np.full(n, pid, dtype=np.int64))
                arrays["SourceIndex"].append(src)
                arrays["ChargeState"].append(cs)
                arrays["KineticEnergy_eV"].append(en)
                arrays["StepIndex"].append(st)
                offset += n
        if not pts_all:
            return None
        poly = pv.PolyData(np.vstack(pts_all))
        poly.lines = np.concatenate(lines).astype(np.int64)
        for name, parts in arrays.items():
            poly.point_data[name] = np.concatenate(parts)
        return poly

    def save_vtp(self, path) -> bool:
        """Write the trajectories to *path*; False if nothing was recorded."""
        poly = self.build_polydata()
        if poly is None:
            return False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        poly.save(path, binary=True)
        return True

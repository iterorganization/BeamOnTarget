# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
# physical_smooth.py
"""
Energy-conserving estimate of the surface power density from the Monte Carlo
tally, with a per-cell uncertainty ("physical" smoothing method).

The radius method in ``smooth_results`` averages each cell over a ball of
neighbours.  This method instead treats the problem as estimating a continuous
surface power-density field q(x) [W/m^2] from a noisy Monte Carlo tally, and
derives the error bar from the same estimator.

1. Energy is conserved.
   The estimator solves ``(M + L) rho' = M rho = P`` on the dual mesh graph,
   with ``M = diag(area)`` and ``L`` a symmetric finite-volume (two-point flux)
   Laplacian.  ``L`` has zero row sums, so ``sum_i A_i rho'_i == sum_i P_i``
   at any bandwidth, on any mesh, to the conjugate-gradient tolerance (the
   achieved error is reported).  The "gather" form ``sum_B P / sum_B A`` used
   by the radius method is normalised per target cell and therefore does not
   conserve the total.

2. No sharp cuts.
   ``(M + L)`` is an M-matrix, so its inverse is non-negative: no ringing, no
   negative overshoot, and a zero-count cell surrounded by hit cells relaxes to
   the local flux.  Diffusion runs along the surface, so power does not leak
   across a gap or onto a shadowed back face the way a Euclidean ball does.

3. The error estimate comes from the smoothing.
   The statistical error is obtained by parametric bootstrap through the same
   operator, so it stays correct under the adaptive bandwidth and the
   feature-angle gate.

Adaptive bandwidth
------------------
``target_rel_error`` is a ceiling on the statistical error.  Every cell starts
at one cell width (nothing below that is resolvable); each pass measures the
error that bandwidth delivers and widens only the cells above the ceiling, by
achieved/target (sigma ~ 1/h), up to ``max_bandwidth_m``.  Well-sampled peaks
keep full resolution; sparse tails get wide kernels.

Monte Carlo counts
------------------
``Deposited_Power_W`` is quantised: ``P = w * N`` with ``w`` the weight of one
macro-particle and ``N`` an integer count.  ``w`` is recovered as the smallest
positive deposit, which gives the per-cell particle count and hence a Poisson
error with no extra metadata.  The Poisson model is calibrated against the
data with an adjacent-pair dispersion statistic.  If the tally is not
quantised (e.g. per-beamlet weights differ, or the deposited energy varies
per particle) a warning is printed and the error bar is less reliable; the
smoothed density and its conservation are unaffected.

Output cell arrays
------------------
``Power_Density_W_m2``             smoothed density (replaced).
``Deposited_Power_W``              ``area * density``, so both arrays agree and
                                   the total is preserved (replaced).
``Power_Density_W_m2_<sp>`` and    per species, through the same final
``Deposited_Power_W_<sp>``         operator; the species sum to the total.
``Power_Density_sigma_stat_W_m2``  1-sigma counting error (bootstrap).
``Power_Density_bias_W_m2``        smoothing error, ``|rho_h - rho_2h|``.
``Power_Density_p95_W_m2``         ``rho + 1.645 * sqrt(sigma^2 + bias^2)``.
``Smoothing_N_eff``                effective macro-particles behind each cell.
``Smoothing_bandwidth_m``          bandwidth applied per cell.

Field data records the input/output power, the conservation error, the
dispersion factor, the macro-particle weight and chi^2/dof.  chi^2/dof is the
Pearson fit of the raw counts to the counts the smoothed field predicts: ~1
means as much smoothing as the statistics support, >1 means real structure is
being erased, <1 means the field still follows the noise.

The input ``Power_Density_W_m2`` is ignored; density is recomputed from the
raw tally and the mesh areas.
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv

POWER_ARRAY = "Deposited_Power_W"
DENSITY_ARRAY = "Power_Density_W_m2"
SIGMA_ARRAY = "Power_Density_sigma_stat_W_m2"
BIAS_ARRAY = "Power_Density_bias_W_m2"
P95_ARRAY = "Power_Density_p95_W_m2"
NEFF_ARRAY = "Smoothing_N_eff"
BW_ARRAY = "Smoothing_bandwidth_m"

# The two physical choices; exposed through the config file.
DEFAULT_TARGET_REL_ERROR = 0.05
"""Ceiling on the 1-sigma statistical error per cell (dimensionless)."""
DEFAULT_MAX_BANDWIDTH_M = 0.04
"""Hard cap on the smoothing length [m]: the smallest hot spot that must
survive, or the component's thermal diffusion length."""

# Numerical settings; rarely need changing.
DEFAULT_MIN_BANDWIDTH_FACTOR = 1.0
"""Bandwidth floor as a multiple of sqrt(cell area).  Raising it does not fill
holes (the statistical rule already widens sparse regions) and only erases
real structure."""
DEFAULT_FEATURE_ANGLE_DEG = 30.0
DEFAULT_FEATURE_ANGLE_SOFT_DEG = 15.0
"""Dihedral gate: full conductance below the soft angle, ramping smoothly to
zero at the feature angle.  Stops power diffusing around a corner onto a
shadowed face.  Mesh-resolution dependent: on a fine mesh a tight fillet
turns several degrees per cell."""
DEFAULT_SUBSTEPS = 3
"""Implicit Euler substeps.  One step leaves the Bessel-K0 kernel of the
resolvent, which is too peaked; three give a near-Gaussian kernel."""
DEFAULT_BOOTSTRAP_REPLICAS = 32
"""Poisson resampling replicas for the error bar (~13 % relative accuracy on
sigma); 0 disables it."""
DEFAULT_REFINE_PASSES = 1
DEFAULT_REFINE_PROBES = 4
"""Bandwidth self-calibration: each pass runs a short bootstrap of this many
replicas and widens cells by achieved/target error."""
DEFAULT_BOOTSTRAP_BLOCK = 4
"""Replicas solved together as one multi-RHS system; peak memory scales as
block x non-zeros x 8 bytes."""
DEFAULT_BAND_FACTOR = 3.0
"""The solve is restricted to this many max-bandwidths around the signal,
with a no-flux boundary, so conservation stays exact."""
DEFAULT_CG_TOL = 1e-7
DEFAULT_CG_MAXITER = 400

# Median of a chi^2_1 variate: turns the median adjacent-pair statistic into a
# robust dispersion factor.  The mean is unusable here because it is dominated
# by the minority of pairs that straddle genuine shadow edges.
_CHI2_1_MEDIAN = 0.4549364231195728
_Z95 = 1.6448536269514722


# ---------------------------------------------------------------------------
#  Geometry (depends only on the mesh)
# ---------------------------------------------------------------------------

def precompute_diffusion_geometry(
        mesh: pv.PolyData,
        feature_angle_deg: float = DEFAULT_FEATURE_ANGLE_DEG,
        feature_angle_soft_deg: float = DEFAULT_FEATURE_ANGLE_SOFT_DEG,
) -> dict | None:
    """Build the dual-graph diffusion operator for a triangle mesh.

    The result depends only on topology and vertex positions, so it can be
    reused for every result file of the same component.  The returned
    transmissibilities carry the geometric factor only (``l_ij / d_ij`` times
    the feature-angle gate); the solve multiplies them by the local
    diffusivity ``kappa = h^2 / 4``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Triangle mesh whose cells are all polygons.
    feature_angle_deg, feature_angle_soft_deg : float
        Dihedral gate.  Conductance across an edge is 1 below the soft angle,
        ramps smoothly (smoothstep) to 0 at the feature angle, and is 0 above.
        The ramp rather than a hard cut avoids a discontinuity at the gate.

    Returns
    -------
    dict or None
        Operator data, or None if the mesh is not a pure triangle mesh.
    """
    if mesh.n_cells == 0 or not mesh.is_all_triangles:
        print("  - [physical] Mesh is not purely triangular - not supported.")
        return None
    tri = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    n_cells = len(tri)
    if n_cells != mesh.n_cells:
        # Vertices, lines or strips would shift the cell-data indices.
        print("  - [physical] Mesh has non-polygon cells - not supported.")
        return None

    pts = np.asarray(mesh.points, dtype=np.float64)
    n_pts = len(pts)
    print(f"  - [physical] Mesh: {n_cells:,} triangles, {n_pts:,} points")

    # Areas, centroids, normals
    v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    centroids = (v0 + v1 + v2) / 3.0
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / twice_area[:, None]
    normals[~np.isfinite(normals).all(axis=1)] = 0.0

    n_degenerate = int((areas <= 0.0).sum())
    if n_degenerate:
        print(f"  - [physical] {n_degenerate:,} zero-area cell(s) excluded; "
              "they carry no energy.")

    # Manifold dual edges: an edge shared by exactly two triangles becomes one
    # dual-graph edge.
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    owner = np.tile(np.arange(n_cells, dtype=np.int64), 3)
    key = e[:, 0] * np.int64(n_pts) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    e = e[order]
    dup = np.where(key[1:] == key[:-1])[0]
    # Drop non-manifold edges (three or more owners): a run longer than two
    # would otherwise contribute several overlapping pairs.
    if len(dup) > 1:
        runs_ok = np.ones(len(dup), dtype=bool)
        triple = np.where(dup[1:] == dup[:-1] + 1)[0]
        if len(triple):
            runs_ok[triple] = False
            runs_ok[triple + 1] = False
        dup = dup[runs_ok]
    ei, ej = owner[dup], owner[dup + 1]
    shared = e[dup]

    edge_len = np.linalg.norm(pts[shared[:, 0]] - pts[shared[:, 1]], axis=1)
    dist = np.linalg.norm(centroids[ei] - centroids[ej], axis=1)

    # Feature-angle gate (smooth ramp, never a hard cut)
    dot = np.clip(np.einsum("ij,ij->i", normals[ei], normals[ej]), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    hard, soft = float(feature_angle_deg), float(feature_angle_soft_deg)
    if hard <= soft:
        hard = soft + 1e-6
    x = np.clip((angle - soft) / (hard - soft), 0.0, 1.0)
    gate = 1.0 - x * x * (3.0 - 2.0 * x)

    # Geometric transmissibility (dimensionless); kappa [m^2] is applied later.
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_geom = gate * edge_len / dist
    bad = ~np.isfinite(trans_geom) | (dist <= 0.0) | (areas[ei] <= 0.0) | (areas[ej] <= 0.0)
    trans_geom[bad] = 0.0

    keep = trans_geom > 0.0
    ei, ej, trans_geom, dist = ei[keep], ej[keep], trans_geom[keep], dist[keep]
    print(f"  - [physical] Dual graph: {len(ei):,} active edges "
          f"({int((gate <= 0.0).sum()):,} closed by the >{hard:g} deg feature gate)")

    return {
        "n_cells": n_cells,
        "n_pts": n_pts,
        "areas": areas,
        "centroids": centroids,
        "edge_i": ei,
        "edge_j": ej,
        "trans_geom": trans_geom,
        "edge_dist": dist,
        "n_degenerate": n_degenerate,
    }


# ---------------------------------------------------------------------------
#  Sparse linear algebra (numpy only)
# ---------------------------------------------------------------------------

def _build_csr_structure(ei, ej, n):
    """Symmetric edge list -> CSR structure, independent of the edge weights.

    Built once and reused for the main, bias and bootstrap operators, which
    differ only in their weights (see ``_csr_values``).
    """
    src = np.concatenate([ei, ej])
    dst = np.concatenate([ej, ei])
    order = np.argsort(src, kind="stable")
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src[order], minlength=n), out=offsets[1:])
    empty = offsets[1:] == offsets[:-1]
    return offsets, dst[order], order, empty


def _csr_values(trans, perm):
    """Edge weights in CSR order for a structure from ``_build_csr_structure``."""
    return np.concatenate([trans, trans])[perm]


def _row_reduce(values, offsets, empty):
    """Per-row sums of a CSR value array; ``values`` may carry a column axis.

    ``np.add.reduceat`` rejects a start index equal to ``len(values)``, which an
    isolated trailing row produces, so starts are clamped and rows without
    entries are zeroed afterwards.
    """
    if len(values) == 0:
        return np.zeros((len(offsets) - 1,) + values.shape[1:])
    starts = np.minimum(offsets[:-1], len(values) - 1)
    acc = np.add.reduceat(values, starts, axis=0)
    if empty.any():
        acc[empty] = 0.0
    return acc


def _spmv(x, diag, offsets, nbr, val, empty):
    """``(M + L) @ X`` for a single vector or an (n, k) block."""
    if x.ndim == 1:
        return diag * x - _row_reduce(val * x[nbr], offsets, empty)
    return diag[:, None] * x - _row_reduce(val[:, None] * x[nbr], offsets, empty)


def _cg(rhs, diag, offsets, nbr, val, empty, n, tol, maxiter, x0=None):
    """Jacobi-preconditioned conjugate gradient for the SPD operator (M + L).

    Returns ``(x, iterations, relative_residual)``.
    """
    block = rhs.ndim == 2
    if x0 is None:
        x = np.zeros_like(rhs)
    else:
        x = np.broadcast_to(x0[:, None] if block and x0.ndim == 1 else x0,
                            rhs.shape).copy()

    def _dot(a, b):
        """Column-wise inner product, or a scalar for a single vector."""
        return (a * b).sum(axis=0) if block else float(a @ b)

    nrm = np.sqrt(_dot(rhs, rhs))
    if not np.any(nrm > 0.0):
        return np.zeros_like(rhs), 0, 0.0
    nrm = np.where(nrm > 0.0, nrm, 1.0) if block else max(nrm, 1e-300)

    r = rhs - _spmv(x, diag, offsets, nbr, val, empty)
    minv = np.where(diag > 0.0, 1.0 / np.where(diag > 0.0, diag, 1.0), 0.0)
    mv = minv[:, None] if block else minv
    z = mv * r
    p = z.copy()
    rz = _dot(r, z)
    resid = float(np.max(np.sqrt(_dot(r, r)) / nrm))
    for k in range(maxiter):
        if resid < tol:
            return x, k, resid
        if not np.any(rz != 0.0):
            break
        ap = _spmv(p, diag, offsets, nbr, val, empty)
        pap = _dot(p, ap)
        if np.any(pap <= 0.0):
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        z = mv * r
        rz_new = _dot(r, z)
        # A converged column has rz == 0; its update is zero either way, so
        # guard the division rather than dropping the column.
        beta = (np.divide(rz_new, rz, out=np.zeros_like(rz_new), where=rz != 0.0)
                if block else rz_new / rz)
        p = z + beta * p
        rz = rz_new
        resid = float(np.max(np.sqrt(_dot(r, r)) / nrm))
    return x, maxiter, resid


def _diffuse(power, areas, csr, trans, n, substeps, tol, maxiter, x0=None):
    """Apply the conservative diffusion operator to a power vector or block.

    Solves ``(M + L/m) rho <- M rho_prev`` m times, so the kernel second
    moment is ``<r^2> = 4 kappa`` regardless of m.  ``power`` is the per-cell
    power [W]; the result is the density [W/m^2], with
    ``sum(areas * result) == sum(power)`` to the CG tolerance.  ``x0``
    warm-starts the first substep.
    """
    offsets, nbr, perm, empty = csr
    val = _csr_values(trans / float(substeps), perm)
    diag = areas + _row_reduce(val, offsets, empty)
    rhs = np.array(power, dtype=np.float64, copy=True)
    block = rhs.ndim == 2
    mass = areas[:, None] if block else areas
    if x0 is None:
        rho = np.zeros_like(rhs)
    else:
        rho = np.broadcast_to(x0[:, None] if block and x0.ndim == 1 else x0,
                              rhs.shape).copy()
    it_total, worst = 0, 0.0
    for _ in range(substeps):
        rho, iters, res = _cg(rhs, diag, offsets, nbr, val, empty,
                              n, tol, maxiter, x0=rho)
        it_total += iters
        worst = max(worst, res)
        rhs = mass * rho           # M rho becomes the next right-hand side
    return rho, it_total, worst


# ---------------------------------------------------------------------------
#  Monte Carlo statistics
# ---------------------------------------------------------------------------

def recover_particle_counts(power: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Recover per-cell macro-particle counts from a quantised power tally.

    Parameters
    ----------
    power : numpy.ndarray
        ``Deposited_Power_W`` per cell.

    Returns
    -------
    counts : numpy.ndarray
        ``P / w`` per cell.
    weight_w : float
        Macro-particle weight ``w`` [W], the smallest positive deposit.
    quality : float
        Fraction of hit cells whose ``P / w`` is within 0.01 of an integer.
        Below ~0.5 the tally is not quantised and the Poisson error model is
        unreliable.
    """
    pos = power[power > 0.0]
    if len(pos) == 0:
        return np.zeros_like(power), 0.0, 0.0
    w = float(pos.min())
    ratio = pos / w
    quality = float((np.abs(ratio - np.round(ratio)) < 0.01).mean())
    counts = np.maximum(power / w, 0.0)
    return counts, w, quality


def estimate_dispersion(
        counts: np.ndarray,
        areas: np.ndarray,
        ei: np.ndarray,
        ej: np.ndarray,
        min_pairs: int = 500,
) -> tuple[float, int]:
    """Calibrate the counting-noise model against the data.

    For two adjacent cells under a locally constant flux,
    ``N_i | (N_i + N_j) = n ~ Binomial(n, p)`` with ``p = A_i / (A_i + A_j)``,
    so ``T = (N_i - n p)^2 / (n p (1 - p))`` is chi^2_1 under Poisson
    statistics.  The median of T is used, because the mean is dominated by
    pairs that straddle genuine shadow edges.

    Returns
    -------
    factor : float
        1 for textbook Poisson, >1 if overdispersed; 1.0 with too few pairs.
    n_pairs : int
        Number of pairs used.
    """
    n = counts[ei] + counts[ej]
    denom = areas[ei] + areas[ej]
    usable = (n >= 10.0) & (denom > 0.0)
    if int(usable.sum()) < min_pairs:
        return 1.0, int(usable.sum())
    p = areas[ei][usable] / denom[usable]
    nn = n[usable]
    var = nn * p * (1.0 - p)
    ok = var > 0.0
    if int(ok.sum()) < min_pairs:
        return 1.0, int(ok.sum())
    t_stat = (counts[ei][usable][ok] - nn[ok] * p[ok]) ** 2 / var[ok]
    factor = float(np.median(t_stat) / _CHI2_1_MEDIAN)
    # Guard against a pathological calibration on tiny samples.
    factor = float(np.clip(factor, 0.25, 10.0))
    return factor, int(ok.sum())


def _bootstrap_sigma(rho, counts, areas, csr, trans, n, substeps,
                     quantum, dispersion, replicas, tol, maxiter, seed,
                     block_size):
    """1-sigma statistical error of the smoothed field, by parametric bootstrap.

    Each replica resamples the counts (Poisson, widened by the dispersion
    factor) and is pushed through the identical operator; the spread of the
    replicas is the error.  Replicas are warm-started from the unperturbed
    solution, so CG converges in a fraction of the iterations.
    """
    rng = np.random.default_rng(seed)
    lam = np.maximum(counts, 0.0) / dispersion
    mean_acc = np.zeros(n)
    sq_acc = np.zeros(n)
    done = 0
    while done < replicas:
        k = min(max(1, int(block_size)), replicas - done)
        p_rep = (rng.poisson(np.repeat(lam[:, None], k, axis=1))
                 .astype(np.float64) * (dispersion * quantum))
        rho_rep, _, _ = _diffuse(p_rep, areas, csr, trans, n,
                                 substeps, tol, maxiter, x0=rho)
        mean_acc += rho_rep.sum(axis=1)
        sq_acc += (rho_rep * rho_rep).sum(axis=1)
        done += k
    mean_acc /= replicas
    var = np.maximum(sq_acc / replicas - mean_acc * mean_acc, 0.0)
    return np.sqrt(var * replicas / max(replicas - 1, 1))


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def apply_physical_smoothing(
        mesh: pv.PolyData,
        target_rel_error: float = DEFAULT_TARGET_REL_ERROR,
        max_bandwidth_m: float = DEFAULT_MAX_BANDWIDTH_M,
        *,
        geometry: dict | None = None,
        min_bandwidth_factor: float = DEFAULT_MIN_BANDWIDTH_FACTOR,
        feature_angle_deg: float = DEFAULT_FEATURE_ANGLE_DEG,
        feature_angle_soft_deg: float = DEFAULT_FEATURE_ANGLE_SOFT_DEG,
        substeps: int = DEFAULT_SUBSTEPS,
        bootstrap_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
        refine_passes: int = DEFAULT_REFINE_PASSES,
        refine_probes: int = DEFAULT_REFINE_PROBES,
        bootstrap_block: int = DEFAULT_BOOTSTRAP_BLOCK,
        band_factor: float = DEFAULT_BAND_FACTOR,
        cg_tol: float = DEFAULT_CG_TOL,
        cg_maxiter: int = DEFAULT_CG_MAXITER,
        seed: int = 12345,
) -> tuple[pv.PolyData, dict | None]:
    """Estimate the power-density field and its uncertainty.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Result mesh with a ``Deposited_Power_W`` cell array.  Not modified.
    target_rel_error : float
        Ceiling on the 1-sigma statistical error per cell.
    max_bandwidth_m : float
        Maximum smoothing length [m].
    geometry : dict, optional
        Output of ``precompute_diffusion_geometry`` for this mesh, to reuse it
        across files of the same component.
    seed : int
        Random seed for the bootstrap, so results are reproducible.

    Other parameters are the numerical settings documented at module level.

    Returns
    -------
    mesh : pyvista.PolyData
        A smoothed copy (see the module docstring for the arrays written), or
        the input unchanged if it cannot be processed.
    stats : dict or None
        Summary for the smoothing log, in the same layout as
        ``smooth_results.apply_smoothing`` plus method-specific keys; None if
        the mesh has no ``Deposited_Power_W``.
    """
    if POWER_ARRAY not in mesh.cell_data:
        return mesh, None
    t0 = time.time()
    out = mesh.copy(deep=True)

    geo = geometry
    if geo is None or geo["n_cells"] != mesh.n_cells:
        geo = precompute_diffusion_geometry(mesh, feature_angle_deg, feature_angle_soft_deg)
    if geo is None:
        return mesh, None

    n_all = geo["n_cells"]
    areas_all = geo["areas"]
    power_all = np.asarray(mesh.cell_data[POWER_ARRAY], dtype=np.float64)
    total_in = float(power_all.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_density = np.where(areas_all > 0.0, power_all / np.where(areas_all > 0.0, areas_all, 1.0), 0.0)
    peak_before = float(raw_density.max()) if raw_density.size else 0.0

    species = [k[len(POWER_ARRAY) + 1:] for k in mesh.cell_data.keys()
               if k.startswith(POWER_ARRAY + "_")]

    stats = {
        "method": "physical",
        "n_cells": n_all,
        "n_smoothed": 0,
        "radius": None,
        "max_cell_area": None,
        "elapsed_s": 0.0,
        "total_power_W": total_in,
        "peak_density_before": peak_before,
        "peak_density_after": peak_before,
        "min_area_m2": float(areas_all.min()) if n_all else 0.0,
        "max_area_m2": float(areas_all.max()) if n_all else 0.0,
        "species": {},
        "target_rel_error": float(target_rel_error),
        "max_bandwidth_m": float(max_bandwidth_m),
    }
    if total_in <= 0.0:
        print("  - [physical] No deposited power in this file - output unchanged.")
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return mesh, stats

    counts_all, quantum, quality = recover_particle_counts(power_all)
    print(f"  - [physical] Macro-particle weight w = {quantum:.6e} W "
          f"(quantisation quality {quality * 100:.1f} %), "
          f"{counts_all.sum():.4g} macro-particles")
    if quality < 0.5:
        print("  - [physical] WARNING: the tally does not look quantised (P = w*N); "
              "the statistical error bar may be unreliable.")

    # Restrict the solve to a band around the signal.  Everything outside is
    # zero for a decaying kernel, and dropping the edges that leave the band
    # imposes a no-flux boundary, so no power can escape.
    active = _active_band(geo, power_all > 0.0, max_bandwidth_m,
                          int(np.ceil(band_factor)))
    n = int(active.sum())
    if n == 0:
        print("  - [physical] Empty active band - output unchanged.")
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return mesh, stats
    idx = np.where(active)[0]
    remap = np.full(n_all, -1, dtype=np.int64)
    remap[idx] = np.arange(n, dtype=np.int64)

    ei_a, ej_a = geo["edge_i"], geo["edge_j"]
    keep = active[ei_a] & active[ej_a]
    ei = remap[ei_a[keep]]
    ej = remap[ej_a[keep]]
    trans_geom = geo["trans_geom"][keep]

    areas = areas_all[idx]
    power = power_all[idx]
    counts = counts_all[idx]
    power_band = float(power.sum())
    print(f"  - [physical] Active band: {n:,} cells / {len(ei):,} edges "
          f"({100.0 * power_band / total_in:.4f} % of the power)")

    dispersion, n_pairs = estimate_dispersion(counts, areas, ei, ej)
    print(f"  - [physical] Dispersion factor {dispersion:.3f} from {n_pairs:,} "
          "adjacent pairs (1.0 = Poisson)")

    csr = _build_csr_structure(ei, ej, n)

    # Adaptive bandwidth: start every cell at the resolution floor (one cell
    # width) and only widen where the measured error exceeds the ceiling.
    h_floor = np.maximum(min_bandwidth_factor * np.sqrt(np.maximum(areas, 0.0)), 1e-9)
    h = np.clip(h_floor, None, max_bandwidth_m)

    def _trans_for(bandwidth):
        """Edge transmissibilities for a per-cell bandwidth field.

        The face coefficient is the harmonic mean of the two cells'
        diffusivity, which keeps the operator symmetric, so conservation and
        the M-matrix property survive the adaptivity.
        """
        kappa = 0.25 * bandwidth * bandwidth
        ki, kj = kappa[ei], kappa[ej]
        return trans_geom * np.where(
            ki + kj > 0.0, 2.0 * ki * kj / np.maximum(ki + kj, 1e-300), 0.0)

    # Each pass measures the error the current bandwidth delivers (short
    # bootstrap) and widens where it exceeds the target: sigma ~ 1/h, so
    # h <- h * achieved/target, clamped to >= 1 so the schedule is monotone.
    rho = None
    for it in range(max(0, int(refine_passes)) + 1):
        trans = _trans_for(h)
        print(f"  - [physical] Bandwidth pass {it}: median {np.median(h) * 1e3:.2f} mm, "
              f"p95 {np.percentile(h, 95) * 1e3:.2f} mm "
              f"(target {target_rel_error * 100:.1f} %, cap {max_bandwidth_m * 1e3:.0f} mm)")
        rho, iters, res = _diffuse(power, areas, csr, trans, n,
                                   substeps, cg_tol, cg_maxiter, x0=rho)
        if it >= int(refine_passes):
            break
        probe = _bootstrap_sigma(rho, counts, areas, csr, trans, n, substeps,
                                 quantum, dispersion, refine_probes,
                                 max(cg_tol, 1e-6), cg_maxiter, seed + 977 * it,
                                 bootstrap_block)
        with np.errstate(divide="ignore", invalid="ignore"):
            achieved = np.where(rho > 0.0, probe / np.maximum(rho, 1e-300), np.nan)
        signal = np.isfinite(achieved) & (power > 0.0)
        if not signal.any():
            break
        global_scale = float(np.median(achieved[signal]) / target_rel_error)
        scale = np.clip(np.where(np.isfinite(achieved),
                                 achieved / target_rel_error, global_scale),
                        1.0, 8.0)
        n_widened = int((scale > 1.01).sum())
        print(f"  - [physical]   achieved {np.median(achieved[signal]) * 100:.2f} % on "
              f"hit cells; widening {n_widened:,} cell(s) (max {scale.max():.2f}x)")
        if n_widened == 0:
            break
        h = np.clip(h * scale, h_floor, max_bandwidth_m)

    energy_out = float((areas * rho).sum())
    cons_err = (energy_out - power_band) / power_band if power_band else 0.0
    print(f"  - [physical] Main solve: {iters} CG iterations (residual {res:.2e}), "
          f"conservation error {cons_err * 100:+.3e} %")
    if res > cg_tol:
        print("  - [physical] WARNING: CG stopped at the iteration cap. Energy is "
              f"still conserved to {cons_err * 100:+.2e} %, but the field is "
              "under-relaxed; lower the maximum bandwidth.")

    # Smoothing bias: bandwidth sensitivity (Richardson)
    rho_2h, _, _ = _diffuse(power, areas, csr, 4.0 * trans, n,
                            substeps, cg_tol, cg_maxiter, x0=rho)
    bias = np.abs(rho - rho_2h)

    # Statistical error: parametric bootstrap through the same operator
    sigma = np.zeros(n)
    if bootstrap_replicas > 1:
        sigma = _bootstrap_sigma(rho, counts, areas, csr, trans, n, substeps,
                                 quantum, dispersion, bootstrap_replicas,
                                 max(cg_tol, 1e-7), cg_maxiter, seed,
                                 bootstrap_block)
        hit = power > 0
        print(f"  - [physical] Bootstrap: {bootstrap_replicas} replicas, median "
              f"relative sigma {_safe_rel(np.median(sigma[hit]), np.median(rho[hit]))}, "
              f"median relative bias {_safe_rel(np.median(bias[hit]), np.median(rho[hit]))}")

    # N_eff = (rho/sigma)^2, guarded: sigma can underflow to 0 in cells the
    # bootstrap never perturbs.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(sigma > 0.0, rho / np.maximum(sigma, 1e-300), 0.0)
    ratio = np.clip(np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e9)
    n_eff = ratio * ratio

    # chi^2/dof: Pearson fit of the raw counts against the counts the smoothed
    # field predicts, lambda_i = rho_i A_i / w.  Zero-count cells are included:
    # testing only hit cells conditions on an upward fluctuation.
    lam_pred = np.maximum(rho, 0.0) * areas / max(quantum, 1e-300)
    test = lam_pred > 0.0
    chi2_dof = (float((((counts[test] - lam_pred[test]) ** 2)
                       / (dispersion * lam_pred[test])).mean())
                if int(test.sum()) > 0 else float("nan"))
    verdict = ("over-smoothed: real structure is being erased" if chi2_dof > 1.5
               else "under-smoothed: the field still follows the noise" if chi2_dof < 0.5
               else "consistent with the statistics")
    print(f"  - [physical] chi2/dof {chi2_dof:.3f} over {int(test.sum()):,} cells ({verdict})")

    # Species: the same final operator.  It is linear, so the smoothed species
    # sum to the smoothed total (to the CG tolerance).
    species_rho = {}
    if species:
        p_sp = np.column_stack([np.asarray(mesh.cell_data[f"{POWER_ARRAY}_{sp}"],
                                           dtype=np.float64)[idx] for sp in species])
        rho_sp, _, _ = _diffuse(p_sp, areas, csr, trans, n, substeps, cg_tol, cg_maxiter)
        species_rho = {sp: rho_sp[:, k] for k, sp in enumerate(species)}

    # Scatter back to the full mesh and write
    total_err = np.sqrt(sigma * sigma + bias * bias)
    full = {
        DENSITY_ARRAY: _scatter(rho, idx, n_all),
        POWER_ARRAY: _scatter(areas * rho, idx, n_all),
        SIGMA_ARRAY: _scatter(sigma, idx, n_all),
        BIAS_ARRAY: _scatter(bias, idx, n_all),
        P95_ARRAY: _scatter(rho + _Z95 * total_err, idx, n_all),
        NEFF_ARRAY: _scatter(n_eff, idx, n_all),
        BW_ARRAY: _scatter(h, idx, n_all),
    }
    for sp, r in species_rho.items():
        full[f"{DENSITY_ARRAY}_{sp}"] = _scatter(r, idx, n_all)
        full[f"{POWER_ARRAY}_{sp}"] = _scatter(areas * r, idx, n_all)

    for name, values in full.items():
        # Keep the stored precision of arrays that already exist.
        dtype = out.cell_data[name].dtype if name in out.cell_data else np.float64
        out.cell_data[name] = values.astype(dtype)
    out.cell_data.active_scalars_name = DENSITY_ARRAY

    power_out = float(full[POWER_ARRAY].sum())
    for name, value in (
            ("smoothing_power_in_W", total_in),
            ("smoothing_power_out_W", power_out),
            ("smoothing_conservation_error", cons_err),
            ("smoothing_dispersion_factor", dispersion),
            ("smoothing_mc_weight_W", quantum),
            ("smoothing_chi2_per_dof", chi2_dof),
            ("smoothing_target_rel_error", float(target_rel_error)),
            ("smoothing_max_bandwidth_m", float(max_bandwidth_m)),
    ):
        out.field_data[name] = np.array([value], dtype=np.float64)

    print(f"  - [physical] Power in {total_in:.4f} W -> out {power_out:.4f} W "
          f"({(power_out - total_in) / total_in * 100:+.3e} %); "
          f"peak {rho.max():.4e} W/m2 (raw P/A peak {peak_before:.4e})")

    for sp in species:
        before = np.asarray(mesh.cell_data[f"{POWER_ARRAY}_{sp}"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_raw = np.where(areas_all > 0.0, before / np.where(areas_all > 0.0, areas_all, 1.0), 0.0)
        stats["species"][sp] = {
            "total_power_W": float(full[f"{POWER_ARRAY}_{sp}"].sum()),
            "peak_density_before": float(sp_raw.max()) if sp_raw.size else 0.0,
            "peak_density_after": float(full[f"{DENSITY_ARRAY}_{sp}"].max()),
        }

    hit = power > 0
    stats.update({
        "n_smoothed": n,
        "elapsed_s": round(time.time() - t0, 2),
        "total_power_W": power_out,
        "peak_density_after": float(rho.max()),
        "conservation_error": cons_err,
        "dispersion_factor": dispersion,
        "mc_weight_W": quantum,
        "quantisation_quality": quality,
        "chi2_per_dof": chi2_dof,
        "median_bandwidth_m": float(np.median(h[hit])) if hit.any() else 0.0,
        "median_rel_sigma": (float(np.median(sigma[hit]) / np.median(rho[hit]))
                             if hit.any() and np.median(rho[hit]) > 0 else float("nan")),
    })
    return out, stats


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _safe_rel(num: float, den: float) -> str:
    return f"{100.0 * num / den:.2f} %" if den > 0 else "n/a"


def _scatter(values: np.ndarray, idx: np.ndarray, n_all: int) -> np.ndarray:
    full = np.zeros(n_all, dtype=np.float64)
    full[idx] = values
    return full


def _active_band(geo: dict, seed_mask: np.ndarray, voxel: float,
                 layers: int) -> np.ndarray:
    """Cells within roughly ``voxel * layers`` of a seed cell, by grid dilation.

    A Euclidean over-estimate of the band is fine: cells pulled in across a
    gap cost a few unknowns and receive zero, because the solve is restricted
    to the gated dual graph and cannot reach them.  The voxel is one bandwidth
    and is dilated ``layers`` times, so the band hugs the signal closely,
    which keeps the linear system small.
    """
    centroids = geo["centroids"]
    if not seed_mask.any():
        return np.zeros(len(centroids), dtype=bool)
    cell = max(voxel, 1e-9)
    origin = centroids.min(axis=0) - cell
    ijk = np.floor((centroids - origin) / cell).astype(np.int64)
    dims = ijk.max(axis=0) + 2
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

    occupied = np.zeros(int(dims.prod()), dtype=bool)
    occupied[flat[seed_mask]] = True
    grid = occupied.reshape(dims)
    for _ in range(max(1, int(layers))):
        nxt = grid.copy()
        for axis in range(3):
            nxt |= np.roll(grid, 1, axis=axis)
            nxt |= np.roll(grid, -1, axis=axis)
        grid = nxt
    return grid.reshape(-1)[flat]

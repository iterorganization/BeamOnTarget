# SPDX-FileCopyrightText: 2025-2026 ITER Organization
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of BeamOnTarget. See LICENSE (GNU LGPL v3) and GPLv3.txt.
"""
Multi-threaded equivalent of

    trimesh.ray.ray_pyembree.RayMeshIntersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False)

for the single-hit ray engine.

Why: profiling a production run (2026-11-06, 40 min) showed ~39 % of the time
in single-threaded embree queries and ~19 % in trimesh turning the hit
locations into an array row by row (list.extend + np.array). embreex releases
the GIL, so the query parallelises across threads; the locations are computed
here with one vectorised planes_lines call.

Results are meant to be IDENTICAL to the trimesh call, not just close:
  * rays are split into contiguous chunks and the per-chunk embree results are
    concatenated in chunk order, so the hit list keeps trimesh's ray-index
    order (downstream np.bincount sums in that order);
  * origins/directions are prepared exactly as trimesh does (float64 copy,
    util.unitize), and the same planes_lines call computes the locations;
  * hits whose plane intersection is invalid are dropped, as trimesh does.

Only the embree query runs in threads; everything else stays in the caller's
thread, so the order of global np.random draws is unchanged.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from trimesh import util, intersections

# Below this many rays per thread the thread overhead is not worth it.
_MIN_RAYS_PER_THREAD = 20_000


def _prepare_scene(intersector):
    """Fetch trimesh's cached embree scene (and the triangle planes) once, in
    the calling thread, and force embree's lazy scene commit with one serial
    query so worker threads never race on it."""
    cached = getattr(intersector, "_bot_threaded_state", None)
    if cached is not None:
        return cached
    scene = intersector._scene
    scene.run(np.zeros((1, 3), dtype=np.float64),
              np.array([[1.0, 0.0, 0.0]], dtype=np.float64))
    state = (scene,
             intersector.mesh.triangles[:, 0, :],
             intersector.mesh.face_normals)
    intersector._bot_threaded_state = state
    return state


def intersects_location_first(intersector, ray_origins, ray_directions, n_threads):
    """Drop-in for ``intersector.intersects_location(..., multiple_hits=False)``.

    Returns (locations, index_ray, index_tri) with the same values, order and
    dtypes as the trimesh call.
    """
    scene, plane_origins, plane_normals = _prepare_scene(intersector)

    # Same input conditioning as trimesh.intersects_id.
    ray_origins = np.array(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)
    if ray_origins.shape != ray_directions.shape:
        raise ValueError("Ray origin and direction don't match!")
    ray_directions = util.unitize(ray_directions)

    n = len(ray_origins)
    k = int(max(1, min(int(n_threads), n // _MIN_RAYS_PER_THREAD)))
    if k <= 1:
        query = scene.run(ray_origins, ray_directions)
    else:
        bounds = np.linspace(0, n, k + 1).astype(np.int64)
        with ThreadPoolExecutor(max_workers=k) as ex:
            # map() returns results in submission order -> ray-index order.
            parts = list(ex.map(
                lambda j: scene.run(ray_origins[bounds[j]:bounds[j + 1]],
                                    ray_directions[bounds[j]:bounds[j + 1]]),
                range(k)))
        query = np.concatenate(parts)

    hit = query != -1
    index_tri = query[hit]
    index_ray = np.nonzero(hit)[0]
    if not hit.any():
        return np.zeros((0, 3), float), index_ray, index_tri

    locations, valid = intersections.planes_lines(
        plane_origins=plane_origins[index_tri],
        plane_normals=plane_normals[index_tri],
        line_origins=ray_origins[index_ray],
        line_directions=ray_directions[index_ray])
    if not valid.all():
        index_ray = index_ray[valid]
        index_tri = index_tri[valid]
    return locations, index_ray, index_tri

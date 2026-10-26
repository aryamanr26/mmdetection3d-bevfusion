# Copyright (c) OpenMMLab. All rights reserved.
"""LiDAR-only local elevation grid.

Fit a ground plane on 1 m cells from the lowest returns, then sample that
plane onto a 50 m x 50 m grid at 0.2 m. Height, variance, slope, and
roughness describe the surface. Obstacle evidence is height above that
surface, so a ramp stays terrain and a curb or vehicle does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

X_MIN = -25.0
Y_MIN = -25.0
RESOLUTION = 0.2
SIZE = 250
COARSE = 1.0
COARSE_SIZE = 50
FINE_PER_COARSE = 5
Z_MIN = -3.5
Z_MAX = 1.0
EGO_RADIUS = 1.0
SEED_BAND = 0.08
INLIER_BAND = 0.08
UPRIGHT_MIN = 0.85
VERTICAL_MAX = 0.707
VERTICAL_DIST = 0.1
FLATNESS_MAX = 0.1
MIN_SEEDS = 3
MIN_SPAN = 0.4
SLOPE_LIMIT_DEG = 25.0
SLOPE_STEP = math.tan(math.radians(SLOPE_LIMIT_DEG)) * COARSE
CURB_STEP = 0.10
CURB_FLAT_DEG = 4.0
GROUND_HINT_Z = -1.7


@dataclass
class ElevationGrid:
    """Five layers on one vehicle-centered grid. Unknown cells are NaN."""

    ground: np.ndarray
    variance: np.ndarray
    slope: np.ndarray
    roughness: np.ndarray
    obstacle: np.ndarray
    resolution: float
    x_min: float
    y_min: float
    size: int
    timings_ms: dict

    @property
    def valid(self):
        return np.isfinite(self.ground)


def _empty(size, timings):
    nan = np.full((size, size), np.nan, dtype=np.float32)
    return ElevationGrid(
        ground=nan.copy(),
        variance=nan.copy(),
        slope=nan.copy(),
        roughness=nan.copy(),
        obstacle=nan.copy(),
        resolution=RESOLUTION,
        x_min=X_MIN,
        y_min=Y_MIN,
        size=size,
        timings_ms=timings,
    )


def _preprocess(points):
    """Keep points inside the window, above the noise floor, outside the ego box."""
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    span = SIZE * RESOLUTION
    keep = (
        (x >= X_MIN) & (x < X_MIN + span)
        & (y >= Y_MIN) & (y < Y_MIN + span)
        & (z > Z_MIN) & (z < Z_MAX)
        & ~((np.abs(x) < EGO_RADIUS) & (np.abs(y) < EGO_RADIUS)))
    return xyz[keep]


def _fit_plane(seeds):
    """Return centroid, upward normal, and flatness, or None."""
    if len(seeds) < MIN_SEEDS:
        return None
    centroid = seeds.mean(axis=0)
    centered = seeds - centroid
    cov = centered.T @ centered / len(seeds)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0].copy()
    if normal[2] < 0.0:
        normal *= -1.0
    flatness = float(evals[0] / max(float(evals.sum()), 1e-12))
    return centroid, normal, flatness


def _plane_z(x, y, centroid, normal):
    return centroid[..., 2] - (
        normal[..., 0] * (x - centroid[..., 0])
        + normal[..., 1] * (y - centroid[..., 1])
    ) / normal[..., 2]


def _seed_buffer(xyz, cell):
    """Lowest points in each coarse cell, packed to a fixed width for one SVD."""
    max_seeds = 32
    zmin = np.full(COARSE_SIZE * COARSE_SIZE, np.inf)
    np.minimum.at(zmin, cell, xyz[:, 2])
    choose = xyz[:, 2] <= zmin[cell] + SEED_BAND
    seeds = xyz[choose]
    seed_cell = cell[choose]
    if len(seeds) == 0:
        empty = np.zeros((0, max_seeds, 3))
        return empty, np.zeros(0, np.int32), np.zeros(0, np.int32)
    order = np.lexsort((seeds[:, 2], seed_cell))
    seeds = seeds[order]
    seed_cell = seed_cell[order]
    uniq, starts, counts = np.unique(seed_cell, return_index=True, return_counts=True)
    width = np.minimum(counts, max_seeds)
    keep = width >= MIN_SEEDS
    uniq, starts, width = uniq[keep], starts[keep], width[keep]
    if len(uniq) == 0:
        empty = np.zeros((0, max_seeds, 3))
        return empty, np.zeros(0, np.int32), np.zeros(0, np.int32)
    local = np.arange(int(width.sum())) - np.repeat(np.cumsum(width) - width, width)
    src = np.repeat(starts, width) + local
    rows = np.repeat(np.arange(len(uniq)), width)
    buf = np.zeros((len(uniq), max_seeds, 3), dtype=np.float64)
    buf[rows, local] = seeds[src]
    return buf, width, uniq


def _planes_from_buffer(buf, width):
    """Batched 3x3 plane fit. Normals point upward."""
    slot = np.arange(buf.shape[1])[None, :] < width[:, None]
    count = width.astype(np.float64)
    masked = np.where(slot[..., None], buf, 0.0)
    centroid = masked.sum(axis=1) / count[:, None]
    centered = np.where(slot[..., None], buf - centroid[:, None, :], 0.0)
    cov = np.einsum('nki,nkj->nij', centered, centered) / count[:, None, None]
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, :, 0].copy()
    normal[normal[:, 2] < 0.0] *= -1.0
    flatness = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-12)
    return centroid, normal, flatness


def _write_planes(valid, height, centroid, normal, within, linear, cen, nrm,
                  within_values):
    if len(linear) == 0:
        return
    iy, ix = np.divmod(np.asarray(linear, dtype=np.int32), COARSE_SIZE)
    center_x = X_MIN + (ix + 0.5) * COARSE
    center_y = Y_MIN + (iy + 0.5) * COARSE
    valid[iy, ix] = True
    height[iy, ix] = _plane_z(center_x, center_y, cen, nrm)
    centroid[iy, ix] = cen
    normal[iy, ix] = nrm
    within[iy, ix] = within_values


def _cell_models(buf, width, uniq):
    """Use a full plane only when the seeds span the cell. Otherwise the
    ground height is the median of the low returns, so a thin LiDAR ring
    cannot tilt the cell.
    """
    cen, nrm, flatness = _planes_from_buffer(buf, width)
    slot = np.arange(buf.shape[1])[None, :] < width[:, None]
    filled = np.where(slot[..., None], buf, np.nan)
    span = np.nanmax(filled, axis=1) - np.nanmin(filled, axis=1)
    med_z = buf[np.arange(len(width)), (width - 1) // 2, 2]
    upright = (nrm[:, 2] >= UPRIGHT_MIN) & (flatness <= FLATNESS_MAX)
    well = upright & (span[:, 0] >= MIN_SPAN) & (span[:, 1] >= MIN_SPAN)
    vertical = nrm[:, 2] < VERTICAL_MAX
    keep = ~vertical
    iy, ix = np.divmod(uniq, COARSE_SIZE)
    center = np.column_stack([
        X_MIN + (ix + 0.5) * COARSE,
        Y_MIN + (iy + 0.5) * COARSE,
        med_z,
    ])
    use_cen = np.where(well[:, None], cen, center)
    use_nrm = np.where(well[:, None], nrm, np.array([0.0, 0.0, 1.0]))
    tilt = np.degrees(np.arccos(np.clip(nrm[:, 2], -1.0, 1.0)))
    within = np.where(well, tilt, 0.0)
    return uniq[keep], use_cen[keep], use_nrm[keep], within[keep], uniq[vertical]


def _gradient_slope(valid, height):
    """Slope of the ground surface from neighboring cell heights."""
    slope = np.full(height.shape, np.nan, dtype=np.float64)
    ys, xs = np.where(valid)
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        sx = sy = 0.0
        if (cx > 0 and cx + 1 < COARSE_SIZE and valid[cy, cx - 1]
                and valid[cy, cx + 1]):
            sx = (height[cy, cx + 1] - height[cy, cx - 1]) / (2.0 * COARSE)
        elif cx + 1 < COARSE_SIZE and valid[cy, cx + 1]:
            sx = (height[cy, cx + 1] - height[cy, cx]) / COARSE
        elif cx > 0 and valid[cy, cx - 1]:
            sx = (height[cy, cx] - height[cy, cx - 1]) / COARSE
        if (cy > 0 and cy + 1 < COARSE_SIZE and valid[cy - 1, cx]
                and valid[cy + 1, cx]):
            sy = (height[cy + 1, cx] - height[cy - 1, cx]) / (2.0 * COARSE)
        elif cy + 1 < COARSE_SIZE and valid[cy + 1, cx]:
            sy = (height[cy + 1, cx] - height[cy, cx]) / COARSE
        elif cy > 0 and valid[cy - 1, cx]:
            sy = (height[cy, cx] - height[cy - 1, cx]) / COARSE
        slope[cy, cx] = math.degrees(math.atan(math.hypot(sx, sy)))
    return slope


def _residual_layers(xyz, cell, valid, centroid, normal):
    """Variance and roughness from points that lie on the accepted plane."""
    nlin = COARSE_SIZE * COARSE_SIZE
    variance = np.full((COARSE_SIZE, COARSE_SIZE), np.nan, dtype=np.float64)
    roughness = np.full((COARSE_SIZE, COARSE_SIZE), np.nan, dtype=np.float64)
    lin_valid = valid.reshape(-1)
    if not lin_valid.any():
        return variance, roughness
    delta = xyz - centroid.reshape(-1, 3)[cell]
    dist = np.abs(np.einsum('ij,ij->i', delta, normal.reshape(-1, 3)[cell]))
    use = lin_valid[cell] & (dist <= INLIER_BAND)
    if not use.any():
        variance[valid] = 0.0
        roughness[valid] = 0.0
        return variance, roughness
    sumsq = np.bincount(cell[use], weights=dist[use] ** 2, minlength=nlin)
    sumr = np.bincount(cell[use], weights=dist[use], minlength=nlin)
    cnt = np.bincount(cell[use], minlength=nlin).astype(np.float64)
    ok = cnt >= 1.0
    second = np.zeros(nlin)
    mean = np.zeros(nlin)
    second[ok] = sumsq[ok] / cnt[ok]
    mean[ok] = sumr[ok] / cnt[ok]
    var = np.full(nlin, np.nan)
    std = np.full(nlin, np.nan)
    var[ok] = second[ok]
    std[ok] = np.sqrt(np.maximum(second[ok] - mean[ok] ** 2, 0.0))
    variance = var.reshape(COARSE_SIZE, COARSE_SIZE)
    roughness = std.reshape(COARSE_SIZE, COARSE_SIZE)
    missing = valid & ~np.isfinite(variance)
    variance[missing] = 0.0
    roughness[missing] = 0.0
    return variance, roughness


def _accept_fit(fit):
    if fit is None:
        return False
    _centroid, normal, flatness = fit
    return normal[2] >= UPRIGHT_MIN and flatness <= FLATNESS_MAX


def _cell_fit(pts):
    """Lowest-seed plane, with one vertical-plane rejection retry."""
    zmin = float(pts[:, 2].min())
    seeds = pts[pts[:, 2] <= zmin + SEED_BAND]
    fit = _fit_plane(seeds)
    if _accept_fit(fit):
        return fit, pts
    if fit is None or fit[1][2] >= VERTICAL_MAX:
        return None, None
    centroid, normal, _flatness = fit
    dist = np.abs((pts - centroid) @ normal)
    kept = pts[dist > VERTICAL_DIST]
    if len(kept) < MIN_SEEDS:
        return None, None
    zmin = float(kept[:, 2].min())
    seeds = kept[kept[:, 2] <= zmin + SEED_BAND]
    fit = _fit_plane(seeds)
    if not _accept_fit(fit):
        return None, None
    return fit, kept


def _neighbor_gate(valid, height, hint):
    """Keep ground that can be reached from the road under the vehicle.

    A ramp grows one cell at a time. A roof or a vehicle deck sits a step
    above that surface, so it stays unknown instead of becoming ground.
    """
    if not valid.any():
        return valid
    source = valid & (np.abs(height - hint) <= 0.30)
    if not source.any():
        med = float(np.nanmedian(height[valid]))
        source = valid & (np.abs(height - med) <= 0.30)
    reach = source.copy()
    queue = list(zip(*np.where(source)))
    head = 0
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while head < len(queue):
        cy, cx = queue[head]
        head += 1
        for dy, dx in offsets:
            ny, nx = cy + dy, cx + dx
            if (ny < 0 or nx < 0 or ny >= COARSE_SIZE or nx >= COARSE_SIZE
                    or reach[ny, nx] or not valid[ny, nx]):
                continue
            if abs(height[ny, nx] - height[cy, cx]) <= SLOPE_STEP:
                reach[ny, nx] = True
                queue.append((ny, nx))
    return reach


def _boundary_evidence(valid, height, slope, obstacle):
    """Mark a step between two flat cells. A ramp is a slope, not a step."""
    flat = valid & np.isfinite(slope) & (slope < CURB_FLAT_DEG)
    term_scale = 1.0 / CURB_STEP
    horizontal = (
        flat[:, :-1] & flat[:, 1:]
        & (np.abs(height[:, 1:] - height[:, :-1]) > CURB_STEP))
    eys, exs = np.where(horizontal)
    if len(eys):
        term = (
            np.abs(height[eys, exs + 1] - height[eys, exs]) * term_scale
        ).astype(np.float32)
        for k in range(FINE_PER_COARSE):
            fy = eys * FINE_PER_COARSE + k
            np.add.at(obstacle, (fy, exs * FINE_PER_COARSE + FINE_PER_COARSE - 1), term)
            np.add.at(obstacle, (fy, (exs + 1) * FINE_PER_COARSE), term)
    vertical = (
        flat[:-1, :] & flat[1:, :]
        & (np.abs(height[1:, :] - height[:-1, :]) > CURB_STEP))
    eys, exs = np.where(vertical)
    if len(eys):
        term = (
            np.abs(height[eys + 1, exs] - height[eys, exs]) * term_scale
        ).astype(np.float32)
        for k in range(FINE_PER_COARSE):
            fx = exs * FINE_PER_COARSE + k
            np.add.at(obstacle, (eys * FINE_PER_COARSE + FINE_PER_COARSE - 1, fx), term)
            np.add.at(obstacle, ((eys + 1) * FINE_PER_COARSE, fx), term)


def estimate_elevation_grid(points):
    """Build the five-layer grid. Timings are milliseconds per stage."""
    import time

    started = time.perf_counter()
    xyz = _preprocess(points)
    preprocess_ms = (time.perf_counter() - started) * 1e3
    fit_started = time.perf_counter()
    if len(xyz) == 0:
        timings = {
            'preprocess': preprocess_ms,
            'ground_fit': 0.0,
            'grid_update': 0.0,
        }
        return _empty(SIZE, timings)

    cx = np.floor((xyz[:, 0] - X_MIN) / COARSE).astype(np.int32)
    cy = np.floor((xyz[:, 1] - Y_MIN) / COARSE).astype(np.int32)
    cell = cy * COARSE_SIZE + cx
    order = np.argsort(cell, kind='stable')
    cell = cell[order]
    xyz = xyz[order]
    _uniq, starts, counts = np.unique(cell, return_index=True, return_counts=True)
    full_start = np.full(COARSE_SIZE * COARSE_SIZE, -1, dtype=np.int32)
    full_count = np.zeros(COARSE_SIZE * COARSE_SIZE, dtype=np.int32)
    full_start[cell[starts]] = starts
    full_count[cell[starts]] = counts

    valid = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=bool)
    height = np.full((COARSE_SIZE, COARSE_SIZE), np.nan, dtype=np.float64)
    centroid = np.zeros((COARSE_SIZE, COARSE_SIZE, 3), dtype=np.float64)
    normal = np.zeros((COARSE_SIZE, COARSE_SIZE, 3), dtype=np.float64)
    within_c = np.full((COARSE_SIZE, COARSE_SIZE), np.nan, dtype=np.float64)

    buf, width, uniq = _seed_buffer(xyz, cell)
    if len(uniq):
        kept, cen, nrm, within_values, retry = _cell_models(buf, width, uniq)
        _write_planes(valid, height, centroid, normal, within_c, kept, cen,
                      nrm, within_values)
        for linear in retry.tolist():
            pts = xyz[full_start[linear]:full_start[linear] + full_count[linear]]
            fit, _pool = _cell_fit(pts)
            if fit is None:
                continue
            fit_cen, fit_nrm, _fit_flat = fit
            tilt = math.degrees(math.acos(float(np.clip(fit_nrm[2], -1.0, 1.0))))
            _write_planes(
                valid, height, centroid, normal, within_c,
                np.asarray([linear]), fit_cen[None], fit_nrm[None],
                np.asarray([tilt]))

    hint = float(np.percentile(xyz[:, 2], 15)) if len(xyz) else GROUND_HINT_Z
    valid = _neighbor_gate(valid, height, hint)
    slope_c = _gradient_slope(valid, height)
    variance_c, rough_c = _residual_layers(xyz, cell, valid, centroid, normal)
    fit_ms = (time.perf_counter() - fit_started) * 1e3

    update_started = time.perf_counter()
    fy, fx = np.mgrid[0:SIZE, 0:SIZE]
    coy = fy // FINE_PER_COARSE
    cox = fx // FINE_PER_COARSE
    known = valid[coy, cox]
    cen = centroid[coy, cox]
    nrm = normal[coy, cox]
    xc = X_MIN + (fx + 0.5) * RESOLUTION
    yc = Y_MIN + (fy + 0.5) * RESOLUTION
    nz = np.where(np.abs(nrm[..., 2]) < 1e-6, np.nan, nrm[..., 2])
    ground = cen[..., 2] - (
        nrm[..., 0] * (xc - cen[..., 0]) + nrm[..., 1] * (yc - cen[..., 1])
    ) / nz
    ground = np.where(known, ground, np.nan).astype(np.float32)
    slope = np.where(known, slope_c[coy, cox], np.nan).astype(np.float32)
    variance = np.where(known, variance_c[coy, cox], np.nan).astype(np.float32)
    roughness = np.where(known, rough_c[coy, cox], np.nan).astype(np.float32)

    obstacle = np.zeros((SIZE, SIZE), dtype=np.float32)
    pix = np.floor((xyz[:, 0] - X_MIN) / RESOLUTION).astype(np.int32)
    piy = np.floor((xyz[:, 1] - Y_MIN) / RESOLUTION).astype(np.int32)
    zg = ground[piy, pix]
    above = np.isfinite(zg) & ((xyz[:, 2] - zg) > CURB_STEP)
    np.add.at(obstacle, (piy[above], pix[above]), 1.0)
    _boundary_evidence(valid, height, within_c, obstacle)
    obstacle = np.where(known, obstacle, np.nan).astype(np.float32)
    update_ms = (time.perf_counter() - update_started) * 1e3

    return ElevationGrid(
        ground=ground,
        variance=variance,
        slope=slope,
        roughness=roughness,
        obstacle=obstacle,
        resolution=RESOLUTION,
        x_min=X_MIN,
        y_min=Y_MIN,
        size=SIZE,
        timings_ms={
            'preprocess': preprocess_ms,
            'ground_fit': fit_ms,
            'grid_update': update_ms,
        },
    )


def _colormap(values, show, vmin, vmax, name, alpha):
    from matplotlib import colormaps

    span = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    color = np.asarray(colormaps[name](norm, bytes=True), dtype=np.uint8).copy()
    color[..., 3] = np.where(show, alpha, 0).astype(np.uint8)
    return color


def layer_rgba(grid, kind):
    """RGBA heatmap for one layer. Unknown cells are fully transparent."""
    valid = grid.valid
    if kind == 'ground':
        values = grid.ground
        finite = values[valid]
        if finite.size:
            vmin, vmax = np.percentile(finite, [5, 95])
        else:
            vmin, vmax = GROUND_HINT_Z, GROUND_HINT_Z + 1.0
        if vmax - vmin < 0.05:
            vmax = vmin + 0.05
        return _colormap(np.nan_to_num(values, nan=vmin), valid, vmin, vmax,
                         'terrain', 170)
    if kind == 'slope':
        values = grid.slope
        return _colormap(np.nan_to_num(values, nan=0.0), valid, 0.0, 15.0,
                         'viridis', 150)
    if kind == 'obstacle':
        values = grid.obstacle
        show = valid & np.isfinite(values) & (values > 0)
        finite = values[show]
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, 1.0)
        return _colormap(np.nan_to_num(values, nan=0.0), show, 0.0, vmax,
                         'inferno', 220)
    raise KeyError(kind)


def self_check():
    """Flat road, an 8 degree ramp, a 0.15 m curb, and a box on the road."""
    step = 0.4
    xs = np.arange(-12.0, 12.0, step)
    ys = np.arange(-12.0, 12.0, step)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    zz = np.full(xx.shape, -1.7, dtype=np.float64)
    ramp = (xx >= 2.0) & (xx <= 10.0) & (yy < 2.0)
    rise = math.tan(math.radians(8.0))
    zz[ramp] = -1.7 + rise * (xx[ramp] - 2.0)
    curb = (xx < 2.0) & (yy >= 2.0) & (yy <= 10.0)
    zz[curb] = -1.55
    cloud = [np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])]
    box_x = np.linspace(-4.0, -3.5, 6)
    box_y = np.linspace(-4.0, -3.5, 6)
    bx, by = np.meshgrid(box_x, box_y, indexing='xy')
    for z in (-1.45, -1.30, -1.15):
        cloud.append(np.column_stack([
            bx.ravel(), by.ravel(), np.full(bx.size, z)]))
    grid = estimate_elevation_grid(np.vstack(cloud))

    def cell_at(x, y):
        ix = int(math.floor((x - X_MIN) / RESOLUTION))
        iy = int(math.floor((y - Y_MIN) / RESOLUTION))
        return iy, ix

    flat = cell_at(-6.0, -6.0)
    ramp_cell = cell_at(6.0, -4.0)
    high = cell_at(-6.0, 6.0)
    low = cell_at(-6.0, 1.5)
    box = cell_at(-3.7, -3.7)
    assert grid.valid[flat], 'flat road cell is unknown'
    assert abs(float(grid.ground[flat]) - (-1.7)) < 0.05, grid.ground[flat]
    assert float(grid.slope[flat]) < 2.0, grid.slope[flat]
    assert float(grid.obstacle[flat]) == 0.0, grid.obstacle[flat]
    assert grid.valid[ramp_cell], 'ramp cell is unknown'
    assert 6.0 <= float(grid.slope[ramp_cell]) <= 10.0, grid.slope[ramp_cell]
    assert float(grid.obstacle[ramp_cell]) < 1.0, grid.obstacle[ramp_cell]
    assert grid.valid[high] and grid.valid[low], 'curb sides missing'
    assert float(grid.ground[high]) > float(grid.ground[low]) + 0.10
    assert float(grid.slope[high]) < 3.0, grid.slope[high]
    # The shared coarse edge is the fine column just inside y=2.
    edge = cell_at(-6.0, 1.9)
    assert float(grid.obstacle[edge]) > 0.0, grid.obstacle[edge]
    assert float(grid.obstacle[box]) > 0.0, grid.obstacle[box]
    assert not np.isfinite(grid.ground[0, 0]), 'empty corner should stay unknown'
    return grid

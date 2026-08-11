# -*- coding: utf-8 -*-
"""Tier-3 (depth + intrinsics + soil plane + fixed ground) 3D canopy geometry.

Ports ``pheno_extract.canopy_geometry`` + ``pheno_extract.geometry.deproject``
/ ``fit_soil_plane``. Computes **plant height** (mean & p95 above the fitted
soil plane), projected area, visible leaf-surface area, and envelope volume,
all in millimetres. Pure ``numpy`` + ``cv2`` — intentionally **shapely-free**
so it is CPU-testable and portable; projected area is the mesh-triangle sum
(an exact area for a non-overlapping height-field mesh).

Fail-closed: any unobservable frame must return ``NaN`` + ``missing_reason``,
never a fabricated metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .calib import CameraIntrinsics, load_rgb_intrinsics
from .base import TraitExtractor, register, INPUT_MASK, INPUT_DEPTH, INPUT_CALIB


# --------------------------------------------------------------------------
# Fixed camera->ground frame + mesh QC config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FixedGroundFrame:
    """Fixed camera->level-ground rotation (no per-frame drift)."""
    rotation_camera_to_ground: Tuple[float, ...] = (
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    authority: str = "fixed_down_capture_contract"

    def rotation(self) -> np.ndarray:
        m = np.asarray(self.rotation_camera_to_ground, dtype=np.float64).reshape(3, 3)
        if not np.allclose(m.T @ m, np.eye(3), atol=1.0e-6):
            raise ValueError("camera_to_ground rotation is not orthogonal")
        if not np.isclose(np.linalg.det(m), 1.0, atol=1.0e-6):
            raise ValueError("camera_to_ground rotation has wrong handedness")
        return m


@dataclass(frozen=True)
class CanopyMeshConfig:
    depth_min_mm: float = 250.0
    depth_max_mm: float = 2500.0
    surface_jump_abs_mm: float = 35.0
    surface_jump_rel: float = 0.03
    envelope_grid_mm: float = 1.0
    maximum_envelope_cells: int = 5_000_000
    minimum_native_depth_fraction: float = 0.75
    minimum_edge_depth_fraction: float = 0.65
    minimum_mesh_vertex_coverage_fraction: float = 0.80
    minimum_mean_canopy_height_mm: float = 1.0
    minimum_p95_canopy_height_mm: float = 2.0
    minimum_height_to_soil_uncertainty_ratio: float = 1.0
    maximum_surface_to_projected_area_ratio: float = 4.0
    surface_normal_radius_pixels: int = 9
    surface_normal_minimum_points: int = 12
    surface_normal_maximum_inclination_degrees: float = 75.0
    minimum_surface_normal_support_fraction: float = 0.80
    maximum_surface_inclination_capped_fraction: float = 0.15


@dataclass
class CanopyGeometryTraits:
    canopy_projected_area_mm2: float = np.nan
    visible_canopy_surface_area_mm2: float = np.nan
    canopy_envelope_volume_mm3: float = np.nan
    mean_canopy_height_mm: float = np.nan
    canopy_height_p95_mm: float = np.nan
    surface_to_projected_area_ratio: float = np.nan
    native_depth_fraction: float = np.nan
    edge_depth_fraction: float = np.nan
    mesh_vertex_coverage_fraction: float = np.nan
    metric_authority: str = ""
    geometry_qc_pass: bool = False
    missing_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Core math
# --------------------------------------------------------------------------
def deproject(depth_mm: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Depth (mm) + intrinsics -> camera-frame XYZ array [H,W,3]."""
    rows, cols = np.indices(depth_mm.shape, dtype=np.float64)
    z = depth_mm.astype(np.float64)
    x = (cols - intrinsics.cx) * z / intrinsics.fx
    y = (rows - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=-1)


def _triangle_indices_for_cell(valid: np.ndarray, xyz: np.ndarray):
    present = np.flatnonzero(valid)
    if len(present) < 3:
        return []
    if len(present) == 3:
        return [tuple(int(v) for v in present)]
    d03 = float(np.linalg.norm(xyz[0] - xyz[3]))
    d12 = float(np.linalg.norm(xyz[1] - xyz[2]))
    return [(0, 1, 3), (0, 3, 2)] if d03 <= d12 else [(0, 1, 2), (1, 3, 2)]


def _pixel_footprint(x, y, depth_mm, intrinsics, rotation) -> np.ndarray:
    corners = np.array([[x - 0.5, y - 0.5], [x + 0.5, y - 0.5],
                        [x + 0.5, y + 0.5], [x - 0.5, y + 0.5]], dtype=np.float64)
    pts = np.column_stack([
        (corners[:, 0] - intrinsics.cx) * depth_mm / intrinsics.fx,
        (corners[:, 1] - intrinsics.cy) * depth_mm / intrinsics.fy,
        np.full(4, depth_mm)])
    return (rotation @ pts.T).T


def _point_height(points: np.ndarray, soil_plane: np.ndarray) -> np.ndarray:
    a, b, c = soil_plane[0], soil_plane[1], soil_plane[2]
    denom = float(np.sqrt(a * a + b * b + 1.0))
    soil_z = a * points[:, 0] + b * points[:, 1] + c  # c holds intercept here
    # NOTE: soil_plane is (a,b,c) for Z = aX + bY + c  -> soil_z uses c as intercept
    return np.maximum((soil_z - points[:, 2]) / denom, 0.0)


def _polygon_area_xy(points: np.ndarray) -> float:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    return float(0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
                            - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))


def _robust_visible_surface_area(*, valid, depth, ground_xyz, layers, intrinsics,
                                  rotation, config: CanopyMeshConfig) -> Tuple[float, float, float]:
    radius = max(int(config.surface_normal_radius_pixels), 1)
    minimum_points = max(int(config.surface_normal_minimum_points), 3)
    minimum_cosine = float(np.cos(np.deg2rad(config.surface_normal_maximum_inclination_degrees)))
    surface_area = 0.0
    supported = 0
    capped = 0
    vy, vx = np.where(valid)
    h, w = valid.shape
    for y, x in zip(vy, vx):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        same = (valid[y0:y1, x0:x1] & (layers[y0:y1, x0:x1] == layers[y, x])
                & (np.abs(depth[y0:y1, x0:x1] - depth[y, x])
                   <= max(config.surface_jump_abs_mm, config.surface_jump_rel * float(depth[y, x]))))
        pts = ground_xyz[y0:y1, x0:x1][same]
        footprint = _pixel_footprint(int(x), int(y), float(depth[y, x]), intrinsics, rotation)
        proj_area = _polygon_area_xy(footprint)
        if len(pts) < minimum_points:
            surface_area += proj_area
            continue
        center = np.median(pts, axis=0)
        design = np.column_stack([pts[:, 0] - center[0], pts[:, 1] - center[1], np.ones(len(pts))])
        resp = pts[:, 2] - center[2]
        coeff = np.linalg.lstsq(design, resp, rcond=None)[0]
        resid = resp - design @ coeff
        mad = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
        inliers = np.abs(resid - np.median(resid)) <= max(2.5 * mad, 1.0)
        if int(inliers.sum()) >= minimum_points:
            coeff = np.linalg.lstsq(design[inliers], resp[inliers], rcond=None)[0]
        cosine = float(1.0 / np.sqrt(1.0 + coeff[0] ** 2 + coeff[1] ** 2))
        if cosine < minimum_cosine:
            capped += 1
        surface_area += proj_area / max(cosine, minimum_cosine)
        supported += 1
    total = max(len(vy), 1)
    return float(surface_area), float(supported / total), float(capped / max(supported, 1))


def _rasterize_envelope(triangles_xy, triangle_heights, fallback_xy, fallback_heights,
                        grid_mm, maximum_cells) -> Tuple[float, float, float]:
    all_xy = [triangles_xy.reshape(-1, 2)]
    all_xy.extend(fallback_xy)
    finite = [p for p in all_xy if p.size and np.isfinite(p).all()]
    if not finite:
        return 0.0, 0.0, 0.0
    stacked = np.concatenate(finite, axis=0)
    minimum = np.floor(stacked.min(axis=0) / grid_mm) * grid_mm
    maximum = np.ceil(stacked.max(axis=0) / grid_mm) * grid_mm
    width, height = np.ceil((maximum - minimum) / grid_mm).astype(int) + 1
    if width <= 0 or height <= 0 or int(width * height) > maximum_cells:
        raise ValueError("envelope integration grid exceeds safety ceiling")
    envelope = np.full((height, width), np.nan, dtype=np.float32)
    for xy, heights in zip(triangles_xy, triangle_heights):
        gx = (xy[:, 0] - minimum[0]) / grid_mm
        gy = (xy[:, 1] - minimum[1]) / grid_mm
        x0, x1 = max(int(np.floor(gx.min())), 0), min(int(np.ceil(gx.max())), width - 1)
        y0, y1 = max(int(np.floor(gy.min())), 0), min(int(np.ceil(gy.max())), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px = minimum[0] + (xx + 0.5) * grid_mm
        py = minimum[1] + (yy + 0.5) * grid_mm
        denom = ((xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0])
                 + (xy[2, 0] - xy[1, 0]) * (xy[0, 1] - xy[2, 1]))
        if abs(float(denom)) < 1.0e-12:
            continue
        w0 = ((xy[1, 1] - xy[2, 1]) * (px - xy[2, 0]) + (xy[2, 0] - xy[1, 0]) * (py - xy[2, 1])) / denom
        w1 = ((xy[2, 1] - xy[0, 1]) * (px - xy[2, 0]) + (xy[0, 0] - xy[2, 0]) * (py - xy[2, 1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1.0e-7) & (w1 >= -1.0e-7) & (w2 >= -1.0e-7)
        if not np.any(inside):
            continue
        interp = w0 * heights[0] + w1 * heights[1] + w2 * heights[2]
        view = envelope[y0:y1 + 1, x0:x1 + 1]
        cur = view[inside]
        vals = interp[inside].astype(np.float32)
        view[inside] = np.where(np.isnan(cur), vals, np.maximum(cur, vals))
    for xy, hv in zip(fallback_xy, fallback_heights):
        polygon = np.rint((xy - minimum) / grid_mm).astype(np.int32)
        local = np.zeros_like(envelope, dtype=np.uint8)
        cv2.fillConvexPoly(local, polygon, 1)
        sel = local > 0
        cur = envelope[sel]
        envelope[sel] = np.where(np.isnan(cur), hv, np.maximum(cur, hv))
    valid = np.isfinite(envelope)
    proj_raster_area = float(valid.sum() * grid_mm * grid_mm)
    volume = float(np.nansum(envelope, dtype=np.float64) * grid_mm * grid_mm)
    mean_h = float(np.nanmean(envelope)) if np.any(valid) else 0.0
    return proj_raster_area, volume, mean_h


# --------------------------------------------------------------------------
# Soil plane fitting (robust IRLS + deterministic RANSAC fallback)
# --------------------------------------------------------------------------
def fit_soil_plane(points: np.ndarray, holdout_fraction: float = 0.25, seed: int = 17,
                   minimum_robust_inlier_ratio: float = 0.18,
                   minimum_robust_holdout_points: int = 100,
                   minimum_robust_spatial_quadrants: int = 3,
                   maximum_robust_slope_norm: float = 0.15,
                   marginal_robust_slope_norm: float = 0.22
                   ) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """Fit soil plane Z = aX + bY + c via IRLS, falling back to deterministic
    RANSAC when the initial fit is poor. Returns (coeffs[a,b,c], holdout_median, audit)."""
    if len(points) < 100:
        return None, None, {"status": "rejected", "reason": "insufficient_soil_points",
                            "point_count": int(len(points))}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    split = max(int(len(points) * (1.0 - holdout_fraction)), 3)
    train, holdout = points[order[:split]], points[order[split:]]
    design = np.column_stack([train[:, 0], train[:, 1], np.ones(len(train))])
    coeff, *_ = np.linalg.lstsq(design, train[:, 2], rcond=None)
    for _ in range(3):
        resid = train[:, 2] - design @ coeff
        mad = np.median(np.abs(resid - np.median(resid)))
        keep = np.abs(resid) <= max(3.0, 3.0 * 1.4826 * mad)
        if int(keep.sum()) < 100:
            break
        coeff, *_ = np.linalg.lstsq(design[keep], train[keep, 2], rcond=None)
    hd = np.column_stack([holdout[:, 0], holdout[:, 1], np.ones(len(holdout))])
    holdout_resid = np.abs(holdout[:, 2] - hd @ coeff)
    median = float(np.median(holdout_resid)) if len(holdout_resid) else None
    initial_slope = float(np.linalg.norm(coeff[:2]))
    method = "iteratively_reweighted_least_squares"
    marginal_accepted = False
    audit: dict = {}

    def _accept(cand):
        r = np.abs(train[:, 2] - design @ cand)
        inl = r <= 6.0
        if int(inl.sum()) < 100:
            return None
        hres = np.abs(holdout[:, 2] - hd @ cand)
        hinl = hres <= 6.0
        iratio = float(np.mean(hinl))
        imed = float(np.median(hres[hinl])) if np.any(hinl) else float("inf")
        if np.any(hinl):
            cxy = np.median(holdout[:, :2], axis=0)
            quad = len({(int(x > cxy[0]), int(y > cxy[1])) for x, y in holdout[hinl, :2]})
        else:
            quad = 0
        sn = float(np.linalg.norm(cand[:2]))
        return cand, iratio, imed, quad, sn

    if median is not None and (median > 8.0 or initial_slope > maximum_robust_slope_norm):
        best_inl = np.zeros(len(train), dtype=bool)
        best_med = float("inf")
        for _ in range(128):
            samp = rng.choice(len(train), size=3, replace=False)
            sd = design[samp]
            if abs(float(np.linalg.det(sd))) < 1.0e-8:
                continue
            cand = np.linalg.solve(sd, train[samp, 2])
            if float(np.linalg.norm(cand[:2])) > maximum_robust_slope_norm:
                continue
            res = _accept(cand)
            if res is None:
                continue
            _, iratio, imed, quad, sn = res
            # Keep the candidate with the most training inliers; break ties by
            # smaller holdout-inlier median residual.
            cand_inl = np.abs(train[:, 2] - design @ cand) <= 6.0
            cand_count = int(cand_inl.sum())
            best_count = int(best_inl.sum())
            if cand_count > best_count or (cand_count == best_count and imed < best_med):
                best_inl = cand_inl
                best_med = imed
        if int(best_inl.sum()) >= 100:
            rob, *_ = np.linalg.lstsq(design[best_inl], train[best_inl, 2], rcond=None)
            hres = np.abs(holdout[:, 2] - hd @ rob)
            hinl = hres <= 6.0
            iratio = float(np.mean(hinl))
            imed = float(np.median(hres[hinl])) if np.any(hinl) else float("inf")
            quad = len({(int(x > np.median(holdout[:, 0])),
                         int(y > np.median(holdout[:, 1]))) for x, y in holdout[hinl, :2]}) if np.any(hinl) else 0
            sn = float(np.linalg.norm(rob[:2]))
            robust_valid = (int(hinl.sum()) >= minimum_robust_holdout_points
                            and iratio >= minimum_robust_inlier_ratio
                            and quad >= minimum_robust_spatial_quadrants
                            and sn <= maximum_robust_slope_norm)
            robust_marginal = (not robust_valid and sn <= marginal_robust_slope_norm
                               and int(hinl.sum()) >= 2 * minimum_robust_holdout_points
                               and iratio >= 2 * minimum_robust_inlier_ratio
                               and quad >= 4 and imed <= 4.0)
            robust_highconf = (not robust_valid and not robust_marginal
                                and sn <= maximum_robust_slope_norm
                                and int(hinl.sum()) >= 4 * minimum_robust_holdout_points
                                and imed <= 3.0 and quad >= 3)
            if robust_valid and imed < median:
                coeff, median, method = rob, imed, "deterministic_ransac_refit"
            elif robust_marginal and imed < median:
                coeff, median, method, marginal_accepted = rob, imed, "deterministic_ransac_marginal_slope", True
            elif robust_highconf and imed < median:
                coeff, median, method = rob, imed, "deterministic_ransac_highconf_inlier"
            audit = {"robust_holdout_inlier_ratio": iratio, "robust_holdout_inlier_median_mm": imed,
                     "robust_spatial_quadrants": quad, "robust_slope_norm": sn}
    final_slope = float(np.linalg.norm(coeff[:2]))
    if final_slope > maximum_robust_slope_norm and not marginal_accepted:
        return None, median, {"status": "rejected", "reason": "soil_plane_slope_exceeded",
                              "point_count": int(len(points)), "holdout_median_mm": median,
                              "method": method, "initial_slope_norm": initial_slope,
                              "final_slope_norm": final_slope, **audit}
    return coeff.astype(np.float64), median, {"status": "accepted", "point_count": int(len(points)),
                                              "holdout_median_mm": median, "method": method,
                                              "initial_slope_norm": initial_slope,
                                              "final_slope_norm": final_slope, **audit}


def soil_candidate_points(depth_mm: np.ndarray, mask: np.ndarray,
                          intrinsics: CameraIntrinsics) -> Optional[np.ndarray]:
    """Soil candidate 3D points: outside the canopy annulus, depth>0."""
    depth = np.asarray(depth_mm, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return None
    ys, xs = np.where(m)
    cx, cy = float(np.median(xs)), float(np.median(ys))
    radius = max(float(np.ptp(xs)), float(np.ptp(ys)), 40.0) * 0.75
    yy, xx = np.indices(m.shape, dtype=np.float64)
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    safe = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    cand = (radial >= radius * 0.78) & (radial <= radius * 1.2) & (~safe) & (depth > 0)
    if int(cand.sum()) < 200:
        cand = (radial <= radius * 1.2) & (~safe) & (depth > 0)
    if int(cand.sum()) < 50:
        return None
    xyz = deproject(depth, intrinsics)
    pts = xyz[cand]
    return pts[np.isfinite(pts).all(axis=1)]


# --------------------------------------------------------------------------
# Main geometry computation
# --------------------------------------------------------------------------
def compute_canopy_geometry(depth_mm: np.ndarray, canopy_mask: np.ndarray,
                            intrinsics: CameraIntrinsics, soil_plane: np.ndarray,
                            *, ground_frame: Optional[FixedGroundFrame] = None,
                            config: Optional[CanopyMeshConfig] = None,
                            soil_plane_uncertainty_mm: Optional[float] = None
                            ) -> CanopyGeometryTraits:
    """Mesh a depth+mask surface and derive height/area/volume (mm).

    ``soil_plane`` is ``[a, b, c]`` for ``Z = aX + bY + c``. Height = distance
    of a point above the soil plane along the ground normal. Raises ValueError
    on degenerate input (caller catches -> fail-closed).
    """
    cfg = config or CanopyMeshConfig()
    frame = ground_frame or FixedGroundFrame()
    rotation = frame.rotation()
    depth = np.asarray(depth_mm, dtype=np.float64)
    mask = np.asarray(canopy_mask, dtype=bool)
    if depth.shape != mask.shape or depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth/mask/intrinsics size mismatch")
    layers = np.zeros(mask.shape, dtype=np.int32)
    valid = (mask & np.isfinite(depth) & (depth >= cfg.depth_min_mm) & (depth <= cfg.depth_max_mm))
    mask_pixels = int(mask.sum())
    if mask_pixels == 0:
        raise ValueError("canopy_mask empty")

    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge = mask & ~eroded
    native_fraction = float(valid.sum() / mask_pixels)
    edge_fraction = float(valid[edge].mean()) if np.any(edge) else native_fraction
    xyz = deproject(depth, intrinsics)
    ground_xyz = np.einsum("ij,hwj->hwi", rotation, xyz)

    tris_cam, tris_gnd, tri_verts = [], [], []
    rejected_jump = 0
    rejected_layer = 0
    ys, xs = np.where(mask)
    y_min, y_max = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 1, mask.shape[0] - 1)
    x_min, x_max = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 1, mask.shape[1] - 1)
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            coords = tuple((y + dy, x + dx) for dy, dx in offsets)
            cell_valid = np.array([valid[c] for c in coords], dtype=bool)
            cell_xyz = np.array([xyz[c] for c in coords])
            for indices in _triangle_indices_for_cell(cell_valid, cell_xyz):
                sel = tuple(coords[i] for i in indices)
                sel_depth = np.array([depth[c] for c in sel])
                thr = max(cfg.surface_jump_abs_mm, cfg.surface_jump_rel * float(np.median(sel_depth)))
                if float(np.ptp(sel_depth)) > thr:
                    rejected_jump += 1
                    continue
                tris_cam.append(np.array([xyz[c] for c in sel]))
                tris_gnd.append(np.array([ground_xyz[c] for c in sel]))
                tri_verts.append(sel)

    if not tris_cam:
        raise ValueError("no canopy triangles passed continuity/layer gates")
    cam_tris = np.asarray(tris_cam)
    gnd_tris = np.asarray(tris_gnd)
    proj_xy = gnd_tris[:, :, :2]
    signed2 = ((proj_xy[:, 1, 0] - proj_xy[:, 0, 0]) * (proj_xy[:, 2, 1] - proj_xy[:, 0, 1])
               - (proj_xy[:, 1, 1] - proj_xy[:, 0, 1]) * (proj_xy[:, 2, 0] - proj_xy[:, 0, 0]))
    nondeg = np.abs(signed2) > 1.0e-8
    if not np.any(nondeg):
        raise ValueError("all ground-projected triangles degenerate")
    cam_tris, gnd_tris, proj_xy = cam_tris[nondeg], gnd_tris[nondeg], proj_xy[nondeg]
    tri_verts = [tri_verts[i] for i in np.flatnonzero(nondeg)]

    used = np.zeros(mask.shape, dtype=bool)
    for verts in tri_verts:
        for c in verts:
            used[c] = True
    mesh_vertex_coverage = float((used & valid).sum() / max(valid.sum(), 1))

    # projected area = sum of |signed projected triangle areas| (exact for a
    # non-overlapping height-field mesh; no shapely needed)
    proj_area = float(np.sum(np.abs(signed2[nondeg]) * 0.5))

    vis_surf, normal_support, capped_frac = _robust_visible_surface_area(
        valid=valid, depth=depth, ground_xyz=ground_xyz, layers=layers,
        intrinsics=intrinsics, rotation=rotation, config=cfg)

    vheights = _point_height(cam_tris.reshape(-1, 3), np.asarray(soil_plane)).reshape(-1, 3)
    _, envelope_vol, mean_h = _rasterize_envelope(
        proj_xy, vheights, [], [], cfg.envelope_grid_mm, cfg.maximum_envelope_cells)
    hvals = _point_height(xyz[valid], np.asarray(soil_plane))
    p95 = float(np.percentile(hvals, 95)) if len(hvals) else 0.0

    cov = []; h_reasons = []; s_reasons = []
    surf_ratio = vis_surf / max(proj_area, 1.0e-12)
    if native_fraction < cfg.minimum_native_depth_fraction:
        cov.append("low_native_depth_coverage")
    if edge_fraction < cfg.minimum_edge_depth_fraction:
        cov.append("low_edge_depth_coverage")
    if mesh_vertex_coverage < cfg.minimum_mesh_vertex_coverage_fraction:
        cov.append("low_mesh_vertex_coverage")
    if mean_h < cfg.minimum_mean_canopy_height_mm:
        h_reasons.append("mean_height_not_above_minimum")
    if p95 < cfg.minimum_p95_canopy_height_mm:
        h_reasons.append("p95_height_not_above_minimum")
    if (soil_plane_uncertainty_mm is not None
            and p95 <= cfg.minimum_height_to_soil_uncertainty_ratio * soil_plane_uncertainty_mm):
        h_reasons.append("p95_height_not_above_soil_uncertainty")
    if surf_ratio > cfg.maximum_surface_to_projected_area_ratio:
        s_reasons.append("implausible_surface_to_projected_ratio")
    if normal_support < cfg.minimum_surface_normal_support_fraction:
        s_reasons.append("low_surface_normal_support")
    if capped_frac > cfg.maximum_surface_inclination_capped_fraction:
        s_reasons.append("high_unresolved_surface_inclination")
    reasons = list(dict.fromkeys(cov + h_reasons + s_reasons))

    return CanopyGeometryTraits(
        canopy_projected_area_mm2=proj_area,
        visible_canopy_surface_area_mm2=vis_surf,
        canopy_envelope_volume_mm3=envelope_vol,
        mean_canopy_height_mm=mean_h,
        canopy_height_p95_mm=p95,
        surface_to_projected_area_ratio=surf_ratio,
        native_depth_fraction=native_fraction,
        edge_depth_fraction=edge_fraction,
        mesh_vertex_coverage_fraction=mesh_vertex_coverage,
        metric_authority="native_d435_only",
        geometry_qc_pass=not reasons,
        missing_reason=";".join(reasons),
    )


def compute_plant_height(depth_mm: np.ndarray, canopy_mask: np.ndarray,
                         intrinsics: CameraIntrinsics,
                         soil_plane: Optional[np.ndarray] = None,
                         *, ground_frame: Optional[FixedGroundFrame] = None,
                         config: Optional[CanopyMeshConfig] = None
                         ) -> CanopyGeometryTraits:
    """Convenience wrapper: fit the soil plane from the canopy annulus (if not
    given) then compute full canopy geometry. Returns :class:`CanopyGeometryTraits`
    with ``canopy_height_p95_mm`` / ``mean_canopy_height_mm`` as the headline."""
    if soil_plane is None:
        pts = soil_candidate_points(depth_mm, canopy_mask, intrinsics)
        if pts is None or len(pts) < 100:
            t = CanopyGeometryTraits()
            t.missing_reason = "soil_plane_unavailable"
            return t
        plane, _, _ = fit_soil_plane(pts)
        if plane is None:
            t = CanopyGeometryTraits()
            t.missing_reason = "soil_plane_fit_failed"
            return t
        soil_plane = plane
    return compute_canopy_geometry(
        depth_mm, canopy_mask, intrinsics, soil_plane,
        ground_frame=ground_frame, config=config)


@register
class CanopyGeometryExtractor(TraitExtractor):
    name = "canopy_3d_geometry"
    description = ("3D canopy geometry from depth + intrinsics: plant height "
                   "(mean / p95 above fitted soil plane), projected area, "
                   "visible surface area, envelope volume (mm). Fail-closed.")
    requires = [INPUT_MASK, INPUT_DEPTH, INPUT_CALIB]
    tier = 3

    def extract(self, *, mask=None, depth=None, calibration=None, **ctx):
        if depth is None or calibration is None:
            raise ValueError("depth and calibration are required for 3D canopy geometry")
        intrinsics = ctx.get("intrinsics")
        if intrinsics is None:
            if isinstance(calibration, CameraIntrinsics):
                intrinsics = calibration
            else:
                depth_arr = np.asarray(depth)
                intrinsics = load_rgb_intrinsics(
                    calibration, int(depth_arr.shape[1]), int(depth_arr.shape[0]))
        soil_plane = ctx.get("soil_plane")
        fixed_ground = ctx.get("fixed_ground") or FixedGroundFrame()
        mesh_config = ctx.get("mesh_config")
        try:
            traits = compute_plant_height(
                np.asarray(depth, dtype=np.float64),
                mask, intrinsics, soil_plane,
                ground_frame=fixed_ground, config=mesh_config,
            )
        except ValueError as exc:
            t = CanopyGeometryTraits()
            t.missing_reason = "l3_failed:%s" % exc
            return t.to_dict()
        return traits.to_dict()

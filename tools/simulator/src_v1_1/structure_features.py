from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import LayoutBlock, SimConfig


STRUCTURE_FEATURES = [
    "top_fill_ratio",
    "top_support_width_frac",
    "top_support_height_frac",
    "top_centroid_dx",
    "top_centroid_dy",
    "front_height_frac",
    "front_slenderness",
    "front_base_width_frac",
    "front_top_width_frac",
    "front_tilt",
    "front_top_heaviness",
    "collapse_margin_proxy",
]


def _clip_range(idx0: int, idx1: int, limit: int) -> tuple[int, int] | None:
    lo = max(0, idx0)
    hi = min(limit - 1, idx1)
    if lo > hi:
        return None
    return lo, hi


def _fill_top_mask(mask: np.ndarray, blocks: list[LayoutBlock], world_half: float) -> None:
    h, w = mask.shape
    span = 2.0 * world_half
    for blk in blocks:
        x0 = int(np.floor(((blk.x - blk.dx / 2.0) + world_half) / span * w))
        x1 = int(np.floor(((blk.x + blk.dx / 2.0) + world_half) / span * w))
        y0 = int(np.floor(((blk.y - blk.dy / 2.0) + world_half) / span * h))
        y1 = int(np.floor(((blk.y + blk.dy / 2.0) + world_half) / span * h))
        xr = _clip_range(x0, x1, w)
        yr = _clip_range(y0, y1, h)
        if xr is None or yr is None:
            continue
        mask[yr[0] : yr[1] + 1, xr[0] : xr[1] + 1] = True


def _fill_front_mask(mask: np.ndarray, blocks: list[LayoutBlock], world_half: float, z_max: float) -> None:
    h, w = mask.shape
    x_span = 2.0 * world_half
    z_span = max(z_max, 1e-6)
    for blk in blocks:
        x0 = int(np.floor(((blk.x - blk.dx / 2.0) + world_half) / x_span * w))
        x1 = int(np.floor(((blk.x + blk.dx / 2.0) + world_half) / x_span * w))
        z0 = int(np.floor((blk.z - blk.dz / 2.0) / z_span * h))
        z1 = int(np.floor((blk.z + blk.dz / 2.0) / z_span * h))
        xr = _clip_range(x0, x1, w)
        zr = _clip_range(z0, z1, h)
        if xr is None or zr is None:
            continue
        # Front view uses image coordinates: top row is high z.
        y0 = h - 1 - zr[1]
        y1 = h - 1 - zr[0]
        yr = _clip_range(y0, y1, h)
        if yr is None:
            continue
        mask[yr[0] : yr[1] + 1, xr[0] : xr[1] + 1] = True


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _band_width_center(mask: np.ndarray, y1: int, y2: int) -> tuple[float, float]:
    h, w = mask.shape
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if y2 <= y1:
        return 0.0, 0.0
    band = mask[y1:y2, :]
    ys, xs = np.where(band)
    if len(xs) == 0:
        return 0.0, 0.0
    width = float(xs.max() - xs.min() + 1) / max(w, 1)
    center = float(xs.mean() / max(w - 1, 1))
    return width, center


def collapse_margin_from_features(features: dict[str, float]) -> float:
    raw = (
        1.20 * features["top_support_width_frac"]
        + 0.90 * features["front_base_width_frac"]
        + 0.50 * features["top_fill_ratio"]
        - 0.75 * abs(features["top_centroid_dx"])
        - 0.55 * abs(features["front_tilt"])
        - 0.20 * features["front_slenderness"]
        - 0.25 * features["front_top_heaviness"]
    )
    return float(np.clip(0.5 + 0.35 * raw, 0.0, 1.0))


def extract_structure_features(layout: list[LayoutBlock], cfg: SimConfig, grid_size: int = 192) -> dict[str, float]:
    if not layout:
        return {name: 0.0 for name in STRUCTURE_FEATURES}

    top_mask = np.zeros((grid_size, grid_size), dtype=bool)
    front_mask = np.zeros((grid_size, grid_size), dtype=bool)

    _fill_top_mask(top_mask, layout, world_half=cfg.world_half)
    z_max = max(2.0 * cfg.world_half, max(blk.z + blk.dz / 2.0 for blk in layout) + cfg.block_edge)
    _fill_front_mask(front_mask, layout, world_half=cfg.world_half, z_max=z_max)

    top_bbox = _bbox(top_mask)
    if top_bbox is None:
        top_fill_ratio = 0.0
        top_support_width_frac = 0.0
        top_support_height_frac = 0.0
        top_centroid_dx = 0.0
        top_centroid_dy = 0.0
    else:
        tx1, ty1, tx2, ty2 = top_bbox
        area = float(top_mask.mean())
        bbox_area = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1))
        ys, xs = np.where(top_mask)
        top_fill_ratio = float(top_mask.sum() / max(bbox_area, 1.0))
        top_support_width_frac = float((tx2 - tx1 + 1) / grid_size)
        top_support_height_frac = float((ty2 - ty1 + 1) / grid_size)
        top_centroid_dx = float((xs.mean() / max(grid_size - 1, 1) - 0.5) * 2.0) if len(xs) else 0.0
        top_centroid_dy = float((ys.mean() / max(grid_size - 1, 1) - 0.5) * 2.0) if len(ys) else 0.0
        _ = area

    front_bbox = _bbox(front_mask)
    if front_bbox is None:
        front_height_frac = 0.0
        front_width_frac = 0.0
        front_slenderness = 0.0
        front_base_width_frac = 0.0
        front_top_width_frac = 0.0
        front_tilt = 0.0
        front_top_heaviness = 0.0
    else:
        fx1, fy1, fx2, fy2 = front_bbox
        bbox_h_px = fy2 - fy1 + 1
        bbox_w_px = fx2 - fx1 + 1
        front_height_frac = float(bbox_h_px / grid_size)
        front_width_frac = float(bbox_w_px / grid_size)
        front_slenderness = float(bbox_h_px / max(bbox_w_px, 1))
        top_band_h = max(1, int(round(0.25 * bbox_h_px)))
        base_band_h = max(1, int(round(0.20 * bbox_h_px)))
        front_top_width_frac, top_center = _band_width_center(front_mask, fy1, fy1 + top_band_h)
        front_base_width_frac, base_center = _band_width_center(front_mask, fy2 - base_band_h + 1, fy2 + 1)
        front_tilt = float(top_center - base_center)

        mid = fy1 + bbox_h_px // 2
        top_pixels = float(front_mask[fy1:mid, :].sum())
        base_pixels = float(front_mask[mid : fy2 + 1, :].sum())
        front_top_heaviness = float(top_pixels / max(top_pixels + base_pixels, 1.0))

    features = {
        "top_fill_ratio": float(top_fill_ratio),
        "top_support_width_frac": float(top_support_width_frac),
        "top_support_height_frac": float(top_support_height_frac),
        "top_centroid_dx": float(top_centroid_dx),
        "top_centroid_dy": float(top_centroid_dy),
        "front_height_frac": float(front_height_frac),
        "front_width_frac": float(front_width_frac),
        "front_slenderness": float(front_slenderness),
        "front_base_width_frac": float(front_base_width_frac),
        "front_top_width_frac": float(front_top_width_frac),
        "front_tilt": float(front_tilt),
        "front_top_heaviness": float(front_top_heaviness),
    }
    features["collapse_margin_proxy"] = collapse_margin_from_features(features)

    return {name: float(features.get(name, 0.0)) for name in STRUCTURE_FEATURES}


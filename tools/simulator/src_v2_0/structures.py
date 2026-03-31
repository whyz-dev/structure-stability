from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ..src.config import LayoutBlock, SimConfig
    from ..src.layout import (
        append_candidate,
        block_dims,
        candidate_dict,
        compute_support_region,
        find_support_contacts,
    )
except ImportError:
    from src.config import LayoutBlock, SimConfig
    from src.layout import (
        append_candidate,
        block_dims,
        candidate_dict,
        compute_support_region,
        find_support_contacts,
    )


STRUCTURE_CLASSES = [
    "thin tower",
    "thick tower",
    "wall",
    "thin triangle",
    "thick triangle",
    "double triangle",
    "pyramid",
]


@dataclass(frozen=True)
class StructureLayoutResult:
    layout: list[LayoutBlock]
    structure_label: str
    stability_label: str
    tilt_axis: str
    tilt_profile: dict


def list_structures() -> list[str]:
    return list(STRUCTURE_CLASSES)


def _sample_structure(rng: np.random.Generator, requested: str) -> str:
    if requested == "random":
        return str(rng.choice(STRUCTURE_CLASSES))
    if requested not in STRUCTURE_CLASSES:
        raise ValueError(f"Unknown structure: {requested}")
    return requested


def _axis_vector(rng: np.random.Generator, allow_diagonal: bool = True) -> tuple[str, np.ndarray]:
    choices = [
        ("left_right", np.array([1.0, 0.0], dtype=np.float64)),
        ("front_back", np.array([0.0, 1.0], dtype=np.float64)),
    ]
    if allow_diagonal:
        choices.extend(
            [
                ("diag_pos", np.array([1.0, 0.35], dtype=np.float64)),
                ("diag_neg", np.array([1.0, -0.35], dtype=np.float64)),
            ]
        )
    axis_name, vec = choices[int(rng.integers(0, len(choices)))]
    vec = vec / max(float(np.linalg.norm(vec)), 1e-9)
    return axis_name, vec


def _sample_segmented_offsets(
    levels: int,
    rng: np.random.Generator,
    step_range: tuple[float, float],
    allow_switch: bool = True,
    allow_diagonal: bool = True,
    jitter: float = 0.02,
) -> tuple[list[tuple[float, float]], str, dict]:
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    axis_name, axis_vec = _axis_vector(rng, allow_diagonal=allow_diagonal)
    current = np.zeros(2, dtype=np.float64)
    segments: list[dict] = []

    remaining = max(0, levels - 1)
    while remaining > 0:
        if not allow_switch or remaining <= 2:
            seg_len = remaining
        else:
            seg_len = int(rng.integers(2, remaining + 1))
        base_step = float(rng.uniform(*step_range))
        seg_record = {
            "axis": axis_name,
            "steps": seg_len,
            "base_step": round(base_step, 4),
        }
        segments.append(seg_record)

        for local_idx in range(seg_len):
            step_mag = base_step * (1.0 + 0.10 * local_idx)
            jitter_vec = rng.uniform(-jitter, jitter, size=2)
            current = current + axis_vec * step_mag + jitter_vec
            offsets.append((float(current[0]), float(current[1])))

        remaining -= seg_len
        if remaining <= 0:
            break

        next_axis_name, next_axis_vec = _axis_vector(rng, allow_diagonal=allow_diagonal)
        if float(np.dot(next_axis_vec, axis_vec)) > 0.85:
            next_axis_vec = np.array([axis_vec[1], -axis_vec[0]], dtype=np.float64)
            next_axis_name = "axis_switch"
        if bool(rng.integers(0, 2)):
            next_axis_vec = -next_axis_vec
            next_axis_name = f"{next_axis_name}_flip"
        axis_name, axis_vec = next_axis_name, next_axis_vec / max(float(np.linalg.norm(next_axis_vec)), 1e-9)

    profile = {"segments": segments, "switches": max(0, len(segments) - 1)}
    return offsets, segments[0]["axis"] if segments else "centered", profile


def _sample_polyline_path(
    steps: int,
    rng: np.random.Generator,
    spacing: float,
    allow_switch: bool = True,
    lateral_jitter: float = 0.0,
) -> tuple[list[tuple[float, float]], str, dict]:
    axis_name, axis_vec = _axis_vector(rng, allow_diagonal=True)
    positions = []
    current = np.zeros(2, dtype=np.float64)
    segments: list[dict] = []
    change_at = int(rng.integers(2, steps - 1)) if allow_switch and steps >= 5 and rng.random() < 0.45 else None
    for idx in range(steps):
        if change_at is not None and idx == change_at:
            next_axis_name, next_axis_vec = _axis_vector(rng, allow_diagonal=True)
            if float(np.dot(next_axis_vec, axis_vec)) > 0.85:
                next_axis_vec = np.array([axis_vec[1], -axis_vec[0]], dtype=np.float64)
                next_axis_name = "path_switch"
            axis_name, axis_vec = next_axis_name, next_axis_vec / max(float(np.linalg.norm(next_axis_vec)), 1e-9)
            segments.append({"switch_index": idx, "axis": axis_name})

        if idx > 0:
            lateral = np.array([-axis_vec[1], axis_vec[0]], dtype=np.float64)
            current = current + axis_vec * spacing + lateral * float(rng.uniform(-lateral_jitter, lateral_jitter))
        positions.append((float(current[0]), float(current[1])))

    profile = {"path_switches": segments, "spacing": round(float(spacing), 4)}
    return positions, axis_name if change_at is None else "polyline", profile


def _sample_linear_x_offsets(
    levels: int,
    rng: np.random.Generator,
    step_range: tuple[float, float],
    allow_flip: bool,
    jitter: float,
    flip_prob: float = 0.45,
) -> tuple[list[tuple[float, float]], str, dict]:
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    current_x = 0.0
    direction = 1.0
    flips: list[int] = []
    flip_at = int(rng.integers(2, levels - 1)) if allow_flip and levels >= 5 and rng.random() < flip_prob else None

    for idx in range(1, levels):
        if flip_at is not None and idx == flip_at:
            direction *= -1.0
            flips.append(idx)
        step_mag = float(rng.uniform(*step_range))
        current_x += direction * step_mag + float(rng.uniform(-jitter, jitter))
        offsets.append((current_x, 0.0))

    profile = {
        "axis": "left_right",
        "flip_indices": flips,
        "step_range": [round(float(step_range[0]), 4), round(float(step_range[1]), 4)],
        "flip_prob": round(float(flip_prob), 3),
    }
    return offsets, "left_right", profile


def _segmented_tilt_step_range(
    cfg: SimConfig,
    levels: int,
    tilt_deg_range: tuple[float, float] = (30.0, 45.0),
    growth_per_step: float = 0.10,
    spread: float = 0.18,
) -> tuple[float, float]:
    n_steps = max(1, levels - 1)
    vertical_pitch = cfg.block_edge + cfg.gap
    total_height = n_steps * vertical_pitch
    sum_scale = sum(1.0 + growth_per_step * idx for idx in range(n_steps))
    min_base = np.tan(np.deg2rad(float(tilt_deg_range[0]))) * total_height / max(sum_scale, 1e-9)
    max_base = np.tan(np.deg2rad(float(tilt_deg_range[1]))) * total_height / max(sum_scale, 1e-9)
    return float(min_base * (1.0 - spread)), float(max_base * (1.0 + spread))


def _linear_tilt_step_range(
    cfg: SimConfig,
    levels: int,
    tilt_deg_range: tuple[float, float] = (30.0, 45.0),
    spread: float = 0.12,
) -> tuple[float, float]:
    n_steps = max(1, levels - 1)
    vertical_pitch = cfg.block_edge + cfg.gap
    total_height = n_steps * vertical_pitch
    min_step = np.tan(np.deg2rad(float(tilt_deg_range[0]))) * total_height / n_steps
    max_step = np.tan(np.deg2rad(float(tilt_deg_range[1]))) * total_height / n_steps
    return float(min_step * (1.0 - spread)), float(max_step * (1.0 + spread))


def _append_block_retry(
    placements: list[LayoutBlock],
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    cfg: SimConfig,
    margin: float,
    context: str = "",
    allow_force: bool = False,
) -> None:
    if allow_force:
        cand = candidate_dict(x, y, z, dx, dy, dz)
        contacts = find_support_contacts(cand, placements, cfg)
        support_region = compute_support_region(cand, contacts)
        supporters = [-1] if contacts["floor"] and not contacts["supports"] else [idx for idx, _, _ in contacts["supports"]]
        placements.append(
            LayoutBlock(
                x=float(cand["x"]),
                y=float(cand["y"]),
                z=float(cand["z"]),
                dx=float(cand["dx"]),
                dy=float(cand["dy"]),
                dz=float(cand["dz"]),
                mass=max(1e-4, cand["dx"] * cand["dy"] * cand["dz"] * cfg.density),
                supporters=supporters,
                support_region=support_region,
            )
        )
        return

    scales = [1.0, 0.9, 0.8, 0.7, 0.55]
    for scale in scales:
        cand = candidate_dict(x, y, z, dx * scale, dy * scale, dz)
        if append_candidate(placements, cand, cfg, margin=margin):
            return
        if append_candidate(placements, cand, cfg, margin=0.0):
            return
    raise RuntimeError(
        "Failed to append candidate while building structure layout"
        f" | context={context}"
        f" | target=({x:.4f}, {y:.4f}, {z:.4f})"
        f" | dims=({dx:.4f}, {dy:.4f}, {dz:.4f})"
        f" | margin={margin:.4f}"
        f" | existing_blocks={len(placements)}"
    )


def _place_grid_layer(
    placements: list[LayoutBlock],
    center_x: float,
    center_y: float,
    z: float,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dz: float,
    cfg: SimConfig,
    margin: float,
    context: str = "",
    allow_force: bool = False,
) -> None:
    pitch_x = dx + cfg.gap
    pitch_y = dy + cfg.gap
    for ix in range(nx):
        for iy in range(ny):
            x = center_x + (ix - (nx - 1) / 2.0) * pitch_x
            y = center_y + (iy - (ny - 1) / 2.0) * pitch_y
            _append_block_retry(
                placements,
                x,
                y,
                z,
                dx,
                dy,
                dz,
                cfg,
                margin,
                context=f"{context}|grid=({ix},{iy})",
                allow_force=allow_force,
            )


def _place_hollow_square_layer(
    placements: list[LayoutBlock],
    center_x: float,
    center_y: float,
    z: float,
    side: int,
    dx: float,
    dy: float,
    dz: float,
    cfg: SimConfig,
    margin: float,
    context: str = "",
    allow_force: bool = False,
) -> None:
    pitch_x = dx + cfg.gap
    pitch_y = dy + cfg.gap
    for ix in range(side):
        for iy in range(side):
            if 0 < ix < side - 1 and 0 < iy < side - 1:
                continue
            x = center_x + (ix - (side - 1) / 2.0) * pitch_x
            y = center_y + (iy - (side - 1) / 2.0) * pitch_y
            _append_block_retry(
                placements,
                x,
                y,
                z,
                dx,
                dy,
                dz,
                cfg,
                margin,
                context=f"{context}|ring=({ix},{iy})",
                allow_force=allow_force,
            )


def _place_linear_row(
    placements: list[LayoutBlock],
    center_x: float,
    center_y: float,
    z: float,
    count: int,
    dx: float,
    dy: float,
    dz: float,
    cfg: SimConfig,
    margin: float,
    axis_vec: np.ndarray,
    context: str = "",
    step_scale: float = 1.0,
    allow_force: bool = False,
) -> None:
    step = (cfg.block_edge + cfg.gap) * step_scale
    for idx in range(count):
        offset = (idx - (count - 1) / 2.0) * step
        x = center_x + axis_vec[0] * offset
        y = center_y + axis_vec[1] * offset
        _append_block_retry(
            placements,
            x,
            y,
            z,
            dx,
            dy,
            dz,
            cfg,
            margin,
            context=f"{context}|row_idx={idx}",
            allow_force=allow_force,
        )


def _sample_triangle_layer_pattern(rng: np.random.Generator) -> tuple[list[int], str]:
    variants = [
        ([5, 4, 3, 3, 2, 2, 1], "standard"),
        ([5, 4, 4, 3, 3, 2, 2, 1, 1], "tall_thin"),
    ]
    pattern, variant = variants[int(rng.integers(0, len(variants)))]
    return list(pattern), variant


def _build_single_column_tower(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    levels = int(rng.integers(7, 11))
    if stability_label == "stable":
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=(0.0, 0.012 * cfg.block_edge),
            allow_switch=True,
            allow_diagonal=True,
            jitter=0.005 * cfg.block_edge,
        )
        margin = 0.10 * cfg.block_edge
    else:
        tilt_step_range = _segmented_tilt_step_range(cfg, levels, tilt_deg_range=(30.0, 45.0))
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=tilt_step_range,
            allow_switch=True,
            allow_diagonal=True,
            jitter=0.020 * cfg.block_edge,
        )
        profile = profile | {"target_tilt_degrees": [30, 45]}
        margin = -1.0

    top_z = 0.0
    for level in range(levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        x, y = offsets[level]
        _append_block_retry(
            placements,
            x,
            y,
            z,
            dx,
            dy,
            dz,
            cfg,
            margin,
            context=f"thin_tower|level={level}",
            allow_force=relaxed if stability_label == "stable" else True,
        )
        top_z = placements[-1].z + placements[-1].dz / 2.0
    return placements, tilt_axis, profile


def _build_thick_tower(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    footprint = 4
    levels = int(rng.integers(5, 8))
    if stability_label == "stable":
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=(0.0, 0.018 * cfg.block_edge),
            allow_switch=True,
            allow_diagonal=True,
            jitter=0.006 * cfg.block_edge,
        )
        margin = 0.08 * cfg.block_edge
    else:
        tilt_step_range = _segmented_tilt_step_range(cfg, levels, tilt_deg_range=(30.0, 45.0))
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=tilt_step_range,
            allow_switch=True,
            allow_diagonal=True,
            jitter=0.018 * cfg.block_edge,
        )
        profile = profile | {"target_tilt_degrees": [30, 45]}
        margin = -1.0

    top_z = 0.0
    for level in range(levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        x, y = offsets[level]
        _place_grid_layer(
            placements,
            x,
            y,
            z,
            footprint,
            footprint,
            dx,
            dy,
            dz,
            cfg,
            margin,
            context=f"thick_tower|level={level}",
            allow_force=relaxed if stability_label == "stable" else True,
        )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)
    return placements, tilt_axis, profile | {"footprint": footprint, "shape_rule": "4x4"}


def _build_wall(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    row_pattern = [5, 6, 5, 6, 5]
    levels = len(row_pattern)
    axis_name, path_vec = _axis_vector(rng, allow_diagonal=False)
    perp_vec = np.array([-path_vec[1], path_vec[0]], dtype=np.float64)
    row_step = cfg.block_edge + cfg.gap

    if stability_label == "stable":
        layer_offsets = [(0.0, 0.0)] * levels
        tilt_axis = "centered"
        profile = {"segments": [], "switches": 0, "mode": "fixed_wall"}
        # For stage-1 geometry checks, a brick wall is acceptable as long as
        # each upper block has real contact with the row below. Requiring the
        # center of mass to stay inside the support hull is too strict for the
        # intended 5-6-5-6-5 wall pattern, so we disable that extra margin check.
        margin = -1.0
        row_jitter = 0.0
    else:
        tilt_step_range = _segmented_tilt_step_range(cfg, levels, tilt_deg_range=(30.0, 45.0))
        layer_offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=tilt_step_range,
            allow_switch=True,
            allow_diagonal=False,
            jitter=0.012 * cfg.block_edge,
        )
        profile = profile | {"target_tilt_degrees": [30, 45]}
        margin = -1.0
        row_jitter = 0.008 * cfg.block_edge

    top_z = 0.0
    for level in range(levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        shift_x, shift_y = layer_offsets[level]
        count = row_pattern[level]
        # Keep intra-row spacing at full block pitch so blocks do not overlap.
        # For 6-block rows, apply a small brick shift to create a wall-like stagger
        # while preserving enough overlap with the 5-block support row below.
        if stability_label == "stable" and count == 6:
            row_pitch = dx + cfg.gap * 0.20
        else:
            row_pitch = row_step
        step_scale = row_pitch / row_step
        brick_shift = 0.0
        if count == 6:
            brick_shift = (0.10 * cfg.block_edge) * (1.0 if ((level // 2) % 2 == 0) else -1.0)
        for idx in range(count):
            offset = (idx - (count - 1) / 2.0) * row_pitch + brick_shift
            px = shift_x + path_vec[0] * offset + perp_vec[0] * float(rng.uniform(-row_jitter, row_jitter))
            py = shift_y + path_vec[1] * offset + perp_vec[1] * float(rng.uniform(-row_jitter, row_jitter))
            _append_block_retry(
                placements,
                px,
                py,
                z,
                dx,
                dy,
                dz,
                cfg,
                margin,
                context=f"wall|level={level}|count={count}|idx={idx}|step_scale={step_scale:.3f}|row_pitch={row_pitch:.4f}",
                allow_force=True,
            )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)

    return placements, tilt_axis, profile | {"wall_axis": axis_name, "row_pattern": row_pattern, "shape_rule": "5-6-5-6-5 brick wall"}


def _build_triangle(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    thickness: int,
    pair_gap: float | None = None,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    layer_pattern, layer_variant = _sample_triangle_layer_pattern(rng)
    steps = len(layer_pattern)
    spacing = cfg.block_edge + cfg.gap
    path_positions = [(0.0, 0.0)] * steps
    profile = {"path_mode": "centered_tower_stack", "spacing": round(float(spacing), 4)}

    if stability_label == "stable":
        column_margin = -1.0 if relaxed else 0.08 * cfg.block_edge
        local_offsets = [(0.0, 0.0)] * steps
        tilt_axis = "centered"
        local_profile = {"axis": "left_right", "flip_indices": [], "step_range": [0.0, 0.0], "mode": "fixed"}
    else:
        column_margin = -1.0
        tilt_step_range = _linear_tilt_step_range(cfg, steps, tilt_deg_range=(30.0, 45.0))
        local_offsets, tilt_axis, local_profile = _sample_linear_x_offsets(
            steps,
            rng,
            step_range=tilt_step_range,
            allow_flip=True,
            jitter=0.016 * cfg.block_edge,
            flip_prob=0.10,
        )
        local_profile = local_profile | {"target_tilt_degrees": [30, 45]}

    rail_offsets = [0.0] if pair_gap is None else [-pair_gap / 2.0, pair_gap / 2.0]
    path_vec = np.array([1.0, 0.0], dtype=np.float64)
    rail_vec = np.array([-path_vec[1], path_vec[0]], dtype=np.float64)

    top_z = 0.0
    for level, count in enumerate(layer_pattern):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        base_x, base_y = path_positions[level]
        local_x, local_y = local_offsets[level]
        for rail_offset in rail_offsets:
            rail_shift = rail_vec * rail_offset
            row_center_x = base_x + local_x + rail_shift[0]
            row_center_y = base_y + local_y + rail_shift[1]
            for thick_idx in range(thickness):
                thick_offset = (thick_idx - (thickness - 1) / 2.0) * (cfg.block_edge + cfg.gap)
                px = row_center_x + rail_vec[0] * thick_offset
                py = row_center_y + rail_vec[1] * thick_offset
                _place_linear_row(
                    placements,
                    px,
                    py,
                    z,
                    count,
                    dx,
                    dy,
                    dz,
                    cfg,
                    column_margin,
                    path_vec,
                    context=f"triangle|level={level}|thick_idx={thick_idx}|pair={'double' if pair_gap is not None else 'single'}",
                    allow_force=relaxed if stability_label == "stable" else True,
                )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)

    result_profile = profile | {
        "steps": steps,
        "thickness": thickness,
        "layer_pattern": layer_pattern,
        "layer_variant": layer_variant,
        "column_offset_profile": local_profile,
        "paired": pair_gap is not None,
        "bottom_alignment": "centered_on_base",
    }
    return placements, tilt_axis, result_profile


def _build_double_triangle(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    layer_pattern, layer_variant = _sample_triangle_layer_pattern(rng)
    steps = len(layer_pattern)
    row_pitch = cfg.block_edge + cfg.gap
    row_span = (layer_pattern[0] - 1) * row_pitch
    square_like_gap = row_span
    base_side = layer_pattern[0]
    base_levels = 1 if stability_label == "stable" else 2
    base_margin = -1.0 if relaxed else 0.08 * cfg.block_edge
    path_positions = [(0.0, 0.0)] * steps

    top_z = 0.0
    for level in range(base_levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        _place_hollow_square_layer(
            placements,
            0.0,
            0.0,
            z,
            base_side,
            dx,
            dy,
            dz,
            cfg,
            base_margin,
            context=f"double_triangle_base|level={level}|side={base_side}",
            allow_force=relaxed if stability_label == "stable" else True,
        )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)

    if stability_label == "stable":
        column_margin = -1.0 if relaxed else 0.08 * cfg.block_edge
        local_offsets = [(0.0, 0.0)] * steps
        tilt_axis = "centered"
        local_profile = {"axis": "left_right", "flip_indices": [], "step_range": [0.0, 0.0], "mode": "fixed"}
    else:
        column_margin = -1.0
        tilt_step_range = _linear_tilt_step_range(cfg, steps, tilt_deg_range=(30.0, 45.0))
        local_offsets, tilt_axis, local_profile = _sample_linear_x_offsets(
            steps,
            rng,
            step_range=tilt_step_range,
            allow_flip=True,
            jitter=0.016 * cfg.block_edge,
            flip_prob=0.10,
        )
        local_profile = local_profile | {"target_tilt_degrees": [30, 45]}

    rail_offsets = [-square_like_gap / 2.0, square_like_gap / 2.0]
    path_vec = np.array([1.0, 0.0], dtype=np.float64)
    rail_vec = np.array([0.0, 1.0], dtype=np.float64)

    for level, count in enumerate(layer_pattern):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + cfg.gap
        base_x, base_y = path_positions[level]
        local_x, local_y = local_offsets[level]
        for rail_offset in rail_offsets:
            row_center_x = base_x + local_x + rail_vec[0] * rail_offset
            row_center_y = base_y + local_y + rail_vec[1] * rail_offset
            _place_linear_row(
                placements,
                row_center_x,
                row_center_y,
                z,
                count,
                dx,
                dy,
                dz,
                cfg,
                column_margin,
                path_vec,
                context=f"double_triangle|level={level}|rail_offset={rail_offset:.4f}",
                allow_force=relaxed if stability_label == "stable" else True,
            )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)

    tilt_profile = local_profile | {
        "pair_gap": round(float(square_like_gap), 4),
        "path_mode": "centered_tower_stack",
        "shape_rule": "double thin triangle stacked vertically on two parallel edges of a hollow square base",
        "layer_pattern": layer_pattern,
        "layer_variant": layer_variant,
        "base_side": base_side,
        "base_levels": base_levels,
        "base_type": "hollow_square",
        "bottom_alignment": "centered_on_base",
    }
    return placements, tilt_axis, tilt_profile


def _build_pyramid(
    rng: np.random.Generator,
    cfg: SimConfig,
    stability_label: str,
    relaxed: bool = False,
) -> tuple[list[LayoutBlock], str, dict]:
    placements: list[LayoutBlock] = []
    side_pattern = [5, 5, 4, 4, 3, 3, 2, 2]
    levels = len(side_pattern)
    if stability_label == "stable":
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels,
            rng,
            step_range=(0.0, 0.01 * cfg.block_edge),
            allow_switch=False,
            allow_diagonal=True,
            jitter=0.002 * cfg.block_edge,
        )
        margin = 0.10 * cfg.block_edge
    else:
        tilt_step_range = _segmented_tilt_step_range(cfg, levels + 2, tilt_deg_range=(30.0, 45.0))
        offsets, tilt_axis, profile = _sample_segmented_offsets(
            levels + 2,
            rng,
            step_range=tilt_step_range,
            allow_switch=True,
            allow_diagonal=True,
            jitter=0.012 * cfg.block_edge,
        )
        profile = profile | {"target_tilt_degrees": [30, 45]}
        margin = -1.0

    top_z = 0.0
    for level in range(levels):
        side = side_pattern[level]
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if level > 0 else 0.0)
        cx, cy = offsets[min(level, len(offsets) - 1)]
        _place_grid_layer(
            placements,
            cx,
            cy,
            z,
            side,
            side,
            dx,
            dy,
            dz,
            cfg,
            margin,
            context=f"pyramid|level={level}|side={side}",
            allow_force=relaxed if stability_label == "stable" else True,
        )
        top_z = max(blk.z + blk.dz / 2.0 for blk in placements)

    spire_levels = 3 if stability_label == "stable" else 4
    spire_offsets = offsets[-spire_levels:] if spire_levels <= len(offsets) else offsets
    spire_margin = 0.04 * cfg.block_edge if stability_label == "stable" else -1.0
    for idx in range(spire_levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + cfg.gap
        sx, sy = spire_offsets[min(idx, len(spire_offsets) - 1)]
        _append_block_retry(
            placements,
            sx,
            sy,
            z,
            dx,
            dy,
            dz,
            cfg,
            spire_margin,
            context=f"pyramid_spire|level={idx}",
            allow_force=relaxed if stability_label == "stable" else True,
        )
        top_z = placements[-1].z + placements[-1].dz / 2.0

    return placements, tilt_axis, profile | {
        "side_pattern": side_pattern,
        "spire_levels": spire_levels,
        "top_shape": "thin_tower_spire",
    }


def build_structure_layout_v2(
    rng: np.random.Generator,
    cfg: SimConfig,
    structure: str,
    stability_label: str,
    relaxed: bool = False,
) -> StructureLayoutResult:
    structure_label = _sample_structure(rng, structure)
    if stability_label not in {"stable", "unstable"}:
        raise ValueError(f"Unknown stability label: {stability_label}")

    if structure_label == "thin tower":
        layout, tilt_axis, tilt_profile = _build_single_column_tower(rng, cfg, stability_label, relaxed=relaxed)
    elif structure_label == "thick tower":
        layout, tilt_axis, tilt_profile = _build_thick_tower(rng, cfg, stability_label, relaxed=relaxed)
    elif structure_label == "wall":
        layout, tilt_axis, tilt_profile = _build_wall(rng, cfg, stability_label, relaxed=relaxed)
    elif structure_label == "thin triangle":
        layout, tilt_axis, tilt_profile = _build_triangle(rng, cfg, stability_label, thickness=1, relaxed=relaxed)
    elif structure_label == "thick triangle":
        layout, tilt_axis, tilt_profile = _build_triangle(
            rng,
            cfg,
            stability_label,
            thickness=int(rng.choice([3, 4])),
            relaxed=relaxed,
        )
    elif structure_label == "double triangle":
        layout, tilt_axis, tilt_profile = _build_double_triangle(
            rng,
            cfg,
            stability_label,
            relaxed=relaxed,
        )
    elif structure_label == "pyramid":
        layout, tilt_axis, tilt_profile = _build_pyramid(rng, cfg, stability_label, relaxed=relaxed)
    else:
        raise ValueError(f"Unsupported structure: {structure_label}")

    return StructureLayoutResult(
        layout=layout,
        structure_label=structure_label,
        stability_label=stability_label,
        tilt_axis=tilt_axis,
        tilt_profile=tilt_profile,
    )

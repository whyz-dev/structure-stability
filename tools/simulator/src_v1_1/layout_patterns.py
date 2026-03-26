from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import LayoutBlock, SimConfig
from src.layout import (
    append_candidate,
    block_dims,
    build_stable_layout,
    build_unstable_bridge_layout,
    build_unstable_cantilever_layout,
    build_unstable_tower_layout,
    candidate_dict,
)


@dataclass(frozen=True)
class PatternSpec:
    name: str
    mode: str


def _stack_vertical(
    placements: list[LayoutBlock],
    x: float,
    y: float,
    levels: int,
    rng: np.random.Generator,
    cfg: SimConfig,
    margin_scale: float = 0.02,
) -> None:
    top_z = 0.0
    for i in range(levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + (cfg.gap if i > 0 else 0.0)
        append_candidate(placements, candidate_dict(x, y, z, dx, dy, dz), cfg, margin=margin_scale * min(dx, dy))
        top_z = placements[-1].z + placements[-1].dz / 2.0


def build_stable_ziggurat_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    side = int(rng.integers(3, 5))
    levels = int(rng.integers(3, 5))
    top_z = 0.0
    for level in range(levels):
        n = max(1, side - level)
        dims = [block_dims(rng, cfg) for _ in range(n * n)]
        sx = max(d[0] for d in dims) + cfg.gap
        sy = max(d[1] for d in dims) + cfg.gap
        max_h = max(d[2] for d in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if level > 0 else 0.0)
        for i, (dx, dy, dz) in enumerate(dims):
            ix = i % n
            iy = i // n
            x = (ix - (n - 1) / 2.0) * sx
            y = (iy - (n - 1) / 2.0) * sy
            append_candidate(placements, candidate_dict(x, y, z, dx, dy, dz), cfg, margin=0.03 * min(dx, dy))
        top_z = z + max_h / 2.0
    return placements


def build_stable_double_column_lintel_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    span = cfg.block_edge * float(rng.uniform(2.1, 2.8))
    left_x = -span / 2.0
    right_x = span / 2.0
    col_levels = int(rng.integers(4, 7))
    _stack_vertical(placements, left_x, 0.0, col_levels, rng, cfg, margin_scale=0.02)
    _stack_vertical(placements, right_x, 0.0, col_levels, rng, cfg, margin_scale=0.02)

    top = max(blk.z + blk.dz / 2.0 for blk in placements)
    dx, dy, dz = block_dims(rng, cfg)
    z = top + dz / 2.0 + cfg.gap
    n_beam = max(2, int(np.ceil(span / max(dx, 1e-6))) + 1)
    for x in np.linspace(left_x, right_x, n_beam):
        append_candidate(placements, candidate_dict(float(x), 0.0, z, dx, dy, dz), cfg, margin=0.0)

    dx2, dy2, dz2 = block_dims(rng, cfg)
    z2 = placements[-1].z + placements[-1].dz / 2.0 + dz2 / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(0.0, 0.0, z2, dx2, dy2, dz2), cfg, margin=0.02 * min(dx2, dy2))
    return placements


def build_stable_ring_core_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    ring_n = int(rng.integers(6, 9))
    radius = cfg.block_edge * float(rng.uniform(1.1, 1.5))
    levels = int(rng.integers(3, 5))

    top_z = 0.0
    for level in range(levels):
        dims = [block_dims(rng, cfg) for _ in range(ring_n)]
        max_h = max(d[2] for d in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if level > 0 else 0.0)
        for i, (dx, dy, dz) in enumerate(dims):
            angle = 2.0 * np.pi * i / ring_n
            x = float(np.cos(angle) * radius)
            y = float(np.sin(angle) * radius)
            append_candidate(placements, candidate_dict(x, y, z, dx, dy, dz), cfg, margin=0.01 * min(dx, dy))
        dxc, dyc, dzc = block_dims(rng, cfg)
        append_candidate(placements, candidate_dict(0.0, 0.0, z, dxc, dyc, dzc), cfg, margin=0.02 * min(dxc, dyc))
        top_z = z + max_h / 2.0
    return placements


def build_stable_buttressed_tower_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    tower_levels = int(rng.integers(5, 8))
    _stack_vertical(placements, 0.0, 0.0, tower_levels, rng, cfg, margin_scale=0.02)

    top = max(blk.z + blk.dz / 2.0 for blk in placements)
    buttress_levels = max(3, tower_levels - 2)
    buttress_offsets = [
        (cfg.block_edge * 1.1, 0.0),
        (-cfg.block_edge * 1.1, 0.0),
        (0.0, cfg.block_edge * 1.1),
        (0.0, -cfg.block_edge * 1.1),
    ]
    for bx, by in buttress_offsets:
        _stack_vertical(placements, bx, by, buttress_levels, rng, cfg, margin_scale=0.01)

    dx, dy, dz = block_dims(rng, cfg)
    z = top + dz / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(0.0, 0.0, z, dx, dy, dz), cfg, margin=0.02 * min(dx, dy))
    return placements


def build_unstable_eccentric_top_mass_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    levels = int(rng.integers(6, 10))
    _stack_vertical(placements, 0.0, 0.0, levels, rng, cfg, margin_scale=0.005)
    top = max(blk.z + blk.dz / 2.0 for blk in placements)

    dx, dy, dz = block_dims(rng, cfg)
    shift = float(rng.uniform(0.28, 0.40) * min(dx, dy))
    z = top + dz / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(shift, 0.0, z, dx, dy, dz), cfg, margin=0.0)
    return placements


def build_unstable_missing_support_bridge_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    span = cfg.block_edge * float(rng.uniform(2.0, 2.6))
    left_x = -span / 2.0
    right_x = span / 2.0
    left_h = int(rng.integers(4, 6))
    right_h = int(rng.integers(2, 4))
    _stack_vertical(placements, left_x, 0.0, left_h, rng, cfg, margin_scale=0.0)
    _stack_vertical(placements, right_x, 0.0, right_h, rng, cfg, margin_scale=0.0)
    top = max(blk.z + blk.dz / 2.0 for blk in placements)

    dx, dy, dz = block_dims(rng, cfg)
    z = top + dz / 2.0 + cfg.gap
    n_beam = max(2, int(np.ceil(span / max(dx, 1e-6))) + 1)
    for x in np.linspace(left_x, right_x, n_beam):
        append_candidate(placements, candidate_dict(float(x), 0.0, z, dx, dy, dz), cfg, margin=0.0)
    return placements


def build_unstable_alternating_offset_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    dx, dy, dz = block_dims(rng, cfg)
    append_candidate(placements, candidate_dict(0.0, 0.0, dz / 2.0, dx, dy, dz), cfg, margin=0.0)
    top_z = dz
    levels = int(rng.integers(8, 12))
    amp = float(rng.uniform(0.18, 0.26) * min(dx, dy))
    for i in range(1, levels):
        ndx, ndy, ndz = block_dims(rng, cfg)
        x = amp * i * (1.0 if i % 2 == 0 else -1.0)
        y = float(rng.uniform(-0.08, 0.08) * i * min(ndx, ndy))
        z = top_z + ndz / 2.0 + cfg.gap
        append_candidate(placements, candidate_dict(x, y, z, ndx, ndy, ndz), cfg, margin=0.0)
        top_z = placements[-1].z + placements[-1].dz / 2.0
    return placements


def build_unstable_tripod_top_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    radius = cfg.block_edge * float(rng.uniform(0.9, 1.2))
    levels = int(rng.integers(3, 5))
    points = []
    for i in range(3):
        angle = 2.0 * np.pi * i / 3.0
        points.append((float(np.cos(angle) * radius), float(np.sin(angle) * radius)))
    for x, y in points:
        _stack_vertical(placements, x, y, levels, rng, cfg, margin_scale=0.0)

    top = max(blk.z + blk.dz / 2.0 for blk in placements)
    dx, dy, dz = block_dims(rng, cfg)
    shift_x = float(rng.uniform(0.18, 0.30) * dx)
    shift_y = float(rng.uniform(-0.22, 0.22) * dy)
    z = top + dz / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(shift_x, shift_y, z, dx, dy, dz), cfg, margin=0.0)
    return placements


PATTERN_BUILDERS: dict[str, tuple[str, callable]] = {
    "stable_default": ("stable", build_stable_layout),
    "stable_ziggurat": ("stable", build_stable_ziggurat_layout),
    "stable_double_column_lintel": ("stable", build_stable_double_column_lintel_layout),
    "stable_ring_core": ("stable", build_stable_ring_core_layout),
    "stable_buttressed_tower": ("stable", build_stable_buttressed_tower_layout),
    "unstable_tower": ("unstable", build_unstable_tower_layout),
    "unstable_bridge": ("unstable", build_unstable_bridge_layout),
    "unstable_cantilever": ("unstable", build_unstable_cantilever_layout),
    "unstable_eccentric_top_mass": ("unstable", build_unstable_eccentric_top_mass_layout),
    "unstable_missing_support_bridge": ("unstable", build_unstable_missing_support_bridge_layout),
    "unstable_alternating_offset": ("unstable", build_unstable_alternating_offset_layout),
    "unstable_tripod_top": ("unstable", build_unstable_tripod_top_layout),
}


def list_patterns(mode: str = "all") -> list[str]:
    if mode == "all":
        return sorted(PATTERN_BUILDERS.keys())
    return sorted([name for name, (m, _) in PATTERN_BUILDERS.items() if m == mode])


def generate_layout_v1_1(
    rng: np.random.Generator,
    mode: str,
    cfg: SimConfig,
    pattern: str | None = None,
) -> tuple[list[LayoutBlock], bool, str]:
    if pattern is not None:
        if pattern not in PATTERN_BUILDERS:
            raise ValueError(f"Unknown pattern: {pattern}")
        p_mode, builder = PATTERN_BUILDERS[pattern]
        if mode in ("stable", "unstable") and p_mode != mode:
            raise ValueError(f"Pattern `{pattern}` is {p_mode} but mode is {mode}")
        layout = builder(rng, cfg)
        return layout, (p_mode == "unstable"), pattern

    if mode == "stable":
        candidates = list_patterns("stable")
    elif mode == "unstable":
        candidates = list_patterns("unstable")
    else:
        chosen_mode = "unstable" if bool(rng.integers(0, 2)) else "stable"
        candidates = list_patterns(chosen_mode)

    pattern_name = str(rng.choice(candidates))
    p_mode, builder = PATTERN_BUILDERS[pattern_name]
    layout = builder(rng, cfg)
    return layout, (p_mode == "unstable"), pattern_name

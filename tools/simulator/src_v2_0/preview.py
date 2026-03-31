from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

try:
    from ..src.config import SimConfig
    from ..src.scene import (
        connect_pybullet,
        instantiate_layout,
        render_front,
        render_top,
        reset_world,
        resolve_initial_overlaps,
    )
except ImportError:
    from src.config import SimConfig
    from src.scene import (
        connect_pybullet,
        instantiate_layout,
        render_front,
        render_top,
        reset_world,
        resolve_initial_overlaps,
    )

from .structures import StructureLayoutResult, build_structure_layout_v2
from .edav2_profile import apply_tone_profile, sample_edav2_render_setup


@dataclass(frozen=True)
class PreviewResult:
    structure_result: StructureLayoutResult
    front_rgb: np.ndarray
    top_rgb: np.ndarray
    render_profile: dict


def render_structure_preview(
    *,
    seed: int,
    structure: str,
    stability_label: str,
    cfg: SimConfig | None = None,
    relaxed: bool = True,
    gui: bool = False,
) -> PreviewResult:
    sim_cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    structure_result = build_structure_layout_v2(
        rng=rng,
        cfg=sim_cfg,
        structure=structure,
        stability_label=stability_label,
        relaxed=relaxed,
    )
    render_setup = sample_edav2_render_setup(rng)

    connect_pybullet(gui=gui)
    try:
        reset_world(sim_cfg)
        body_ids = instantiate_layout(structure_result.layout, rng, sim_cfg)
        resolve_initial_overlaps(body_ids, sim_cfg)
        front_rgb = render_front(
            sim_cfg,
            cam=render_setup.front_camera,
            light_direction=render_setup.light_direction,
            light_color=render_setup.light_color,
        )
        top_rgb = render_top(
            sim_cfg,
            cam=render_setup.top_camera,
            light_direction=render_setup.light_direction,
            light_color=render_setup.light_color,
        )
    finally:
        p.disconnect()

    return PreviewResult(
        structure_result=structure_result,
        front_rgb=apply_tone_profile(front_rgb, render_setup.front_tone),
        top_rgb=apply_tone_profile(top_rgb, render_setup.top_tone),
        render_profile=render_setup.to_metadata(),
    )

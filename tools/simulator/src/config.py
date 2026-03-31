from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimConfig:
    width: int = 384
    height: int = 384
    fps: int = 30
    frames: int = 300
    dt: float = 1.0 / 240.0
    steps_per_frame: int = 8

    gravity: float = -9.81
    world_half: float = 1.6

    density: float = 650.0
    restitution: float = 0.0
    lateral_friction: float = 0.90
    spinning_friction: float = 0.12
    rolling_friction: float = 0.04
    linear_damping: float = 0.18
    angular_damping: float = 0.30

    block_edge: float = 0.24
    edge_jitter: float = 0.0
    gap: float = 0.005

    min_stable_blocks: int = 10
    max_stable_blocks: int = 16
    min_unstable_blocks: int = 8
    mode_match_max_tries: int = 6
    presim_steps: int = 90

    ground_texture: str = "checker_blue.png"
    light_dir: tuple[float, float, float] = (-1.0, 0.35, 1.0)


@dataclass
class CameraSpec:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov: float


@dataclass
class LayoutBlock:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    mass: float
    supporters: list[int]
    support_region: dict

from __future__ import annotations

from .config import CameraSpec


def _look_target() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.75)


def front_camera() -> CameraSpec:
    return CameraSpec(
        eye=(5.83, -4.62, 3.30),
        target=_look_target(),
        up=(0.0, 0.0, 1.0),
        fov=42.0,
    )


def top_camera() -> CameraSpec:
    return CameraSpec(
        eye=(0.0, 0.0, 9.20),
        target=_look_target(),
        up=(0.0, 1.0, 0.0),
        fov=36.0,
    )

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

try:
    from ..src.config import CameraSpec
except ImportError:
    from src.config import CameraSpec


@dataclass(frozen=True)
class ToneProfile:
    hue_shift_deg: float
    saturation_gain: float
    value_gain: float
    gamma: float


@dataclass(frozen=True)
class RenderSetup:
    front_camera: CameraSpec
    top_camera: CameraSpec
    light_direction: tuple[float, float, float]
    light_color: tuple[float, float, float]
    front_tone: ToneProfile
    top_tone: ToneProfile
    front_pose: dict
    top_pose: dict

    def to_metadata(self) -> dict:
        return {
            "front_camera": {
                "eye": list(self.front_camera.eye),
                "target": list(self.front_camera.target),
                "up": list(self.front_camera.up),
                "fov": self.front_camera.fov,
            },
            "top_camera": {
                "eye": list(self.top_camera.eye),
                "target": list(self.top_camera.target),
                "up": list(self.top_camera.up),
                "fov": self.top_camera.fov,
            },
            "light": {
                "direction": list(self.light_direction),
                "color": list(self.light_color),
            },
            "front_tone": {
                "hue_shift_deg": self.front_tone.hue_shift_deg,
                "saturation_gain": self.front_tone.saturation_gain,
                "value_gain": self.front_tone.value_gain,
                "gamma": self.front_tone.gamma,
            },
            "top_tone": {
                "hue_shift_deg": self.top_tone.hue_shift_deg,
                "saturation_gain": self.top_tone.saturation_gain,
                "value_gain": self.top_tone.value_gain,
                "gamma": self.top_tone.gamma,
            },
            "front_pose": self.front_pose,
            "top_pose": self.top_pose,
            "reference": "edav2_dev_test_proxy",
        }


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = max(float(np.linalg.norm(arr)), 1e-9)
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _direction_from_angles(yaw_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    elev = math.radians(elevation_deg)
    x = math.cos(yaw) * math.cos(elev)
    y = math.sin(yaw) * math.cos(elev)
    z = math.sin(elev)
    return _normalize((x, y, z))


def _top_up_vector(roll_deg: float) -> tuple[float, float, float]:
    roll = math.radians(roll_deg)
    return _normalize((math.sin(roll), math.cos(roll), 0.0))


def _camera_up_with_roll(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    roll_deg: float,
) -> tuple[float, float, float]:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    look = target_v - eye_v
    look = look / max(float(np.linalg.norm(look)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(look, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(look, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    base_up = np.cross(right, look)
    base_up = base_up / max(float(np.linalg.norm(base_up)), 1e-9)
    roll = math.radians(roll_deg)
    up = base_up * math.cos(roll) + right * math.sin(roll)
    return _normalize((float(up[0]), float(up[1]), float(up[2])))


def _orbit_camera(
    *,
    target: tuple[float, float, float],
    radius: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    fov: float,
) -> CameraSpec:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    radial_xy = radius * math.cos(pitch)
    eye = (
        float(target[0] + radial_xy * math.cos(yaw)),
        float(target[1] + radial_xy * math.sin(yaw)),
        float(target[2] + radius * math.sin(pitch)),
    )
    up = _camera_up_with_roll(eye, target, roll_deg)
    return CameraSpec(eye=eye, target=target, up=up, fov=float(fov))


def sample_edav2_render_setup(rng: np.random.Generator) -> RenderSetup:
    # edav2 notes:
    # - front vp_pitch_proxy should move down from old generated_v2 (~0.456) toward dev/test (~0.389)
    # - front shadow angle clusters near 145-170 degrees wrapped
    # - brightness should be lower than old train-like renders, with slightly higher saturation
    front_target = (
        float(rng.uniform(-0.10, 0.10)),
        float(rng.uniform(-0.10, 0.10)),
        float(rng.uniform(0.68, 0.90)),
    )
    front_yaw_deg = float(rng.uniform(0.0, 360.0))
    front_pitch_deg = float(rng.uniform(12.0, 28.0))
    front_roll_deg = float(rng.uniform(-5.0, 5.0))
    front_radius = float(rng.uniform(6.2, 7.4))
    front_height_boost = float(rng.uniform(0.0, 1.15))
    front_fov = float(rng.uniform(39.0, 45.0))
    front_camera = _orbit_camera(
        target=front_target,
        radius=front_radius,
        yaw_deg=front_yaw_deg,
        pitch_deg=front_pitch_deg,
        roll_deg=front_roll_deg,
        fov=front_fov,
    )
    if front_height_boost > 0.0:
        raised_eye = (
            front_camera.eye[0],
            front_camera.eye[1],
            front_camera.eye[2] + front_height_boost,
        )
        front_camera = CameraSpec(
            eye=raised_eye,
            target=front_camera.target,
            up=_camera_up_with_roll(raised_eye, front_camera.target, front_roll_deg),
            fov=front_camera.fov,
        )

    top_roll_deg = float(rng.uniform(-7.0, 7.0))
    top_camera = CameraSpec(
        eye=(
            float(rng.uniform(-0.35, 0.35)),
            float(rng.uniform(-0.35, 0.35)),
            float(rng.uniform(8.75, 9.60)),
        ),
        target=(
            float(rng.uniform(-0.08, 0.08)),
            float(rng.uniform(-0.08, 0.08)),
            float(rng.uniform(0.70, 0.88)),
        ),
        up=_top_up_vector(top_roll_deg),
        fov=float(rng.uniform(34.0, 39.0)),
    )

    light_direction = _direction_from_angles(
        yaw_deg=float(rng.uniform(145.0, 170.0)),
        elevation_deg=float(rng.uniform(30.0, 46.0)),
    )
    light_intensity = float(rng.uniform(0.82, 0.96))
    light_color = (light_intensity, light_intensity, light_intensity)

    front_tone = ToneProfile(
        hue_shift_deg=float(rng.uniform(-4.0, 4.0)),
        saturation_gain=float(rng.uniform(1.18, 1.42)),
        value_gain=float(rng.uniform(0.88, 0.97)),
        gamma=float(rng.uniform(1.00, 1.10)),
    )
    top_tone = ToneProfile(
        hue_shift_deg=float(rng.uniform(-3.0, 3.0)),
        saturation_gain=float(rng.uniform(1.05, 1.22)),
        value_gain=float(rng.uniform(0.90, 1.00)),
        gamma=float(rng.uniform(0.98, 1.08)),
    )

    return RenderSetup(
        front_camera=front_camera,
        top_camera=top_camera,
        light_direction=light_direction,
        light_color=light_color,
        front_tone=front_tone,
        top_tone=top_tone,
        front_pose={
            "yaw_deg": front_yaw_deg,
            "pitch_deg": front_pitch_deg,
            "roll_deg": front_roll_deg,
            "radius": front_radius,
            "height_boost": front_height_boost,
        },
        top_pose={
            "roll_deg": top_roll_deg,
        },
    )


def apply_tone_profile(rgb: np.ndarray, tone: ToneProfile) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + tone.hue_shift_deg / 2.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * tone.saturation_gain, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * tone.value_gain, 0.0, 255.0)
    toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    toned = np.clip(np.power(np.clip(toned, 0.0, 1.0), tone.gamma), 0.0, 1.0)
    return np.clip(toned * 255.0, 0.0, 255.0).astype(np.uint8)

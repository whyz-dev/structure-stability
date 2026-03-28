from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_data

from .camera import front_camera, top_camera
from .config import CameraSpec, LayoutBlock, SimConfig


def _rand_color(rng: np.random.Generator) -> tuple[float, float, float, float]:
    color = rng.uniform(0.30, 0.92, size=3)
    return float(color[0]), float(color[1]), float(color[2]), 1.0


def _compute_mass(dx: float, dy: float, dz: float, density: float) -> float:
    return max(1e-4, dx * dy * dz * density)


def connect_pybullet(gui: bool = False) -> int:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return cid


def reset_world(cfg: SimConfig) -> None:
    p.resetSimulation()
    p.setGravity(0.0, 0.0, cfg.gravity)
    p.setPhysicsEngineParameter(
        fixedTimeStep=cfg.dt,
        numSolverIterations=80,
        numSubSteps=0,
        enableConeFriction=1,
        deterministicOverlappingPairs=1,
    )
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    except p.error:
        pass

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(
        plane_id,
        -1,
        restitution=0.0,
        lateralFriction=1.0,
        spinningFriction=0.0,
        rollingFriction=0.0,
    )
    try:
        tex = p.loadTexture(cfg.ground_texture)
        p.changeVisualShape(plane_id, -1, textureUniqueId=tex, rgbaColor=[1.0, 1.0, 1.0, 1.0])
    except p.error:
        pass


def create_block(
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float, float],
    cfg: SimConfig,
) -> int:
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=color,
        specularColor=[0.35, 0.35, 0.35],
    )
    body = p.createMultiBody(
        baseMass=_compute_mass(size[0], size[1], size[2], cfg.density),
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )
    p.changeDynamics(
        body,
        -1,
        restitution=cfg.restitution,
        lateralFriction=cfg.lateral_friction,
        spinningFriction=cfg.spinning_friction,
        rollingFriction=cfg.rolling_friction,
        linearDamping=cfg.linear_damping,
        angularDamping=cfg.angular_damping,
    )
    return body


def instantiate_layout(layout: list[LayoutBlock], rng: np.random.Generator, cfg: SimConfig) -> list[int]:
    return [
        create_block((blk.x, blk.y, blk.z), (blk.dx, blk.dy, blk.dz), _rand_color(rng), cfg)
        for blk in layout
    ]


def resolve_initial_overlaps(
    body_ids: list[int],
    cfg: SimConfig,
    max_iters: int = 100,
    min_penetration: float = 0.001,
) -> None:
    for _ in range(max_iters):
        moved = False
        for i, body_id in enumerate(body_ids):
            required_lift = 0.0
            for other in body_ids[:i]:
                for pt in p.getClosestPoints(body_id, other, distance=0.0):
                    penetration = -float(pt[8])
                    if penetration > min_penetration:
                        required_lift = max(required_lift, penetration + cfg.gap)
            if required_lift <= 0.0:
                continue
            pos, orn = p.getBasePositionAndOrientation(body_id)
            p.resetBasePositionAndOrientation(body_id, [pos[0], pos[1], pos[2] + required_lift], orn)
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            moved = True
        if not moved:
            break


def run_presim_stability_check(body_ids: list[int], cfg: SimConfig, stable_mode: bool) -> bool:
    init_pose = [p.getBasePositionAndOrientation(body_id) for body_id in body_ids]
    max_disp = 0.0
    max_tilt = 0.0
    max_speed = 0.0

    for _ in range(cfg.presim_steps):
        p.stepSimulation()
        for body_id, (init_pos, init_orn) in zip(body_ids, init_pose):
            pos, orn = p.getBasePositionAndOrientation(body_id)
            lin_vel, _ = p.getBaseVelocity(body_id)
            max_disp = max(max_disp, float(np.linalg.norm(np.array(pos) - np.array(init_pos))))
            tilt = float(np.linalg.norm(np.array(p.getEulerFromQuaternion(orn))[:2] - np.array(p.getEulerFromQuaternion(init_orn))[:2]))
            max_tilt = max(max_tilt, tilt)
            max_speed = max(max_speed, float(np.linalg.norm(lin_vel)))

    for body_id, (init_pos, init_orn) in zip(body_ids, init_pose):
        p.resetBasePositionAndOrientation(body_id, init_pos, init_orn)
        p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    if stable_mode:
        return max_disp < 0.02 and max_tilt < 0.08 and max_speed < 0.08
    return True


def step_world(cfg: SimConfig) -> None:
    for _ in range(cfg.steps_per_frame):
        p.stepSimulation()


def _camera_matrices(cam: CameraSpec, cfg: SimConfig) -> tuple[list[float], list[float]]:
    view = p.computeViewMatrix(
        cameraEyePosition=cam.eye,
        cameraTargetPosition=cam.target,
        cameraUpVector=cam.up,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam.fov,
        aspect=float(cfg.width) / float(cfg.height),
        nearVal=0.02,
        farVal=20.0,
    )
    return view, proj


def _capture_camera(
    view: list[float],
    proj: list[float],
    cfg: SimConfig,
    shadow: int,
    light_direction: tuple[float, float, float] | None = None,
    light_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
    light_dir = list(light_direction or cfg.light_dir)
    light_rgb = list(light_color or (1.0, 1.0, 1.0))
    try:
        _, _, rgba, _, _ = p.getCameraImage(
            width=cfg.width,
            height=cfg.height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            shadow=shadow,
            lightDirection=light_dir,
            lightColor=light_rgb,
        )
    except p.error:
        _, _, rgba, _, _ = p.getCameraImage(
            width=cfg.width,
            height=cfg.height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            shadow=0,
            lightDirection=light_dir,
            lightColor=light_rgb,
        )
    return np.asarray(rgba, dtype=np.uint8).reshape(cfg.height, cfg.width, 4)[..., :3]


def render_camera(
    cam: CameraSpec,
    cfg: SimConfig,
    shadow: int,
    light_direction: tuple[float, float, float] | None = None,
    light_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
    view, proj = _camera_matrices(cam, cfg)
    return _capture_camera(
        view,
        proj,
        cfg,
        shadow=shadow,
        light_direction=light_direction,
        light_color=light_color,
    )


def render_front(
    cfg: SimConfig,
    cam: CameraSpec | None = None,
    light_direction: tuple[float, float, float] | None = None,
    light_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
    return render_camera(
        cam or front_camera(),
        cfg,
        shadow=1,
        light_direction=light_direction,
        light_color=light_color,
    )


def render_top(
    cfg: SimConfig,
    cam: CameraSpec | None = None,
    light_direction: tuple[float, float, float] | None = None,
    light_color: tuple[float, float, float] | None = None,
) -> np.ndarray:
    return render_camera(
        cam or top_camera(),
        cfg,
        shadow=0,
        light_direction=light_direction,
        light_color=light_color,
    )

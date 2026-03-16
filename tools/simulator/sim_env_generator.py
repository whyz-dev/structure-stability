from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data


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
    restitution: float = 0.04
    lateral_friction: float = 0.82
    spinning_friction: float = 0.08
    rolling_friction: float = 0.02
    linear_damping: float = 0.03
    angular_damping: float = 0.06

    block_edge: float = 0.24
    edge_jitter: float = 0.10
    gap: float = 0.003
    min_floors: int = 4
    max_floors: int = 8

    unstable_push_scale: float = 1.25
    mode_match_max_tries: int = 6
    ground_texture: str = "checker_blue.png"
    light_dir: tuple[float, float, float] = (-0.55, 0.28, 1.85)


@dataclass
class CameraSpec:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov: float


@dataclass
class BlockNode:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    mass: float
    supporters: list[int]
    support_region: dict


def _rand_color(rng: np.random.Generator) -> tuple[float, float, float, float]:
    color = rng.uniform(0.30, 0.92, size=3)
    return float(color[0]), float(color[1]), float(color[2]), 1.0


def _look_target() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.75)


def _front_camera() -> CameraSpec:
    return CameraSpec(
        eye=(5.83, -4.62, 3.30),
        target=_look_target(),
        up=(0.0, 0.0, 1.0),
        fov=42.0,
    )


def _top_camera() -> CameraSpec:
    return CameraSpec(
        eye=(0.0, 0.0, 9.20),
        target=_look_target(),
        up=(0.0, 1.0, 0.0),
        fov=36.0,
    )


def _compute_mass(dx: float, dy: float, dz: float, density: float) -> float:
    return max(1e-4, dx * dy * dz * density)


def _block_dims(rng: np.random.Generator, cfg: SimConfig) -> tuple[float, float, float]:
    scale = rng.uniform(1.0 - cfg.edge_jitter, 1.0 + cfg.edge_jitter)
    dx = cfg.block_edge * scale
    dy = cfg.block_edge * rng.uniform(0.92, 1.08) * scale
    dz = cfg.block_edge * rng.uniform(0.86, 1.02) * scale
    return dx, dy, dz


def _tile_corner_anchor() -> tuple[float, float]:
    return (0.0, 0.0)


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
    pos: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    color: tuple[float, float, float, float],
    cfg: SimConfig,
) -> int:
    half_extents = [dx / 2.0, dy / 2.0, dz / 2.0]
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color,
        specularColor=[0.35, 0.35, 0.35],
    )
    body = p.createMultiBody(
        baseMass=_compute_mass(dx, dy, dz, cfg.density),
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=pos.tolist(),
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


def _clip_xy(x: float, y: float, dx: float, dy: float, cfg: SimConfig) -> tuple[float, float]:
    x = float(np.clip(x, -cfg.world_half + dx / 2.0, cfg.world_half - dx / 2.0))
    y = float(np.clip(y, -cfg.world_half + dy / 2.0, cfg.world_half - dy / 2.0))
    return x, y


def _append_block(
    body_ids: list[int],
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    rng: np.random.Generator,
    cfg: SimConfig,
) -> None:
    x, y = _clip_xy(x, y, dx, dy, cfg)
    body_ids.append(
        create_block(
            pos=np.array([x, y, z], dtype=np.float64),
            dx=dx,
            dy=dy,
            dz=dz,
            color=_rand_color(rng),
            cfg=cfg,
        )
    )


def resolve_initial_overlaps(
    body_ids: list[int],
    cfg: SimConfig,
    max_iters: int = 160,
    min_penetration: float = 0.003,
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
            p.resetBasePositionAndOrientation(
                body_id,
                [pos[0], pos[1], pos[2] + required_lift],
                orn,
            )
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            moved = True
        if not moved:
            break


def _rect_from_xywh(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2.0, x + w / 2.0, y - h / 2.0, y + h / 2.0)


def _rect_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    xmin = max(a[0], b[0])
    xmax = min(a[1], b[1])
    ymin = max(a[2], b[2])
    ymax = min(a[3], b[3])
    if xmin >= xmax or ymin >= ymax:
        return None
    return xmin, xmax, ymin, ymax


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[1] - rect[0]) * max(0.0, rect[3] - rect[2])


def _rect_corners(rect: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    return [
        (rect[0], rect[2]),
        (rect[0], rect[3]),
        (rect[1], rect[2]),
        (rect[1], rect[3]),
    ]


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> np.ndarray:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    lower: list[tuple[float, float]] = []
    for pt in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)

    upper: list[tuple[float, float]] = []
    for pt in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_in_convex_polygon_with_margin(point: tuple[float, float], hull: np.ndarray, margin: float) -> bool:
    if hull.shape[0] < 3:
        return False
    pxy = np.asarray(point, dtype=np.float64)
    for i in range(hull.shape[0]):
        a = hull[i]
        b = hull[(i + 1) % hull.shape[0]]
        edge = b - a
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        signed_dist = float(np.dot(pxy - a, normal / norm))
        if signed_dist < margin:
            return False
    return True


def propose_block_candidate(
    placements: list[BlockNode],
    rng: np.random.Generator,
    cfg: SimConfig,
) -> dict:
    dx, dy, dz = _block_dims(rng, cfg)
    if not placements:
        ax, ay = _tile_corner_anchor()
        return {"x": ax, "y": ay, "z": dz / 2.0, "dx": dx, "dy": dy, "dz": dz}

    tops = np.array([blk.z + blk.dz / 2.0 for blk in placements], dtype=np.float64)
    top_q = float(np.quantile(tops, 0.55))
    support_pool = [i for i, top in enumerate(tops) if top >= top_q]
    primary = int(rng.choice(support_pool))
    support_indices = [primary]

    peers = []
    for idx in support_pool:
        if idx == primary:
            continue
        same_level = abs(float(tops[idx] - tops[primary])) <= 0.02
        near_xy = math.hypot(placements[idx].x - placements[primary].x, placements[idx].y - placements[primary].y)
        if same_level and near_xy <= (placements[idx].dx + placements[primary].dx) * 1.4:
            peers.append(idx)
    if peers and rng.random() < 0.45:
        support_indices.append(int(rng.choice(peers)))

    sx = float(np.mean([placements[idx].x for idx in support_indices]))
    sy = float(np.mean([placements[idx].y for idx in support_indices]))
    top_z = float(np.mean([tops[idx] for idx in support_indices]))
    jitter = 0.10 * min(dx, dy)
    return {
        "x": sx + float(rng.uniform(-jitter, jitter)),
        "y": sy + float(rng.uniform(-jitter, jitter)),
        "z": top_z + dz / 2.0,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def find_support_contacts(candidate: dict, placements: list[BlockNode], cfg: SimConfig) -> dict:
    candidate_rect = _rect_from_xywh(candidate["x"], candidate["y"], candidate["dx"], candidate["dy"])
    candidate_bottom = float(candidate["z"] - candidate["dz"] / 2.0)
    z_tol = max(0.01, cfg.gap * 3.0)
    area_tol = max(1e-4, candidate["dx"] * candidate["dy"] * 0.03)

    supports: list[tuple[int, tuple[float, float, float, float], float]] = []
    for idx, blk in enumerate(placements):
        blk_top = blk.z + blk.dz / 2.0
        if abs(candidate_bottom - blk_top) > z_tol:
            continue
        inter = _rect_intersection(candidate_rect, _rect_from_xywh(blk.x, blk.y, blk.dx, blk.dy))
        if inter is None:
            continue
        area = _rect_area(inter)
        if area >= area_tol:
            supports.append((idx, inter, area))

    floor_contact = abs(candidate_bottom) <= z_tol
    return {"floor": floor_contact, "supports": supports}


def compute_support_region(candidate: dict, contacts: dict) -> dict:
    if contacts["floor"] and not contacts["supports"]:
        rect = _rect_from_xywh(candidate["x"], candidate["y"], candidate["dx"], candidate["dy"])
        patches = [rect]
    else:
        patches = [item[1] for item in contacts["supports"]]
    if not patches:
        return {"patches": [], "hull": np.zeros((0, 2), dtype=np.float64)}

    corners: list[tuple[float, float]] = []
    for patch in patches:
        corners.extend(_rect_corners(patch))
    return {"patches": patches, "hull": _convex_hull(corners)}


def is_point_inside_support_with_margin(point: tuple[float, float], support_region: dict, margin: float) -> bool:
    hull = support_region.get("hull")
    if not isinstance(hull, np.ndarray) or hull.shape[0] < 3:
        return False
    return _point_in_convex_polygon_with_margin(point, hull, margin)


def compute_substructure_com(block_idx: int, placements: list[BlockNode], children: dict[int, list[int]]) -> np.ndarray:
    stack = [block_idx]
    seen: set[int] = set()
    total_mass = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        blk = placements[idx]
        total_mass += blk.mass
        weighted += blk.mass * np.array([blk.x, blk.y, blk.z], dtype=np.float64)
        stack.extend(children.get(idx, []))
    return weighted / max(total_mass, 1e-9)


def validate_load_path(placements: list[BlockNode]) -> bool:
    if len(placements) <= 1:
        return False

    children: dict[int, list[int]] = {i: [] for i in range(len(placements))}
    degree = [0 for _ in placements]
    for idx, blk in enumerate(placements):
        for supporter in blk.supporters:
            if supporter >= 0:
                children[supporter].append(idx)
                degree[idx] += 1
                degree[supporter] += 1

    if any(d == 0 for d in degree):
        return False

    memo: dict[int, bool] = {}

    def reaches_floor(idx: int) -> bool:
        if idx in memo:
            return memo[idx]
        if -1 in placements[idx].supporters:
            memo[idx] = True
            return True
        memo[idx] = any(s >= 0 and reaches_floor(s) for s in placements[idx].supporters)
        return memo[idx]

    if not all(reaches_floor(i) for i in range(len(placements))):
        return False

    for idx, blk in enumerate(placements):
        if -1 in blk.supporters:
            continue
        com = compute_substructure_com(idx, placements, children)
        margin = 0.08 * min(blk.dx, blk.dy)
        if not is_point_inside_support_with_margin((float(com[0]), float(com[1])), blk.support_region, margin):
            return False
    return True


def run_presim_stability_check(body_ids: list[int], stable_mode: bool) -> bool:
    init_pose = [p.getBasePositionAndOrientation(body_id) for body_id in body_ids]
    max_disp = 0.0
    max_tilt = 0.0
    max_speed = 0.0

    for _ in range(90):
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


def _is_aabb_overlap(candidate: dict, blk: BlockNode) -> bool:
    ax0, ax1 = candidate["x"] - candidate["dx"] / 2.0, candidate["x"] + candidate["dx"] / 2.0
    ay0, ay1 = candidate["y"] - candidate["dy"] / 2.0, candidate["y"] + candidate["dy"] / 2.0
    az0, az1 = candidate["z"] - candidate["dz"] / 2.0, candidate["z"] + candidate["dz"] / 2.0
    bx0, bx1 = blk.x - blk.dx / 2.0, blk.x + blk.dx / 2.0
    by0, by1 = blk.y - blk.dy / 2.0, blk.y + blk.dy / 2.0
    bz0, bz1 = blk.z - blk.dz / 2.0, blk.z + blk.dz / 2.0
    return (ax0 < bx1 and ax1 > bx0) and (ay0 < by1 and ay1 > by0) and (az0 < bz1 and az1 > bz0)


def _build_stable_pyramid(body_ids: list[int], rng: np.random.Generator, cfg: SimConfig) -> None:
    ax, ay = _tile_corner_anchor()
    floors = int(rng.integers(max(4, cfg.min_floors), min(9, cfg.max_floors) + 1))
    base_side = int(rng.choice([4, 5, 6]))
    top_z = 0.0
    first_done = False

    for layer in range(floors):
        side = max(1, base_side - (layer // 2))
        dims = [_block_dims(rng, cfg) for _ in range(side * side)]
        sx = max(d[0] for d in dims) + cfg.gap
        sy = max(d[1] for d in dims) + cfg.gap
        max_h = max(d[2] for d in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if layer > 0 else 0.0)

        for i in range(side * side):
            dx, dy, dz = dims[i]
            ix = i % side
            iy = i // side
            if not first_done:
                x, y = ax, ay
                first_done = True
            else:
                x = ax + (ix - (side - 1) / 2.0) * sx
                y = ay + (iy - (side - 1) / 2.0) * sy
            _append_block(body_ids, x, y, z, dx, dy, dz, rng, cfg)
        top_z = z + max_h / 2.0


def generate_stable_structure(rng: np.random.Generator, cfg: SimConfig) -> list[int]:
    ax, ay = _tile_corner_anchor()
    for _ in range(24):
        placements: list[BlockNode] = []
        target_blocks = int(rng.integers(10, 16))

        dx0, dy0, dz0 = _block_dims(rng, cfg)
        base = {"x": ax, "y": ay, "z": dz0 / 2.0, "dx": dx0, "dy": dy0, "dz": dz0}
        placements.append(
            BlockNode(
                x=base["x"],
                y=base["y"],
                z=base["z"],
                dx=dx0,
                dy=dy0,
                dz=dz0,
                mass=_compute_mass(dx0, dy0, dz0, cfg.density),
                supporters=[-1],
                support_region=compute_support_region(base, {"floor": True, "supports": []}),
            )
        )

        fail_streak = 0
        while len(placements) < target_blocks and fail_streak < target_blocks * 10:
            cand = propose_block_candidate(placements, rng, cfg)
            contacts = find_support_contacts(cand, placements, cfg)
            if not contacts["supports"]:
                fail_streak += 1
                continue

            support_region = compute_support_region(cand, contacts)
            margin = 0.14 * min(cand["dx"], cand["dy"])
            if not is_point_inside_support_with_margin((cand["x"], cand["y"]), support_region, margin):
                fail_streak += 1
                continue
            if any(_is_aabb_overlap(cand, blk) for blk in placements):
                fail_streak += 1
                continue

            placements.append(
                BlockNode(
                    x=float(cand["x"]),
                    y=float(cand["y"]),
                    z=float(cand["z"]),
                    dx=float(cand["dx"]),
                    dy=float(cand["dy"]),
                    dz=float(cand["dz"]),
                    mass=_compute_mass(cand["dx"], cand["dy"], cand["dz"], cfg.density),
                    supporters=[idx for idx, _, _ in contacts["supports"]],
                    support_region=support_region,
                )
            )
            fail_streak = 0

        if len(placements) < 8 or not validate_load_path(placements):
            continue

        body_ids = [
            create_block(
                pos=np.array([blk.x, blk.y, blk.z], dtype=np.float64),
                dx=blk.dx,
                dy=blk.dy,
                dz=blk.dz,
                color=_rand_color(rng),
                cfg=cfg,
            )
            for blk in placements
        ]
        if run_presim_stability_check(body_ids, stable_mode=True):
            return body_ids
        for body_id in body_ids:
            p.removeBody(body_id)

    body_ids: list[int] = []
    _build_stable_pyramid(body_ids, rng, cfg)
    return body_ids


def enforce_block_adjacency(body_ids: list[int], cfg: SimConfig) -> None:
    if len(body_ids) <= 1:
        return

    def degrees() -> dict[int, int]:
        deg = {body_id: 0 for body_id in body_ids}
        for i, body_id in enumerate(body_ids):
            for other in body_ids[i + 1 :]:
                if p.getClosestPoints(body_id, other, distance=max(0.006, cfg.gap + 0.002)):
                    deg[body_id] += 1
                    deg[other] += 1
        return deg

    for _ in range(3):
        deg = degrees()
        lonely = [body_id for body_id, value in deg.items() if value == 0]
        if not lonely:
            break
        for body_id in lonely:
            pos, orn = p.getBasePositionAndOrientation(body_id)
            aabb_min, aabb_max = p.getAABB(body_id)
            self_h = float(aabb_max[2] - aabb_min[2])

            nearest = None
            best = 1e9
            for other in body_ids:
                if other == body_id:
                    continue
                other_pos, _ = p.getBasePositionAndOrientation(other)
                dxy = (other_pos[0] - pos[0]) ** 2 + (other_pos[1] - pos[1]) ** 2
                if dxy < best:
                    best = dxy
                    nearest = other
            if nearest is None:
                continue

            other_pos, _ = p.getBasePositionAndOrientation(nearest)
            _, other_max = p.getAABB(nearest)
            p.resetBasePositionAndOrientation(
                body_id,
                [other_pos[0], other_pos[1], float(other_max[2]) + self_h / 2.0],
                orn,
            )
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def _build_pyramid(body_ids: list[int], rng: np.random.Generator, cfg: SimConfig, unstable: bool) -> None:
    ax, ay = _tile_corner_anchor()
    floors = int(rng.integers(cfg.min_floors, cfg.max_floors + 1))
    width = int(rng.choice([5, 6, 7]))
    depth = int(rng.choice([4, 5, 6]))
    top_z = 0.0
    first_done = False

    for layer in range(floors):
        cols = max(1, width - layer)
        rows = max(1, depth - layer)
        dims = [_block_dims(rng, cfg) for _ in range(cols * rows)]
        sx = max(d[0] for d in dims) + cfg.gap
        sy = max(d[1] for d in dims) + cfg.gap
        max_h = max(d[2] for d in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if layer > 0 else 0.0)

        for i in range(cols * rows):
            dx, dy, dz = dims[i]
            ix = i % cols
            iy = i // cols
            if not first_done:
                x, y = ax, ay
                first_done = True
            else:
                x = ax + (ix - (cols - 1) / 2.0) * sx
                y = ay + (iy - (rows - 1) / 2.0) * sy
            if unstable and layer >= floors - 2:
                x += float(rng.uniform(-0.018, 0.018))
                y += float(rng.uniform(-0.018, 0.018))
            _append_block(body_ids, x, y, z, dx, dy, dz, rng, cfg)
        top_z = z + max_h / 2.0


def _build_tower(body_ids: list[int], rng: np.random.Generator, cfg: SimConfig, unstable: bool) -> None:
    ax, ay = _tile_corner_anchor()
    floors = int(rng.integers(max(5, cfg.min_floors), min(7, cfg.max_floors) + 1))
    top_z = 0.0
    for layer in range(floors):
        count = 1 if layer > 2 else int(rng.integers(1, 3))
        dims = [_block_dims(rng, cfg) for _ in range(count)]
        max_h = max(d[2] for d in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if layer > 0 else 0.0)
        spread = 0.012 + 0.004 * min(layer, 5)
        for i, (dx, dy, dz) in enumerate(dims):
            if layer == 0 and i == 0:
                x, y = ax, ay
            else:
                x = ax + float(rng.uniform(-spread, spread))
                y = ay + float(rng.uniform(-spread, spread))
            if unstable and layer > floors // 2:
                x += float(rng.uniform(-0.02, 0.02))
                y += float(rng.uniform(-0.02, 0.02))
            _append_block(body_ids, x, y, z, dx, dy, dz, rng, cfg)
        top_z = z + max_h / 2.0


def _build_cantilever(body_ids: list[int], rng: np.random.Generator, cfg: SimConfig) -> None:
    _build_pyramid(body_ids, rng, cfg, unstable=True)
    positions = np.array([p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids], dtype=np.float64)
    top_idx = int(np.argmax(positions[:, 2]))
    tx, ty, tz = positions[top_idx]
    arm_len = int(rng.integers(3, 6))
    theta = float(rng.uniform(-0.45 * math.pi, 0.25 * math.pi))
    dxdir, dydir = math.cos(theta), math.sin(theta)
    for k in range(arm_len):
        dx, dy, dz = _block_dims(rng, cfg)
        step = 0.58 * max(dx, dy)
        x = float(tx + dxdir * step * (k + 1))
        y = float(ty + dydir * step * (k + 1))
        z = float(tz + dz * (0.08 * k + 0.03))
        _append_block(body_ids, x, y, z, dx, dy, dz, rng, cfg)


def create_structure(rng: np.random.Generator, unstable: bool, cfg: SimConfig) -> list[int]:
    if not unstable:
        return generate_stable_structure(rng, cfg)

    body_ids: list[int] = []
    choice = int(rng.choice(3, p=np.array([0.34, 0.26, 0.40], dtype=np.float64)))
    if choice == 0:
        _build_pyramid(body_ids, rng, cfg, unstable=True)
    elif choice == 1:
        _build_tower(body_ids, rng, cfg, unstable=True)
    else:
        _build_cantilever(body_ids, rng, cfg)
    return body_ids


def apply_initial_perturbation(
    body_ids: list[int],
    unstable: bool,
    sim_rng: np.random.Generator,
    cfg: SimConfig,
) -> None:
    positions = np.array([p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids], dtype=np.float64)
    center_xy = np.mean(positions[:, :2], axis=0)
    z_med = float(np.median(positions[:, 2]))

    for body_id, pos in zip(body_ids, positions):
        radial = pos[:2] - center_xy
        norm = float(np.linalg.norm(radial))
        radial = radial / norm if norm > 1e-8 else np.array([1.0, 0.0], dtype=np.float64)

        if unstable:
            lin = np.array([sim_rng.uniform(-0.015, 0.015), sim_rng.uniform(-0.015, 0.015), 0.0], dtype=np.float64)
        else:
            lin = np.zeros(3, dtype=np.float64)
        ang = np.zeros(3, dtype=np.float64)

        if unstable:
            if pos[2] >= z_med:
                lin[:2] += radial * sim_rng.uniform(0.12, 0.24) * cfg.unstable_push_scale
                lin[2] -= sim_rng.uniform(0.02, 0.08)
            else:
                lin[:2] += radial * sim_rng.uniform(0.02, 0.08) * cfg.unstable_push_scale

        p.resetBaseVelocity(body_id, linearVelocity=lin.tolist(), angularVelocity=ang.tolist())


def settle_prepass(steps: int = 30) -> None:
    for _ in range(steps):
        p.stepSimulation()


def step_world(cfg: SimConfig) -> None:
    for _ in range(cfg.steps_per_frame):
        p.stepSimulation()


def render_camera(cam: CameraSpec, cfg: SimConfig, shadow: int) -> np.ndarray:
    view = p.computeViewMatrix(
        cameraEyePosition=cam.eye,
        cameraTargetPosition=cam.target,
        cameraUpVector=cam.up,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=cam.fov,
        aspect=float(cfg.width) / float(cfg.height),
        nearVal=0.02,
        farVal=120.0,
    )
    try:
        _, _, rgba, _, _ = p.getCameraImage(
            width=cfg.width,
            height=cfg.height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            shadow=shadow,
            lightDirection=list(cfg.light_dir),
            lightColor=[1.0, 1.0, 1.0],
        )
    except p.error:
        _, _, rgba, _, _ = p.getCameraImage(
            width=cfg.width,
            height=cfg.height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            shadow=0,
            lightDirection=list(cfg.light_dir),
            lightColor=[1.0, 1.0, 1.0],
        )
    return np.asarray(rgba, dtype=np.uint8).reshape(cfg.height, cfg.width, 4)[..., :3]


def render_front(cfg: SimConfig) -> np.ndarray:
    return render_camera(_front_camera(), cfg, shadow=1)


def render_top(cfg: SimConfig) -> np.ndarray:
    return render_camera(_top_camera(), cfg, shadow=0)


def _get_body_state(body_id: int) -> dict:
    pos, orn = p.getBasePositionAndOrientation(body_id)
    lin, ang = p.getBaseVelocity(body_id)
    return {
        "pos": np.array(pos, dtype=np.float64),
        "orn": np.array(orn, dtype=np.float64),
        "lin": np.array(lin, dtype=np.float64),
        "ang": np.array(ang, dtype=np.float64),
    }


def detect_unstable(body_ids: list[int], init_positions: np.ndarray, final_positions: np.ndarray) -> bool:
    states = [_get_body_state(body_id) for body_id in body_ids]
    lin = np.array([np.linalg.norm(state["lin"]) for state in states], dtype=np.float64)
    ang = np.array([np.linalg.norm(state["ang"]) for state in states], dtype=np.float64)

    z_drop = init_positions[:, 2] - final_positions[:, 2]
    drop_count = int(np.sum(z_drop > 0.035))
    mean_drop = float(np.mean(z_drop))
    top_drop = float(np.max(init_positions[:, 2]) - np.max(final_positions[:, 2]))

    init_xy = init_positions[:, :2]
    final_xy = final_positions[:, :2]
    init_center = np.mean(init_xy, axis=0)
    final_center = np.mean(final_xy, axis=0)
    init_spread = float(np.mean(np.linalg.norm(init_xy - init_center, axis=1)))
    final_spread = float(np.mean(np.linalg.norm(final_xy - final_center, axis=1)))
    spread_gain = final_spread - init_spread

    return (
        drop_count >= max(3, len(body_ids) // 4)
        or mean_drop > 0.045
        or top_drop > 0.11
        or spread_gain > 0.05
        or int(np.sum(lin > 0.06)) >= max(2, len(body_ids) // 5)
        or int(np.sum(ang > 0.50)) >= max(2, len(body_ids) // 5)
    )


def run_single(
    sample_id: str,
    out_root: Path,
    seed: int,
    mode: str,
    cfg: SimConfig,
    sim_seed: int | None = None,
    gui: bool = False,
) -> dict:
    out_dir = out_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target_label = mode if mode in ("stable", "unstable") else None
    max_tries = cfg.mode_match_max_tries if target_label is not None else 1

    attempt_used = 0
    unstable_requested = False
    actual_sim_seed = 0
    detected_label = "stable"
    final_label = "stable"
    num_blocks = 0

    for attempt in range(max_tries):
        attempt_used = attempt
        scene_seed = seed + attempt * 100003
        rng = np.random.default_rng(scene_seed)

        if target_label == "stable":
            unstable_requested = False
        elif target_label == "unstable":
            unstable_requested = True
        else:
            unstable_requested = bool(rng.integers(0, 2))

        connect_pybullet(gui=gui)
        try:
            reset_world(cfg)
            body_ids = create_structure(rng, unstable_requested, cfg)
            resolve_initial_overlaps(
                body_ids,
                cfg,
                max_iters=(80 if unstable_requested else 30),
                min_penetration=(0.0015 if unstable_requested else 0.003),
            )
            enforce_block_adjacency(body_ids, cfg)
            resolve_initial_overlaps(body_ids, cfg, max_iters=40, min_penetration=0.0015)
            num_blocks = len(body_ids)
            settle_prepass(steps=(40 if not unstable_requested else 20))

            base_sim_seed = (scene_seed * 1664525 + 1013904223) & 0xFFFFFFFF if sim_seed is None else sim_seed + attempt * 100003
            actual_sim_seed = int(base_sim_seed)
            apply_initial_perturbation(body_ids, unstable_requested, np.random.default_rng(actual_sim_seed), cfg)

            video_path = out_dir / "simulation.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                cfg.fps,
                (cfg.width, cfg.height),
            )
            first_front = render_front(cfg)
            first_top = render_top(cfg)
            writer.write(cv2.cvtColor(first_front, cv2.COLOR_RGB2BGR))

            init_positions = np.array([p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids], dtype=np.float64)
            init_mean_z = float(np.mean(init_positions[:, 2]))
            init_top_z = float(np.max(init_positions[:, 2]))

            for _ in range(max(0, cfg.frames - 1)):
                step_world(cfg)
                writer.write(cv2.cvtColor(render_front(cfg), cv2.COLOR_RGB2BGR))
            writer.release()

            cv2.imwrite(str(out_dir / "front.png"), cv2.cvtColor(first_front, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / "top.png"), cv2.cvtColor(first_top, cv2.COLOR_RGB2BGR))

            final_positions = np.array([p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids], dtype=np.float64)
            final_mean_z = float(np.mean(final_positions[:, 2]))
            final_top_z = float(np.max(final_positions[:, 2]))
            detected_label = "unstable" if (
                detect_unstable(body_ids, init_positions, final_positions)
                or (init_mean_z - final_mean_z) > 0.05
                or (init_top_z - final_top_z) > 0.10
            ) else "stable"
            final_label = detected_label
        finally:
            p.disconnect()

        if target_label is None or detected_label == target_label or attempt == max_tries - 1:
            break

    meta = {
        "id": sample_id,
        "seed": seed,
        "scene_seed": seed + attempt_used * 100003,
        "sim_seed": actual_sim_seed,
        "mode_requested": mode,
        "unstable_requested": unstable_requested,
        "label": final_label,
        "detected_label": detected_label,
        "mode_match": (target_label is None or detected_label == target_label),
        "mode_attempts": attempt_used + 1,
        "fps": cfg.fps,
        "frames": cfg.frames,
        "resolution": [cfg.width, cfg.height],
        "render": "pybullet_hardware_opengl",
        "physics": "pybullet_rigid_body",
        "num_blocks": num_blocks,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate structure-collapse simulations using PyBullet rigid-body physics. "
            "Outputs: front.png, top.png, simulation.mp4, meta.json"
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--data-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prefix", type=str, default="SIM")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-seed", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["stable", "unstable", "random"], default="random")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "random":
        n_unstable = args.count // 2
        modes = ["unstable"] * n_unstable + ["stable"] * (args.count - n_unstable)
        mode_rng = np.random.default_rng(args.seed ^ 0xA5A5)
        mode_rng.shuffle(modes)
    else:
        modes = [args.mode] * args.count

    rows = []
    for i, mode_i in enumerate(modes):
        sample_id = f"{args.prefix}_{args.start_index + i:04d}"
        meta = run_single(
            sample_id=sample_id,
            out_root=args.out_root,
            seed=args.seed + i,
            mode=mode_i,
            cfg=cfg,
            sim_seed=(None if args.sim_seed is None else args.sim_seed + i),
            gui=args.gui,
        )
        rows.append(meta)
        print(f"generated: {sample_id} -> {meta['label']}")

    with (args.out_root / "generated_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"saved summary: {args.out_root / 'generated_summary.json'}")


if __name__ == "__main__":
    main()

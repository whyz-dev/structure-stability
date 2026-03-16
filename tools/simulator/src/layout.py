from __future__ import annotations

import math

import numpy as np

from .config import LayoutBlock, SimConfig


def _compute_mass(dx: float, dy: float, dz: float, density: float) -> float:
    return max(1e-4, dx * dy * dz * density)


def block_dims(rng: np.random.Generator, cfg: SimConfig) -> tuple[float, float, float]:
    scale = rng.uniform(1.0 - cfg.edge_jitter, 1.0 + cfg.edge_jitter)
    dx = cfg.block_edge * scale
    dy = cfg.block_edge * rng.uniform(0.92, 1.08) * scale
    dz = cfg.block_edge * rng.uniform(0.86, 1.02) * scale
    return dx, dy, dz


def candidate_dict(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> dict:
    return {"x": x, "y": y, "z": z, "dx": dx, "dy": dy, "dz": dz}


def rect_from_xywh(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2.0, x + w / 2.0, y - h / 2.0, y + h / 2.0)


def rect_intersection(
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


def rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[1] - rect[0]) * max(0.0, rect[3] - rect[2])


def rect_corners(rect: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    return [
        (rect[0], rect[2]),
        (rect[0], rect[3]),
        (rect[1], rect[2]),
        (rect[1], rect[3]),
    ]


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[tuple[float, float]]) -> np.ndarray:
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


def point_inside_support_with_margin(point: tuple[float, float], hull: np.ndarray, margin: float) -> bool:
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


def _is_aabb_overlap(candidate: dict, blk: LayoutBlock) -> bool:
    ax0, ax1 = candidate["x"] - candidate["dx"] / 2.0, candidate["x"] + candidate["dx"] / 2.0
    ay0, ay1 = candidate["y"] - candidate["dy"] / 2.0, candidate["y"] + candidate["dy"] / 2.0
    az0, az1 = candidate["z"] - candidate["dz"] / 2.0, candidate["z"] + candidate["dz"] / 2.0
    bx0, bx1 = blk.x - blk.dx / 2.0, blk.x + blk.dx / 2.0
    by0, by1 = blk.y - blk.dy / 2.0, blk.y + blk.dy / 2.0
    bz0, bz1 = blk.z - blk.dz / 2.0, blk.z + blk.dz / 2.0
    return (ax0 < bx1 and ax1 > bx0) and (ay0 < by1 and ay1 > by0) and (az0 < bz1 and az1 > bz0)


def _clipped_xy(x: float, y: float, dx: float, dy: float, cfg: SimConfig) -> tuple[float, float]:
    x = float(np.clip(x, -cfg.world_half + dx / 2.0, cfg.world_half - dx / 2.0))
    y = float(np.clip(y, -cfg.world_half + dy / 2.0, cfg.world_half - dy / 2.0))
    return x, y


def find_support_contacts(candidate: dict, placements: list[LayoutBlock], cfg: SimConfig) -> dict:
    c_rect = rect_from_xywh(candidate["x"], candidate["y"], candidate["dx"], candidate["dy"])
    c_bottom = float(candidate["z"] - candidate["dz"] / 2.0)
    z_tol = max(0.01, cfg.gap * 3.0)
    area_tol = max(1e-4, candidate["dx"] * candidate["dy"] * 0.03)

    supports: list[tuple[int, tuple[float, float, float, float], float]] = []
    for idx, blk in enumerate(placements):
        blk_top = blk.z + blk.dz / 2.0
        if abs(c_bottom - blk_top) > z_tol:
            continue
        inter = rect_intersection(c_rect, rect_from_xywh(blk.x, blk.y, blk.dx, blk.dy))
        if inter is None:
            continue
        area = rect_area(inter)
        if area >= area_tol:
            supports.append((idx, inter, area))

    floor_contact = abs(c_bottom) <= z_tol
    return {"floor": floor_contact, "supports": supports}


def compute_support_region(candidate: dict, contacts: dict) -> dict:
    patches = [item[1] for item in contacts["supports"]]
    if contacts["floor"] and not patches:
        patches = [rect_from_xywh(candidate["x"], candidate["y"], candidate["dx"], candidate["dy"])]
    if not patches:
        return {"patches": [], "hull": np.zeros((0, 2), dtype=np.float64)}

    points: list[tuple[float, float]] = []
    for patch in patches:
        points.extend(rect_corners(patch))
    return {"patches": patches, "hull": convex_hull(points)}


def append_candidate(placements: list[LayoutBlock], candidate: dict, cfg: SimConfig, margin: float) -> bool:
    candidate["x"], candidate["y"] = _clipped_xy(candidate["x"], candidate["y"], candidate["dx"], candidate["dy"], cfg)
    contacts = find_support_contacts(candidate, placements, cfg)
    if placements and not contacts["supports"] and not contacts["floor"]:
        return False

    support_region = compute_support_region(candidate, contacts)
    if margin >= 0.0 and not point_inside_support_with_margin(
        (candidate["x"], candidate["y"]),
        support_region["hull"],
        margin,
    ):
        return False
    if any(_is_aabb_overlap(candidate, blk) for blk in placements):
        return False

    supporters = [-1] if contacts["floor"] and not contacts["supports"] else [idx for idx, _, _ in contacts["supports"]]
    placements.append(
        LayoutBlock(
            x=float(candidate["x"]),
            y=float(candidate["y"]),
            z=float(candidate["z"]),
            dx=float(candidate["dx"]),
            dy=float(candidate["dy"]),
            dz=float(candidate["dz"]),
            mass=_compute_mass(candidate["dx"], candidate["dy"], candidate["dz"], cfg.density),
            supporters=supporters,
            support_region=support_region,
        )
    )
    return True


def compute_substructure_com(block_idx: int, placements: list[LayoutBlock], children: dict[int, list[int]]) -> np.ndarray:
    stack = [block_idx]
    visited: set[int] = set()
    total_mass = 0.0
    weighted = np.zeros(3, dtype=np.float64)
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        blk = placements[idx]
        total_mass += blk.mass
        weighted += blk.mass * np.array([blk.x, blk.y, blk.z], dtype=np.float64)
        stack.extend(children.get(idx, []))
    return weighted / max(total_mass, 1e-9)


def validate_load_path(placements: list[LayoutBlock]) -> bool:
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

    if any(val == 0 for val in degree):
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
        if not point_inside_support_with_margin((float(com[0]), float(com[1])), blk.support_region["hull"], margin):
            return False
    return True


def propose_stable_candidate(placements: list[LayoutBlock], rng: np.random.Generator, cfg: SimConfig) -> dict:
    dx, dy, dz = block_dims(rng, cfg)
    if not placements:
        return candidate_dict(0.0, 0.0, dz / 2.0, dx, dy, dz)

    tops = np.array([blk.z + blk.dz / 2.0 for blk in placements], dtype=np.float64)
    top_threshold = float(np.quantile(tops, 0.55))
    support_pool = [i for i, top in enumerate(tops) if top >= top_threshold]
    primary = int(rng.choice(support_pool))
    support_indices = [primary]

    peers = []
    for idx in support_pool:
        if idx == primary:
            continue
        same_level = abs(float(tops[idx] - tops[primary])) <= 0.02
        near_xy = math.hypot(placements[idx].x - placements[primary].x, placements[idx].y - placements[primary].y)
        if same_level and near_xy <= 1.4 * (placements[idx].dx + placements[primary].dx):
            peers.append(idx)
    if peers and rng.random() < 0.45:
        support_indices.append(int(rng.choice(peers)))

    sx = float(np.mean([placements[idx].x for idx in support_indices]))
    sy = float(np.mean([placements[idx].y for idx in support_indices]))
    top_z = float(np.mean([tops[idx] for idx in support_indices]))
    jitter = 0.10 * min(dx, dy)
    return candidate_dict(
        sx + float(rng.uniform(-jitter, jitter)),
        sy + float(rng.uniform(-jitter, jitter)),
        top_z + dz / 2.0,
        dx,
        dy,
        dz,
    )


def fallback_stable_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    base_side = int(rng.choice([2, 3]))
    floors = int(rng.integers(4, 6))
    top_z = 0.0
    for layer in range(floors):
        side = max(1, base_side - layer // 2)
        dims = [block_dims(rng, cfg) for _ in range(side * side)]
        sx = max(dim[0] for dim in dims) + cfg.gap
        sy = max(dim[1] for dim in dims) + cfg.gap
        max_h = max(dim[2] for dim in dims)
        z = top_z + max_h / 2.0 + (cfg.gap if layer > 0 else 0.0)
        for i, (dx, dy, dz) in enumerate(dims):
            ix = i % side
            iy = i // side
            x = (ix - (side - 1) / 2.0) * sx
            y = (iy - (side - 1) / 2.0) * sy
            append_candidate(placements, candidate_dict(x, y, z, dx, dy, dz), cfg, margin=0.02 * min(dx, dy))
        top_z = z + max_h / 2.0
    return placements


def build_stable_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    for _ in range(24):
        placements: list[LayoutBlock] = []
        dx0, dy0, dz0 = block_dims(rng, cfg)
        append_candidate(placements, candidate_dict(0.0, 0.0, dz0 / 2.0, dx0, dy0, dz0), cfg, margin=0.0)

        target_blocks = int(rng.integers(cfg.min_stable_blocks, cfg.max_stable_blocks + 1))
        fail_streak = 0
        while len(placements) < target_blocks and fail_streak < target_blocks * 10:
            cand = propose_stable_candidate(placements, rng, cfg)
            margin = 0.14 * min(cand["dx"], cand["dy"])
            if append_candidate(placements, cand, cfg, margin):
                fail_streak = 0
            else:
                fail_streak += 1

        if len(placements) >= cfg.min_stable_blocks and validate_load_path(placements):
            return placements

    return fallback_stable_layout(rng, cfg)


def build_unstable_tower_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    dx, dy, dz = block_dims(rng, cfg)
    append_candidate(placements, candidate_dict(0.0, 0.0, dz / 2.0, dx, dy, dz), cfg, margin=0.0)

    direction = np.asarray(rng.choice([[1.0, 0.2], [0.9, -0.3], [0.6, 0.6]], size=1)[0], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    offset_scale = float(rng.uniform(0.22, 0.32))
    last_x, last_y = 0.0, 0.0
    top_z = dz

    floors = int(rng.integers(8, 13))
    for level in range(1, floors):
        ndx, ndy, ndz = block_dims(rng, cfg)
        step = offset_scale * min(ndx, ndy)
        lateral = direction * step * level
        x = float(last_x + lateral[0])
        y = float(last_y + lateral[1])
        z = float(top_z + ndz / 2.0 + cfg.gap)
        cand = candidate_dict(x, y, z, ndx, ndy, ndz)
        margin = 0.005 * min(ndx, ndy)
        if not append_candidate(placements, cand, cfg, margin):
            x = float(last_x + lateral[0] * 0.7)
            y = float(last_y + lateral[1] * 0.7)
            append_candidate(placements, candidate_dict(x, y, z, ndx, ndy, ndz), cfg, margin=0.0)
        last_x, last_y = placements[-1].x, placements[-1].y
        top_z = placements[-1].z + placements[-1].dz / 2.0
    return placements


def build_unstable_bridge_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    gap = cfg.block_edge * float(rng.uniform(1.2, 1.6))
    height = int(rng.integers(4, 6))

    left_x = -gap / 2.0
    right_x = gap / 2.0
    top_left = 0.0
    top_right = 0.0
    for _ in range(height):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_left + dz / 2.0 + (cfg.gap if top_left > 0.0 else 0.0)
        append_candidate(placements, candidate_dict(left_x, 0.0, z, dx, dy, dz), cfg, margin=0.0)
        top_left = placements[-1].z + placements[-1].dz / 2.0

        dx, dy, dz = block_dims(rng, cfg)
        z = top_right + dz / 2.0 + (cfg.gap if top_right > 0.0 else 0.0)
        append_candidate(placements, candidate_dict(right_x, 0.0, z, dx, dy, dz), cfg, margin=0.0)
        top_right = placements[-1].z + placements[-1].dz / 2.0

    beam_dx = gap + cfg.block_edge * float(rng.uniform(1.4, 1.8))
    beam_dy = cfg.block_edge * float(rng.uniform(0.90, 1.05))
    beam_dz = cfg.block_edge * float(rng.uniform(0.92, 1.02))
    beam_z = max(top_left, top_right) + beam_dz / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(0.0, 0.0, beam_z, beam_dx, beam_dy, beam_dz), cfg, margin=0.004 * beam_dy)

    dx, dy, dz = block_dims(rng, cfg)
    topper_x = float(rng.uniform(0.18, 0.30) * beam_dx)
    topper_z = placements[-1].z + placements[-1].dz / 2.0 + dz / 2.0 + cfg.gap
    append_candidate(placements, candidate_dict(topper_x, 0.0, topper_z, dx, dy, dz), cfg, margin=0.0)
    return placements


def build_unstable_cantilever_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    placements: list[LayoutBlock] = []
    side = 2
    dims0 = [block_dims(rng, cfg) for _ in range(side * side)]
    sx = max(dim[0] for dim in dims0) + cfg.gap
    sy = max(dim[1] for dim in dims0) + cfg.gap
    max_h = max(dim[2] for dim in dims0)
    z0 = max_h / 2.0
    for i, (dx, dy, dz) in enumerate(dims0):
        ix = i % side
        iy = i // side
        x = (ix - (side - 1) / 2.0) * sx
        y = (iy - (side - 1) / 2.0) * sy
        append_candidate(placements, candidate_dict(x, y, z0, dx, dy, dz), cfg, margin=0.0)

    top_z = max(blk.z + blk.dz / 2.0 for blk in placements)
    spine_levels = int(rng.integers(3, 5))
    spine_x = 0.0
    spine_y = 0.0
    for _ in range(spine_levels):
        dx, dy, dz = block_dims(rng, cfg)
        z = top_z + dz / 2.0 + cfg.gap
        append_candidate(placements, candidate_dict(spine_x, spine_y, z, dx, dy, dz), cfg, margin=0.02 * min(dx, dy))
        top_z = placements[-1].z + placements[-1].dz / 2.0
        spine_x = placements[-1].x
        spine_y = placements[-1].y

    arm_len = int(rng.integers(3, 6))
    direction = np.asarray([1.0, float(rng.uniform(-0.35, 0.35))], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    arm_x, arm_y = spine_x, spine_y
    for idx in range(arm_len):
        dx, dy, dz = block_dims(rng, cfg)
        support_limit = 0.42 * min(dx, dy)
        arm_x += float(direction[0] * support_limit)
        arm_y += float(direction[1] * support_limit)
        z = top_z + dz / 2.0 + (cfg.gap if idx == 0 else 0.0)
        append_candidate(placements, candidate_dict(arm_x, arm_y, z, dx, dy, dz), cfg, margin=0.0)
        top_z = placements[-1].z + placements[-1].dz / 2.0
    return placements


def _sample_unstable_layout(rng: np.random.Generator, cfg: SimConfig) -> list[LayoutBlock]:
    builders = [build_unstable_tower_layout, build_unstable_bridge_layout, build_unstable_cantilever_layout]
    fallback: list[LayoutBlock] | None = None
    for _ in range(12):
        builder = builders[int(rng.integers(0, len(builders)))]
        layout = builder(rng, cfg)
        if fallback is None or len(layout) > len(fallback):
            fallback = layout
        if len(layout) >= cfg.min_unstable_blocks:
            return layout
    return fallback if fallback is not None else build_unstable_tower_layout(rng, cfg)


def generate_layout(rng: np.random.Generator, mode: str, cfg: SimConfig) -> tuple[list[LayoutBlock], bool]:
    if mode == "stable":
        return build_stable_layout(rng, cfg), False
    if mode == "unstable":
        return _sample_unstable_layout(rng, cfg), True

    unstable = bool(rng.integers(0, 2))
    if unstable:
        return _sample_unstable_layout(rng, cfg), True
    return build_stable_layout(rng, cfg), False

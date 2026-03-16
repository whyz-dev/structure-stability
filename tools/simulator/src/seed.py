from __future__ import annotations

import numpy as np


def build_mode_schedule(mode: str, count: int, seed: int) -> list[str]:
    if mode != "random":
        return [mode] * count

    n_unstable = count // 2
    modes = ["unstable"] * n_unstable + ["stable"] * (count - n_unstable)
    rng = np.random.default_rng(seed ^ 0xA5A5)
    rng.shuffle(modes)
    return modes


def scene_seed(base_seed: int, attempt: int) -> int:
    return base_seed + attempt * 100003


def sim_seed(scene_seed_value: int, base_sim_seed: int | None, attempt: int) -> int:
    if base_sim_seed is None:
        return int((scene_seed_value * 1664525 + 1013904223) & 0xFFFFFFFF)
    return int(base_sim_seed + attempt * 100003)

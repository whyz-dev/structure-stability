from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

try:
    from ..src.config import SimConfig
    from ..src.files import ensure_sample_dir, open_writer, save_meta, save_sample_images, save_summary
    from ..src.scene import (
        connect_pybullet,
        instantiate_layout,
        render_front,
        render_top,
        reset_world,
        resolve_initial_overlaps,
        run_presim_stability_check,
        step_world,
    )
    from ..src.seed import build_mode_schedule, scene_seed, sim_seed
except ImportError:
    from src.config import SimConfig
    from src.files import ensure_sample_dir, open_writer, save_meta, save_sample_images, save_summary
    from src.scene import (
        connect_pybullet,
        instantiate_layout,
        render_front,
        render_top,
        reset_world,
        resolve_initial_overlaps,
        run_presim_stability_check,
        step_world,
    )
    from src.seed import build_mode_schedule, scene_seed, sim_seed

from .structures import build_structure_layout_v2, list_structures
from .edav2_profile import apply_tone_profile, sample_edav2_render_setup


def detect_unstable(body_ids: list[int], init_positions: np.ndarray, final_positions: np.ndarray) -> bool:
    lin = []
    ang = []
    for body_id in body_ids:
        lv, av = p.getBaseVelocity(body_id)
        lin.append(float(np.linalg.norm(lv)))
        ang.append(float(np.linalg.norm(av)))

    z_drop = init_positions[:, 2] - final_positions[:, 2]
    init_xy = init_positions[:, :2]
    final_xy = final_positions[:, :2]
    init_center = np.mean(init_xy, axis=0)
    final_center = np.mean(final_xy, axis=0)
    init_spread = float(np.mean(np.linalg.norm(init_xy - init_center, axis=1)))
    final_spread = float(np.mean(np.linalg.norm(final_xy - final_center, axis=1)))

    return (
        int(np.sum(z_drop > 0.035)) >= max(3, len(body_ids) // 4)
        or float(np.mean(z_drop)) > 0.045
        or float(np.max(init_positions[:, 2]) - np.max(final_positions[:, 2])) > 0.11
        or (final_spread - init_spread) > 0.05
        or int(np.sum(np.asarray(lin) > 0.06)) >= max(2, len(body_ids) // 5)
        or int(np.sum(np.asarray(ang) > 0.50)) >= max(2, len(body_ids) // 5)
    )


def run_single(
    sample_id: str,
    out_root: Path,
    seed: int,
    mode: str,
    structure: str,
    cfg: SimConfig,
    gui: bool = False,
) -> dict:
    out_dir = ensure_sample_dir(out_root, sample_id)
    target_label = mode if mode in ("stable", "unstable") else None
    max_tries = cfg.mode_match_max_tries if target_label is not None else 1

    final_meta: dict | None = None
    for attempt in range(max_tries):
        scene_seed_value = scene_seed(seed, attempt)
        rng = np.random.default_rng(scene_seed_value)
        stability_label = mode if target_label else str(rng.choice(["stable", "unstable"]))
        result = build_structure_layout_v2(rng, cfg, structure=structure, stability_label=stability_label)
        render_setup = sample_edav2_render_setup(rng)

        connect_pybullet(gui=gui)
        try:
            reset_world(cfg)
            body_ids = instantiate_layout(result.layout, rng, cfg)
            resolve_initial_overlaps(body_ids, cfg)

            if not run_presim_stability_check(body_ids, cfg, stable_mode=(result.stability_label == "stable")):
                if target_label == "stable" and attempt < max_tries - 1:
                    continue

            actual_sim_seed = sim_seed(scene_seed_value)

            writer = open_writer(out_dir / "simulation.mp4", cfg)
            first_front = render_front(
                cfg,
                cam=render_setup.front_camera,
                light_direction=render_setup.light_direction,
                light_color=render_setup.light_color,
            )
            first_top = render_top(
                cfg,
                cam=render_setup.top_camera,
                light_direction=render_setup.light_direction,
                light_color=render_setup.light_color,
            )
            first_front = apply_tone_profile(first_front, render_setup.front_tone)
            first_top = apply_tone_profile(first_top, render_setup.top_tone)
            writer.write(cv2.cvtColor(first_front, cv2.COLOR_RGB2BGR))

            init_positions = np.array(
                [p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids],
                dtype=np.float64,
            )
            init_mean_z = float(np.mean(init_positions[:, 2]))
            init_top_z = float(np.max(init_positions[:, 2]))

            for _ in range(max(0, cfg.frames - 1)):
                step_world(cfg)
                frame = render_front(
                    cfg,
                    cam=render_setup.front_camera,
                    light_direction=render_setup.light_direction,
                    light_color=render_setup.light_color,
                )
                frame = apply_tone_profile(frame, render_setup.front_tone)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            save_sample_images(out_dir, first_front, first_top)

            final_positions = np.array(
                [p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids],
                dtype=np.float64,
            )
            final_mean_z = float(np.mean(final_positions[:, 2]))
            final_top_z = float(np.max(final_positions[:, 2]))
            detected_label = "unstable" if (
                detect_unstable(body_ids, init_positions, final_positions)
                or (init_mean_z - final_mean_z) > 0.05
                or (init_top_z - final_top_z) > 0.10
            ) else "stable"

            final_meta = {
                "id": sample_id,
                "seed": seed,
                "scene_seed": scene_seed_value,
                "sim_seed": actual_sim_seed,
                "mode_requested": mode,
                "structure_requested": structure,
                "stability_label_requested": stability_label,
                "structure_label": result.structure_label,
                "tilt_axis": result.tilt_axis,
                "tilt_profile": result.tilt_profile,
                "render_profile": render_setup.to_metadata(),
                "label": detected_label,
                "detected_label": detected_label,
                "mode_match": (target_label is None or detected_label == target_label),
                "mode_attempts": attempt + 1,
                "fps": cfg.fps,
                "frames": cfg.frames,
                "resolution": [cfg.width, cfg.height],
                "render": "pybullet_hardware_opengl",
                "physics": "pybullet_rigid_body",
                "num_blocks": len(body_ids),
                "generator_version": "v2.0_structure_first",
            }
        finally:
            p.disconnect()

        if target_label is None or final_meta["detected_label"] == target_label or attempt == max_tries - 1:
            break

    if final_meta is None:
        raise RuntimeError("Failed to generate a valid sample")

    save_meta(out_dir, final_meta)
    return final_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generator v2.0: structure-first simulation generation for 7 structure classes "
            "with stable/unstable labels."
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/generated_v2_0"))
    parser.add_argument("--prefix", type=str, default="SIM20")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["stable", "unstable", "random"], default="random")
    parser.add_argument("--structure", type=str, default="random")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--list-structures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.list_structures:
        for name in list_structures():
            print(name)
        return

    rows = []
    mode_schedule = build_mode_schedule(args.mode, args.count, args.seed)
    for i, mode_i in enumerate(mode_schedule):
        sample_id = f"{args.prefix}_{args.start_index + i:04d}"
        meta = run_single(
            sample_id=sample_id,
            out_root=args.out_root,
            seed=args.seed + i,
            mode=mode_i,
            structure=args.structure,
            cfg=cfg,
            gui=args.gui,
        )
        rows.append(meta)
        print(
            f"generated: {sample_id} -> {meta['label']} "
            f"(structure={meta['structure_label']}, tilt={meta['tilt_axis']})"
        )

    save_summary(args.out_root, rows)
    print(f"saved summary: {args.out_root / 'generated_summary.json'}")


if __name__ == "__main__":
    main()

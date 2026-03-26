from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p

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

from .layout_patterns import generate_layout_v1_1, list_patterns
from .profile import FeatureStats, StructureProfileBank
from .structure_features import STRUCTURE_FEATURES, extract_structure_features


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CSV = ROOT / "outputs" / "physics_feature_analysis_v2" / "class_analysis_features.csv"


@dataclass(frozen=True)
class MatchConfig:
    quantile_low: float = 0.15
    quantile_high: float = 0.85
    score_threshold: float = 1.65
    max_violations: int = 2
    hard_margin_ratio: float = 0.12
    layout_match_max_tries: int = 40
    feature_grid_size: int = 192


FEATURE_WEIGHTS = {
    "top_fill_ratio": 1.4,
    "top_support_width_frac": 1.2,
    "top_support_height_frac": 1.1,
    "top_centroid_dx": 1.0,
    "top_centroid_dy": 0.8,
    "front_height_frac": 0.8,
    "front_slenderness": 1.0,
    "front_base_width_frac": 1.3,
    "front_top_width_frac": 1.0,
    "front_tilt": 1.2,
    "front_top_heaviness": 1.1,
    "collapse_margin_proxy": 1.4,
}


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


def choose_profile_split(domain_profile: str, rng: np.random.Generator) -> str:
    if domain_profile == "train_like":
        return "train"
    if domain_profile == "dev_like":
        return "dev"
    return "train" if float(rng.random()) < 0.5 else "dev"


def _match_score(
    features: dict[str, float],
    target: dict[str, FeatureStats],
    cfg: MatchConfig,
) -> tuple[bool, float, int]:
    total_weight = 0.0
    total_score = 0.0
    violations = 0
    for name in STRUCTURE_FEATURES:
        if name not in target or name not in features:
            continue
        stat = target[name]
        value = float(features[name])
        width = max(stat.high - stat.low, 1e-4)
        hard_margin = cfg.hard_margin_ratio * width + 1e-4
        if value < (stat.low - hard_margin) or value > (stat.high + hard_margin):
            violations += 1
        z = abs(value - stat.median) / max(stat.iqr, 1e-4)
        weight = float(FEATURE_WEIGHTS.get(name, 1.0))
        total_score += weight * z
        total_weight += weight
    if total_weight <= 0.0:
        return True, 0.0, 0
    score = float(total_score / total_weight)
    ok = (violations <= cfg.max_violations) and (score <= cfg.score_threshold)
    return ok, score, int(violations)


def select_layout_with_matching(
    rng: np.random.Generator,
    mode: str,
    sim_cfg: SimConfig,
    pattern: str | None,
    domain_profile: str,
    match_cfg: MatchConfig,
    profile_bank: StructureProfileBank | None,
    enable_match: bool,
) -> tuple[list, bool, str, str, int, bool, float, int, dict[str, float]]:
    profile_split = choose_profile_split(domain_profile, rng)
    best_layout = None
    best_unstable = False
    best_pattern = "unknown"
    best_score = float("inf")
    best_violations = 999
    best_features: dict[str, float] = {}

    tries = match_cfg.layout_match_max_tries if enable_match else 1
    for layout_try in range(1, tries + 1):
        layout, unstable_requested, generated_pattern = generate_layout_v1_1(rng, mode, sim_cfg, pattern=pattern)
        features = extract_structure_features(layout, sim_cfg, grid_size=match_cfg.feature_grid_size)
        label_name = "unstable" if unstable_requested else "stable"

        if (not enable_match) or (profile_bank is None) or (not profile_bank.is_ready()):
            return layout, unstable_requested, generated_pattern, profile_split, layout_try, True, 0.0, 0, features

        target = profile_bank.get_target(profile_split, label_name)
        if target is None:
            return layout, unstable_requested, generated_pattern, profile_split, layout_try, True, 0.0, 0, features

        ok, score, violations = _match_score(features, target, match_cfg)
        if score < best_score:
            best_layout = layout
            best_unstable = unstable_requested
            best_pattern = generated_pattern
            best_score = score
            best_violations = violations
            best_features = features
        if ok:
            return (
                layout,
                unstable_requested,
                generated_pattern,
                profile_split,
                layout_try,
                True,
                score,
                violations,
                features,
            )

    if best_layout is None:
        layout, unstable_requested, generated_pattern = generate_layout_v1_1(rng, mode, sim_cfg, pattern=pattern)
        features = extract_structure_features(layout, sim_cfg, grid_size=match_cfg.feature_grid_size)
        return layout, unstable_requested, generated_pattern, profile_split, tries, False, float("nan"), 999, features
    return (
        best_layout,
        best_unstable,
        best_pattern,
        profile_split,
        tries,
        False,
        float(best_score),
        int(best_violations),
        best_features,
    )


def run_single(
    sample_id: str,
    out_root: Path,
    seed: int,
    mode: str,
    sim_cfg: SimConfig,
    pattern: str | None,
    domain_profile: str,
    match_cfg: MatchConfig,
    profile_bank: StructureProfileBank | None,
    enable_match: bool = True,
    gui: bool = False,
) -> dict:
    out_dir = ensure_sample_dir(out_root, sample_id)
    target_label = mode if mode in ("stable", "unstable") else None
    max_tries = sim_cfg.mode_match_max_tries if target_label is not None else 1

    final_meta: dict | None = None
    for attempt in range(max_tries):
        scene_seed_value = scene_seed(seed, attempt)
        rng = np.random.default_rng(scene_seed_value)
        mode_for_layout = mode if target_label else "random"
        (
            layout,
            unstable_requested,
            generated_pattern,
            profile_split,
            layout_match_tries,
            layout_match_ok,
            layout_match_score,
            layout_match_violations,
            layout_features,
        ) = select_layout_with_matching(
            rng=rng,
            mode=mode_for_layout,
            sim_cfg=sim_cfg,
            pattern=pattern,
            domain_profile=domain_profile,
            match_cfg=match_cfg,
            profile_bank=profile_bank,
            enable_match=enable_match,
        )

        connect_pybullet(gui=gui)
        try:
            reset_world(sim_cfg)
            body_ids = instantiate_layout(layout, rng, sim_cfg)
            resolve_initial_overlaps(body_ids, sim_cfg)

            if not run_presim_stability_check(body_ids, sim_cfg, stable_mode=not unstable_requested):
                if target_label == "stable" and attempt < max_tries - 1:
                    continue

            actual_sim_seed = sim_seed(scene_seed_value)
            writer = open_writer(out_dir / "simulation.mp4", sim_cfg)
            first_front = render_front(sim_cfg)
            first_top = render_top(sim_cfg)
            writer.write(cv2.cvtColor(first_front, cv2.COLOR_RGB2BGR))

            init_positions = np.array(
                [p.getBasePositionAndOrientation(body_id)[0] for body_id in body_ids],
                dtype=np.float64,
            )
            init_mean_z = float(np.mean(init_positions[:, 2]))
            init_top_z = float(np.max(init_positions[:, 2]))

            for _ in range(max(0, sim_cfg.frames - 1)):
                step_world(sim_cfg)
                frame = render_front(sim_cfg)
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
                "pattern_requested": pattern,
                "pattern_generated": generated_pattern,
                "unstable_requested": unstable_requested,
                "label": detected_label,
                "detected_label": detected_label,
                "mode_match": (target_label is None or detected_label == target_label),
                "mode_attempts": attempt + 1,
                "fps": sim_cfg.fps,
                "frames": sim_cfg.frames,
                "resolution": [sim_cfg.width, sim_cfg.height],
                "render": "pybullet_hardware_opengl",
                "physics": "pybullet_rigid_body",
                "num_blocks": len(body_ids),
                "generator_version": "v1.1_test",
                "domain_profile": domain_profile,
                "profile_split": profile_split,
                "layout_match_enabled": bool(enable_match),
                "layout_match_ok": bool(layout_match_ok),
                "layout_match_tries": int(layout_match_tries),
                "layout_match_score": None if np.isnan(layout_match_score) else round(float(layout_match_score), 6),
                "layout_match_violations": int(layout_match_violations),
                "layout_features": {k: round(float(v), 6) for k, v in layout_features.items()},
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
            "Generator v1.1 (test): structure-profile matched layout generation "
            "with rejection sampling over structural features."
        )
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/generated_v1_1_test"))
    parser.add_argument("--prefix", type=str, default="SIM11")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["stable", "unstable", "random"], default="random")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--domain-profile", choices=["train_like", "dev_like", "mixed"], default="mixed")
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--all-patterns-once", action="store_true")
    parser.add_argument("--list-patterns", action="store_true")
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--quantile-low", type=float, default=0.15)
    parser.add_argument("--quantile-high", type=float, default=0.85)
    parser.add_argument("--match-score-threshold", type=float, default=1.65)
    parser.add_argument("--match-max-violations", type=int, default=2)
    parser.add_argument("--match-hard-margin-ratio", type=float, default=0.12)
    parser.add_argument("--layout-match-max-tries", type=int, default=40)
    parser.add_argument("--feature-grid-size", type=int, default=192)
    parser.add_argument("--no-layout-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_cfg = SimConfig()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.list_patterns:
        print("stable patterns:")
        for name in list_patterns("stable"):
            print(f"  - {name}")
        print("unstable patterns:")
        for name in list_patterns("unstable"):
            print(f"  - {name}")
        return

    match_cfg = MatchConfig(
        quantile_low=float(args.quantile_low),
        quantile_high=float(args.quantile_high),
        score_threshold=float(args.match_score_threshold),
        max_violations=int(args.match_max_violations),
        hard_margin_ratio=float(args.match_hard_margin_ratio),
        layout_match_max_tries=int(args.layout_match_max_tries),
        feature_grid_size=int(args.feature_grid_size),
    )
    profile_bank = StructureProfileBank(
        csv_path=args.feature_csv,
        quantile_low=match_cfg.quantile_low,
        quantile_high=match_cfg.quantile_high,
    )

    rows = []
    if args.all_patterns_once:
        if args.mode == "stable":
            sequence = [(name, "stable") for name in list_patterns("stable")]
        elif args.mode == "unstable":
            sequence = [(name, "unstable") for name in list_patterns("unstable")]
        else:
            sequence = (
                [(name, "stable") for name in list_patterns("stable")]
                + [(name, "unstable") for name in list_patterns("unstable")]
            )
    else:
        mode_schedule = build_mode_schedule(args.mode, args.count, args.seed)
        sequence = [(args.pattern, mode_i) for mode_i in mode_schedule]

    for i, (pattern_name, mode_i) in enumerate(sequence):
        sample_id = f"{args.prefix}_{args.start_index + i:04d}"
        meta = run_single(
            sample_id=sample_id,
            out_root=args.out_root,
            seed=args.seed + i,
            mode=mode_i,
            sim_cfg=sim_cfg,
            pattern=pattern_name,
            domain_profile=args.domain_profile,
            match_cfg=match_cfg,
            profile_bank=profile_bank,
            enable_match=not args.no_layout_match,
            gui=args.gui,
        )
        rows.append(meta)
        print(
            f"generated: {sample_id} -> {meta['label']} "
            f"(pattern={meta['pattern_generated']}, profile={meta['profile_split']}, "
            f"match={meta['layout_match_ok']}, score={meta['layout_match_score']})"
        )

    save_summary(args.out_root, rows)
    print(f"saved summary: {args.out_root / 'generated_summary.json'}")


if __name__ == "__main__":
    main()

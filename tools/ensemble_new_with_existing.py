from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def find_best_pair(
    new_dev_path: Path,
    existing_dirs: list[Path],
    out_dir: Path,
    prefer_dir_name: str = "dev_prob_ensemble_current",
) -> pd.DataFrame:
    new_df = pd.read_csv(new_dev_path)
    new_df = new_df[new_df["source_domain"] == 1][["id", "label_int", "pred_cal"]].rename(columns={"pred_cal": "prob_new"})

    all_paths: list[Path] = []
    for d in existing_dirs:
        if d.exists():
            all_paths.extend(sorted(d.glob("*_dev_probs.csv")))

    picked: dict[str, Path] = {}
    for p in all_paths:
        key = p.name
        if key not in picked or prefer_dir_name in str(p):
            picked[key] = p

    rows: list[dict[str, float | str]] = []
    for name, p in sorted(picked.items()):
        try:
            old_df = pd.read_csv(p)[["id", "prob"]]
        except Exception:
            continue

        merged = new_df.merge(old_df, on="id", how="inner")
        if len(merged) != len(new_df):
            continue

        y = merged["label_int"].to_numpy(dtype=np.int64)
        p_new = np.clip(merged["prob_new"].to_numpy(dtype=np.float64), 1e-15, 1 - 1e-15)
        p_old = np.clip(merged["prob"].to_numpy(dtype=np.float64), 1e-15, 1 - 1e-15)

        new_loss = float(log_loss(y, p_new, labels=[0, 1]))
        old_loss = float(log_loss(y, p_old, labels=[0, 1]))

        best_alpha_new = 0.0
        best_pair_loss = float("inf")
        for alpha_new in np.linspace(0.0, 1.0, 201):
            blend = np.clip(alpha_new * p_new + (1.0 - alpha_new) * p_old, 1e-15, 1 - 1e-15)
            loss = float(log_loss(y, blend, labels=[0, 1]))
            if loss < best_pair_loss:
                best_pair_loss = loss
                best_alpha_new = float(alpha_new)

        best_single = min(new_loss, old_loss)
        rows.append(
            {
                "other_model_file": name,
                "other_path": str(p),
                "new_single_logloss": new_loss,
                "other_single_logloss": old_loss,
                "best_alpha_new": best_alpha_new,
                "best_alpha_other": 1.0 - best_alpha_new,
                "best_pair_logloss": best_pair_loss,
                "best_single_logloss": best_single,
                "improvement_vs_best_single": best_single - best_pair_loss,
            }
        )

    result = pd.DataFrame(rows).sort_values(["best_pair_logloss", "improvement_vs_best_single"], ascending=[True, False]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "pair_search_new_vs_existing.csv", index=False)
    return result


def write_submission_blend(submission_a: Path, submission_b: Path, w_a: float, out_path: Path) -> None:
    a = pd.read_csv(submission_a)[["id", "unstable_prob"]].rename(columns={"unstable_prob": "pa"})
    b = pd.read_csv(submission_b)[["id", "unstable_prob"]].rename(columns={"unstable_prob": "pb"})
    merged = a.merge(b, on="id", how="inner")
    merged["unstable_prob"] = w_a * merged["pa"] + (1.0 - w_a) * merged["pb"]
    merged["stable_prob"] = 1.0 - merged["unstable_prob"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["id", "unstable_prob", "stable_prob"]].to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find best ensemble between new physics-like dev oof and existing dev-prob models.")
    parser.add_argument(
        "--new-dev-oof",
        type=Path,
        default=Path("outputs/physics_solution_like_train_design_strict_v2/oof_valid.csv"),
    )
    parser.add_argument(
        "--existing-devprob-dirs",
        type=Path,
        nargs="+",
        default=[Path("outputs/dev_prob_ensemble_v1_6_refresh"), Path("outputs/dev_prob_ensemble_current")],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ensemble_with_physics_like_v1"))
    parser.add_argument("--submission-a", type=Path, default=Path("outputs/submissions/teacher_regularization_v3.2.csv"))
    parser.add_argument("--submission-b", type=Path, default=Path("outputs/submissions/teacher_regularization_v3.1.csv"))
    parser.add_argument(
        "--submission-out",
        type=Path,
        default=Path("outputs/submissions/ensemble_devbest_v3.2_0.725_v3.1_0.275.csv"),
    )
    parser.add_argument("--submission-weight-a", type=float, default=0.725)
    args = parser.parse_args()

    pair_df = find_best_pair(args.new_dev_oof, args.existing_devprob_dirs, args.out_dir)
    if pair_df.empty:
        raise RuntimeError("No compatible dev-prob files found.")

    best = pair_df.iloc[0].to_dict()
    print("BEST_PAIR:", best)

    write_submission_blend(
        submission_a=args.submission_a,
        submission_b=args.submission_b,
        w_a=float(args.submission_weight_a),
        out_path=args.submission_out,
    )
    print(f"SAVED_SUBMISSION: {args.submission_out}")


if __name__ == "__main__":
    main()

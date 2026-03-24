from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss


def read_notebook_history(path: Path) -> dict:
    df = pd.read_csv(path)
    if "dev_logloss" not in df.columns:
        raise ValueError(f"`dev_logloss` column not found in {path}")
    best_idx = df["dev_logloss"].astype(float).idxmin()
    best_row = df.loc[best_idx]
    result = {
        "source": str(path),
        "kind": "notebook_history",
        "metric_name": "dev_logloss",
        "metric_value": float(best_row["dev_logloss"]),
    }
    if "epoch" in df.columns:
        result["epoch"] = int(best_row["epoch"])
    return result


def read_physics_design_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "design_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "kind": "physics_solution_train_design",
        "metric_name": "valid_logloss",
        "metric_value": float(data["valid_logloss"]),
    }


def read_physics_cv_metrics(run_dir: Path) -> dict | None:
    oof_path = run_dir / "oof_all.csv"
    if not oof_path.exists():
        return None
    df = pd.read_csv(oof_path)
    required = {"source_domain", "label_int", "pred_cal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {oof_path}: {sorted(missing)}")
    dev_oof = df[df["source_domain"] == 1].copy()
    if dev_oof.empty:
        raise ValueError(f"No dev rows found in {oof_path}")
    metric = log_loss(dev_oof["label_int"].values, dev_oof["pred_cal"].values, labels=[0, 1])
    return {
        "source": str(oof_path),
        "kind": "physics_solution_cv_dev_oof",
        "metric_name": "dev_oof_logloss",
        "metric_value": float(metric),
        "rows": int(len(dev_oof)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare notebook dev logloss with physics_solution dev metrics.")
    parser.add_argument("--notebook-history", type=Path, required=True, help="Path to notebook history CSV.")
    parser.add_argument(
        "--physics-run-dir",
        type=Path,
        required=True,
        help="Run directory from tools/physics_solution train-design or cv-train.",
    )
    args = parser.parse_args()

    rows: list[dict] = [read_notebook_history(args.notebook_history)]

    design_metrics = read_physics_design_metrics(args.physics_run_dir)
    if design_metrics is not None:
        rows.append(design_metrics)

    cv_metrics = read_physics_cv_metrics(args.physics_run_dir)
    if cv_metrics is not None:
        rows.append(cv_metrics)

    if len(rows) == 1:
        raise FileNotFoundError(
            f"No comparable physics_solution outputs found in {args.physics_run_dir}. "
            "Expected `design_metrics.json` or `oof_all.csv`."
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("metric_value", ascending=True).reset_index(drop=True)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()

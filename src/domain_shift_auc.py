from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
FEATURE_CSV = ROOT / "outputs" / "eda_cv" / "cv_features_max150.csv"
OUT_JSON = ROOT / "outputs" / "eda_cv" / "domain_shift_auc_summary.json"


def build_sample_level_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["split", "sample_id", "label"]
    value_cols = [c for c in feat_df.columns if c not in meta_cols + ["view", "image_path"]]

    wide_parts = []
    for view in sorted(feat_df["view"].unique()):
        part = feat_df.loc[feat_df["view"] == view, meta_cols + value_cols].copy()
        rename_map = {col: f"{view}_{col}" for col in value_cols}
        part = part.rename(columns=rename_map)
        wide_parts.append(part)

    sample_df = wide_parts[0]
    for part in wide_parts[1:]:
        sample_df = sample_df.merge(part, on=meta_cols, how="outer")

    # Cross-view deltas often capture camera / renderer inconsistencies.
    common_value_cols = [
        col
        for col in value_cols
        if f"front_{col}" in sample_df.columns and f"top_{col}" in sample_df.columns
    ]
    for col in common_value_cols:
        sample_df[f"delta_{col}"] = sample_df[f"front_{col}"] - sample_df[f"top_{col}"]

    return sample_df


def make_models(numeric_cols: list[str]) -> dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ]
    )

    return {
        "logreg": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "rf": Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            ("num", SimpleImputer(strategy="median"), numeric_cols),
                        ]
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_binary_domain_task(
    sample_df: pd.DataFrame,
    neg_split: str,
    pos_split: str,
    n_splits: int = 5,
) -> dict:
    task_df = sample_df.loc[sample_df["split"].isin([neg_split, pos_split])].copy()
    task_df["target"] = (task_df["split"] == pos_split).astype(int)

    feature_cols = [c for c in task_df.columns if c not in ["split", "sample_id", "label", "target"]]
    feature_cols = [c for c in feature_cols if task_df[c].notna().any()]
    X = task_df[feature_cols]
    y = task_df["target"].to_numpy()

    models = make_models(feature_cols)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    result = {
        "task": f"{neg_split}_vs_{pos_split}",
        "neg_split": neg_split,
        "pos_split": pos_split,
        "n_samples": int(len(task_df)),
        "class_counts": task_df["target"].value_counts().sort_index().to_dict(),
        "models": {},
    }

    for model_name, model in models.items():
        fold_aucs = []
        oof_pred = np.zeros(len(task_df), dtype=np.float64)

        for train_idx, valid_idx in cv.split(X, y):
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            oof_pred[valid_idx] = pred
            fold_aucs.append(float(roc_auc_score(y[valid_idx], pred)))

        model.fit(X, y)
        if model_name == "logreg":
            coefs = model.named_steps["clf"].coef_[0]
            importances = pd.Series(np.abs(coefs), index=feature_cols).sort_values(ascending=False)
        else:
            importances = pd.Series(
                model.named_steps["clf"].feature_importances_,
                index=feature_cols,
            ).sort_values(ascending=False)

        result["models"][model_name] = {
            "fold_auc": [round(v, 6) for v in fold_aucs],
            "mean_auc": round(float(np.mean(fold_aucs)), 6),
            "std_auc": round(float(np.std(fold_aucs)), 6),
            "oof_auc": round(float(roc_auc_score(y, oof_pred)), 6),
            "top_features": [
                {"feature": feature, "importance": round(float(score), 6)}
                for feature, score in importances.head(12).items()
            ],
        }

    return result


def main() -> None:
    feat_df = pd.read_csv(FEATURE_CSV)
    feat_df = feat_df.loc[feat_df["split"] != "generated_v2"].copy()
    sample_df = build_sample_level_features(feat_df)

    tasks = [
        ("dev", "test"),
        ("train", "dev"),
        ("train", "test"),
    ]

    results = [evaluate_binary_domain_task(sample_df, neg_split, pos_split) for neg_split, pos_split in tasks]
    summary = {
        "feature_csv": str(FEATURE_CSV),
        "sample_shape": list(sample_df.shape),
        "tasks": results,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved {OUT_JSON}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

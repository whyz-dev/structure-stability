from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentations import build_default_transforms
from models import MultiViewBidirectionalCrossAttention, MultiViewBidirectionalCrossAttentionConfig
from preprocessing import PreprocessConfig, MultiViewPreprocessor


@dataclass(frozen=True)
class EvalPreprocessPolicy:
    name: str
    apply_brightness_on_dev: bool = False
    apply_rotation_on_dev: bool = False


class EvalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path,
        transform,
        preprocessor: MultiViewPreprocessor | None,
        policy: EvalPreprocessPolicy,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessor = preprocessor
        self.policy = policy
        self.label_map = {"stable": 0, "unstable": 1}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        views = []
        for view in ("front", "top"):
            image_path = self.root_dir / sample_id / f"{view}.png"
            image = Image.open(image_path).convert("RGB")
            if self.preprocessor is not None:
                split = "raw"
                if self.policy.apply_brightness_on_dev:
                    split = "train"
                image = self.preprocessor.apply(
                    image,
                    split=split,
                    view=view,
                    image_path=image_path,
                )
                if self.policy.apply_rotation_on_dev and view == "top":
                    image = self.preprocessor.apply(
                        image,
                        split="dev",
                        view=view,
                        image_path=image_path,
                    )
            views.append(self.transform(image))

        label = self.label_map[row["label"]]
        return views, label


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_dir: Path,
    policy: EvalPreprocessPolicy,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = build_default_transforms(320)

    preprocessor = None
    if policy.apply_brightness_on_dev or policy.apply_rotation_on_dev:
        preprocessor = MultiViewPreprocessor(
            data_dir=data_dir,
            cfg=PreprocessConfig(
                enable_brightness=policy.apply_brightness_on_dev,
                enable_top_rotation=policy.apply_rotation_on_dev,
            ),
        )

    dev_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")
    dataset = EvalDataset(
        df=dev_df,
        root_dir=data_dir / "dev",
        transform=eval_transform,
        preprocessor=preprocessor,
        policy=policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiViewBidirectionalCrossAttention(MultiViewBidirectionalCrossAttentionConfig()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["ema_state_dict"] if "ema_state_dict" in checkpoint else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for views, labels in loader:
            views = [v.to(device) for v in views]
            logits = model(views).view(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.asarray(all_probs, dtype=np.float64)
    all_labels = np.asarray(all_labels, dtype=np.float64)
    clipped = np.clip(all_probs, 1e-15, 1 - 1e-15)
    logloss = -np.mean(all_labels * np.log(clipped) + (1 - all_labels) * np.log(1 - clipped))
    acc = np.mean((all_probs > 0.5) == all_labels)

    row: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "policy": policy.name,
        "val_logloss": float(logloss),
        "val_acc": float(acc),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "saved_val_logloss": checkpoint.get("val_logloss"),
        "saved_val_acc": checkpoint.get("val_acc"),
    }
    row.update({f"policy_{k}": v for k, v in asdict(policy).items() if k != "name"})
    return row


def build_default_policies() -> list[EvalPreprocessPolicy]:
    return [
        EvalPreprocessPolicy(name="raw"),
        EvalPreprocessPolicy(name="rotation_only", apply_rotation_on_dev=True),
        EvalPreprocessPolicy(name="brightness_only", apply_brightness_on_dev=True),
        EvalPreprocessPolicy(
            name="brightness_and_rotation",
            apply_brightness_on_dev=True,
            apply_rotation_on_dev=True,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-glob",
        default="outputs/weights/baseline_v2.2*.pt",
        help="Glob pattern for checkpoints to evaluate.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-csv",
        default="outputs/eda_preprocessing/v22_checkpoint_preprocessing_ablation.csv",
    )
    args = parser.parse_args()

    data_dir = (ROOT / args.data_dir).resolve()
    checkpoint_paths = sorted(ROOT.glob(args.checkpoint_glob))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found for glob: {args.checkpoint_glob}")

    rows = []
    for checkpoint_path in checkpoint_paths:
        print(f"\n### {checkpoint_path}")
        for policy in build_default_policies():
            row = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                policy=policy,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "policy": row["policy"],
                        "val_logloss": round(float(row["val_logloss"]), 6),
                        "val_acc": round(float(row["val_acc"]), 4),
                    },
                    ensure_ascii=False,
                )
            )

    result_df = pd.DataFrame(rows).sort_values(["checkpoint", "val_logloss", "policy"]).reset_index(drop=True)
    output_path = (ROOT / args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

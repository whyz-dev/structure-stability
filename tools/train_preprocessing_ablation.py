from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentations import build_default_transforms
from models import (
    EMAConfig,
    ModelEMA,
    MultiViewBidirectionalCrossAttention,
    MultiViewBidirectionalCrossAttentionConfig,
)
from preprocessing import MultiViewPreprocessor, PreprocessConfig
from reproducibility import make_generator, seed_everything, seed_worker


@dataclass(frozen=True)
class Policy:
    name: str
    enable_brightness: bool
    enable_top_rotation: bool


class MultiViewDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False, preprocessor=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.preprocessor = preprocessor
        self.label_map = {"stable": 0, "unstable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        split = str(row["split"]) if "split" in self.df.columns else ("test" if self.is_test else "train")

        base_dir = self.root_dir
        if ("sample_dir" in self.df.columns) and pd.notna(row.get("sample_dir", np.nan)):
            base_dir = str(row["sample_dir"])

        folder_path = Path(base_dir) / sample_id
        views = []
        for name in ("front", "top"):
            img_path = folder_path / f"{name}.png"
            image = Image.open(img_path).convert("RGB")
            if self.preprocessor is not None:
                image = self.preprocessor.apply(image, split=split, view=name, image_path=img_path)
            if self.transform:
                image = self.transform(image)
            views.append(image)

        if self.is_test:
            return views

        label = self.label_map[row["label"]]
        return views, label


def _extract_frame_by_sec(cap, sec, fps, frame_count):
    frame_idx = int(max(0, min(frame_count - 1, round(sec * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _extract_last_frame(cap, frame_count):
    last_idx = max(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _video_aug_cache_signature(cfg):
    keys = [
        "SEED",
        "UNSTABLE_START_MIN_SEC",
        "UNSTABLE_START_MAX_SEC",
        "UNSTABLE_FRAMES_MIN",
        "UNSTABLE_FRAMES_MAX",
        "STABLE_END_MIN_SEC",
        "STABLE_END_MAX_SEC",
        "STABLE_FRAMES_PER_VIDEO",
    ]
    return {k: cfg.get(k) for k in keys}


def build_video_augmented_df(train_df, data_dir, cfg):
    train_root = data_dir / "train"
    aug_root = data_dir / "train_video_aug"
    aug_root.mkdir(parents=True, exist_ok=True)

    cache_csv = aug_root / "aug_df.csv"
    cache_meta = aug_root / "cache_meta.json"
    cache_sig = _video_aug_cache_signature(cfg)

    if cfg.get("VIDEO_AUG_CACHE", True) and cache_csv.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text())
            if meta.get("signature") == cache_sig:
                cached_df = pd.read_csv(cache_csv)
                if not cached_df.empty:
                    cached_df["sample_dir"] = str(aug_root)
                    cached_df["split"] = "train"
                    print(f"video aug cache hit: {len(cached_df)} samples from {cache_csv}")
                    return cached_df
        except Exception as exc:
            print(f"video aug cache read failed. rebuild cache. ({exc})")

    for p in aug_root.glob("AUGV_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    rng = np.random.default_rng(cfg["SEED"])
    stable_rows = []
    unstable_rows = []
    saved_idx = 0

    def save_aug(img, label):
        nonlocal saved_idx
        aug_id = f"AUGV_{saved_idx:07d}"
        out_dir = aug_root / aug_id
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_dir / "front.png")
        Image.fromarray(img).save(out_dir / "top.png")
        row = {"id": aug_id, "label": label, "sample_dir": str(aug_root), "split": "train"}
        saved_idx += 1
        return row

    for sample_id in tqdm(train_df["id"].tolist(), desc="Video aug stable(last-frame)", dynamic_ncols=True, ascii=True):
        video_path = train_root / sample_id / "simulation.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            continue
        img = _extract_last_frame(cap, frame_count)
        cap.release()
        if img is not None:
            stable_rows.append(save_aug(img, "stable"))

    unstable_ids = train_df.loc[train_df["label"] == "unstable", "id"].tolist()
    for sample_id in tqdm(unstable_ids, desc="Video aug unstable(0.5~1.0s)", dynamic_ncols=True, ascii=True):
        video_path = train_root / sample_id / "simulation.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps is None or fps <= 0 or frame_count <= 1:
            cap.release()
            continue
        duration = frame_count / fps
        low = cfg["UNSTABLE_START_MIN_SEC"]
        high = min(cfg["UNSTABLE_START_MAX_SEC"], max(0.0, duration - 1.0 / fps))
        if high <= low:
            cap.release()
            continue
        n_unstable = int(rng.integers(cfg["UNSTABLE_FRAMES_MIN"], cfg["UNSTABLE_FRAMES_MAX"] + 1))
        unstable_secs = rng.uniform(low, high, size=n_unstable)
        for sec in unstable_secs:
            img = _extract_frame_by_sec(cap, float(sec), fps, frame_count)
            if img is not None:
                unstable_rows.append(save_aug(img, "unstable"))
        cap.release()

    stable_df = pd.DataFrame(stable_rows)
    unstable_df = pd.DataFrame(unstable_rows)
    if stable_df.empty or unstable_df.empty:
        return pd.DataFrame(columns=["id", "label", "sample_dir", "split"])

    target_unstable = len(stable_df)
    if len(unstable_df) >= target_unstable:
        unstable_bal = unstable_df.sample(n=target_unstable, random_state=cfg["SEED"])
    else:
        unstable_bal = unstable_df.sample(n=target_unstable, replace=True, random_state=cfg["SEED"])

    aug_df = pd.concat([stable_df, unstable_bal], ignore_index=True)
    aug_df = aug_df.sample(frac=1.0, random_state=cfg["SEED"]).reset_index(drop=True)
    if cfg.get("VIDEO_AUG_CACHE", True):
        aug_df.to_csv(cache_csv, index=False)
        cache_meta.write_text(json.dumps({"signature": cache_sig}, ensure_ascii=False, indent=2))
    return aug_df


def mixup_multiview_batch(views, labels, alpha=0.2):
    if alpha <= 0:
        return views, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(labels.size(0), device=labels.device)
    mixed_views = [lam * v + (1.0 - lam) * v[index, :] for v in views]
    return mixed_views, labels, labels[index], lam


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2, mixup_prob=0.5, ema=None):
    model.train()
    train_loss = 0.0
    for views, labels in tqdm(loader, desc="Training", dynamic_ncols=True, ascii=True):
        views = [v.to(device) for v in views]
        labels = labels.to(device).float()
        optimizer.zero_grad()
        if mixup_alpha > 0 and np.random.rand() < mixup_prob:
            mixed_views, labels_a, labels_b, lam = mixup_multiview_batch(views, labels, alpha=mixup_alpha)
            outputs = model(mixed_views).view(-1)
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(views).view(-1)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        train_loss += loss.item()
    return train_loss / max(len(loader), 1)


def validate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for views, labels in tqdm(loader, desc="Validation", dynamic_ncols=True, ascii=True):
            views = [v.to(device) for v in views]
            outputs = model(views).view(-1)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)
    p = np.clip(all_probs, 1e-15, 1 - 1e-15)
    logloss = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))
    acc = np.mean((all_probs > 0.5) == all_labels)
    return float(logloss), float(acc)


def maybe_limit_df(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.copy()
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def build_policy(name: str) -> Policy:
    table = {
        "raw": Policy("raw", False, False),
        "rotation_only": Policy("rotation_only", False, True),
        "brightness_only": Policy("brightness_only", True, False),
        "brightness_and_rotation": Policy("brightness_and_rotation", True, True),
    }
    if name not in table:
        raise ValueError(f"Unknown policy: {name}")
    return table[name]


def run_experiment(args, policy: Policy, device: torch.device) -> dict[str, object]:
    cfg = {
        "IMG_SIZE": args.img_size,
        "EPOCHS": args.epochs,
        "LEARNING_RATE": args.learning_rate,
        "BATCH_SIZE": args.batch_size,
        "SEED": args.seed,
        "NUM_WORKERS": args.num_workers,
        "WEIGHT_DECAY": args.weight_decay,
        "MIXUP_ALPHA": args.mixup_alpha,
        "MIXUP_PROB": args.mixup_prob,
        "MIN_LR": args.min_lr,
        "EMA_DECAY": args.ema_decay,
        "EMA_USE_FOR_EVAL": True,
        "VIDEO_AUG_ENABLE": args.video_aug,
        "VIDEO_AUG_CACHE": True,
        "UNSTABLE_START_MIN_SEC": 0.5,
        "UNSTABLE_START_MAX_SEC": 1.0,
        "UNSTABLE_FRAMES_MIN": 2,
        "UNSTABLE_FRAMES_MAX": 3,
        "STABLE_END_MIN_SEC": 9.0,
        "STABLE_END_MAX_SEC": 10.0,
        "STABLE_FRAMES_PER_VIDEO": 2,
    }

    seed_everything(cfg["SEED"])
    data_dir = (ROOT / "data").resolve()
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")

    train_df = maybe_limit_df(train_df, args.max_train_samples, args.seed)
    val_df = maybe_limit_df(val_df, args.max_dev_samples, args.seed + 1)

    train_transform, test_transform = build_default_transforms(cfg["IMG_SIZE"])
    preprocessor = MultiViewPreprocessor(
        data_dir,
        PreprocessConfig(
            enable_brightness=policy.enable_brightness,
            enable_top_rotation=policy.enable_top_rotation,
        ),
    )

    train_df_for_train = train_df.copy()
    train_df_for_train["sample_dir"] = str(data_dir / "train")
    train_df_for_train["split"] = "train"

    if cfg["VIDEO_AUG_ENABLE"]:
        aug_df = build_video_augmented_df(train_df, data_dir, cfg)
        if not aug_df.empty:
            if args.max_aug_samples is not None and args.max_aug_samples > 0 and len(aug_df) > args.max_aug_samples:
                aug_df = aug_df.sample(n=args.max_aug_samples, random_state=args.seed).reset_index(drop=True)
            train_df_for_train = pd.concat([train_df_for_train, aug_df], ignore_index=True)

    val_df_for_eval = val_df.copy()
    val_df_for_eval["sample_dir"] = str(data_dir / "dev")
    val_df_for_eval["split"] = "dev"

    train_dataset = MultiViewDataset(train_df_for_train, str(data_dir / "train"), train_transform, False, preprocessor)
    val_dataset = MultiViewDataset(val_df_for_eval, str(data_dir / "dev"), test_transform, False, preprocessor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["SEED"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["SEED"] + 1),
    )

    model = MultiViewBidirectionalCrossAttention(MultiViewBidirectionalCrossAttentionConfig()).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"], eta_min=cfg["MIN_LR"])
    ema = ModelEMA(model, EMAConfig(decay=cfg["EMA_DECAY"]))

    best_logloss = float("inf")
    best_acc = 0.0
    best_epoch = -1
    patience_left = args.early_stopping_patience
    history = []

    out_dir = (ROOT / "outputs" / "eda_preprocessing").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ablation_{policy.name}.pt"

    for epoch in range(1, cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=cfg["MIXUP_ALPHA"],
            mixup_prob=cfg["MIXUP_PROB"],
            ema=ema,
        )
        eval_model = ema.ema_model if cfg["EMA_USE_FOR_EVAL"] else model
        val_logloss, val_acc = validate(eval_model, val_loader, device)
        scheduler.step()

        improved = val_logloss < (best_logloss - args.early_stopping_min_delta)
        if improved:
            best_logloss = val_logloss
            best_acc = val_acc
            best_epoch = epoch
            patience_left = args.early_stopping_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_logloss": val_logloss,
                    "val_acc": val_acc,
                    "policy": policy.name,
                    "cfg": cfg,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1

        row = {
            "policy": policy.name,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_logloss": float(val_logloss),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "patience_left": patience_left,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if patience_left < 0:
            print(f"early stopping triggered for {policy.name} at epoch {epoch}")
            break

    hist_path = out_dir / f"ablation_{policy.name}_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    return {
        "policy": policy.name,
        "enable_brightness": policy.enable_brightness,
        "enable_top_rotation": policy.enable_top_rotation,
        "best_epoch": best_epoch,
        "best_val_logloss": best_logloss,
        "best_val_acc": best_acc,
        "history_csv": str(hist_path),
        "checkpoint_path": str(ckpt_path),
        "train_rows": len(train_df_for_train),
        "dev_rows": len(val_df_for_eval),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["raw", "rotation_only", "brightness_only", "brightness_and_rotation"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-aug", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-aug-samples", type=int)
    parser.add_argument("--summary-csv", default="outputs/eda_preprocessing/preprocessing_ablation_summary.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    summary_rows = []
    for name in args.policies:
        policy = build_policy(name)
        print(f"\n===== policy={policy.name} brightness={policy.enable_brightness} rotation={policy.enable_top_rotation} =====")
        summary_rows.append(run_experiment(args, policy, device))

    summary_df = pd.DataFrame(summary_rows).sort_values("best_val_logloss").reset_index(drop=True)
    summary_path = (ROOT / args.summary_csv).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

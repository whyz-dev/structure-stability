from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import cv2
except Exception:
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
PHYSICS_DIR = ROOT / "tools" / "physics_solution"
for path in (SRC_DIR, PHYSICS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from augmentations import build_default_transforms
from models import (
    EMAConfig,
    ModelEMA,
    MultiViewBidirectionalCrossAttention,
    MultiViewBidirectionalCrossAttentionConfig,
    MultiViewCrossAttention,
    MultiViewCrossAttentionConfig,
    MultiViewDualEncoderBidirectionalCrossAttention,
    MultiViewDualEncoderBidirectionalCrossAttentionConfig,
    MultiViewFeatureFusion,
    MultiViewFeatureFusionConfig,
)
from reproducibility import make_generator, seed_everything, seed_worker

try:
    from checkerboard_rectification import CheckerboardTopNormConfig, CheckerboardTopNormalizer
except Exception:
    CheckerboardTopNormConfig = None
    CheckerboardTopNormalizer = None


DEFAULT_CFG: dict[str, object] = {
    "IMG_SIZE": 320,
    "EPOCHS": 100,
    "LEARNING_RATE": 3e-4,
    "MIN_LR": 1e-6,
    "BATCH_SIZE": 64,
    "SEED": 42,
    "NUM_WORKERS": 0 if os.name == "nt" else 16,
    "USE_AMP": True,
    "WEIGHT_DECAY": 1e-4,
    "EMA_DECAY": 0.999,
    "EMA_USE_FOR_EVAL": True,
    "PATIENCE": 10,
    "PRETRAINED": True,
    "BACKBONE_NAME": "efficientnetv2_rw_s",
    "CHECKERBOARD_TOP_NORMALIZE": True,
    "VIDEO_AUG_ENABLE": True,
    "VIDEO_AUG_CACHE": True,
    "UNSTABLE_START_MIN_SEC": 0.5,
    "UNSTABLE_START_MAX_SEC": 1.0,
    "UNSTABLE_FRAMES_MIN": 2,
    "UNSTABLE_FRAMES_MAX": 3,
    "STABLE_END_MIN_SEC": 9.0,
    "STABLE_END_MAX_SEC": 10.0,
    "STABLE_FRAMES_PER_VIDEO": 2,
    "MAX_TRAIN_SAMPLES": None,
    "MAX_VAL_SAMPLES": None,
    "EXPERIMENT_NAME": "Architecture_Comparison",
    "EXPERIMENT_VERSION": "v1.0",
    "RUN_MODELS": [
        "feature_fusion",
        "cross_attention",
        "bidirectional_cross_attention",
        "dual_encoder_bidirectional_cross_attention",
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    build: Callable[[dict[str, object]], nn.Module]
    description: str


def _safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_model_specs() -> list[ModelSpec]:
    def build_feature_fusion(cfg: dict[str, object]) -> nn.Module:
        config = MultiViewFeatureFusionConfig(
            backbone_name=str(cfg["BACKBONE_NAME"]),
            pretrained=bool(cfg["PRETRAINED"]),
        )
        return MultiViewFeatureFusion(config)

    def build_cross_attention(cfg: dict[str, object]) -> nn.Module:
        config = MultiViewCrossAttentionConfig(
            backbone_name=str(cfg["BACKBONE_NAME"]),
            pretrained=bool(cfg["PRETRAINED"]),
        )
        return MultiViewCrossAttention(config)

    def build_bidirectional(cfg: dict[str, object]) -> nn.Module:
        config = MultiViewBidirectionalCrossAttentionConfig(
            backbone_name=str(cfg["BACKBONE_NAME"]),
            pretrained=bool(cfg["PRETRAINED"]),
        )
        return MultiViewBidirectionalCrossAttention(config)

    def build_dual_encoder(cfg: dict[str, object]) -> nn.Module:
        config = MultiViewDualEncoderBidirectionalCrossAttentionConfig(
            front_backbone_name=str(cfg["BACKBONE_NAME"]),
            top_backbone_name=str(cfg["BACKBONE_NAME"]),
            pretrained=bool(cfg["PRETRAINED"]),
        )
        return MultiViewDualEncoderBidirectionalCrossAttention(config)

    return [
        ModelSpec(
            name="feature_fusion",
            display_name="MultiViewFeatureFusion",
            build=build_feature_fusion,
            description="Shared CNN features with concat(front, top, abs diff, product).",
        ),
        ModelSpec(
            name="cross_attention",
            display_name="MultiViewCrossAttention",
            build=build_cross_attention,
            description="Single-direction front-query/top-key-value cross-attention.",
        ),
        ModelSpec(
            name="bidirectional_cross_attention",
            display_name="MultiViewBidirectionalCrossAttention",
            build=build_bidirectional,
            description="Shared encoder with bidirectional front<->top cross-attention.",
        ),
        ModelSpec(
            name="dual_encoder_bidirectional_cross_attention",
            display_name="MultiViewDualEncoderBidirectionalCrossAttention",
            build=build_dual_encoder,
            description="Separate front/top encoders with bidirectional cross-attention.",
        ),
    ]


class MultiViewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: Path | str,
        transform=None,
        is_test: bool = False,
        checkerboard_top_normalize: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        self.label_map = {"stable": 0, "unstable": 1}

        if checkerboard_top_normalize:
            if CheckerboardTopNormalizer is None or CheckerboardTopNormConfig is None:
                raise ImportError("checkerboard_rectification.py could not be imported.")
            self.top_normalizer = CheckerboardTopNormalizer(CheckerboardTopNormConfig(enabled=True))
        else:
            self.top_normalizer = None

    def __len__(self) -> int:
        return len(self.df)

    def _sample_dir(self, row: pd.Series) -> Path:
        if "sample_dir" in row and pd.notna(row["sample_dir"]):
            return Path(str(row["sample_dir"]))
        return self.root_dir

    def _load_img(self, sample_id: str, view: str, sample_dir: Path) -> Image.Image:
        path = sample_dir / sample_id / f"{view}.png"
        image = Image.open(path).convert("RGB")

        is_video_aug = sample_id.startswith("AUGV_")
        if view == "top" and self.top_normalizer is not None and not is_video_aug:
            image = self.top_normalizer.normalize(path, image)
        return image

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])
        sample_dir = self._sample_dir(row)

        views = []
        for view in ("front", "top"):
            image = self._load_img(sample_id, view, sample_dir)
            if self.transform is not None:
                image = self.transform(image)
            views.append(image)

        if self.is_test:
            return views

        label = self.label_map[str(row["label"])]
        return views, label


def _extract_frame_by_sec(cap: cv2.VideoCapture, sec: float, fps: float, frame_count: int):
    frame_idx = int(max(0, min(frame_count - 1, round(sec * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _extract_last_frame(cap: cv2.VideoCapture, frame_count: int):
    last_idx = max(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _video_aug_cache_signature(cfg: dict[str, object]) -> dict[str, object]:
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


def detect_collapse_start(
    cap: cv2.VideoCapture,
    fps: float,
    frame_count: int,
    motion_threshold: float = 0.02,
) -> float | None:
    prev_frame = None
    sample_interval = max(1, int(fps * 0.2))

    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev_frame is not None:
            diff = float(np.mean(np.abs(gray - prev_frame)))
            if diff > motion_threshold:
                return frame_idx / fps
        prev_frame = gray
    return None


def build_video_augmented_df(train_df: pd.DataFrame, data_dir: Path, cfg: dict[str, object]) -> pd.DataFrame:
    if cv2 is None:
        raise ImportError("opencv-python is required when VIDEO_AUG_ENABLE=True. Install cv2 or set VIDEO_AUG_ENABLE=False.")

    train_root = data_dir / "train"
    aug_root = data_dir / "train_video_aug"
    aug_root.mkdir(parents=True, exist_ok=True)

    cache_csv = aug_root / "aug_df.csv"
    cache_meta = aug_root / "cache_meta.json"
    cache_sig = _video_aug_cache_signature(cfg)

    if cfg.get("VIDEO_AUG_CACHE", True) and cache_csv.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("signature") == cache_sig:
                cached_df = pd.read_csv(cache_csv)
                if not cached_df.empty:
                    cached_df["sample_dir"] = str(aug_root)
                    print(f"video aug cache hit: {len(cached_df)} samples from {cache_csv}")
                    return cached_df
        except Exception as exc:
            print(f"video aug cache read failed. rebuilding cache. ({exc})")

    for path in aug_root.glob("AUGV_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    rng = np.random.default_rng(int(cfg["SEED"]))
    stable_rows: list[dict[str, object]] = []
    unstable_rows: list[dict[str, object]] = []
    saved_idx = 0

    def save_aug(img: np.ndarray, label: str) -> dict[str, object]:
        nonlocal saved_idx
        aug_id = f"AUGV_{saved_idx:07d}"
        out_dir = aug_root / aug_id
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_dir / "front.png")
        Image.fromarray(img).save(out_dir / "top.png")
        saved_idx += 1
        return {"id": aug_id, "label": label, "sample_dir": str(aug_root)}

    for sample_id in tqdm(train_df["id"].tolist(), desc="Video aug stable(last-frame)", dynamic_ncols=True):
        video_path = train_root / str(sample_id) / "simulation.mp4"
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
    for sample_id in tqdm(unstable_ids, desc="Video aug unstable(collapse-detect)", dynamic_ncols=True):
        video_path = train_root / str(sample_id) / "simulation.mp4"
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 1:
            cap.release()
            continue

        duration = frame_count / fps
        collapse_sec = detect_collapse_start(cap, fps, frame_count)
        if collapse_sec is None:
            low = float(cfg["UNSTABLE_START_MIN_SEC"])
            high = min(float(cfg["UNSTABLE_START_MAX_SEC"]), max(0.0, duration - 1.0 / fps))
        else:
            low = max(0.0, collapse_sec - 0.3)
            high = min(duration, collapse_sec + 0.5)

        if high <= low:
            cap.release()
            continue

        n_unstable = int(rng.integers(int(cfg["UNSTABLE_FRAMES_MIN"]), int(cfg["UNSTABLE_FRAMES_MAX"]) + 1))
        for sec in rng.uniform(low, high, size=n_unstable):
            img = _extract_frame_by_sec(cap, float(sec), fps, frame_count)
            if img is not None:
                unstable_rows.append(save_aug(img, "unstable"))
        cap.release()

    stable_df = pd.DataFrame(stable_rows)
    unstable_df = pd.DataFrame(unstable_rows)
    if stable_df.empty or unstable_df.empty:
        print("video aug warning: stable or unstable generated set is empty.")
        return pd.DataFrame(columns=["id", "label", "sample_dir"])

    target_unstable = len(stable_df)
    if len(unstable_df) >= target_unstable:
        unstable_balanced = unstable_df.sample(n=target_unstable, random_state=int(cfg["SEED"]))
    else:
        unstable_balanced = unstable_df.sample(n=target_unstable, replace=True, random_state=int(cfg["SEED"]))

    aug_df = pd.concat([stable_df, unstable_balanced], ignore_index=True)
    aug_df = aug_df.sample(frac=1.0, random_state=int(cfg["SEED"])).reset_index(drop=True)

    if cfg.get("VIDEO_AUG_CACHE", True):
        aug_df.to_csv(cache_csv, index=False)
        cache_meta.write_text(json.dumps({"signature": cache_sig}, ensure_ascii=False, indent=2), encoding="utf-8")
    return aug_df


def maybe_limit_df(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.copy()
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def prepare_dataframes(cfg: dict[str, object], data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(data_dir / "dev.csv", encoding="utf-8-sig")

    train_df = maybe_limit_df(train_df, cfg.get("MAX_TRAIN_SAMPLES"), int(cfg["SEED"]))
    val_df = maybe_limit_df(val_df, cfg.get("MAX_VAL_SAMPLES"), int(cfg["SEED"]) + 1)

    train_df_for_train = train_df.copy()
    train_df_for_train["sample_dir"] = str(data_dir / "train")

    if cfg.get("VIDEO_AUG_ENABLE", False):
        aug_df = build_video_augmented_df(train_df, data_dir, cfg)
        if not aug_df.empty:
            train_df_for_train = pd.concat([train_df_for_train, aug_df], ignore_index=True)
            print(f"video aug added: {len(aug_df)} samples")
        else:
            print("video aug added: 0 samples")

    val_df_for_eval = val_df.copy()
    val_df_for_eval["sample_dir"] = str(data_dir / "dev")

    print("Final train class ratio:")
    print(train_df_for_train["label"].value_counts())
    print("Validation class ratio:")
    print(val_df_for_eval["label"].value_counts())
    return train_df_for_train, val_df_for_eval


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, object],
    data_dir: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_transform, test_transform = build_default_transforms(int(cfg["IMG_SIZE"]))
    normalize_top = bool(cfg.get("CHECKERBOARD_TOP_NORMALIZE", True))

    train_dataset = MultiViewDataset(
        train_df,
        data_dir / "train",
        transform=train_transform,
        is_test=False,
        checkerboard_top_normalize=normalize_top,
    )
    val_dataset = MultiViewDataset(
        val_df,
        data_dir / "dev",
        transform=test_transform,
        is_test=False,
        checkerboard_top_normalize=normalize_top,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg["SEED"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(cfg["NUM_WORKERS"]),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg["SEED"]) + 1),
    )
    return train_loader, val_loader


def _model_logits(model: nn.Module, views: list[torch.Tensor]) -> torch.Tensor:
    outputs = model(views)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    return outputs.view(-1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: dict[str, object],
    ema: ModelEMA | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int = 0,
) -> float:
    model.train()
    amp_enabled = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for views, labels in pbar:
        views = [v.to(device, non_blocking=True) for v in views]
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = _model_logits(model, views)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        epoch_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(len(loader), 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for views, labels in tqdm(loader, desc="Validation", dynamic_ncols=True):
            views = [v.to(device, non_blocking=True) for v in views]
            labels = labels.to(device, non_blocking=True).float()
            logits = _model_logits(model, views)
            loss = criterion(logits, labels)

            val_loss += float(loss.item())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0).astype(np.float64)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.float64)
    clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    logloss = -np.mean(labels_np * np.log(clipped) + (1 - labels_np) * np.log(1 - clipped))
    acc = np.mean((probs > 0.5) == labels_np)
    return float(val_loss / max(len(loader), 1)), float(logloss), float(acc)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_model(
    spec: ModelSpec,
    cfg: dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_dir: Path,
    output_dirs: dict[str, Path],
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    seed_everything(int(cfg["SEED"]))
    train_loader, val_loader = make_dataloaders(train_df, val_df, cfg, data_dir, device)

    model = spec.build(cfg).to(device)
    parameter_count = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["LEARNING_RATE"]),
        weight_decay=float(cfg["WEIGHT_DECAY"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["EPOCHS"]),
        eta_min=float(cfg["MIN_LR"]),
    )
    ema = ModelEMA(model, EMAConfig(decay=float(cfg["EMA_DECAY"])))
    amp_enabled = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    safe_name = _safe_name(spec.name)
    checkpoint_path = output_dirs["weights"] / f"best_model_{safe_name}.pt"
    history_path = output_dirs["reports"] / f"history_{safe_name}.csv"

    print("=" * 88)
    print(f"Model: {spec.display_name}")
    print(spec.description)
    print(f"Parameters: {parameter_count:,}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 88)

    best_epoch = -1
    best_logloss = float("inf")
    best_acc = 0.0
    early_stop_counter = 0
    history_rows: list[dict[str, object]] = []
    started = time.time()

    for epoch in range(1, int(cfg["EPOCHS"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            ema=ema,
            scaler=scaler,
            epoch=epoch,
        )
        eval_model = ema.ema_model if bool(cfg["EMA_USE_FOR_EVAL"]) else model
        val_loss, val_logloss, val_acc = validate(eval_model, val_loader, criterion, device)

        improved = val_logloss < best_logloss
        if improved:
            best_logloss = val_logloss
            best_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": spec.name,
                    "display_name": spec.display_name,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_logloss": val_logloss,
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                checkpoint_path,
            )
            print(f"  -> Best model saved: {checkpoint_path} (epoch={epoch}, val_logloss={val_logloss:.6f})")
        else:
            early_stop_counter += 1
            print(f"  -> No improvement. EarlyStopping counter: {early_stop_counter}/{cfg['PATIENCE']}")

        lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "model": spec.name,
                "display_name": spec.display_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_logloss": val_logloss,
                "val_acc": val_acc,
                "best_logloss": best_logloss,
                "best_acc": best_acc,
                "lr": lr,
                "improved": improved,
                "early_stop_counter": early_stop_counter,
            }
        )

        print(
            f"Epoch [{epoch}] LR={lr:.8f} Train={train_loss:.4f} "
            f"ValLogLoss={val_logloss:.6f} ValAcc={val_acc:.4f}"
        )

        if early_stop_counter >= int(cfg["PATIENCE"]):
            print(f"! Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

        scheduler.step()

    elapsed_sec = time.time() - started
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_path, index=False)

    result = {
        "model": spec.name,
        "display_name": spec.display_name,
        "description": spec.description,
        "best_epoch": best_epoch,
        "best_val_logloss": best_logloss,
        "best_val_acc": best_acc,
        "parameter_count": parameter_count,
        "elapsed_min": elapsed_sec / 60.0,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
    }

    del model, optimizer, scheduler, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result, history_df


def make_output_dirs(cfg: dict[str, object]) -> dict[str, Path]:
    experiment_name = str(cfg["EXPERIMENT_NAME"])
    version = str(cfg["EXPERIMENT_VERSION"])
    dirs = {
        "weights": ROOT / "outputs" / "weights" / experiment_name / version,
        "reports": ROOT / "outputs" / "reports" / experiment_name / version,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_comparison(
    cfg: dict[str, object] | None = None,
    selected_model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    run_cfg = dict(DEFAULT_CFG)
    if cfg is not None:
        run_cfg.update(cfg)

    data_dir = ROOT / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data folder not found: {data_dir}")

    selected_model_names = selected_model_names or list(run_cfg["RUN_MODELS"])
    specs_by_name = {spec.name: spec for spec in get_model_specs()}
    missing = [name for name in selected_model_names if name not in specs_by_name]
    if missing:
        raise ValueError(f"Unknown model names: {missing}. Available: {sorted(specs_by_name)}")

    seed_everything(int(run_cfg["SEED"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dirs = make_output_dirs(run_cfg)

    print(f"ROOT: {ROOT}")
    print(f"DATA_DIR: {data_dir}")
    print(f"device: {device}")
    print(f"output reports: {output_dirs['reports']}")
    print(f"output weights: {output_dirs['weights']}")

    train_df, val_df = prepare_dataframes(run_cfg, data_dir)
    results: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}

    for model_name in selected_model_names:
        result, history_df = train_single_model(
            specs_by_name[model_name],
            run_cfg,
            train_df,
            val_df,
            data_dir,
            output_dirs,
            device,
        )
        results.append(result)
        histories[model_name] = history_df

    results_df = pd.DataFrame(results).sort_values("best_val_logloss").reset_index(drop=True)
    results_path = output_dirs["reports"] / "architecture_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")
    return results_df, histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare structure-stability model architectures.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-video-aug", action="store_true")
    parser.add_argument("--no-checkerboard", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(DEFAULT_CFG)
    if args.epochs is not None:
        cfg["EPOCHS"] = args.epochs
    if args.batch_size is not None:
        cfg["BATCH_SIZE"] = args.batch_size
    if args.num_workers is not None:
        cfg["NUM_WORKERS"] = args.num_workers
    if args.backbone is not None:
        cfg["BACKBONE_NAME"] = args.backbone
    if args.no_pretrained:
        cfg["PRETRAINED"] = False
    if args.no_video_aug:
        cfg["VIDEO_AUG_ENABLE"] = False
    if args.no_checkerboard:
        cfg["CHECKERBOARD_TOP_NORMALIZE"] = False
    if args.max_train_samples is not None:
        cfg["MAX_TRAIN_SAMPLES"] = args.max_train_samples
    if args.max_val_samples is not None:
        cfg["MAX_VAL_SAMPLES"] = args.max_val_samples

    selected = None
    if args.models:
        selected = [name.strip() for name in args.models.split(",") if name.strip()]

    results_df, _ = run_comparison(cfg, selected)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

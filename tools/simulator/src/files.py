from __future__ import annotations

import json
from pathlib import Path

import cv2

from .config import SimConfig


def ensure_sample_dir(out_root: Path, sample_id: str) -> Path:
    out_dir = out_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def open_writer(video_path: Path, cfg: SimConfig) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.fps,
        (cfg.width, cfg.height),
    )


def save_sample_images(out_dir: Path, front_rgb, top_rgb) -> None:
    cv2.imwrite(str(out_dir / "front.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "top.png"), cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR))


def save_meta(out_dir: Path, meta: dict) -> None:
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_summary(out_root: Path, rows: list[dict]) -> None:
    with (out_root / "generated_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

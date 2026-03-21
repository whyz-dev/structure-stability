from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_BRIGHTNESS_CACHE = ROOT / "outputs" / "eda_preprocessing" / "baseline_v22_brightness_stats.csv"


@dataclass(frozen=True)
class PreprocessConfig:
    enable_brightness: bool = True
    enable_top_rotation: bool = True
    rotation_pad_value: int = 128
    ring_ratio: float = 0.10
    rot_line_min: int = 10
    rot_conf_min: float = 0.20


def read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def brightness_mean_from_rgb(rgb: np.ndarray) -> float:
    hsv = rgb_to_hsv(rgb)
    return float(hsv[:, :, 2].mean())


def make_quantile_mapper(source: np.ndarray, target: np.ndarray, grid_size: int = 1024):
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    q = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    source_q = np.quantile(source, q)
    target_q = np.quantile(target, q)

    def map_values(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        quantiles = np.interp(values, source_q, q, left=0.0, right=1.0)
        return np.interp(quantiles, q, target_q).astype(np.float32)

    return map_values


def adjust_brightness_rgb(rgb: np.ndarray, target_mean: float) -> np.ndarray:
    hsv = rgb_to_hsv(rgb).astype(np.float32)
    value = hsv[:, :, 2]
    source_mean = float(value.mean())
    scale = float(target_mean / max(source_mean, 1e-6))
    hsv[:, :, 2] = np.clip(value * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def rotate_rgb(rgb: np.ndarray, angle_deg: float, pad_value: int = 128) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value),
    )


def _catalog_image_rows(data_dir: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "sample_submission.csv")

    rows: list[dict[str, object]] = []
    split_specs = [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]
    for split, frame in split_specs:
        split_dir = data_dir / split
        for sample_id in frame["id"].astype(str):
            for view in ("front", "top"):
                image_path = split_dir / sample_id / f"{view}.png"
                if image_path.exists():
                    rows.append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "view": view,
                            "image_path": str(image_path),
                        }
                    )
    return pd.DataFrame(rows)


def _build_brightness_stats(data_dir: Path, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv)
        expected_cols = {"split", "sample_id", "view", "image_path", "brightness_mean"}
        if expected_cols.issubset(cached.columns):
            return cached

    image_df = _catalog_image_rows(data_dir)
    rows = []
    for row in image_df.itertuples(index=False):
        rgb = read_rgb(row.image_path)
        rows.append(
            {
                "split": row.split,
                "sample_id": row.sample_id,
                "view": row.view,
                "image_path": row.image_path,
                "brightness_mean": brightness_mean_from_rgb(rgb),
            }
        )

    brightness_df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    brightness_df.to_csv(cache_csv, index=False)
    return brightness_df


def _estimate_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    s_thr = float(np.percentile(sat, 60.0))
    g_thr = float(np.percentile(gray, 45.0))
    v_thr = float(np.percentile(val, 35.0))

    mask = ((sat > s_thr) | (gray < g_thr) | (val < v_thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return (mask > 0).astype(np.uint8)

    best = 1
    best_area = 0
    h, w = gray.shape
    for idx in range(1, n_labels):
        x, y, ww, hh, area = stats[idx].tolist()
        if area > best_area and ww > 8 and hh > 8 and area < 0.995 * h * w:
            best = idx
            best_area = area
    return (labels == best).astype(np.uint8)


def _ring_mask(h: int, w: int, ratio: float) -> np.ndarray:
    radius = max(1, int(round(min(h, w) * ratio)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:radius, :] = 1
    mask[-radius:, :] = 1
    mask[:, :radius] = 1
    mask[:, -radius:] = 1
    return mask


def _line_angles(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180.0, threshold=30, minLineLength=24, maxLineGap=6)
    if lines is None or len(lines) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    angles = []
    lengths = []
    for line in lines[:400]:
        x1, y1, x2, y2 = line[0].tolist()
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        line_len = float(np.hypot(dx, dy))
        if line_len < 8:
            continue
        angle = (float(np.degrees(np.arctan2(dy, dx))) + 180.0) % 180.0
        angles.append(angle)
        lengths.append(line_len)
    if not angles:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(angles, dtype=np.float32), np.asarray(lengths, dtype=np.float32)


def estimate_top_rotation(rgb: np.ndarray, cfg: PreprocessConfig) -> dict[str, float | bool | int | str]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = _estimate_mask(rgb)
    bg = (1 - fg).astype(np.uint8)
    if int(bg.sum()) < int(0.03 * h * w):
        bg = _ring_mask(h, w, cfg.ring_ratio)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = (mag / (mag.max() + 1e-6) * 255.0).astype(np.uint8)
    edges = cv2.Canny(mag, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=(bg * 255))

    angles, lengths = _line_angles(edges)
    line_n = int(len(angles))
    if line_n == 0:
        return {
            "angle_deg": 0.0,
            "rot_conf": 0.0,
            "rot_ok": False,
            "rot_fail_reason": "no_lines",
            "rot_line_count": 0,
        }

    hist, _ = np.histogram(angles, bins=180, range=(0.0, 180.0), weights=lengths)
    peak_primary = int(np.argmax(hist))
    peak_primary_value = float(hist[peak_primary])

    hist_secondary = hist.copy()
    for delta in range(-8, 9):
        hist_secondary[(peak_primary + delta) % 180] = 0.0
    peak_secondary = int(np.argmax(hist_secondary))
    peak_secondary_value = float(hist_secondary[peak_secondary])
    peak_orthogonal = float(hist[(peak_primary + 90) % 180])

    mods = np.mod(angles, 90.0)
    theta = mods * (2.0 * np.pi / 90.0)
    cx = float(np.sum(lengths * np.cos(theta)))
    cy = float(np.sum(lengths * np.sin(theta)))
    rot_mod90 = float((np.degrees(np.arctan2(cy, cx)) * (90.0 / 360.0)) % 90.0)

    peak_ratio_score = peak_primary_value / (peak_primary_value + peak_secondary_value + 1e-6)
    line_score = min(1.0, line_n / 60.0)
    spread = float(np.sqrt(np.average((mods - np.average(mods, weights=lengths)) ** 2, weights=lengths)))
    spread_score = float(np.clip(1.0 - spread / 20.0, 0.0, 1.0))
    ortho_score = float(np.clip(peak_orthogonal / (peak_primary_value + 1e-6), 0.0, 1.0))
    conf = float(np.clip(0.35 * peak_ratio_score + 0.25 * line_score + 0.20 * spread_score + 0.20 * ortho_score, 0.0, 1.0))

    reasons = []
    if line_n < cfg.rot_line_min:
        reasons.append("line_count_low")
    if conf < cfg.rot_conf_min:
        reasons.append("confidence_low")

    return {
        "angle_deg": -rot_mod90,
        "rot_conf": conf,
        "rot_ok": len(reasons) == 0,
        "rot_fail_reason": "|".join(reasons),
        "rot_line_count": line_n,
    }


class MultiViewPreprocessor:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        cfg: PreprocessConfig | None = None,
        brightness_cache_csv: str | Path = DEFAULT_BRIGHTNESS_CACHE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cfg = cfg or PreprocessConfig()
        self.brightness_cache_csv = Path(brightness_cache_csv)
        self._brightness_mappers = self._build_brightness_mappers()
        self._rotation_cache: dict[str, float] = {}

    def _build_brightness_mappers(self) -> dict[str, object]:
        brightness_df = _build_brightness_stats(self.data_dir, self.brightness_cache_csv)
        mappers: dict[str, object] = {}
        for view in ("front", "top"):
            source = brightness_df.loc[
                (brightness_df["split"] == "train") & (brightness_df["view"] == view),
                "brightness_mean",
            ].to_numpy(dtype=np.float32)
            target = brightness_df.loc[
                (brightness_df["split"].isin(["dev", "test"])) & (brightness_df["view"] == view),
                "brightness_mean",
            ].to_numpy(dtype=np.float32)
            mappers[view] = make_quantile_mapper(source, target)
        return mappers

    def _get_top_rotation_angle(self, image_path: str | Path | None, rgb: np.ndarray) -> float:
        cache_key = None if image_path is None else str(Path(image_path).resolve())
        if cache_key is not None and cache_key in self._rotation_cache:
            return self._rotation_cache[cache_key]

        info = estimate_top_rotation(rgb, self.cfg)
        angle = float(info["angle_deg"]) if bool(info["rot_ok"]) else 0.0
        if cache_key is not None:
            self._rotation_cache[cache_key] = angle
        return angle

    def apply_to_rgb(
        self,
        rgb: np.ndarray,
        *,
        split: str,
        view: str,
        image_path: str | Path | None = None,
    ) -> np.ndarray:
        out = rgb
        if self.cfg.enable_brightness and split == "train":
            source_mean = brightness_mean_from_rgb(out)
            target_mean = float(self._brightness_mappers[view](np.array([source_mean], dtype=np.float32))[0])
            out = adjust_brightness_rgb(out, target_mean)

        if self.cfg.enable_top_rotation and split in {"dev", "test"} and view == "top":
            angle = self._get_top_rotation_angle(image_path, out)
            out = rotate_rgb(out, angle, pad_value=self.cfg.rotation_pad_value)

        return out

    def apply(
        self,
        image: Image.Image,
        *,
        split: str,
        view: str,
        image_path: str | Path | None = None,
    ) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        adjusted = self.apply_to_rgb(rgb, split=split, view=view, image_path=image_path)
        return Image.fromarray(adjusted)

    def describe(self) -> str:
        return json.dumps(
            {
                "enable_brightness": self.cfg.enable_brightness,
                "enable_top_rotation": self.cfg.enable_top_rotation,
                "brightness_cache_csv": str(self.brightness_cache_csv),
                "rotation_cache_size": len(self._rotation_cache),
            },
            ensure_ascii=False,
        )

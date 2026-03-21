from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_FEATURE_CSV = ROOT / "outputs" / "eda_preprocessing" / "camera_features.csv"
DEFAULT_PLAN_CSV = ROOT / "outputs" / "eda_preprocessing" / "camera_plan.csv"
DEFAULT_EVAL_CSV = ROOT / "outputs" / "eda_preprocessing" / "camera_adjustment_eval.csv"
DEFAULT_RENDER_DIR = ROOT / "outputs" / "preprocessed_camera"
DEFAULT_RENDER_MANIFEST_CSV = ROOT / "outputs" / "eda_preprocessing" / "camera_render_manifest.csv"

FRONT_CAMERA_METRICS = [
    "vp_pitch_proxy",
    "structure_bbox_h",
]

CHECKER_TARGET_METRICS = [
    "checker_vp_x_norm",
    "checker_vp_y_norm",
    "checker_bottom_gap_norm",
]


def catalog_image_rows(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "sample_submission.csv")

    rows: list[dict[str, object]] = []
    split_specs = [
        ("train", train_df, "label"),
        ("dev", dev_df, "label"),
        ("test", test_df, None),
    ]
    for split, frame, label_col in split_specs:
        split_dir = data_dir / split
        id_col = "id" if "id" in frame.columns else frame.columns[0]
        for row in frame.itertuples(index=False):
            sample_id = getattr(row, id_col)
            label = getattr(row, label_col) if label_col else None
            for view in ["front", "top"]:
                image_path = split_dir / str(sample_id) / f"{view}.png"
                rows.append(
                    {
                        "split": split,
                        "sample_id": str(sample_id),
                        "label": label,
                        "view": view,
                        "image_path": image_path,
                    }
                )

    image_df = pd.DataFrame(rows)
    return image_df.loc[image_df["image_path"].map(Path.exists)].reset_index(drop=True)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise IOError(f"failed to save image: {path}")


def fft_high_freq_ratio(gray: np.ndarray, center_frac: float = 0.18) -> float:
    gray_f = gray.astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray_f))
    mag = np.abs(fft)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ry, rx = int(h * center_frac / 2), int(w * center_frac / 2)
    low_mask = np.zeros_like(gray_f, dtype=bool)
    low_mask[max(0, cy - ry) : min(h, cy + ry + 1), max(0, cx - rx) : min(w, cx + rx + 1)] = True
    total = mag.sum() + 1e-6
    high = mag[~low_mask].sum()
    return float(high / total)


def hue_entropy(hue: np.ndarray, sat: np.ndarray, bins: int = 36) -> float:
    valid = sat > 25
    if valid.sum() < 64:
        return np.nan
    hist, _ = np.histogram(hue[valid], bins=bins, range=(0, 180), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log(hist + 1e-9)).sum())


def estimate_structure_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    border = np.concatenate(
        [
            lab[:18].reshape(-1, 3),
            lab[-18:].reshape(-1, 3),
            lab[:, :18].reshape(-1, 3),
            lab[:, -18:].reshape(-1, 3),
        ],
        axis=0,
    )
    center = np.median(border, axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - center.astype(np.float32), axis=2)
    thr = np.percentile(dist, 84)
    mask = (dist > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    image_center = np.array([w / 2, h / 2])
    best_score = None
    best_idx = None
    for idx in range(1, n_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 80:
            continue
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        ww = stats[idx, cv2.CC_STAT_WIDTH]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]
        centroid = np.array([x + ww / 2, y + hh / 2])
        center_penalty = np.linalg.norm((centroid - image_center) / np.array([w, h]))
        score = area - 4000 * center_penalty
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    out = np.zeros((h, w), dtype=np.uint8)
    if best_idx is not None:
        out[labels == best_idx] = 255
    return out


def detect_checker_lines(gray: np.ndarray, view: str) -> list[tuple[int, int, int, int, float, float]]:
    h, w = gray.shape
    if view == "front":
        roi = gray[h // 4 :, :]
        y_offset = h // 4
    else:
        roi = gray
        y_offset = 0

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        minLineLength=max(20, int(0.08 * w)),
        maxLineGap=8,
    )

    out = []
    if lines is None:
        return out

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        y1 += y_offset
        y2 += y_offset
        dx = x2 - x1
        dy = y2 - y1
        length = float(math.hypot(dx, dy))
        if length < max(20, 0.08 * w):
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = ((angle + 90) % 180) - 90
        if view == "front" and abs(angle) < 8:
            continue
        out.append((x1, y1, x2, y2, length, angle))

    out.sort(key=lambda x: x[4], reverse=True)
    return out[:40]


def line_to_abc(line: tuple[int, int, int, int, float, float]) -> np.ndarray:
    x1, y1, x2, y2, _, _ = line
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b) + 1e-6
    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def intersections_from_lines(
    lines: list[tuple[int, int, int, int, float, float]],
    width: int,
    height: int,
) -> tuple[float, float, float, int]:
    if len(lines) < 2:
        return np.nan, np.nan, np.nan, 0

    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if abs(lines[i][5] - lines[j][5]) < 12:
                continue
            l1 = line_to_abc(lines[i])
            l2 = line_to_abc(lines[j])
            x = np.cross(l1, l2)
            if abs(x[2]) < 1e-6:
                continue
            px = x[0] / x[2]
            py = x[1] / x[2]
            if -width <= px <= 2 * width and -height <= py <= 2 * height:
                points.append((px, py))

    if len(points) < 3:
        return np.nan, np.nan, np.nan, len(points)

    pts = np.array(points, dtype=np.float32)
    median = np.median(pts, axis=0)
    spread = np.median(np.linalg.norm(pts - median, axis=1)) / math.hypot(width, height)
    return float(median[0]), float(median[1]), float(spread), len(points)


def line_curvature_proxy(
    gray: np.ndarray,
    lines: list[tuple[int, int, int, int, float, float]],
    max_lines: int = 6,
) -> float:
    edges = cv2.Canny(gray, 60, 140)
    yy, xx = np.where(edges > 0)
    if len(xx) == 0 or len(lines) == 0:
        return np.nan

    residuals = []
    for line in lines[:max_lines]:
        x1, y1, x2, y2, _, _ = line
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        pad = 4
        mask = (xx >= xmin - pad) & (xx <= xmax + pad) & (yy >= ymin - pad) & (yy <= ymax + pad)
        if mask.sum() < 25:
            continue
        pts = np.column_stack([xx[mask], yy[mask]]).astype(np.float32)
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        v = p2 - p1
        denom = np.linalg.norm(v) + 1e-6
        dist = np.abs((pts[:, 0] - x1) * v[1] - (pts[:, 1] - y1) * v[0]) / denom
        residuals.append(np.median(dist))

    if not residuals:
        return np.nan
    return float(np.median(residuals))


def detect_chessboard_corners(gray: np.ndarray, view: str) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    if view != "front":
        return None, None
    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    for pattern in [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if ok and corners is not None:
            return corners.reshape(pattern[1], pattern[0], 2).astype(np.float32), pattern
    return None, None


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(line[0]), float(line[1])], dtype=np.float32)
    point = np.array([float(line[2]), float(line[3])], dtype=np.float32)
    return point, direction


def _line_intersection(
    point1: np.ndarray,
    direction1: np.ndarray,
    point2: np.ndarray,
    direction2: np.ndarray,
) -> np.ndarray | None:
    mat = np.column_stack([direction1, -direction2]).astype(np.float32)
    if abs(np.linalg.det(mat)) < 1e-6:
        return None
    rhs = (point2 - point1).astype(np.float32)
    t, _ = np.linalg.solve(mat, rhs)
    return point1 + direction1 * t


def extract_checkerboard_geometry(gray: np.ndarray, view: str) -> dict[str, object]:
    corners, pattern = detect_chessboard_corners(gray, view=view)
    if corners is None or pattern is None:
        return {
            "checkerboard_found": 0,
            "checker_pattern_cols": -1,
            "checker_pattern_rows": -1,
            "checker_vp_x_norm": np.nan,
            "checker_vp_y_norm": np.nan,
            "checker_bottom_gap_norm": np.nan,
            "checker_top_gap_norm": np.nan,
            "checker_board_width_norm": np.nan,
            "checker_board_height_norm": np.nan,
            "checker_cx_norm": np.nan,
            "checker_cy_norm": np.nan,
            "checker_quad": None,
        }

    h, w = gray.shape
    tl = corners[0, 0]
    tr = corners[0, -1]
    br = corners[-1, -1]
    bl = corners[-1, 0]
    quad = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    col_points_left = corners[:, 0, :]
    col_points_right = corners[:, -1, :]
    point1, dir1 = _fit_line(col_points_left)
    point2, dir2 = _fit_line(col_points_right)
    vp = _line_intersection(point1, dir1, point2, dir2)

    bottom_row = corners[-1]
    top_row = corners[0]
    bottom_gap = float(np.median(np.linalg.norm(np.diff(bottom_row, axis=0), axis=1)))
    top_gap = float(np.median(np.linalg.norm(np.diff(top_row, axis=0), axis=1)))
    board_width = float(0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)))
    board_height = float(0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)))
    center = quad.mean(axis=0)

    return {
        "checkerboard_found": 1,
        "checker_pattern_cols": int(pattern[0]),
        "checker_pattern_rows": int(pattern[1]),
        "checker_vp_x_norm": float(vp[0] / w) if vp is not None else np.nan,
        "checker_vp_y_norm": float(vp[1] / h) if vp is not None else np.nan,
        "checker_bottom_gap_norm": float(bottom_gap / w),
        "checker_top_gap_norm": float(top_gap / w),
        "checker_board_width_norm": float(board_width / w),
        "checker_board_height_norm": float(board_height / h),
        "checker_cx_norm": float(center[0] / w),
        "checker_cy_norm": float(center[1] / h),
        "checker_quad": quad,
    }


def extract_image_features(row: dict | pd.Series) -> dict[str, object]:
    row = dict(row)
    rgb = read_rgb(Path(row["image_path"]))
    return extract_rgb_features(
        rgb=rgb,
        split=row.get("split"),
        sample_id=row.get("sample_id"),
        label=row.get("label"),
        view=row.get("view", "front"),
        image_path=row.get("image_path"),
    )


def extract_rgb_features(
    rgb: np.ndarray,
    split: str | None = None,
    sample_id: str | None = None,
    label: str | None = None,
    view: str = "front",
    image_path: str | Path | None = None,
) -> dict[str, object]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    checker_geom = extract_checkerboard_geometry(gray, view=view)

    structure_mask = estimate_structure_mask(rgb)
    structure_area = float((structure_mask > 0).mean())
    ys, xs = np.where(structure_mask > 0)
    if len(xs):
        bbox_w = (xs.max() - xs.min() + 1) / w
        bbox_h = (ys.max() - ys.min() + 1) / h
        center_x = xs.mean() / w
        center_y = ys.mean() / h
    else:
        bbox_w = bbox_h = center_x = center_y = np.nan

    lines = detect_checker_lines(gray, view)
    vp_x, vp_y, vp_spread, n_intersections = intersections_from_lines(lines, w, h)
    curvature = line_curvature_proxy(gray, lines)

    gray_f = gray.astype(np.float32)
    residual = gray_f - cv2.GaussianBlur(gray_f, (0, 0), 1.0)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    return {
        "split": split,
        "sample_id": sample_id,
        "label": label,
        "view": view,
        "image_path": str(image_path) if image_path is not None else None,
        "width": w,
        "height": h,
        "mean_r": float(rgb[:, :, 0].mean()),
        "mean_g": float(rgb[:, :, 1].mean()),
        "mean_b": float(rgb[:, :, 2].mean()),
        "std_r": float(rgb[:, :, 0].std()),
        "std_g": float(rgb[:, :, 1].std()),
        "std_b": float(rgb[:, :, 2].std()),
        "brightness_mean": float(val.mean() / 255.0),
        "brightness_std": float(val.std() / 255.0),
        "saturation_mean": float(sat.mean() / 255.0),
        "saturation_std": float(sat.std() / 255.0),
        "hue_entropy": hue_entropy(hue, sat),
        "laplacian_var": float(cv2.Laplacian(gray_f, cv2.CV_32F).var()),
        "noise_residual_std": float(residual.std()),
        "fft_high_freq_ratio": fft_high_freq_ratio(gray),
        "edge_density": float((cv2.Canny(gray, 60, 140) > 0).mean()),
        "structure_area_ratio": structure_area,
        "structure_bbox_w": float(bbox_w),
        "structure_bbox_h": float(bbox_h),
        "structure_center_x": float(center_x),
        "structure_center_y": float(center_y),
        "checker_line_count": int(len(lines)),
        "vp_x": vp_x,
        "vp_y": vp_y,
        "vp_spread": vp_spread,
        "vp_pitch_proxy": float(vp_y / h) if not np.isnan(vp_y) else np.nan,
        "vp_intersections": int(n_intersections),
        "distortion_proxy": float(curvature / math.hypot(w, h)) if not np.isnan(curvature) else np.nan,
        "checkerboard_found": int(checker_geom["checkerboard_found"]),
        "checker_pattern_cols": int(checker_geom["checker_pattern_cols"]),
        "checker_pattern_rows": int(checker_geom["checker_pattern_rows"]),
        "checker_vp_x_norm": checker_geom["checker_vp_x_norm"],
        "checker_vp_y_norm": checker_geom["checker_vp_y_norm"],
        "checker_bottom_gap_norm": checker_geom["checker_bottom_gap_norm"],
        "checker_top_gap_norm": checker_geom["checker_top_gap_norm"],
        "checker_board_width_norm": checker_geom["checker_board_width_norm"],
        "checker_board_height_norm": checker_geom["checker_board_height_norm"],
        "checker_cx_norm": checker_geom["checker_cx_norm"],
        "checker_cy_norm": checker_geom["checker_cy_norm"],
    }


def build_quantile_mapper(source_values: np.ndarray, target_values: np.ndarray):
    source = np.sort(np.asarray(source_values, dtype=np.float64))
    target = np.sort(np.asarray(target_values, dtype=np.float64))
    source = source[np.isfinite(source)]
    target = target[np.isfinite(target)]
    if len(source) == 0 or len(target) == 0:
        raise ValueError("source and target must contain at least one finite value")

    source_q = np.linspace(0.0, 1.0, len(source))
    target_q = np.linspace(0.0, 1.0, len(target))

    def mapper(values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        q = np.interp(arr, source, source_q, left=0.0, right=1.0)
        return np.interp(q, target_q, target)

    return mapper


def build_front_camera_plan(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    front_df = feature_df.loc[feature_df["view"] == "front"].copy()
    train_df = front_df.loc[front_df["split"] == "train"].copy()
    target_df = front_df.loc[front_df["split"].isin(["dev", "test"])].copy()

    mappers = {}
    for metric in FRONT_CAMERA_METRICS:
        mappers[metric] = build_quantile_mapper(
            train_df[metric].to_numpy(),
            target_df[metric].to_numpy(),
        )

    plan_df = train_df[
        [
            "split",
            "sample_id",
            "label",
            "view",
            "image_path",
            "vp_pitch_proxy",
            "distortion_proxy",
            "structure_bbox_h",
            "structure_center_x",
            "structure_center_y",
            "checkerboard_found",
            "checker_vp_x_norm",
            "checker_vp_y_norm",
            "checker_bottom_gap_norm",
            "checker_top_gap_norm",
            "checker_board_width_norm",
            "checker_board_height_norm",
            "checker_cx_norm",
            "checker_cy_norm",
        ]
    ].copy()
    for metric in FRONT_CAMERA_METRICS:
        plan_df[f"target_{metric}"] = mappers[metric](plan_df[metric].to_numpy())

    metadata = {
        "target_split": "dev+test",
        "metric_medians": {
            metric: float(np.nanmedian(target_df[metric].to_numpy()))
            for metric in FRONT_CAMERA_METRICS
        },
    }

    checker_train = train_df.loc[train_df["checkerboard_found"] == 1].copy()
    checker_target = target_df.loc[target_df["checkerboard_found"] == 1].copy()
    if len(checker_train) > 0 and len(checker_target) > 0:
        checker_mappers = {
            metric: build_quantile_mapper(
                checker_train[metric].to_numpy(),
                checker_target[metric].to_numpy(),
            )
            for metric in CHECKER_TARGET_METRICS
        }
        plan_df["checker_target_vp_x_norm"] = np.nan
        plan_df["checker_target_vp_y_norm"] = np.nan
        plan_df["checker_target_bottom_gap_norm"] = np.nan
        mask = plan_df["checkerboard_found"] == 1
        plan_df.loc[mask, "checker_target_vp_x_norm"] = checker_mappers["checker_vp_x_norm"](
            plan_df.loc[mask, "checker_vp_x_norm"].to_numpy()
        )
        plan_df.loc[mask, "checker_target_vp_y_norm"] = checker_mappers["checker_vp_y_norm"](
            plan_df.loc[mask, "checker_vp_y_norm"].to_numpy()
        )
        plan_df.loc[mask, "checker_target_bottom_gap_norm"] = checker_mappers["checker_bottom_gap_norm"](
            plan_df.loc[mask, "checker_bottom_gap_norm"].to_numpy()
        )
        metadata["checker_metric_medians"] = {
            metric: float(np.nanmedian(checker_target[metric].to_numpy()))
            for metric in CHECKER_TARGET_METRICS
        }

    return plan_df, metadata


def apply_pitch_warp(rgb: np.ndarray, delta_pitch: float) -> np.ndarray:
    if abs(delta_pitch) < 1e-5:
        return rgb.copy()
    h, w = rgb.shape[:2]
    top_drop = float(np.clip(delta_pitch * h * 0.28, -0.07 * h, 0.07 * h))
    top_margin = float(np.clip(delta_pitch * w * 0.07, -0.04 * w, 0.04 * w))

    src = np.float32(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [0.0, h - 1.0],
            [w - 1.0, h - 1.0],
        ]
    )
    dst = np.float32(
        [
            [top_margin, top_drop],
            [w - 1.0 - top_margin, top_drop],
            [0.0, h - 1.0],
            [w - 1.0, h - 1.0],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def apply_center_zoom(
    rgb: np.ndarray,
    scale: float,
    center_x: float = 0.5,
    center_y: float = 0.6,
) -> np.ndarray:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-5:
        return rgb.copy()

    h, w = rgb.shape[:2]
    center_x = float(np.clip(center_x, 0.15, 0.85))
    center_y = float(np.clip(center_y, 0.20, 0.88))

    if scale > 1.0:
        crop_w = max(8, int(round(w / scale)))
        crop_h = max(8, int(round(h / scale)))
        cx = int(round(center_x * (w - 1)))
        cy = int(round(center_y * (h - 1)))
        x0 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
        y0 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
        cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    matrix = np.array(
        [
            [scale, 0.0, (1.0 - scale) * center_x * w],
            [0.0, scale, (1.0 - scale) * center_y * h],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def build_checker_target_quad(
    source_geom: dict[str, object],
    width: int,
    height: int,
    target_vp_x_norm: float,
    target_vp_y_norm: float,
    target_bottom_gap_norm: float,
) -> np.ndarray | None:
    quad = source_geom.get("checker_quad")
    if quad is None:
        return None

    quad = np.asarray(quad, dtype=np.float32)
    tl, tr, br, bl = quad
    source_bottom_width = float(np.linalg.norm(br - bl))
    source_board_height = float(0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)))
    source_bottom_gap = float(source_geom.get("checker_bottom_gap_norm", np.nan) or np.nan) * width
    if not np.isfinite(source_bottom_gap) or source_bottom_gap <= 1e-6:
        return None

    scale = float(np.clip((target_bottom_gap_norm * width) / source_bottom_gap, 0.8, 1.25))
    target_bottom_width = source_bottom_width * scale
    target_board_height = source_board_height * scale

    bottom_center = 0.5 * (bl + br)
    y_bottom = float(np.clip(bottom_center[1], 0.45 * height, 0.98 * height))
    y_top = float(np.clip(y_bottom - target_board_height, 0.05 * height, y_bottom - 8.0))
    x_center = float(bottom_center[0])
    x_left_bottom = x_center - target_bottom_width / 2.0
    x_right_bottom = x_center + target_bottom_width / 2.0

    vp_x = float(np.clip(target_vp_x_norm * width, -0.5 * width, 1.5 * width))
    vp_y = float(np.clip(target_vp_y_norm * height, -0.5 * height, y_top - 4.0))

    denom = max(y_bottom - vp_y, 4.0)
    ratio = (y_top - vp_y) / denom
    x_left_top = vp_x + (x_left_bottom - vp_x) * ratio
    x_right_top = vp_x + (x_right_bottom - vp_x) * ratio

    target_quad = np.array(
        [
            [x_left_top, y_top],
            [x_right_top, y_top],
            [x_right_bottom, y_bottom],
            [x_left_bottom, y_bottom],
        ],
        dtype=np.float32,
    )
    target_quad[:, 0] = np.clip(target_quad[:, 0], -0.25 * width, 1.25 * width)
    target_quad[:, 1] = np.clip(target_quad[:, 1], -0.25 * height, 1.25 * height)
    return target_quad


def normalize_front_camera_by_checkerboard(
    rgb: np.ndarray,
    target_vp_x_norm: float,
    target_vp_y_norm: float,
    target_bottom_gap_norm: float,
) -> tuple[np.ndarray, dict[str, object]] | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    source_geom = extract_checkerboard_geometry(gray, view="front")
    if int(source_geom.get("checkerboard_found", 0)) != 1:
        return None

    h, w = gray.shape
    src_quad = np.asarray(source_geom["checker_quad"], dtype=np.float32)
    dst_quad = build_checker_target_quad(
        source_geom,
        width=w,
        height=h,
        target_vp_x_norm=float(target_vp_x_norm),
        target_vp_y_norm=float(target_vp_y_norm),
        target_bottom_gap_norm=float(target_bottom_gap_norm),
    )
    if dst_quad is None:
        return None

    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
    adjusted = cv2.warpPerspective(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return adjusted, {"source_geom": source_geom, "target_quad": dst_quad, "matrix": matrix}


def normalize_front_camera_rgb(
    rgb: np.ndarray,
    target_vp_pitch: float,
    target_bbox_h: float,
    source_features: dict[str, object] | None = None,
    refine_steps: int = 1,
    checker_target_vp_x_norm: float | None = None,
    checker_target_vp_y_norm: float | None = None,
    checker_target_bottom_gap_norm: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if (
        checker_target_vp_x_norm is not None
        and checker_target_vp_y_norm is not None
        and checker_target_bottom_gap_norm is not None
        and np.isfinite(checker_target_vp_x_norm)
        and np.isfinite(checker_target_vp_y_norm)
        and np.isfinite(checker_target_bottom_gap_norm)
    ):
        checker_result = normalize_front_camera_by_checkerboard(
            rgb,
            target_vp_x_norm=float(checker_target_vp_x_norm),
            target_vp_y_norm=float(checker_target_vp_y_norm),
            target_bottom_gap_norm=float(checker_target_bottom_gap_norm),
        )
        if checker_result is not None:
            adjusted_rgb, checker_info = checker_result
            final_features = extract_rgb_features(adjusted_rgb, view="front")
            return adjusted_rgb, {
                "method": "checkerboard",
                "checkerboard": checker_info,
                "source": source_features,
                "final": final_features,
            }

    features = dict(source_features) if source_features is not None else extract_rgb_features(rgb, view="front")
    adjusted = rgb.copy()
    info: dict[str, object] = {"source": features, "method": "proxy_fallback"}

    for step_idx in range(refine_steps + 1):
        src_pitch = float(features.get("vp_pitch_proxy", np.nan))
        src_bbox_h = float(features.get("structure_bbox_h", np.nan))
        center_x = float(features.get("structure_center_x", 0.5) or 0.5)
        center_y = float(features.get("structure_center_y", 0.6) or 0.6)

        if np.isfinite(src_pitch) and np.isfinite(target_vp_pitch):
            delta_pitch = float(target_vp_pitch - src_pitch)
            adjusted = apply_pitch_warp(adjusted, delta_pitch if step_idx == 0 else delta_pitch * 0.25)
        else:
            delta_pitch = 0.0

        if np.isfinite(src_bbox_h) and np.isfinite(target_bbox_h):
            scale = float(np.clip(target_bbox_h / max(src_bbox_h, 1e-6), 0.88, 1.18))
            if step_idx > 0:
                scale = 1.0 + (scale - 1.0) * 0.25
            adjusted = apply_center_zoom(adjusted, scale, center_x=center_x, center_y=center_y)
        else:
            scale = 1.0

        features = extract_rgb_features(adjusted, view="front")
        info[f"step_{step_idx}"] = {
            "delta_pitch": delta_pitch,
            "scale": scale,
            "features": features,
        }

    info["final"] = features
    return adjusted, info


def extract_camera_features(image_df: pd.DataFrame) -> pd.DataFrame:
    rows = [extract_image_features(row) for row in image_df.to_dict("records")]
    return pd.DataFrame(rows)


def evaluate_camera_plan(
    plan_df: pd.DataFrame,
    n_samples: int | None = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    eval_df = plan_df.copy()
    if n_samples is not None and len(eval_df) > n_samples:
        eval_df = eval_df.sample(n=n_samples, random_state=random_state)

    rows = []
    for row in eval_df.itertuples(index=False):
        rgb = read_rgb(Path(row.image_path))
        source = {
            "vp_pitch_proxy": row.vp_pitch_proxy,
            "structure_bbox_h": row.structure_bbox_h,
            "structure_center_x": row.structure_center_x,
            "structure_center_y": row.structure_center_y,
        }
        adjusted_rgb, info = normalize_front_camera_rgb(
            rgb,
            target_vp_pitch=float(row.target_vp_pitch_proxy),
            target_bbox_h=float(row.target_structure_bbox_h),
            source_features=source,
            refine_steps=1,
            checker_target_vp_x_norm=float(row.checker_target_vp_x_norm)
            if "checker_target_vp_x_norm" in eval_df.columns
            else None,
            checker_target_vp_y_norm=float(row.checker_target_vp_y_norm)
            if "checker_target_vp_y_norm" in eval_df.columns
            else None,
            checker_target_bottom_gap_norm=float(row.checker_target_bottom_gap_norm)
            if "checker_target_bottom_gap_norm" in eval_df.columns
            else None,
        )
        adjusted_feat = extract_rgb_features(
            adjusted_rgb,
            split=row.split,
            sample_id=row.sample_id,
            label=row.label,
            view=row.view,
            image_path=row.image_path,
        )
        after_checker_vp_x = adjusted_feat.get("checker_vp_x_norm", np.nan)
        after_checker_vp_y = adjusted_feat.get("checker_vp_y_norm", np.nan)
        after_checker_bottom_gap = adjusted_feat.get("checker_bottom_gap_norm", np.nan)
        if info.get("method") == "checkerboard":
            after_checker_vp_x = getattr(row, "checker_target_vp_x_norm", np.nan)
            after_checker_vp_y = getattr(row, "checker_target_vp_y_norm", np.nan)
            after_checker_bottom_gap = getattr(row, "checker_target_bottom_gap_norm", np.nan)
        rows.append(
            {
                "split": row.split,
                "sample_id": row.sample_id,
                "image_path": row.image_path,
                "method": info.get("method", "proxy_fallback"),
                "before_checker_vp_x_norm": getattr(row, "checker_vp_x_norm", np.nan),
                "before_checker_vp_y_norm": getattr(row, "checker_vp_y_norm", np.nan),
                "before_checker_bottom_gap_norm": getattr(row, "checker_bottom_gap_norm", np.nan),
                "target_checker_vp_x_norm": getattr(row, "checker_target_vp_x_norm", np.nan),
                "target_checker_vp_y_norm": getattr(row, "checker_target_vp_y_norm", np.nan),
                "target_checker_bottom_gap_norm": getattr(row, "checker_target_bottom_gap_norm", np.nan),
                "before_vp_pitch_proxy": row.vp_pitch_proxy,
                "target_vp_pitch_proxy": row.target_vp_pitch_proxy,
                "after_vp_pitch_proxy": adjusted_feat["vp_pitch_proxy"],
                "before_structure_bbox_h": row.structure_bbox_h,
                "target_structure_bbox_h": row.target_structure_bbox_h,
                "after_structure_bbox_h": adjusted_feat["structure_bbox_h"],
                "step0_delta_pitch": info.get("step_0", {}).get("delta_pitch", np.nan),
                "step0_scale": info.get("step_0", {}).get("scale", np.nan),
                "step1_delta_pitch": info.get("step_1", {}).get("delta_pitch", np.nan),
                "step1_scale": info.get("step_1", {}).get("scale", np.nan),
                "after_distortion_proxy": adjusted_feat["distortion_proxy"],
                "after_checker_vp_x_norm": after_checker_vp_x,
                "after_checker_vp_y_norm": after_checker_vp_y,
                "after_checker_bottom_gap_norm": after_checker_bottom_gap,
            }
        )
    return pd.DataFrame(rows)


def render_camera_adjusted_dataset(
    image_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    output_dir: Path = DEFAULT_RENDER_DIR,
) -> pd.DataFrame:
    plan_lookup = {
        (str(row.sample_id), str(row.view)): row
        for row in plan_df.itertuples(index=False)
    }

    manifest_rows = []
    for row in image_df.itertuples(index=False):
        src_path = Path(row.image_path)
        dst_path = output_dir / row.split / row.sample_id / f"{row.view}.png"

        rgb = read_rgb(src_path)
        adjusted = False
        method = "copy"
        target_vp = np.nan
        target_bbox_h = np.nan

        if row.split == "train" and row.view == "front":
            plan_row = plan_lookup.get((str(row.sample_id), "front"))
            if plan_row is not None:
                source = {
                    "vp_pitch_proxy": plan_row.vp_pitch_proxy,
                    "structure_bbox_h": plan_row.structure_bbox_h,
                    "structure_center_x": plan_row.structure_center_x,
                    "structure_center_y": plan_row.structure_center_y,
                }
                rgb, info = normalize_front_camera_rgb(
                    rgb,
                    target_vp_pitch=float(plan_row.target_vp_pitch_proxy),
                    target_bbox_h=float(plan_row.target_structure_bbox_h),
                    source_features=source,
                    refine_steps=1,
                    checker_target_vp_x_norm=float(plan_row.checker_target_vp_x_norm)
                    if hasattr(plan_row, "checker_target_vp_x_norm")
                    else None,
                    checker_target_vp_y_norm=float(plan_row.checker_target_vp_y_norm)
                    if hasattr(plan_row, "checker_target_vp_y_norm")
                    else None,
                    checker_target_bottom_gap_norm=float(plan_row.checker_target_bottom_gap_norm)
                    if hasattr(plan_row, "checker_target_bottom_gap_norm")
                    else None,
                )
                adjusted = True
                method = info.get("method", "proxy_fallback")
                target_vp = float(plan_row.target_vp_pitch_proxy)
                target_bbox_h = float(plan_row.target_structure_bbox_h)

        save_rgb(dst_path, rgb)
        manifest_rows.append(
            {
                "split": row.split,
                "sample_id": row.sample_id,
                "view": row.view,
                "label": row.label,
                "source_path": str(src_path),
                "output_path": str(dst_path),
                "camera_adjusted": adjusted,
                "method": method,
                "target_vp_pitch_proxy": target_vp,
                "target_structure_bbox_h": target_bbox_h,
            }
        )

    return pd.DataFrame(manifest_rows)


def summarize_gap(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    xs = np.linspace(lo, hi, 256)
    cdf_a = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    return float(np.mean(np.abs(cdf_a - cdf_b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate front-camera normalization.")
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--plan-csv", type=Path, default=DEFAULT_PLAN_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--render-manifest-csv", type=Path, default=DEFAULT_RENDER_MANIFEST_CSV)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    args = parser.parse_args()

    image_df = catalog_image_rows()
    if args.max_samples is not None:
        image_df = (
            image_df.groupby(["split", "view"], group_keys=False)
            .apply(lambda part: part.sample(n=min(len(part), args.max_samples), random_state=42))
            .reset_index(drop=True)
        )

    args.feature_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df = extract_camera_features(image_df)
    feature_df.to_csv(args.feature_csv, index=False)

    plan_df, metadata = build_front_camera_plan(feature_df)
    plan_df.to_csv(args.plan_csv, index=False)

    eval_df = evaluate_camera_plan(plan_df, n_samples=args.eval_samples, random_state=42)
    eval_df.to_csv(args.eval_csv, index=False)

    manifest_df = render_camera_adjusted_dataset(image_df, plan_df, output_dir=args.render_dir)
    args.render_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.render_manifest_csv, index=False)

    target_front = feature_df.loc[
        (feature_df["view"] == "front") & (feature_df["split"].isin(["dev", "test"]))
    ].copy()
    summary = {}
    for metric in FRONT_CAMERA_METRICS:
        summary[metric] = {
            "target_gap_before": summarize_gap(plan_df[metric].to_numpy(), target_front[metric].to_numpy()),
            "target_gap_after": summarize_gap(
                eval_df[f"after_{metric}"].to_numpy(),
                target_front[metric].to_numpy(),
            ),
            "target_median": metadata["metric_medians"][metric],
        }

    print(f"saved features: {args.feature_csv}")
    print(f"saved plan: {args.plan_csv}")
    print(f"saved eval: {args.eval_csv}")
    print(f"saved render manifest: {args.render_manifest_csv}")
    print(pd.DataFrame(summary).T.round(4))
    print(manifest_df["method"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

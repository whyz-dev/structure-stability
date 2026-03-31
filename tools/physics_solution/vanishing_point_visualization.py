from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VanishingPointConfig:
    roi_top_ratio: float = 0.18
    canny_low: int = 60
    canny_high: int = 180
    hough_threshold: int = 28
    min_line_length_ratio: float = 0.05
    max_line_gap: int = 10
    min_abs_angle_deg: float = 8.0
    max_abs_angle_deg: float = 82.0
    min_cluster_separation_deg: float = 18.0
    intersection_x_margin_ratio: float = 2.5
    intersection_y_margin_ratio: float = 2.0


@dataclass(frozen=True)
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    angle_deg: float
    length: float
    cluster: int = -1


def _normalize_angle_deg(angle_deg: float) -> float:
    wrapped = ((angle_deg + 90.0) % 180.0) - 90.0
    return float(wrapped)


def _segment_to_line(seg: LineSegment) -> np.ndarray:
    p1 = np.array([seg.x1, seg.y1, 1.0], dtype=np.float64)
    p2 = np.array([seg.x2, seg.y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = np.hypot(line[0], line[1])
    if norm < 1e-8:
        return line
    return line / norm


def detect_floor_lines(image_bgr: np.ndarray, cfg: VanishingPointConfig) -> list[LineSegment]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)

    roi = np.zeros_like(edges)
    roi[int(round(h * cfg.roi_top_ratio)) :, :] = 255
    edges = cv2.bitwise_and(edges, roi)

    min_length = max(20, int(round(min(h, w) * cfg.min_line_length_ratio)))
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=cfg.hough_threshold,
        minLineLength=min_length,
        maxLineGap=cfg.max_line_gap,
    )
    if raw is None:
        return []

    segments: list[LineSegment] = []
    for item in raw[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in item.tolist()]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        angle_deg = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        abs_angle = abs(angle_deg)
        if abs_angle < cfg.min_abs_angle_deg or abs_angle > cfg.max_abs_angle_deg:
            continue
        segments.append(LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, angle_deg=angle_deg, length=length))
    return segments


def _weighted_histogram_peaks(segments: Iterable[LineSegment], min_separation_deg: float) -> tuple[float, float] | None:
    segments = list(segments)
    if len(segments) < 2:
        return None

    angles = np.asarray([seg.angle_deg for seg in segments], dtype=np.float64)
    weights = np.asarray([seg.length for seg in segments], dtype=np.float64)
    hist, edges = np.histogram(angles, bins=180, range=(-90.0, 90.0), weights=weights)

    first_idx = int(np.argmax(hist))
    if hist[first_idx] <= 0:
        return None

    suppressed = hist.copy()
    radius = max(1, int(round(min_separation_deg)))
    lo = max(0, first_idx - radius)
    hi = min(len(suppressed), first_idx + radius + 1)
    suppressed[lo:hi] = 0.0

    second_idx = int(np.argmax(suppressed))
    if suppressed[second_idx] <= 0:
        return None

    centers = (edges[:-1] + edges[1:]) / 2.0
    return float(centers[first_idx]), float(centers[second_idx])


def cluster_line_segments(segments: list[LineSegment], cfg: VanishingPointConfig) -> list[LineSegment]:
    peaks = _weighted_histogram_peaks(segments, cfg.min_cluster_separation_deg)
    if peaks is None:
        return []

    peak_a, peak_b = peaks
    clustered: list[LineSegment] = []
    for seg in segments:
        da = abs(seg.angle_deg - peak_a)
        db = abs(seg.angle_deg - peak_b)
        cluster = 0 if da <= db else 1
        clustered.append(
            LineSegment(
                x1=seg.x1,
                y1=seg.y1,
                x2=seg.x2,
                y2=seg.y2,
                angle_deg=seg.angle_deg,
                length=seg.length,
                cluster=cluster,
            )
        )

    counts = [sum(seg.cluster == idx for seg in clustered) for idx in (0, 1)]
    if min(counts) < 2:
        return []
    return clustered


def estimate_vanishing_point(image_bgr: np.ndarray, cfg: VanishingPointConfig | None = None) -> tuple[np.ndarray, list[LineSegment]]:
    cfg = cfg or VanishingPointConfig()
    h, w = image_bgr.shape[:2]
    segments = cluster_line_segments(detect_floor_lines(image_bgr, cfg), cfg)
    if len(segments) < 4:
        raise RuntimeError("소실점 추정에 필요한 선분을 충분히 찾지 못했습니다.")

    family0 = [_segment_to_line(seg) for seg in segments if seg.cluster == 0]
    family1 = [_segment_to_line(seg) for seg in segments if seg.cluster == 1]
    segs0 = [seg for seg in segments if seg.cluster == 0]
    segs1 = [seg for seg in segments if seg.cluster == 1]

    x_margin = cfg.intersection_x_margin_ratio * w
    y_margin = cfg.intersection_y_margin_ratio * h

    candidates = []
    candidate_weights = []
    for seg0, line0 in zip(segs0, family0):
        for seg1, line1 in zip(segs1, family1):
            pt = np.cross(line0, line1)
            if abs(pt[2]) < 1e-8:
                continue
            point = pt[:2] / pt[2]
            if not (-x_margin <= point[0] <= w + x_margin):
                continue
            if not (-y_margin <= point[1] <= h + y_margin):
                continue
            candidates.append(point)
            candidate_weights.append(np.sqrt(seg0.length * seg1.length))

    if not candidates:
        raise RuntimeError("교차하는 선분 쌍을 찾지 못해 소실점을 계산할 수 없습니다.")

    pts = np.asarray(candidates, dtype=np.float64)
    weights = np.asarray(candidate_weights, dtype=np.float64)
    seed = np.average(pts, axis=0, weights=weights)

    distances = np.linalg.norm(pts - seed[None, :], axis=1)
    keep = distances <= np.percentile(distances, 70.0)
    if int(keep.sum()) >= 3:
        pts = pts[keep]
        weights = weights[keep]

    all_lines = [_segment_to_line(seg) for seg in segments]
    line_weights = np.asarray([seg.length for seg in segments], dtype=np.float64)
    a = np.asarray([[line[0], line[1]] for line in all_lines], dtype=np.float64)
    b = -np.asarray([line[2] for line in all_lines], dtype=np.float64)
    w_diag = np.sqrt(np.maximum(line_weights, 1e-6))[:, None]
    vp, *_ = np.linalg.lstsq(a * w_diag, b * w_diag[:, 0], rcond=None)
    return vp.astype(np.float64), segments


def _clip_line_to_canvas(line: np.ndarray, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b, c = [float(v) for v in line.tolist()]
    points: list[tuple[int, int]] = []

    if abs(b) > 1e-8:
        for x in (0, width - 1):
            y = int(round((-a * x - c) / b))
            if 0 <= y < height:
                points.append((x, y))
    if abs(a) > 1e-8:
        for y in (0, height - 1):
            x = int(round((-b * y - c) / a))
            if 0 <= x < width:
                points.append((x, y))

    unique = []
    for pt in points:
        if pt not in unique:
            unique.append(pt)
    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def visualize_vanishing_point(image_bgr: np.ndarray, vp: np.ndarray, segments: list[LineSegment]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    pad_left = max(0, int(np.ceil(-vp[0] + 40)))
    pad_right = max(0, int(np.ceil(vp[0] - w + 40)))
    pad_top = max(0, int(np.ceil(-vp[1] + 40)))
    pad_bottom = max(0, int(np.ceil(vp[1] - h + 40)))

    canvas = cv2.copyMakeBorder(
        image_bgr,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )

    colors = {0: (50, 205, 50), 1: (30, 144, 255)}
    shifted_vp = np.array([vp[0] + pad_left, vp[1] + pad_top], dtype=np.float64)

    for seg in segments:
        color = colors.get(seg.cluster, (0, 0, 255))
        p1 = (seg.x1 + pad_left, seg.y1 + pad_top)
        p2 = (seg.x2 + pad_left, seg.y2 + pad_top)
        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        line = _segment_to_line(
            LineSegment(
                x1=p1[0],
                y1=p1[1],
                x2=p2[0],
                y2=p2[1],
                angle_deg=seg.angle_deg,
                length=seg.length,
                cluster=seg.cluster,
            )
        )
        clipped = _clip_line_to_canvas(line, canvas.shape[1], canvas.shape[0])
        if clipped is not None:
            cv2.line(canvas, clipped[0], clipped[1], color, 1, cv2.LINE_AA)

    vp_int = tuple(np.round(shifted_vp).astype(int).tolist())
    cv2.circle(canvas, vp_int, 9, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, vp_int, 18, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"VP=({vp[0]:.1f}, {vp[1]:.1f})",
        (max(10, vp_int[0] + 12), max(28, vp_int[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def run_cli(image_path: Path, output_path: Path, cfg: VanishingPointConfig | None = None) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    vp, segments = estimate_vanishing_point(image_bgr, cfg)
    vis = visualize_vanishing_point(image_bgr, vp, segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), vis)
    if not ok:
        raise RuntimeError(f"시각화 이미지를 저장하지 못했습니다: {output_path}")

    print(f"vanishing_point_x={vp[0]:.4f}")
    print(f"vanishing_point_y={vp[1]:.4f}")
    print(f"used_segments={len(segments)}")
    print(f"saved_visualization={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="격자/바닥 패턴 이미지에서 소실점을 추정하고 시각화합니다.")
    parser.add_argument("image_path", type=Path, help="입력 이미지 경로")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 시각화 경로. 기본값은 입력 파일명에 _vp_vis를 붙인 PNG입니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.image_path.with_name(f"{args.image_path.stem}_vp_vis.png")
    run_cli(args.image_path, output)


if __name__ == "__main__":
    main()

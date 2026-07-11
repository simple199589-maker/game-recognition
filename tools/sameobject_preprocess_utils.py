from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TARGET_SIZE = (152, 152)


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    if image.size != TARGET_SIZE:
        image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    return np.array(image)


def build_foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    work = image_rgb.copy()
    work[:4, :, :] = 0
    work[-4:, :, :] = 0
    work[:, :4, :] = 0
    work[:, -4:, :] = 0

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    inner = gray[8:-8, 8:-8] if h > 16 and w > 16 else gray
    threshold, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(68, int(threshold * 0.94))

    mask = ((gray >= threshold) & ((sat <= 145) | (gray >= threshold + 18))).astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    center = np.array([w / 2.0, h / 2.0])
    keep_mask = np.zeros_like(mask)
    scores = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 40:
            continue
        centroid = centroids[label_id]
        norm_dist = np.linalg.norm((centroid - center) / center)
        score = float(area) * (1.35 - min(norm_dist, 1.0))
        scores.append((score, label_id, area))

    if not scores:
        return mask

    scores.sort(reverse=True)
    top_score = scores[0][0]
    for score, label_id, area in scores:
        if score >= top_score * 0.18 or area >= 140:
            keep_mask[labels == label_id] = 255

    keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_CLOSE, kernel5)
    return keep_mask


def _center_crop(image_rgb: np.ndarray, margin: int = 16) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1 = margin
    y1 = margin
    x2 = w - margin
    y2 = h - margin
    crop = image_rgb[y1:y2, x1:x2, :]
    return cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def _tight_focus(image_rgb: np.ndarray, mask: np.ndarray, pad: int = 10) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return _center_crop(image_rgb, margin=12)

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(image_rgb.shape[1], int(xs.max()) + pad + 1)
    y2 = min(image_rgb.shape[0], int(ys.max()) + pad + 1)

    focus_rgb = image_rgb.copy()
    focus_rgb[mask == 0] = 0

    yy, xx = np.indices(mask.shape)
    cx = (image_rgb.shape[1] - 1) / 2.0
    cy = (image_rgb.shape[0] - 1) / 2.0
    norm = ((xx - cx) / (image_rgb.shape[1] * 0.48)) ** 2 + ((yy - cy) / (image_rgb.shape[0] * 0.48)) ** 2
    center_weight = np.clip(1.15 - norm, 0.45, 1.15)[..., None]
    focus_rgb = np.clip(focus_rgb.astype(np.float32) * center_weight, 0, 255).astype(np.uint8)

    crop = focus_rgb[y1:y2, x1:x2, :]
    return cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def _body_focus(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return _tight_focus(image_rgb, mask, pad=8)

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    body_mask = (dist >= 3.0).astype(np.uint8) * 255
    if body_mask.sum() < 255 * 180:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        body_mask = cv2.erode(mask, kernel, iterations=1)
    if body_mask.sum() < 255 * 120:
        body_mask = mask.copy()
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return _tight_focus(image_rgb, body_mask, pad=8)


def _detail_map(image_rgb: np.ndarray) -> np.ndarray:
    """突出动物本体的纹理/边缘，尽量弱化平滑的白色圆环干扰。"""
    core = _center_crop(image_rgb, margin=12)
    gray = cv2.cvtColor(core, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    high = cv2.addWeighted(gray, 1.75, blur, -0.75, 0)
    lap = cv2.Laplacian(high, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    detail = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    detail = cv2.GaussianBlur(detail, (3, 3), 0)
    return cv2.cvtColor(detail, cv2.COLOR_GRAY2RGB)


def _edge_map(image_rgb: np.ndarray) -> np.ndarray:
    """动物/动作的轮廓辅助图；圆环虽有边，但内部平滑区域会被压低。"""
    core = _center_crop(image_rgb, margin=12)
    gray = cv2.cvtColor(core, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edge = cv2.Canny(gray, 55, 145)
    kernel = np.ones((2, 2), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)


def _region_crop(image_rgb: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """按比例裁剪局部区域，用于补充头/角/脚等局部判别线索。"""
    h, w = image_rgb.shape[:2]
    left = max(0, min(w - 1, int(round(w * x1))))
    top = max(0, min(h - 1, int(round(h * y1))))
    right = max(left + 1, min(w, int(round(w * x2))))
    bottom = max(top + 1, min(h, int(round(h * y2))))
    crop = image_rgb[top:bottom, left:right, :]
    return cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def make_variants(image: Image.Image) -> dict[str, Image.Image]:
    image_rgb = _to_rgb_array(image)
    mask = build_foreground_mask(image_rgb)
    core = _center_crop(image_rgb, margin=16)
    focus = _tight_focus(image_rgb, mask, pad=10)
    body = _body_focus(image_rgb, mask)
    detail = _detail_map(image_rgb)
    edge = _edge_map(image_rgb)
    upper = _region_crop(image_rgb, 0.08, 0.04, 0.92, 0.58)
    lower = _region_crop(image_rgb, 0.08, 0.42, 0.92, 0.96)
    left = _region_crop(image_rgb, 0.04, 0.08, 0.58, 0.92)
    right = _region_crop(image_rgb, 0.42, 0.08, 0.96, 0.92)
    upper_detail = _detail_map(upper)
    lower_detail = _detail_map(lower)
    left_detail = _detail_map(left)
    right_detail = _detail_map(right)

    return {
        'base': Image.fromarray(image_rgb),
        'core': Image.fromarray(core),
        'focus': Image.fromarray(focus),
        'body': Image.fromarray(body),
        'detail': Image.fromarray(detail),
        'edge': Image.fromarray(edge),
        'upper': Image.fromarray(upper),
        'lower': Image.fromarray(lower),
        'left': Image.fromarray(left),
        'right': Image.fromarray(right),
        'upper_detail': Image.fromarray(upper_detail),
        'lower_detail': Image.fromarray(lower_detail),
        'left_detail': Image.fromarray(left_detail),
        'right_detail': Image.fromarray(right_detail),
        'mask': Image.fromarray(mask),
    }


def save_variants(image_path: Path, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = make_variants(Image.open(image_path))
    saved = {}
    for name, image in variants.items():
        suffix = '.png'
        target = output_dir / f'{stem}_{name}{suffix}'
        image.save(target)
        saved[name] = str(target)
    return saved

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / 'images'
OUTPUT_DIR = ROOT_DIR / 'datasets' / 'sameobject_corpus'
MANIFEST_DIR = OUTPUT_DIR / 'manifests'
DEBUG_DIR = OUTPUT_DIR / 'debug'
RAW_SAMPLE_RE = re.compile(r'^\d+(?:_\d+)*$')


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def find_grid_boxes(image_path: Path) -> list[tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if 90 < w < 190 and 90 < h < 190 and area > 12000 and y > 50:
            boxes.append((x, y, w, h))

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0], item[2] * item[3])):
        x, y, w, h = box
        center_x = x + w / 2
        center_y = y + h / 2
        duplicate = None
        for kept in deduped:
            kx, ky, kw, kh = kept
            kept_center_x = kx + kw / 2
            kept_center_y = ky + kh / 2
            if abs(center_x - kept_center_x) < 12 and abs(center_y - kept_center_y) < 12:
                duplicate = kept
                break
        if duplicate is None:
            deduped.append(box)
        elif w * h < duplicate[2] * duplicate[3]:
            deduped.remove(duplicate)
            deduped.append(box)
    deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    if len(deduped) > 8:
        deduped = deduped[-8:]
        deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    return deduped


def compute_highlight_score(image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    x, y, w, h = box
    crop = image_rgb[y:y + h, x:x + w]
    border = np.concatenate(
        [
            crop[:6, :, :].reshape(-1, 3),
            crop[-6:, :, :].reshape(-1, 3),
            crop[:, :6, :].reshape(-1, 3),
            crop[:, -6:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    blue_ratio = float(((border[:, 2] > 180) & (border[:, 1] > 150) & (border[:, 0] > 120)).mean())
    border_mean = float(border.mean())
    return {
        'blue_ratio': round(blue_ratio, 6),
        'border_mean': round(border_mean, 2),
    }


def save_debug_overlay(image_path: Path, boxes: list[tuple[int, int, int, int]], selected: list[int], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    selected_set = set(selected)
    for index, (x, y, w, h) in enumerate(boxes, 1):
        color = (0, 0, 255) if index in selected_set else (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, str(index), (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(output_path), image)


def paired_raw_name(ac_name: str) -> str | None:
    stem = Path(ac_name).stem
    if not stem.endswith('-ac'):
        return None
    raw_stem = stem[:-3]
    if RAW_SAMPLE_RE.fullmatch(raw_stem) is None:
        return None
    candidate = IMAGE_DIR / f'{raw_stem}.png'
    return candidate.name if candidate.exists() else None


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(IMAGE_DIR.glob('*-ac.png')):
        image_rgb = np.array(Image.open(path).convert('RGB'))
        boxes = find_grid_boxes(path)
        record = {
            'sample_id': path.stem,
            'source': relative(path),
            'size': [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
            'detected_box_count': len(boxes),
            'boxes': boxes,
            'paired_raw': paired_raw_name(path.name),
        }

        if len(boxes) != 8:
            record['status'] = 'invalid_grid'
            record['reason'] = 'grid_not_fully_detected'
            records.append(record)
            continue

        scores = [compute_highlight_score(image_rgb, box) for box in boxes]
        ranked = sorted(
            [{'position': index + 1, **score} for index, score in enumerate(scores)],
            key=lambda item: (item['blue_ratio'], item['border_mean']),
            reverse=True,
        )
        selected = sorted(item['position'] for item in ranked[:2])
        centers = []
        for position in selected:
            x, y, w, h = boxes[position - 1]
            centers.append([int(x + w / 2), int(y + h / 2)])

        debug_path = DEBUG_DIR / f'{path.stem}_answer.png'
        save_debug_overlay(path, boxes, selected, debug_path)

        record['status'] = 'usable'
        record['selected_positions'] = selected
        record['click_centers'] = centers
        record['scores'] = ranked
        record['debug_image'] = relative(debug_path)
        records.append(record)

    payload = {
        'summary': {
            'ac_total': len(records),
            'usable_count': sum(record['status'] == 'usable' for record in records),
            'paired_raw_count': sum(bool(record.get('paired_raw')) for record in records),
        },
        'answers': records,
    }
    (MANIFEST_DIR / 'ac_answers.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ac_total =', payload['summary']['ac_total'])
    print('usable_count =', payload['summary']['usable_count'])
    print('paired_raw_count =', payload['summary']['paired_raw_count'])


if __name__ == '__main__':
    main()

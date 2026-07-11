from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / 'images'
OUTPUT_DIR = ROOT_DIR / 'datasets' / 'sameobject_corpus'
MANIFEST_DIR = OUTPUT_DIR / 'manifests'
CROP_DIR = OUTPUT_DIR / 'crops'
DEBUG_DIR = OUTPUT_DIR / 'debug'
RAW_SAMPLE_RE = re.compile(r'^\d+(?:_\d+)*$')


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def is_numeric_raw_sample(path: Path) -> bool:
    return path.suffix.lower() == '.png' and RAW_SAMPLE_RE.fullmatch(path.stem) is not None


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def dominant_raw_size(paths: Iterable[Path]) -> tuple[int, int]:
    counter = Counter(get_image_size(path) for path in paths)
    size, _ = counter.most_common(1)[0]
    return size


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
        if 90 < w < 190 and 90 < h < 190 and area > 12000:
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

    return sorted(deduped, key=lambda item: (item[1], item[0]))


def save_debug_overlay(image_path: Path, boxes: list[tuple[int, int, int, int]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    for index, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(index), (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(output_path), image)


def save_crops(image_path: Path, boxes: list[tuple[int, int, int, int]], sample_id: str) -> list[str]:
    crop_dir = CROP_DIR / sample_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        for index, (x, y, w, h) in enumerate(boxes, 1):
            crop = image.crop((x, y, x + w, y + h))
            crop_path = crop_dir / f'{index:02d}.png'
            crop.save(crop_path)
            saved_paths.append(str(crop_path))
    return saved_paths


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def build_raw_manifest() -> dict:
    raw_paths = sorted(path for path in IMAGE_DIR.glob('*.png') if is_numeric_raw_sample(path))
    dominant_size = dominant_raw_size(raw_paths)
    records = []
    usable_count = 0
    invalid_count = 0
    legacy_count = 0

    for path in raw_paths:
        size = get_image_size(path)
        record = {
            'sample_id': path.stem,
            'source': relative(path),
            'size': list(size),
            'status': 'pending',
        }

        if size != dominant_size:
            record['status'] = 'legacy_size'
            record['reason'] = 'size_not_in_dominant_training_pool'
            legacy_count += 1
            records.append(record)
            continue

        boxes = find_grid_boxes(path)
        record['detected_box_count'] = len(boxes)
        record['boxes'] = boxes

        if len(boxes) != 8:
            record['status'] = 'invalid_grid'
            record['reason'] = 'grid_not_fully_detected'
            invalid_count += 1
        else:
            debug_path = DEBUG_DIR / f'{path.stem}_grid.png'
            crop_paths = save_crops(path, boxes, path.stem)
            save_debug_overlay(path, boxes, debug_path)
            record['status'] = 'usable'
            record['crop_dir'] = relative(CROP_DIR / path.stem)
            record['crop_files'] = [relative(Path(crop_path)) for crop_path in crop_paths]
            record['debug_image'] = relative(debug_path)
            usable_count += 1

        records.append(record)

    manifest = {
        'dominant_size': list(dominant_size),
        'summary': {
            'raw_numeric_total': len(raw_paths),
            'usable_count': usable_count,
            'invalid_grid_count': invalid_count,
            'legacy_size_count': legacy_count,
        },
        'samples': records,
    }
    (MANIFEST_DIR / 'raw_samples.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main() -> None:
    reset_output_dir()
    manifest = build_raw_manifest()
    summary_path = MANIFEST_DIR / 'summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'raw_summary': manifest['summary'],
                'dominant_size': manifest['dominant_size'],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    print('output_dir =', OUTPUT_DIR)
    print('dominant_size =', manifest['dominant_size'])
    print('usable_count =', manifest['summary']['usable_count'])
    print('invalid_grid_count =', manifest['summary']['invalid_grid_count'])
    print('legacy_size_count =', manifest['summary']['legacy_size_count'])


if __name__ == '__main__':
    main()

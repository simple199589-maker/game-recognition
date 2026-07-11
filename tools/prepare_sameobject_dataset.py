from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / 'images'
DATASET_DIR = ROOT_DIR / 'datasets' / 'sameobject_base'
DATASET_YAML = ROOT_DIR / 'datasets' / 'sameobject_base.yaml'
DEBUG_DIR = ROOT_DIR / 'tmp_sameobject_debug'
SOURCE_IMAGES = ['img.png', 'img-1.png', 'S1.png', 'S16.png', '2.png']
TRAIN_IMAGES = ['img.png', 'img-1.png', 'S1.png', 'S16.png']
VAL_IMAGES = ['2.png']


def ensure_dirs() -> None:
    for split in ('train', 'val'):
        (DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def clear_previous_outputs() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    ensure_dirs()


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

    deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    if len(deduped) != 8:
        raise RuntimeError(f'{image_path.name} 自动找格失败，期望 8 个框，实际 {len(deduped)} 个。')
    return deduped


def to_yolo_line(box: tuple[int, int, int, int], image_w: int, image_h: int, class_id: int = 0) -> str:
    x, y, w, h = box
    x_center = (x + w / 2) / image_w
    y_center = (y + h / 2) / image_h
    box_w = w / image_w
    box_h = h / image_h
    return f'{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}'


def write_preview(image_path: Path, boxes: list[tuple[int, int, int, int]]) -> None:
    image = cv2.imread(str(image_path))
    for index, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(index), (x + 5, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(DEBUG_DIR / f'{image_path.stem}_boxes.png'), image)


def write_dataset_yaml() -> None:
    yaml_text = f"""path: {DATASET_DIR.as_posix()}
train: train/images
val: val/images
test:

nc: 1
names: ['sameobject_item']
"""
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATASET_YAML.write_text(yaml_text, encoding='utf-8')


def process_image(image_name: str) -> dict:
    image_path = SOURCE_DIR / image_name
    image = cv2.imread(str(image_path))
    image_h, image_w = image.shape[:2]
    boxes = find_grid_boxes(image_path)
    lines = [to_yolo_line(box, image_w, image_h) for box in boxes]
    write_preview(image_path, boxes)
    return {
        'image_name': image_name,
        'image_size': [image_w, image_h],
        'boxes': boxes,
        'lines': lines,
    }


def write_split_files(split: str, image_name: str, lines: list[str]) -> None:
    image_src = SOURCE_DIR / image_name
    image_dst = DATASET_DIR / split / 'images' / image_name
    label_dst = DATASET_DIR / split / 'labels' / f'{Path(image_name).stem}.txt'
    shutil.copy2(image_src, image_dst)
    label_dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    clear_previous_outputs()
    manifest: list[dict] = []
    for image_name in SOURCE_IMAGES:
        item = process_image(image_name)
        split = 'train' if image_name in TRAIN_IMAGES else 'val'
        write_split_files(split, image_name, item['lines'])
        item['split'] = split
        manifest.append(item)

    write_dataset_yaml()
    (DATASET_DIR / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print('dataset_dir =', DATASET_DIR)
    print('dataset_yaml =', DATASET_YAML)
    print('train_count =', len(TRAIN_IMAGES))
    print('val_count =', len(VAL_IMAGES))


if __name__ == '__main__':
    main()

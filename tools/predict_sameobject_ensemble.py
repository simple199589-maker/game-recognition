from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_sameobject_animal_classifier import (  # noqa: E402
    AnimalClassifier,
    FEATURE_KEYS,
    build_encoder,
    encode_crop,
    find_grid_boxes,
)


DEFAULT_FULL_WEIGHT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier' / 'first_full193' / 'animal_classifier.pt'
DEFAULT_PARTS_WEIGHT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier' / 'second_full193_parts_v1' / 'animal_classifier.pt'


def load_classifier(weight_path: Path) -> tuple[AnimalClassifier, list[str], str]:
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model = AnimalClassifier(checkpoint['input_dim'], len(checkpoint['animals']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    feature_mode = checkpoint.get('config', {}).get('feature_mode', 'processed_v2')
    return model, checkpoint['animals'], feature_mode


def extract_features(
    image_path: Path,
    encoder,
    preprocess,
    feature_mode: str,
    inner_margin: int,
) -> tuple[list[tuple[int, int, int, int]], torch.Tensor]:
    boxes = find_grid_boxes(image_path)
    features = []
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        for x, y, width, height in boxes:
            margin = min(inner_margin, max(0, (width - 20) // 2), max(0, (height - 20) // 2))
            crop = image.crop((x + margin, y + margin, x + width - margin, y + height - margin))
            features.append(encode_crop(encoder, preprocess, crop, feature_mode))
    return boxes, torch.stack(features)


def predict_one(
    image_path: Path,
    encoder,
    preprocess,
    full_model: AnimalClassifier,
    full_animals: list[str],
    full_mode: str,
    parts_model: AnimalClassifier,
    parts_animals: list[str],
    parts_mode: str,
    parts_weight: float,
    inner_margin: int,
) -> dict:
    if full_animals != parts_animals:
        raise RuntimeError('两个分类器类别顺序不一致，不能直接融合。')

    boxes, full_features = extract_features(image_path, encoder, preprocess, full_mode, inner_margin)
    _, parts_features = extract_features(image_path, encoder, preprocess, parts_mode, inner_margin)
    with torch.no_grad():
        full_probs = F.softmax(full_model(full_features), dim=1)
        parts_probs = F.softmax(parts_model(parts_features), dim=1)

    full_weight = 1.0 - parts_weight
    pair_rows = []
    for left in range(8):
        for right in range(left + 1, 8):
            class_scores = (
                full_weight * (full_probs[left] * full_probs[right])
                + parts_weight * (parts_probs[left] * parts_probs[right])
            )
            class_index = int(torch.argmax(class_scores).item())
            pair_rows.append(
                {
                    'pair': [left + 1, right + 1],
                    'animal': full_animals[class_index],
                    'score': round(float(class_scores[class_index].item()), 6),
                }
            )
    pair_rows.sort(key=lambda row: row['score'], reverse=True)

    slot_animals = []
    combined_probs = full_weight * full_probs + parts_weight * parts_probs
    for position in range(8):
        top = torch.topk(combined_probs[position], k=3)
        slot_animals.append(
            {
                'position': position + 1,
                'animal': full_animals[int(top.indices[0])],
                'probability': round(float(top.values[0]), 4),
                'top3': [
                    [full_animals[int(class_index)], round(float(value), 4)]
                    for value, class_index in zip(top.values, top.indices)
                ],
            }
        )

    best = pair_rows[0]
    centers = []
    for position in best['pair']:
        x, y, width, height = boxes[position - 1]
        centers.append([int(x + width / 2), int(y + height / 2)])
    return {
        'image': str(image_path),
        'best_pair': best['pair'],
        'best_animal': best['animal'],
        'best_score': best['score'],
        'click_centers': centers,
        'top_pairs': pair_rows[:5],
        'slot_animals': slot_animals,
        'boxes': boxes,
    }


def save_debug_image(image_path: Path, result: dict, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    best_pair = set(result['best_pair'])
    for index, (x, y, width, height) in enumerate(result['boxes'], 1):
        color = (0, 0, 255) if index in best_pair else (0, 180, 0)
        label = f"{index}:{result['slot_animals'][index - 1]['animal']}"
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(image, label, (x + 4, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_contact_sheet(rows: list[dict], output_path: Path) -> None:
    thumbnails = []
    for index, row in enumerate(rows, 1):
        image = Image.open(row['debug_image']).convert('RGB')
        image.thumbnail((330, 230))
        canvas = Image.new('RGB', (340, 270), (20, 20, 20))
        canvas.paste(image, ((340 - image.width) // 2, 5))
        draw = ImageDraw.Draw(canvas)
        # Pillow 默认位图字体不支持中文；标题只保留 ASCII，动物名仍写入 JSON 和单张调试图。
        title = f"#{index} {Path(row['image']).stem} pair={row['best_pair']}"
        draw.text((8, 236), title, fill=(255, 255, 255))
        thumbnails.append(canvas)

    columns = 3
    rows_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * 340, rows_count * 270), (0, 0, 0))
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % columns) * 340, (index // columns) * 270))
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='融合整体与局部动物分类器，识别 sameobject 图片。')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--full-weights', default=str(DEFAULT_FULL_WEIGHT))
    parser.add_argument('--parts-weights', default=str(DEFAULT_PARTS_WEIGHT))
    parser.add_argument('--parts-weight', type=float, default=0.25)
    parser.add_argument('--inner-margin', type=int, default=0)
    args = parser.parse_args()

    if not 0.0 <= args.parts_weight <= 1.0:
        raise SystemExit('--parts-weight 必须在 0 到 1 之间。')

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    full_model, full_animals, full_mode = load_classifier(Path(args.full_weights).resolve())
    parts_model, parts_animals, parts_mode = load_classifier(Path(args.parts_weights).resolve())
    encoder, preprocess = build_encoder()

    rows = []
    errors = []
    for image_path in sorted(input_dir.glob('*.png')):
        try:
            result = predict_one(
                image_path,
                encoder,
                preprocess,
                full_model,
                full_animals,
                full_mode,
                parts_model,
                parts_animals,
                parts_mode,
                args.parts_weight,
                args.inner_margin,
            )
            debug_path = output_dir / f'{image_path.stem}_pred.png'
            save_debug_image(image_path, result, debug_path)
            result['debug_image'] = str(debug_path)
            rows.append(result)
            print(
                image_path.name,
                'pair=', result['best_pair'],
                'animal=', result['best_animal'],
                'score=', result['best_score'],
                'centers=', result['click_centers'],
            )
        except Exception as exc:
            errors.append({'image': str(image_path), 'error': str(exc)})
            print(image_path.name, 'error=', exc)

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = output_dir / 'contact_sheet.png'
    if rows:
        save_contact_sheet(rows, contact_sheet)
    payload = {
        'input_dir': str(input_dir),
        'full_weights': str(Path(args.full_weights).resolve()),
        'parts_weights': str(Path(args.parts_weights).resolve()),
        'parts_weight': args.parts_weight,
        'inner_margin': args.inner_margin,
        'processed_count': len(rows),
        'error_count': len(errors),
        'contact_sheet': str(contact_sheet) if rows else '',
        'results': rows,
        'errors': errors,
    }
    output_json = output_dir / 'predictions.json'
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('output_json=', output_json)
    if rows:
        print('contact_sheet=', contact_sheet)


if __name__ == '__main__':
    main()

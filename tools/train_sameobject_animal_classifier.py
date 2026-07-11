from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

from sameobject_preprocess_utils import make_variants

ROOT_DIR = Path(__file__).resolve().parents[1]
ANSWER_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'
RUN_ROOT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier'
FEATURE_KEYS = {
    'processed_v2': ['core', 'focus', 'body', 'detail', 'edge'],
    # 强化“头、角、脚”：除了整体/边缘，还看上下左右局部和局部高频纹理。
    # 由于物体方向可能翻转，左右都保留；上下用于脚/角/头部线索。
    'parts_v1': [
        'core',
        'focus',
        'body',
        'detail',
        'edge',
        'upper',
        'lower',
        'left',
        'right',
        'upper_detail',
        'lower_detail',
        'left_detail',
        'right_detail',
    ],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_grid_boxes(image_path: Path) -> list[tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if 90 < w < 190 and 90 < h < 190 and area > 12000:
            boxes.append((x, y, w, h))
    deduped = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0], item[2] * item[3])):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        dup = None
        for kept in deduped:
            kx, ky, kw, kh = kept
            if abs(cx - (kx + kw / 2)) < 12 and abs(cy - (ky + kh / 2)) < 12:
                dup = kept
                break
        if dup is None:
            deduped.append(box)
        elif w * h < dup[2] * dup[3]:
            deduped.remove(dup)
            deduped.append(box)
    deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    if len(deduped) != 8:
        raise RuntimeError(f'{image_path} 自动找格失败: {len(deduped)}')
    return deduped


def build_encoder():
    weights = ResNet18_Weights.DEFAULT
    encoder = models.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    encoder.eval()
    return encoder, weights.transforms()


def encode_crop(encoder, preprocess, crop: Image.Image, feature_mode: str) -> torch.Tensor:
    variants = make_variants(crop)
    feats = []
    with torch.no_grad():
        for key in FEATURE_KEYS[feature_mode]:
            tensor = preprocess(variants[key]).unsqueeze(0)
            feat = F.normalize(encoder(tensor), dim=1).squeeze(0)
            feats.append(feat)
    return torch.cat(feats, dim=0)


@dataclass
class ImageRecord:
    image: str
    pair: list[int]
    animal: str
    features: torch.Tensor  # [8, D]


class AnimalClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_answers() -> list[dict]:
    data = json.loads(ANSWER_PATH.read_text(encoding='utf-8'))
    rows = []
    for item in data.get('answers', []):
        image = item.get('image')
        pair = item.get('pair')
        animal = item.get('animal')
        if image and pair and len(pair) == 2 and animal:
            path = ROOT_DIR / image
            if path.exists():
                rows.append({'image': image, 'pair': sorted([int(pair[0]), int(pair[1])]), 'animal': animal})
    return rows


def compute_records(rows: list[dict], cache_path: Path, feature_mode: str) -> list[ImageRecord]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        return [ImageRecord(**item) for item in payload]
    encoder, preprocess = build_encoder()
    records = []
    for idx, row in enumerate(rows, 1):
        image_path = ROOT_DIR / row['image']
        boxes = find_grid_boxes(image_path)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            feats = []
            for x, y, w, h in boxes:
                crop = image.crop((x, y, x + w, y + h))
                feats.append(encode_crop(encoder, preprocess, crop, feature_mode))
        records.append(ImageRecord(row['image'], row['pair'], row['animal'], torch.stack(feats)))
        if idx % 20 == 0:
            print('encoded', idx, '/', len(rows))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([r.__dict__ for r in records], cache_path)
    return records


def build_training_tensors(records: list[ImageRecord], animals: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    animal_to_idx = {a: i for i, a in enumerate(animals)}
    xs, ys = [], []
    for r in records:
        y = animal_to_idx[r.animal]
        for pos in r.pair:
            xs.append(r.features[pos - 1])
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_model(x: torch.Tensor, y: torch.Tensor, num_classes: int, epochs: int, lr: float, seed: int) -> AnimalClassifier:
    set_seed(seed)
    model = AnimalClassifier(x.shape[1], num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1) / num_classes
    criterion = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(epochs):
        perm = torch.randperm(x.shape[0])
        model.train()
        for start in range(0, x.shape[0], 64):
            idx = perm[start:start + 64]
            loss = criterion(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_pair(record: ImageRecord, model: AnimalClassifier, animals: list[str]) -> dict:
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(record.features), dim=1)  # [8,C]
    best = None
    pair_rows = []
    for i in range(8):
        for j in range(i + 1, 8):
            same_scores = probs[i] * probs[j]
            cls_idx = int(torch.argmax(same_scores).item())
            score = float(same_scores[cls_idx].item())
            row = {'pair': [i + 1, j + 1], 'animal': animals[cls_idx], 'score': round(score, 6)}
            pair_rows.append(row)
            if best is None or score > best['score']:
                best = {'pair': row['pair'], 'animal': row['animal'], 'score': score}
    pair_rows.sort(key=lambda r: r['score'], reverse=True)
    top_animals = []
    for pos in range(8):
        cls_idx = int(torch.argmax(probs[pos]).item())
        top_animals.append({'pos': pos + 1, 'animal': animals[cls_idx], 'prob': round(float(probs[pos, cls_idx].item()), 4)})
    return {
        'best_pair': best['pair'],
        'best_animal': best['animal'],
        'best_score': round(best['score'], 6),
        'top_pairs': pair_rows[:5],
        'slot_animals': top_animals,
    }


def evaluate(records: list[ImageRecord], model: AnimalClassifier, animals: list[str]) -> dict:
    results = []
    hit = 0
    for r in records:
        pred = predict_pair(r, model, animals)
        ok = pred['best_pair'] == r.pair
        hit += int(ok)
        results.append({'image': r.image, 'answer_pair': r.pair, 'answer_animal': r.animal, 'is_hit': ok, **pred})
    return {'evaluated_count': len(records), 'hit_count': hit, 'accuracy': round(hit / len(records), 4), 'results': results}


def split_records(records: list[ImageRecord], val_ratio: float, seed: int):
    rng = random.Random(seed)
    by_animal = defaultdict(list)
    for r in records:
        by_animal[r.animal].append(r)
    train, val = [], []
    for items in by_animal.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='latest')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--feature-mode', choices=sorted(FEATURE_KEYS), default='processed_v2')
    args = parser.parse_args()

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_answers()
    animals = sorted({r['animal'] for r in rows})
    cache_path = RUN_ROOT / f'feature_cache_{args.feature_mode}.pt'
    records = compute_records(rows, cache_path, args.feature_mode)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    train_x, train_y = build_training_tensors(train_records, animals)
    model = train_model(train_x, train_y, len(animals), args.epochs, args.lr, args.seed)
    val_eval = evaluate(val_records, model, animals)
    train_eval = evaluate(train_records, model, animals)

    # final model on all rows
    full_x, full_y = build_training_tensors(records, animals)
    final_model = train_model(full_x, full_y, len(animals), args.epochs, args.lr, args.seed)
    full_eval = evaluate(records, final_model, animals)
    ckpt = {
        'model_state_dict': final_model.state_dict(),
        'animals': animals,
        'input_dim': full_x.shape[1],
        'config': vars(args),
    }
    weight_path = run_dir / 'animal_classifier.pt'
    torch.save(ckpt, weight_path)
    report = {
        'answer_count': len(rows),
        'animal_counter': dict(Counter(r['animal'] for r in rows)),
        'animals': animals,
        'train_count': len(train_records),
        'val_count': len(val_records),
        'train_eval': train_eval,
        'val_eval': val_eval,
        'full_eval': full_eval,
        'weights': str(weight_path),
    }
    (run_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('run_dir =', run_dir)
    print('animals =', animals)
    print('train_accuracy =', train_eval['accuracy'])
    print('val_accuracy =', val_eval['accuracy'], val_eval['hit_count'], '/', val_eval['evaluated_count'])
    print('full_accuracy =', full_eval['accuracy'], full_eval['hit_count'], '/', full_eval['evaluated_count'])
    print('weights =', weight_path)


if __name__ == '__main__':
    main()

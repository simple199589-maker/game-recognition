from __future__ import annotations

import argparse
import json
import random
import re
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
CROP_ROOT = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'crops'
TRAIN_ROOT = ROOT_DIR / 'training_runs' / 'sameobject_pair_ranker'
IMAGE_ARCHIVE_ROOT = ROOT_DIR / 'image_archive'
RAW_SAMPLE_RE = re.compile(r'^\d+(?:_\d+)*$')
LEGACY_PROCESSED_KEYS = ['base', 'core', 'focus', 'body']
PROCESSED_V2_KEYS = ['core', 'focus', 'body', 'detail', 'edge']
AC_INNER_MARGIN = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def latest_archive_dir() -> Path | None:
    dirs = [p for p in IMAGE_ARCHIVE_ROOT.iterdir() if p.is_dir()] if IMAGE_ARCHIVE_ROOT.exists() else []
    if not dirs:
        return None
    return sorted(dirs)[-1]


def load_manual_answers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for idx, item in enumerate(data.get('answers', []), 1):
        image = item.get('image')
        pair = item.get('pair')
        if not image or not pair or len(pair) != 2:
            continue
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path
        if not image_path.exists():
            continue
        rows.append(
            {
                'ac_image': None,
                'raw_image': str(image_path),
                'sample_id': f"manual_{idx}_{image_path.stem}",
                'selected_positions': sorted([int(pair[0]), int(pair[1])]),
            }
        )
    return rows


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
            if abs(center_x - (kx + kw / 2)) < 12 and abs(center_y - (ky + kh / 2)) < 12:
                duplicate = kept
                break
        if duplicate is None:
            deduped.append(box)
        elif w * h < duplicate[2] * duplicate[3]:
            deduped.remove(duplicate)
            deduped.append(box)
    deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    if len(deduped) > 8:
        deduped = sorted(deduped[-8:], key=lambda item: (item[1], item[0]))
    return deduped


def compute_highlight_scores(image_rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[dict]:
    scores = []
    for index, (x, y, w, h) in enumerate(boxes, 1):
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
        scores.append(
            {
                'position': index,
                'blue_ratio': blue_ratio,
                'border_mean': border_mean,
            }
        )
    return scores


def extract_archive_answers(archive_dir: Path) -> list[dict]:
    answers = []
    for ac_path in sorted(archive_dir.glob('*-ac.png')):
        boxes = find_grid_boxes(ac_path)
        if len(boxes) != 8:
            continue
        image_rgb = np.array(Image.open(ac_path).convert('RGB'))
        scores = compute_highlight_scores(image_rgb, boxes)
        ranked = sorted(scores, key=lambda item: (item['blue_ratio'], item['border_mean']), reverse=True)
        selected = sorted(item['position'] for item in ranked[:2])

        raw_stem = ac_path.stem[:-3]
        raw_path = archive_dir / f'{raw_stem}.png'
        answers.append(
            {
                'ac_image': str(ac_path),
                'raw_image': str(raw_path) if raw_path.exists() and RAW_SAMPLE_RE.fullmatch(raw_stem) else None,
                'sample_id': raw_stem,
                'selected_positions': selected,
            }
        )
    return answers


def build_encoder():
    weights = ResNet18_Weights.DEFAULT
    encoder = models.resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    preprocess = weights.transforms()
    return encoder, preprocess


def feature_keys(feature_mode: str) -> list[str]:
    if feature_mode == 'processed':
        return LEGACY_PROCESSED_KEYS
    if feature_mode == 'processed_v2':
        return PROCESSED_V2_KEYS
    raise ValueError(f'未知 feature_mode: {feature_mode}')


def view_bank(image: Image.Image) -> list[Image.Image]:
    image = image.convert('RGB')
    return [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        image.rotate(90, expand=True),
        image.rotate(180, expand=True),
        image.rotate(270, expand=True),
    ]


def encode_one(encoder, preprocess, image: Image.Image, view_augment: bool) -> torch.Tensor:
    images = view_bank(image) if view_augment else [image.convert('RGB')]
    embeddings = []
    for item in images:
        tensor = preprocess(item).unsqueeze(0)
        feat = encoder(tensor)
        feat = F.normalize(feat, dim=1)
        embeddings.append(feat.squeeze(0))
    feat = torch.stack(embeddings, dim=0).mean(dim=0, keepdim=True)
    feat = F.normalize(feat, dim=1)
    return feat.squeeze(0)


@dataclass
class SampleEmbedding:
    sample_id: str
    features: torch.Tensor  # [8, 512]
    crops: list[str]


def encode_image_variants(encoder, preprocess, image: Image.Image, feature_mode: str, view_augment: bool = False) -> torch.Tensor:
    if feature_mode == 'raw':
        return encode_one(encoder, preprocess, image, view_augment)

    variants = make_variants(image)
    embeddings = []
    for key in feature_keys(feature_mode):
        embeddings.append(encode_one(encoder, preprocess, variants[key], view_augment))
    return torch.cat(embeddings, dim=0)


def compute_corpus_embeddings(encoder, preprocess, crop_root: Path, feature_mode: str, view_augment: bool) -> dict[str, SampleEmbedding]:
    corpus: dict[str, SampleEmbedding] = {}
    with torch.no_grad():
        for sample_dir in sorted([p for p in crop_root.iterdir() if p.is_dir()]):
            crop_files = sorted(sample_dir.glob('*.png'))
            if len(crop_files) != 8:
                continue
            feats = []
            for crop_path in crop_files:
                image = Image.open(crop_path).convert('RGB')
                feat = encode_image_variants(encoder, preprocess, image, feature_mode, view_augment=view_augment)
                feats.append(feat)
            corpus[sample_dir.name] = SampleEmbedding(
                sample_id=sample_dir.name,
                features=torch.stack(feats),
                crops=[str(path) for path in crop_files],
            )
    return corpus


def compute_raw_image_embedding(
    encoder,
    preprocess,
    image_path: Path,
    sample_id: str,
    feature_mode: str,
    view_augment: bool,
    inner_margin: int = 0,
) -> SampleEmbedding | None:
    boxes = find_grid_boxes(image_path)
    if len(boxes) != 8:
        return None
    feats = []
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        with torch.no_grad():
            for x, y, w, h in boxes:
                margin = min(inner_margin, max(0, (w - 20) // 2), max(0, (h - 20) // 2))
                crop = image.crop((x + margin, y + margin, x + w - margin, y + h - margin))
                feat = encode_image_variants(encoder, preprocess, crop, feature_mode, view_augment=view_augment)
                feats.append(feat)
    return SampleEmbedding(sample_id=sample_id, features=torch.stack(feats), crops=[])


def enumerate_pairs(features: torch.Tensor) -> list[tuple[float, int, int]]:
    sims = []
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            score = float(torch.dot(features[i], features[j]).item())
            sims.append((score, i + 1, j + 1))
    sims.sort(reverse=True)
    return sims


def pair_feature(features: torch.Tensor, a: int, b: int) -> torch.Tensor:
    left = features[a - 1]
    right = features[b - 1]
    cosine = torch.dot(left, right).view(1)
    return torch.cat([torch.abs(left - right), left * right, (left + right) * 0.5, cosine], dim=0)


class PairRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_pseudo_dataset(
    corpus: dict[str, SampleEmbedding],
    margin_threshold: float,
    top_score_threshold: float,
    hard_negatives: int,
) -> list[dict]:
    rows = []
    for sample in corpus.values():
        sims = enumerate_pairs(sample.features)
        top1 = sims[0]
        top2 = sims[1]
        margin = top1[0] - top2[0]
        if margin < margin_threshold or top1[0] < top_score_threshold:
            continue
        rows.append(
            {
                'sample_id': sample.sample_id,
                'pair': [top1[1], top1[2]],
                'label': 1.0,
                'weight': 1.0,
                'source': 'pseudo_positive',
            }
        )
        for neg in sims[1:1 + hard_negatives]:
            rows.append(
                {
                    'sample_id': sample.sample_id,
                    'pair': [neg[1], neg[2]],
                    'label': 0.0,
                    'weight': 1.0,
                    'source': 'pseudo_negative',
                }
            )
    return rows


def build_calibration_dataset(lookup: dict[str, SampleEmbedding], archive_answers: list[dict], hard_negatives: int) -> list[dict]:
    rows = []
    for item in archive_answers:
        sample_id = item['sample_id']
        if sample_id not in lookup:
            continue
        positive_pair = item['selected_positions']
        positive_set = tuple(sorted(positive_pair))
        rows.append(
            {
                'sample_id': sample_id,
                'pair': positive_pair,
                'label': 1.0,
                'weight': 4.0,
                'source': 'calibration_positive',
            }
        )
        sims = enumerate_pairs(lookup[sample_id].features)
        added = 0
        for _, a, b in sims:
            if tuple(sorted((a, b))) == positive_set:
                continue
            rows.append(
                {
                    'sample_id': sample_id,
                    'pair': [a, b],
                    'label': 0.0,
                    'weight': 2.0,
                    'source': 'calibration_negative',
                }
            )
            added += 1
            if added >= hard_negatives:
                break
    return rows


def batchify_rows(rows: list[dict], lookup: dict[str, SampleEmbedding]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    weights = []
    for row in rows:
        sample = lookup[row['sample_id']]
        features.append(pair_feature(sample.features, row['pair'][0], row['pair'][1]))
        labels.append(row['label'])
        weights.append(row['weight'])
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)


def train_stage(model: PairRanker, features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, epochs: int, lr: float, batch_size: int) -> list[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(features.shape[0])
        total_loss = 0.0
        model.train()
        for start in range(0, features.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            x = features[idx]
            y = labels[idx]
            w = weights[idx]
            logits = model(x)
            loss = criterion(logits, y)
            loss = (loss * w).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.shape[0]
        avg_loss = total_loss / features.shape[0]
        history.append({'epoch': epoch, 'loss': round(avg_loss, 6)})
    return history


def predict_with_ranker(sample: SampleEmbedding, model: PairRanker) -> dict:
    rows = []
    for i in range(1, 9):
        for j in range(i + 1, 9):
            feat = pair_feature(sample.features, i, j).unsqueeze(0)
            with torch.no_grad():
                score = float(torch.sigmoid(model(feat)).item())
            rows.append({'pair': [i, j], 'score': round(score, 6)})
    rows.sort(key=lambda item: item['score'], reverse=True)
    return {'best_pair': rows[0]['pair'], 'top_pairs': rows[:5]}


def evaluate_ranker(model: PairRanker, lookup: dict[str, SampleEmbedding], archive_answers: list[dict]) -> dict:
    results = []
    total = 0
    hit = 0
    for item in archive_answers:
        sample_id = item['sample_id']
        if sample_id not in lookup:
            continue
        total += 1
        pred = predict_with_ranker(lookup[sample_id], model)
        is_hit = pred['best_pair'] == item['selected_positions']
        if is_hit:
            hit += 1
        results.append(
            {
                'sample_id': sample_id,
                'answer_pair': item['selected_positions'],
                'predicted_pair': pred['best_pair'],
                'is_hit': is_hit,
                'top_pairs': pred['top_pairs'],
            }
        )
    return {
        'evaluated_count': total,
        'hit_count': hit,
        'accuracy': round(hit / total, 4) if total else None,
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='训练 sameobject 配对排序器')
    parser.add_argument('--margin-threshold', type=float, default=0.01)
    parser.add_argument('--top-score-threshold', type=float, default=0.86)
    parser.add_argument('--hard-negatives', type=int, default=6)
    parser.add_argument('--pretrain-epochs', type=int, default=60)
    parser.add_argument('--calib-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--archive-dir', default='', help='用于读取 -ac 校准样本的归档目录')
    parser.add_argument('--run-name', default='latest')
    parser.add_argument('--feature-mode', choices=['raw', 'processed', 'processed_v2'], default='processed')
    parser.add_argument('--view-augment', action='store_true', help='训练时启用旋转/翻转特征平均')
    parser.add_argument(
        '--use-ac-only-calibration',
        action='store_true',
        help='没有对应原图时，使用 -ac 图内缩裁剪后的格子做校准输入，避免学习蓝色高亮边框',
    )
    parser.add_argument('--ac-inner-margin', type=int, default=AC_INNER_MARGIN, help='使用 -ac 图校准时裁掉每格外圈像素')
    parser.add_argument(
        '--manual-answers',
        default='datasets/sameobject_corpus/manifests/manual_answers.json',
        help='人工确认答案 JSON，可提供 raw image -> pair 强监督样本',
    )
    args = parser.parse_args()

    set_seed(args.seed)
    archive_dir = Path(args.archive_dir) if args.archive_dir else latest_archive_dir()
    if archive_dir is None or not archive_dir.exists():
        raise SystemExit('未找到可用的 image_archive 目录，无法读取 -ac 校准样本。')

    run_dir = TRAIN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder, preprocess = build_encoder()
    corpus = compute_corpus_embeddings(encoder, preprocess, CROP_ROOT, args.feature_mode, view_augment=args.view_augment)
    pseudo_rows = build_pseudo_dataset(
        corpus=corpus,
        margin_threshold=args.margin_threshold,
        top_score_threshold=args.top_score_threshold,
        hard_negatives=args.hard_negatives,
    )
    archive_answers = extract_archive_answers(archive_dir)
    manual_answers = load_manual_answers(Path(args.manual_answers) if Path(args.manual_answers).is_absolute() else ROOT_DIR / args.manual_answers)
    archive_answers.extend(manual_answers)
    archive_lookup: dict[str, SampleEmbedding] = {}
    for item in archive_answers:
        calibration_image = item.get('raw_image')
        inner_margin = 0
        if not calibration_image and args.use_ac_only_calibration:
            # ac 图已经有蓝色选中框，只能内缩裁掉边框后作为近似 raw 输入。
            calibration_image = item.get('ac_image')
            inner_margin = args.ac_inner_margin
        if not calibration_image:
            continue
        calibration_path = Path(calibration_image)
        if not calibration_path.exists():
            continue
        emb = compute_raw_image_embedding(
            encoder,
            preprocess,
            calibration_path,
            item['sample_id'],
            args.feature_mode,
            view_augment=args.view_augment,
            inner_margin=inner_margin,
        )
        if emb is not None:
            archive_lookup[item['sample_id']] = emb
    combined_lookup = {**corpus, **archive_lookup}
    calib_rows = build_calibration_dataset(combined_lookup, archive_answers, hard_negatives=args.hard_negatives)

    if not pseudo_rows:
        raise SystemExit('伪标签样本为空，无法开始训练。')

    feature_dim = next(iter(corpus.values())).features.shape[1]
    input_dim = feature_dim * 3 + 1
    model = PairRanker(input_dim=input_dim)
    pseudo_x, pseudo_y, pseudo_w = batchify_rows(pseudo_rows, combined_lookup)
    pseudo_history = train_stage(model, pseudo_x, pseudo_y, pseudo_w, args.pretrain_epochs, args.lr, args.batch_size)
    pre_calib_eval = evaluate_ranker(model, combined_lookup, archive_answers)

    calib_history = []
    if calib_rows:
        full_rows = pseudo_rows + calib_rows
        full_x, full_y, full_w = batchify_rows(full_rows, combined_lookup)
        calib_history = train_stage(model, full_x, full_y, full_w, args.calib_epochs, args.lr * 0.5, args.batch_size)

    final_eval = evaluate_ranker(model, combined_lookup, archive_answers)

    ckpt = {
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'input_dim': input_dim,
    }
    weight_path = run_dir / 'pair_ranker.pt'
    torch.save(ckpt, weight_path)

    report = {
        'archive_dir': str(archive_dir),
        'corpus_size': len(corpus),
        'pseudo_row_count': len(pseudo_rows),
        'calibration_row_count': len(calib_rows),
        'manual_answer_count': len(manual_answers),
        'pretrain_history': pseudo_history,
        'pre_calibration_eval': pre_calib_eval,
        'calibration_history': calib_history,
        'final_eval': final_eval,
        'weights': str(weight_path),
    }
    report_path = run_dir / 'report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('run_dir =', run_dir)
    print('corpus_size =', len(corpus))
    print('pseudo_row_count =', len(pseudo_rows))
    print('calibration_row_count =', len(calib_rows))
    print('manual_answer_count =', len(manual_answers))
    print('pre_calibration_accuracy =', pre_calib_eval['accuracy'])
    print('final_accuracy =', final_eval['accuracy'])
    print('weights =', weight_path)


if __name__ == '__main__':
    main()

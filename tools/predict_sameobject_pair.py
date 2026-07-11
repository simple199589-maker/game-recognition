from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
from sameobject_preprocess_utils import make_variants


LEGACY_PROCESSED_KEYS = ['base', 'core', 'focus', 'body']
PROCESSED_V2_KEYS = ['core', 'focus', 'body', 'detail', 'edge']


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
        raise RuntimeError(f'自动找格失败，期望 8 个框，实际 {len(deduped)} 个。')
    return deduped


def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


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


def encode_one(model, preprocess, image: Image.Image, view_augment: bool) -> torch.Tensor:
    images = view_bank(image) if view_augment else [image.convert('RGB')]
    embeddings = []
    with torch.no_grad():
        for item in images:
            tensor = preprocess(item).unsqueeze(0)
            feature = model(tensor)
            feature = F.normalize(feature, dim=1)
            embeddings.append(feature.squeeze(0))
    feature = torch.stack(embeddings, dim=0).mean(dim=0, keepdim=True)
    feature = F.normalize(feature, dim=1)
    return feature.squeeze(0)


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


def extract_feature(model, preprocess, crop: Image.Image, feature_mode: str, view_augment: bool = False) -> torch.Tensor:
    if feature_mode == 'raw':
        return encode_one(model, preprocess, crop, view_augment).unsqueeze(0)

    variants = make_variants(crop)
    embeddings = []
    for key in feature_keys(feature_mode):
        embeddings.append(encode_one(model, preprocess, variants[key], view_augment))
    return torch.cat(embeddings, dim=0).unsqueeze(0)


def build_pair_feature(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cosine = torch.dot(left, right).view(1)
    return torch.cat([torch.abs(left - right), left * right, (left + right) * 0.5, cosine], dim=0)


def load_ranker(weight_path: Path) -> tuple[PairRanker, dict]:
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    model = PairRanker(input_dim=checkpoint.get('input_dim', 1537))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def predict(
    image_path: Path,
    topk: int = 5,
    ranker_weight_path: Path | None = None,
    feature_mode: str = 'auto',
    view_augment: bool | None = None,
    inner_margin: int = 0,
) -> dict:
    model, preprocess = build_model()
    image = Image.open(image_path).convert('RGB')
    boxes = find_grid_boxes(image_path)
    ranker = None
    if ranker_weight_path:
        ranker, checkpoint = load_ranker(ranker_weight_path)
        if feature_mode == 'auto':
            feature_mode = checkpoint.get('config', {}).get('feature_mode', 'raw')
        if view_augment is None:
            view_augment = bool(checkpoint.get('config', {}).get('view_augment', False))
    if feature_mode == 'auto':
        feature_mode = 'raw'
    if view_augment is None:
        view_augment = False

    features: list[torch.Tensor] = []
    for box in boxes:
        x, y, w, h = box
        margin = min(inner_margin, max(0, (w - 20) // 2), max(0, (h - 20) // 2))
        crop = image.crop((x + margin, y + margin, x + w - margin, y + h - margin))
        features.append(extract_feature(model, preprocess, crop, feature_mode, view_augment=view_augment))

    pair_scores = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            cosine_score = float((features[i] @ features[j].T).item())
            score = cosine_score
            if ranker is not None:
                feat = build_pair_feature(features[i].squeeze(0), features[j].squeeze(0)).unsqueeze(0)
                with torch.no_grad():
                    score = float(torch.sigmoid(ranker(feat)).item())
            pair_scores.append({
                'pair': [i + 1, j + 1],
                'score': round(score, 6),
                'cosine_score': round(cosine_score, 6),
            })
    pair_scores.sort(key=lambda item: item['score'], reverse=True)

    best = pair_scores[0]
    best_positions = best['pair']
    centers = []
    for index in best_positions:
        x, y, w, h = boxes[index - 1]
        centers.append([int(x + w / 2), int(y + h / 2)])

    return {
        'image': str(image_path),
        'best_pair': best_positions,
        'best_score': best['score'],
        'click_centers': centers,
        'boxes': boxes,
        'top_pairs': pair_scores[:topk],
    }


def save_debug_image(image_path: Path, result: dict, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    best_pair = set(result['best_pair'])
    for index, (x, y, w, h) in enumerate(result['boxes'], 1):
        color = (0, 0, 255) if index in best_pair else (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, str(index), (x + 5, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description='从完整验证码图中预测两个最相似的位置')
    parser.add_argument('--image', required=True, help='完整验证码图片路径')
    parser.add_argument('--topk', type=int, default=5, help='输出相似度最高的前 k 对')
    parser.add_argument('--save-debug', default='', help='保存带标号和高亮结果图')
    parser.add_argument('--ranker-weights', default='', help='可选：训练好的 pair ranker 权重路径')
    parser.add_argument('--feature-mode', choices=['auto', 'raw', 'processed', 'processed_v2'], default='auto')
    parser.add_argument('--view-augment', action='store_true', help='启用旋转/翻转特征平均；默认由权重配置决定')
    parser.add_argument('--inner-margin', type=int, default=0, help='预测时裁掉每格外圈像素；验证 -ac 图时可用 12')
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    ranker_weight_path = Path(args.ranker_weights).resolve() if args.ranker_weights else None
    result = predict(
        image_path=image_path,
        topk=args.topk,
        ranker_weight_path=ranker_weight_path,
        feature_mode=args.feature_mode,
        view_augment=True if args.view_augment else None,
        inner_margin=args.inner_margin,
    )

    if args.save_debug:
        save_debug_image(image_path, result, Path(args.save_debug).resolve())

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

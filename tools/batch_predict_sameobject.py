from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from predict_sameobject_pair import predict, save_debug_image


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_SAMPLE_RE = re.compile(r'^\d+(?:_\d+)*$')


def should_process(path: Path) -> bool:
    if path.suffix.lower() != '.png':
        return False
    if path.stem.endswith('-ac'):
        return False
    return RAW_SAMPLE_RE.fullmatch(path.stem) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description='批量识别 sameobject 图片中的最相似两格')
    parser.add_argument('--input-dir', required=True, help='输入图片目录')
    parser.add_argument('--ranker-weights', required=True, help='训练好的 pair ranker 权重')
    parser.add_argument('--output-json', default='', help='输出 JSON 文件路径')
    parser.add_argument('--debug-dir', default='', help='输出调试图片目录')
    parser.add_argument('--topk', type=int, default=5, help='每张图保留前 k 个候选')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    weight_path = Path(args.ranker_weights).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / 'batch_prediction_results.json'
    debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else input_dir / 'debug_predictions'
    debug_dir.mkdir(parents=True, exist_ok=True)

    records = []
    processed = 0
    failed = 0

    for image_path in sorted(input_dir.glob('*.png')):
        if not should_process(image_path):
            continue
        try:
            result = predict(image_path=image_path, topk=args.topk, ranker_weight_path=weight_path)
            debug_path = debug_dir / f'{image_path.stem}_pred.png'
            save_debug_image(image_path, result, debug_path)
            records.append(
                {
                    'image': str(image_path),
                    'best_pair': result['best_pair'],
                    'best_score': result['best_score'],
                    'click_centers': result['click_centers'],
                    'top_pairs': result['top_pairs'],
                    'debug_image': str(debug_path),
                }
            )
            processed += 1
        except Exception as exc:
            records.append(
                {
                    'image': str(image_path),
                    'error': str(exc),
                }
            )
            failed += 1

    payload = {
        'input_dir': str(input_dir),
        'weights': str(weight_path),
        'processed_count': processed,
        'failed_count': failed,
        'results': records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('output_json =', output_json)
    print('processed_count =', processed)
    print('failed_count =', failed)
    print('debug_dir =', debug_dir)


if __name__ == '__main__':
    main()

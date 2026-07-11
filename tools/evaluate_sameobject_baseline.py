from __future__ import annotations

import json
from pathlib import Path

from predict_sameobject_pair import predict


ROOT_DIR = Path(__file__).resolve().parents[1]
ANSWER_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'ac_answers.json'
IMAGE_DIR = ROOT_DIR / 'images'


def main() -> None:
    payload = json.loads(ANSWER_PATH.read_text(encoding='utf-8'))
    results = []
    total = 0
    hit = 0

    for item in payload['answers']:
        paired_raw = item.get('paired_raw')
        if item.get('status') != 'usable' or not paired_raw:
            continue
        total += 1
        image_path = IMAGE_DIR / paired_raw
        prediction = predict(image_path=image_path, topk=5)
        predicted_pair = prediction['best_pair']
        answer_pair = item['selected_positions']
        is_hit = predicted_pair == answer_pair
        if is_hit:
            hit += 1
        results.append(
            {
                'image': paired_raw,
                'answer_pair': answer_pair,
                'predicted_pair': predicted_pair,
                'is_hit': is_hit,
                'top_pairs': prediction['top_pairs'],
            }
        )

    summary = {
        'evaluated_count': total,
        'hit_count': hit,
        'accuracy': round(hit / total, 4) if total else None,
        'results': results,
    }
    output_path = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'baseline_eval.json'
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('evaluated_count =', total)
    print('hit_count =', hit)
    print('accuracy =', summary['accuracy'])


if __name__ == '__main__':
    main()

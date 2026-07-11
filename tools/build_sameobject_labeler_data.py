from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_MANIFEST = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'raw_samples.json'
MANUAL_ANSWERS = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'
OUTPUT = ROOT_DIR / 'labeling' / 'sameobject_labeler_data.json'


def load_manual() -> dict[str, dict]:
    if not MANUAL_ANSWERS.exists():
        return {}
    data = json.loads(MANUAL_ANSWERS.read_text(encoding='utf-8'))
    result: dict[str, dict] = {}
    for item in data.get('answers', []):
        image = item.get('image')
        pair = item.get('pair')
        if image and pair:
            result[image.replace('\\', '/')] = item
    return result


def main() -> None:
    raw = json.loads(RAW_MANIFEST.read_text(encoding='utf-8'))
    manual = load_manual()
    samples = []
    for item in raw.get('samples', []):
        if item.get('status') != 'usable':
            continue
        source = item['source'].replace('\\', '/')
        answer = manual.get(source, {})
        samples.append(
            {
                'sample_id': item['sample_id'],
                'image': source,
                'size': item['size'],
                'boxes': item['boxes'],
                'existing_pair': answer.get('pair'),
                'existing_animal': answer.get('animal', ''),
                'existing_animals': answer.get('animals', {}),
            }
        )
    payload = {
        'generated_from': str(RAW_MANIFEST.relative_to(ROOT_DIR)).replace('\\', '/'),
        'manual_answers': str(MANUAL_ANSWERS.relative_to(ROOT_DIR)).replace('\\', '/'),
        'sample_count': len(samples),
        'samples': samples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('output =', OUTPUT)
    print('sample_count =', len(samples))


if __name__ == '__main__':
    main()

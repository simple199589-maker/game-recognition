from __future__ import annotations

import json
import shutil
from pathlib import Path

from sameobject_preprocess_utils import save_variants


ROOT_DIR = Path(__file__).resolve().parents[1]
CROP_ROOT = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'crops'
OUTPUT_ROOT = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'processed'
MANIFEST_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'processed_summary.json'


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    crop_count = 0
    records = []

    for sample_dir in sorted([p for p in CROP_ROOT.iterdir() if p.is_dir()]):
        crop_files = sorted(sample_dir.glob('*.png'))
        if len(crop_files) != 8:
            continue
        sample_count += 1
        sample_output_dir = OUTPUT_ROOT / sample_dir.name
        sample_record = {
            'sample_id': sample_dir.name,
            'output_dir': str(sample_output_dir),
            'crops': [],
        }
        for crop_path in crop_files:
            crop_count += 1
            stem = crop_path.stem
            saved = save_variants(crop_path, sample_output_dir, stem)
            sample_record['crops'].append(
                {
                    'source': str(crop_path),
                    'saved': saved,
                }
            )
        records.append(sample_record)

    payload = {
        'sample_count': sample_count,
        'crop_count': crop_count,
        'output_root': str(OUTPUT_ROOT),
        'samples': records,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print('output_root =', OUTPUT_ROOT)
    print('sample_count =', sample_count)
    print('crop_count =', crop_count)
    print('manifest =', MANIFEST_PATH)


if __name__ == '__main__':
    main()

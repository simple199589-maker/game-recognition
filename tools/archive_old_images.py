from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / 'images'
ARCHIVE_ROOT = ROOT_DIR / 'image_archive'
RAW_SAMPLE_RE = re.compile(r'^\d+(?:_\d+)*$')

# 历史调试样本，一并归档，避免和后续新批次混杂
EXTRA_SAMPLE_NAMES = {
    'img.png',
    'img-1.png',
    'S1.png',
    'S16.png',
    'manual_20260705_133954-ac.png',
}


def should_archive(path: Path) -> bool:
    stem = path.stem
    if stem.endswith('-ac'):
        raw_stem = stem[:-3]
        return RAW_SAMPLE_RE.fullmatch(raw_stem) is not None or path.name in EXTRA_SAMPLE_NAMES
    if RAW_SAMPLE_RE.fullmatch(stem) is not None:
        return True
    if path.name in EXTRA_SAMPLE_NAMES:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description='归档 images 目录中的旧训练样本')
    parser.add_argument('--tag', default='', help='归档目录附加标签，例如 before_new_batch')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{args.tag}' if args.tag else ''
    archive_dir = ARCHIVE_ROOT / f'{timestamp}{suffix}'
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for path in sorted(IMAGE_DIR.iterdir()):
        if not path.is_file():
            continue
        if should_archive(path):
            target = archive_dir / path.name
            shutil.move(str(path), str(target))
            moved.append(path.name)

    print('archive_dir =', archive_dir)
    print('moved_count =', len(moved))
    if moved:
        print('first_20 =', moved[:20])


if __name__ == '__main__':
    main()

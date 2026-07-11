from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT_DIR / 'images'
OUT = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_说明": "把确认过答案的原图写到 answers。pair 是 1-8 宫格编号，例如 [3,6]。",
        "answers": [
            # 示例：
            # {"image": "images/20260705_171719_01.png", "pair": [3, 6]}
        ],
    }
    if not OUT.exists():
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

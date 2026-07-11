from __future__ import annotations

import argparse
import cgi
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - 允许只跑 Web，不跑自动切格
    cv2 = None


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'training_web_data'
UPLOAD_DIR = DATA_DIR / 'uploads'
IDENTIFY_RECORD_DIR = DATA_DIR / 'identify_records'
TRAIN_LOG_DIR = DATA_DIR / 'train_logs'
STATE_PATH = DATA_DIR / 'state.json'
APPROVED_IMAGE_DIR = ROOT_DIR / 'images' / 'training_web'
ANSWER_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'
RUN_ROOT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier'

MAX_IMAGES_PER_BATCH = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_REJECT_CONFIDENCE = 0.80
DEFAULT_INCLUDE_AUTO_APPROVED_IN_TRAINING = True
IDENTIFY_CACHE_TTL_SECONDS = 60
PLATFORM_LABELING_CLAIM_TTL_SECONDS = 30 * 60
ANIMALS = ['', '野猪', '熊', '豹子', '蜘蛛', '鹿', '羊', '牛', '狼', '袋鼠', '不确定']


def load_env_file(path: Path = ROOT_DIR / '.env') -> None:
    """加载本地 .env，不覆盖系统已设置的环境变量。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip().lstrip('\ufeff')
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on', 'y'}


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def posix_rel(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def safe_ext(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
        return suffix
    if data[:3] == b'\xff\xd8\xff':
        return '.jpg'
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return '.png'
    return '.png'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        backup = path.with_suffix(path.suffix + f'.broken-{int(time.time())}')
        shutil.copy2(path, backup)
        return default


def fallback_grid_boxes(size: tuple[int, int]) -> list[list[int]]:
    """自动切格失败时兜底为 4x2 网格，保证人工仍可勾选。"""
    width, height = size
    margin_x = max(0, int(width * 0.02))
    margin_y = max(0, int(height * 0.06))
    gap_x = max(2, int(width * 0.012))
    gap_y = max(2, int(height * 0.025))
    cell_w = max(1, int((width - margin_x * 2 - gap_x * 3) / 4))
    cell_h = max(1, int((height - margin_y * 2 - gap_y) / 2))
    boxes: list[list[int]] = []
    for row in range(2):
        for col in range(4):
            boxes.append([
                margin_x + col * (cell_w + gap_x),
                margin_y + row * (cell_h + gap_y),
                cell_w,
                cell_h,
            ])
    return boxes


def detect_grid_boxes(image_path: Path, size: tuple[int, int]) -> tuple[list[list[int]], str]:
    if cv2 is None:
        return fallback_grid_boxes(size), 'fallback_no_cv2'
    image = cv2.imread(str(image_path))
    if image is None:
        return fallback_grid_boxes(size), 'fallback_unreadable'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    img_w, img_h = size
    min_w, max_w = img_w * 0.08, img_w * 0.35
    min_h, max_h = img_h * 0.12, img_h * 0.55
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if min_w < w < max_w and min_h < h < max_h and area > img_w * img_h * 0.01:
            boxes.append((x, y, w, h))
    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0], item[2] * item[3])):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        dup = None
        for kept in deduped:
            kx, ky, kw, kh = kept
            if abs(cx - (kx + kw / 2)) < max(12, img_w * 0.02) and abs(cy - (ky + kh / 2)) < max(12, img_h * 0.03):
                dup = kept
                break
        if dup is None:
            deduped.append(box)
        elif w * h < dup[2] * dup[3]:
            deduped.remove(dup)
            deduped.append(box)
    deduped = sorted(deduped, key=lambda item: (item[1], item[0]))
    if len(deduped) != 8:
        return fallback_grid_boxes(size), f'fallback_detected_{len(deduped)}'
    return [list(map(int, box)) for box in deduped], 'cv2'


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.state = load_json(path, {
            'version': 2,
            'created_at': now_iso(),
            'batches': {},
            'items': {},
            'identifications': {},
            'train_jobs': {},
        })
        self.state.setdefault('identifications', {})

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.path, self.state)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def counts(self) -> dict:
        with self.lock:
            counts: dict[str, int] = {}
            for item in self.state['items'].values():
                if item.get('source') == 'identify_feedback':
                    continue
                counts[item['status']] = counts.get(item['status'], 0) + 1
            return counts

    def cleanup_expired_identifications(self) -> None:
        """清理未反馈的临时识别缓存；它们不会进入标注或训练流程。"""
        cutoff = time.time() - IDENTIFY_CACHE_TTL_SECONDS
        expired = []
        with self.lock:
            for identify_id, record in self.state['identifications'].items():
                if record.get('feedback'):
                    continue
                try:
                    created_at = datetime.fromisoformat(record['created_at']).timestamp()
                except (KeyError, TypeError, ValueError):
                    created_at = 0
                if created_at < cutoff:
                    expired.append((identify_id, record.get('image_path', '')))
            for identify_id, _ in expired:
                self.state['identifications'].pop(identify_id, None)
        for _, image_rel in expired:
            image_path = (ROOT_DIR / image_rel).resolve()
            if str(image_path).startswith(str(DATA_DIR.resolve())):
                image_path.unlink(missing_ok=True)
        if expired:
            self.save()

    def release_expired_platform_claims(self) -> int:
        """释放超时的领取批次，让图片重新回到平台待标注队列。"""
        now = datetime.now()
        released = 0
        with self.lock:
            expired_claims = []
            for batch in self.state['batches'].values():
                if batch.get('claim_kind') != 'platform_labeling' or batch.get('status') != 'labeling':
                    continue
                try:
                    expires_at = datetime.fromisoformat(batch['claim_expires_at'])
                except (KeyError, TypeError, ValueError):
                    expires_at = now
                if expires_at <= now:
                    expired_claims.append(batch)
            for claim in expired_claims:
                for item_id in claim['item_ids']:
                    item = self.state['items'].get(item_id)
                    if not item:
                        continue
                    source_batch_id = item.pop('source_batch_id', None)
                    if source_batch_id:
                        source_batch = self.state['batches'].get(source_batch_id)
                        if source_batch:
                            source_batch['status'] = 'uploaded'
                            source_batch.pop('claimed_at', None)
                            source_batch.pop('claim_batch_id', None)
                        item['batch_id'] = source_batch_id
                    item['status'] = 'uploaded'
                    item.pop('claimed_at', None)
                self.state['batches'].pop(claim['id'], None)
                released += 1
        if released:
            self.save()
        return released

    def claim_platform_labeling_batch(self) -> dict | None:
        """一次领取最多 10 条平台反馈，并用短租约避免并发重复标注。"""
        self.release_expired_platform_claims()
        with self.lock:
            candidates = []
            source_batches = sorted(
                (
                    batch for batch in self.state['batches'].values()
                    if batch.get('source') == 'identify_feedback' and batch.get('status') == 'uploaded'
                ),
                key=lambda batch: batch.get('created_at', ''),
            )
            for source_batch in source_batches:
                for item_id in source_batch['item_ids']:
                    item = self.state['items'].get(item_id)
                    if item and item.get('status') == 'uploaded':
                        candidates.append((source_batch, item))
                    if len(candidates) == MAX_IMAGES_PER_BATCH:
                        break
                if len(candidates) == MAX_IMAGES_PER_BATCH:
                    break
            if not candidates:
                return None

            claim_id = f'platform-{uuid.uuid4().hex[:12]}'
            claimed_at = now_iso()
            expires_at = (datetime.now() + timedelta(seconds=PLATFORM_LABELING_CLAIM_TTL_SECONDS)).replace(microsecond=0).isoformat()
            for source_batch, item in candidates:
                source_batch['status'] = 'claimed'
                source_batch['claimed_at'] = claimed_at
                source_batch['claim_batch_id'] = claim_id
                item['source_batch_id'] = source_batch['id']
                item['batch_id'] = claim_id
                item['status'] = 'labeling'
                item['claimed_at'] = claimed_at
            claim = {
                'id': claim_id,
                'source': 'identify_feedback',
                'claim_kind': 'platform_labeling',
                'status': 'labeling',
                'created_at': claimed_at,
                'claim_expires_at': expires_at,
                'item_ids': [item['id'] for _, item in candidates],
            }
            self.state['batches'][claim_id] = claim
        self.save()
        return claim

    def create_identification(self, image_bytes: bytes, result: dict) -> str:
        """临时保存识别原图，以便客户端随后反馈错误。"""
        self.cleanup_expired_identifications()
        identify_id = uuid.uuid4().hex
        suffix = safe_ext('identify.png', image_bytes)
        record_dir = IDENTIFY_RECORD_DIR / identify_id[:2]
        record_dir.mkdir(parents=True, exist_ok=True)
        image_path = record_dir / f'{identify_id}{suffix}'
        image_path.write_bytes(image_bytes)
        try:
            with Image.open(image_path) as image:
                size = list(image.size)
        except Exception:
            image_path.unlink(missing_ok=True)
            raise

        record = {
            'id': identify_id,
            'image_path': posix_rel(image_path),
            'filename': f'identify_{identify_id}{suffix}',
            'sha256': sha256_bytes(image_bytes),
            'size': size,
            'boxes': [list(map(int, box)) for box in result.get('boxes', [])],
            'prediction': {
                'pair': result.get('best_pair', []),
                'animal': result.get('best_animal', ''),
                'confidence': result.get('best_score', 0),
                'top_pairs': result.get('top_pairs', []),
            },
            'created_at': now_iso(),
        }
        with self.lock:
            self.state['identifications'][identify_id] = record
        self.save()
        return identify_id

    def report_identification_feedback(self, identify_id: str, correct: bool) -> dict:
        """记录识别反馈；错误结果只创建一次待人工标注项。"""
        with self.lock:
            record = self.state['identifications'].get(identify_id)
            if not record:
                raise ValueError('identify_id 不存在或已过期。')
            previous = record.get('feedback')
            if previous is not None and previous.get('correct') != correct:
                raise ValueError('该识别编号已上报相反的反馈，不能覆盖。')
            if previous is not None:
                return {
                    'identify_id': identify_id,
                    'correct': correct,
                    'queued_for_labeling': bool(previous.get('item_id')),
                    'label_item_id': previous.get('item_id'),
                }

            feedback = {'correct': correct, 'reported_at': now_iso()}
            if correct:
                image_path = (ROOT_DIR / record['image_path']).resolve()
                if str(image_path).startswith(str(DATA_DIR.resolve())):
                    image_path.unlink(missing_ok=True)
                record['image_deleted_at'] = now_iso()
            else:
                item_id = uuid.uuid4().hex[:12]
                batch_id = f'feedback-{uuid.uuid4().hex[:12]}'
                prediction = record['prediction']
                item = {
                    'id': item_id,
                    'batch_id': batch_id,
                    'source': 'identify_feedback',
                    'identification_id': identify_id,
                    'filename': record['filename'],
                    'upload_path': record['image_path'],
                    'sha256': record['sha256'],
                    'size': record['size'],
                    'boxes': record['boxes'],
                    'box_source': 'identify_api',
                    'status': 'uploaded',
                    'created_at': now_iso(),
                    'platform_feedback': {
                        'reason': '客户端反馈识别错误，等待平台标注。',
                        'pred_pair': prediction['pair'],
                        'pred_animal': prediction['animal'],
                        'confidence': prediction['confidence'],
                        'top_pairs': prediction['top_pairs'],
                    },
                }
                self.state['batches'][batch_id] = {
                    'id': batch_id,
                    'source': 'identify_feedback',
                    'status': 'uploaded',
                    'created_at': now_iso(),
                    'item_ids': [item_id],
                }
                self.state['items'][item_id] = item
                feedback['item_id'] = item_id
            record['feedback'] = feedback
        self.save()
        return {
            'identify_id': identify_id,
            'correct': correct,
            'queued_for_labeling': not correct,
            'label_item_id': feedback.get('item_id'),
        }


def auto_promote_matched_items(store: StateStore) -> int:
    """把历史待审中“用户提交和模型预测完全一致”的样本自动转为待训练。"""
    changed = 0
    with store.lock:
        for item in store.state['items'].values():
            if item.get('status') != 'needs_admin':
                continue
            screen = item.get('screen') or {}
            pred_pair = sorted([int(x) for x in screen.get('pred_pair') or []])
            user_pair = sorted([int(x) for x in item.get('pair') or []])
            pred_animal = str(screen.get('pred_animal') or '').strip()
            user_animal = str(item.get('animal') or '').strip()
            if pred_pair and pred_pair == user_pair and pred_animal and pred_animal == user_animal:
                item['status'] = 'approved'
                item['review_note'] = item.get('review_note') or 'auto-approved: migrated matched screen result'
                item['reviewed_at'] = item.get('reviewed_at') or now_iso()
                item['reviewed_by'] = item.get('reviewed_by') or 'auto_screen_migration'
                screen['reason'] = '用户答案与模型预测一致，自动通过，进入待训练。'
                screen['ok'] = True
                changed += 1
    if changed:
        store.save()
    return changed


class ScreeningWorker:
    def __init__(self, store: StateStore, reject_confidence: float, identifier=None) -> None:
        self.store = store
        self.reject_confidence = reject_confidence
        self.lock = threading.Lock()
        self.identifier = identifier
        self.identifier_error = ''

    def queue_batch(self, batch_id: str) -> None:
        thread = threading.Thread(target=self.screen_batch, args=(batch_id,), daemon=True)
        thread.start()

    def get_identifier(self):
        if self.identifier is not None or self.identifier_error:
            return self.identifier
        with self.lock:
            if self.identifier is not None or self.identifier_error:
                return self.identifier
            try:
                sys.path.insert(0, str(ROOT_DIR / 'tools'))
                from predict_sameobject_ensemble import DEFAULT_FULL_WEIGHT, DEFAULT_PARTS_WEIGHT
                from sameobject_api import Identifier

                self.identifier = Identifier(DEFAULT_FULL_WEIGHT, DEFAULT_PARTS_WEIGHT, parts_weight=0.25)
            except Exception as exc:
                self.identifier_error = repr(exc)
                print('screening model unavailable:', self.identifier_error)
        return self.identifier

    def screen_batch(self, batch_id: str) -> None:
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                return
            batch['status'] = 'screening'
            batch['screen_started_at'] = now_iso()
            for item_id in batch['item_ids']:
                self.store.state['items'][item_id]['status'] = 'screening'
        self.store.save()

        identifier = self.get_identifier()
        with self.store.lock:
            item_ids = list(self.store.state['batches'][batch_id]['item_ids'])

        for item_id in item_ids:
            try:
                with self.store.lock:
                    item = self.store.state['items'][item_id]
                    image_path = ROOT_DIR / item['upload_path']
                    user_pair = sorted([int(x) for x in item['pair']])
                if identifier is None:
                    screen = {
                        'ok': True,
                        'mode': 'manual_fallback',
                        'reason': f'模型不可用，直接进入人工审核：{self.identifier_error}',
                    }
                    status = 'needs_admin'
                else:
                    result = identifier.identify(image_path.read_bytes())
                    pred_pair = sorted([int(x) for x in result.get('best_pair', [])])
                    score = float(result.get('best_score') or 0.0)
                    pred_animal = str(result.get('best_animal') or '').strip()
                    user_animal = str(item.get('animal') or '').strip()
                    same_pair = pred_pair == user_pair
                    same_animal = bool(user_animal) and user_animal == pred_animal
                    screen = {
                        'ok': (same_pair and same_animal) or ((not same_pair) and score < self.reject_confidence),
                        'mode': 'model',
                        'pred_pair': pred_pair,
                        'pred_animal': pred_animal,
                        'confidence': score,
                        'top_pairs': result.get('top_pairs', []),
                    }
                    if same_pair and same_animal:
                        screen['reason'] = '用户答案与模型预测一致，自动通过，进入待训练。'
                        status = 'approved'
                        with self.store.lock:
                            item = self.store.state['items'][item_id]
                            item['review_note'] = 'auto-approved: pair/animal matched model prediction'
                            item['reviewed_at'] = now_iso()
                            item['reviewed_by'] = 'auto_screen'
                    elif same_pair:
                        screen['reason'] = f'答案格一致但动物类别不一致，进入人工审核。用户={user_animal or "空"}，预测={pred_animal or "空"}'
                        status = 'needs_admin'
                    elif score >= self.reject_confidence:
                        screen['reason'] = f'高置信度不一致，疑似乱选，自动拦截。阈值={self.reject_confidence}'
                        status = 'auto_rejected'
                    else:
                        screen['reason'] = '模型不确定或低置信度冲突，进入人工审核。'
                        status = 'needs_admin'
                with self.store.lock:
                    item = self.store.state['items'][item_id]
                    item['screen'] = screen
                    item['status'] = status
                    item['screened_at'] = now_iso()
                self.store.save()
            except Exception as exc:
                traceback.print_exc()
                with self.store.lock:
                    item = self.store.state['items'][item_id]
                    item['status'] = 'needs_admin'
                    item['screen'] = {
                        'ok': True,
                        'mode': 'error_fallback',
                        'reason': f'初筛异常，转人工审核：{exc!r}',
                    }
                    item['screened_at'] = now_iso()
                self.store.save()

        with self.store.lock:
            batch = self.store.state['batches'][batch_id]
            batch['status'] = 'screened'
            batch['screen_finished_at'] = now_iso()
        self.store.save()


class TrainingRunner:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.lock = threading.Lock()

    @staticmethod
    def is_auto_approved(item: dict) -> bool:
        return str(item.get('reviewed_by') or '').startswith('auto_screen')

    def export_approved_to_manifest(self, include_auto_approved: bool) -> int:
        with self.store.lock:
            approved = [
                item for item in self.store.state['items'].values()
                if item['status'] == 'approved' and (include_auto_approved or not self.is_auto_approved(item))
            ]
        if not approved:
            raise RuntimeError('没有可导出的已通过数据。若只剩自动通过样本，请打开“包含自动通过样本”。')

        ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest = load_json(ANSWER_PATH, {'_说明': 'pair 是正确两个格子；animals 是 1-8 每格动物类别。', 'answers': []})
        answers = manifest.get('answers', [])
        by_image = {row.get('image'): row for row in answers if row.get('image')}

        APPROVED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        exported = 0
        with self.store.lock:
            for item in approved:
                src = ROOT_DIR / item['upload_path']
                ext = Path(item['upload_path']).suffix or '.png'
                dst = APPROVED_IMAGE_DIR / f"{item['id']}{ext}"
                if src.exists():
                    shutil.copy2(src, dst)
                image_rel = posix_rel(dst)
                item['training_image'] = image_rel
                row = {
                    'image': image_rel,
                    'pair': sorted([int(x) for x in item['pair']]),
                    'animal': item.get('animal', ''),
                }
                animals = {str(k): v for k, v in (item.get('animals') or {}).items() if v}
                if animals:
                    row['animals'] = animals
                if item.get('review_note'):
                    row['note'] = item['review_note']
                by_image[image_rel] = row
                exported += 1
            self.store.save()

        if ANSWER_PATH.exists():
            backup = ANSWER_PATH.with_suffix(ANSWER_PATH.suffix + f'.bak-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
            shutil.copy2(ANSWER_PATH, backup)
        merged = dict(manifest)
        merged['answers'] = sorted(by_image.values(), key=lambda row: row.get('image', ''))
        atomic_write_json(ANSWER_PATH, merged)
        return exported

    def start(self, run_name: str, feature_mode: str, epochs: int, val_ratio: float, include_auto_approved: bool) -> dict:
        with self.lock:
            with self.store.lock:
                for job in self.store.state['train_jobs'].values():
                    if job.get('status') == 'running':
                        raise RuntimeError('已有训练任务正在运行。')
                blocking = [item for item in self.store.state['items'].values() if item['status'] in {'screening', 'needs_admin'}]
                if blocking:
                    raise RuntimeError(f'还有 {len(blocking)} 条数据未完成审核，不能开始训练。')
            exported = self.export_approved_to_manifest(include_auto_approved)
            job_id = uuid.uuid4().hex[:12]
            log_path = TRAIN_LOG_DIR / f'{job_id}.log'
            TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = RUN_ROOT / f'feature_cache_{feature_mode}.pt'
            if cache_path.exists():
                cache_path.unlink()
            args = [
                sys.executable,
                str(ROOT_DIR / 'tools' / 'train_sameobject_animal_classifier.py'),
                '--run-name', run_name,
                '--feature-mode', feature_mode,
                '--epochs', str(epochs),
                '--val-ratio', str(val_ratio),
            ]
            log_file = log_path.open('w', encoding='utf-8', errors='replace')
            process = subprocess.Popen(
                args,
                cwd=str(ROOT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job = {
                'id': job_id,
                'status': 'running',
                'created_at': now_iso(),
                'run_name': run_name,
                'feature_mode': feature_mode,
                'epochs': epochs,
                'val_ratio': val_ratio,
                'include_auto_approved': include_auto_approved,
                'exported_count': exported,
                'log_path': posix_rel(log_path),
                'pid': process.pid,
                'command': args,
            }
            with self.store.lock:
                self.store.state['train_jobs'][job_id] = job
            self.store.save()

            def wait_job() -> None:
                try:
                    code = process.wait()
                finally:
                    log_file.close()
                with self.store.lock:
                    current = self.store.state['train_jobs'][job_id]
                    current['status'] = 'succeeded' if code == 0 else 'failed'
                    current['returncode'] = code
                    current['finished_at'] = now_iso()
                self.store.save()

            threading.Thread(target=wait_job, daemon=True).start()
            return job


USER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SameObject 训练标注</title>
  <style>
    :root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#1f6feb;--green:#238636}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",Arial,sans-serif;overflow:hidden}
    header{height:56px;background:#161b22;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;padding:8px 12px;white-space:nowrap}.brand{font-weight:800;font-size:16px}.meta{color:var(--muted)}a{color:#58a6ff}
    button,input{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit;min-height:34px}button{cursor:pointer}button:not(.box):hover{background:#30363d}button:focus,input:focus{outline:2px solid rgba(88,166,255,.45)}button.primary{background:var(--blue);border-color:#58a6ff}button.success{background:var(--green);border-color:#2ea043}button:disabled{opacity:.55;cursor:not-allowed}
    main{height:calc(100vh - 56px);display:grid;grid-template-columns:minmax(620px,1fr) 360px;gap:10px;padding:10px;overflow:hidden}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;min-height:0}.stage{display:flex;flex-direction:column;overflow:hidden}.topline{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line)}
    .viewer{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;overflow:auto;background:#05080c;border-radius:0 0 10px 10px}.imageWrap{position:relative;display:inline-block;max-width:100%;max-height:100%}#captcha{display:block;max-width:100%;max-height:calc(100vh - 128px);height:auto;width:auto;user-select:none}
    .box{position:absolute;border:3px solid #2ea043;border-radius:10px;background:rgba(46,160,67,.06);cursor:pointer;min-width:28px;min-height:28px}.box:hover{background:rgba(46,160,67,.12);border-color:#56d364}.box.selected{border-color:#ff4d4f;background:rgba(255,77,79,.2);box-shadow:0 0 14px rgba(255,77,79,.7)}.box.selected:hover{background:rgba(255,77,79,.24);border-color:#ff7b72}.box .num{position:absolute;left:3px;top:2px;background:rgba(0,0,0,.72);border-radius:5px;padding:0 5px;font-weight:800}.box .tag{position:absolute;right:3px;bottom:2px;background:rgba(31,111,235,.9);border-radius:5px;padding:0 5px;font-size:12px}
    aside{padding:10px;overflow:auto}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}.pair{font-weight:800}.ok{color:#3fb950}.bad{color:#ff7b72}.yellow{color:#f2cc60}.animalGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.animalGrid button.active{background:var(--blue);border-color:#58a6ff}.thumbs{display:flex;gap:6px;overflow:auto;padding:7px 0}.thumb{border:2px solid var(--line);border-radius:8px;background:#0d1117;padding:2px;cursor:pointer;position:relative}.thumb.active{border-color:#58a6ff}.thumb.done:after{content:"";position:absolute;right:3px;top:3px;width:9px;height:9px;border-radius:50%;background:#3fb950}.thumb img{display:block;width:58px;height:36px;object-fit:cover;border-radius:5px}.hint{font-size:13px;color:#c9d1d9;line-height:1.65}.kbd{background:#30363d;border-radius:4px;padding:1px 5px;font-family:Consolas,monospace}.status{white-space:pre-wrap;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px;min-height:70px;color:var(--muted)}
    @media(max-width:980px){body{overflow:auto}main{height:auto;grid-template-columns:1fr}.viewer{min-height:420px}#captcha{max-height:70vh}}
  </style>
</head>
<body>
<header><span class="brand">SameObject 训练标注</span><input id="fileInput" type="file" accept="image/*" multiple style="max-width:250px"><button id="uploadBtn" class="primary">上传</button><button id="loadPlatformBtn">领取平台待标注</button><button id="prevBtn">上一张 A</button><button id="nextBtn">下一张 D</button><button id="submitBtn" class="success" disabled>提交整轮 Ctrl+Enter</button><span id="progress" class="meta">未上传</span><span style="flex:1"></span><a href="/admin">管理员</a></header>
<main><section class="panel stage"><div class="topline"><div><b id="title">等待上传</b> <span id="fileMeta" class="meta"></span></div><div class="pair" id="pairText">未选择</div></div><div class="viewer"><div id="imageWrap" class="imageWrap"><img id="captcha" alt="当前标注图片"></div></div></section><aside class="panel"><div class="row"><b>动物类别</b><span id="animalText" class="yellow">未选择</span></div><div id="animalGrid" class="animalGrid"></div><div class="row"><input id="animalOther" placeholder="自定义类别" style="flex:1;min-width:0"><button id="applyOtherBtn">应用</button></div><div class="row"><button id="clearPairBtn">清空当前</button><button id="refreshBtn" disabled>刷新初筛</button></div><div class="thumbs" id="thumbs"></div><div class="hint"><p><b>快捷键</b>：<span class="kbd">1-8</span> 选答案格，<span class="kbd">A/←</span> 上一张，<span class="kbd">D/→</span> 下一张，<span class="kbd">Enter</span> 下一张，<span class="kbd">Ctrl+Enter</span> 提交整轮。</p><p>每轮最多 10 张；所有图片都选好两个格子并填写类别后才可提交。</p></div><div id="status" class="status">请选择图片上传。</div></aside></main>
<script>
const animals = ['', '野猪','熊','豹子','蜘蛛','鹿','羊','牛','狼','袋鼠','不确定']; const PLATFORM_BATCH_KEY='sameobject_platform_labeling_batch'; let batch=null, idx=0; const answers=new Map(); const $=id=>document.getElementById(id);
function setStatus(t,cls=''){ $('status').className='status '+cls; $('status').textContent=t; } function esc(s){ return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); } async function api(url,opts={}){ const r=await fetch(url,opts); const d=await r.json().catch(()=>({})); if(!r.ok||d.code) throw new Error(d.message||r.statusText); return d.data??d; }
function current(){ return batch?.items[idx]; } function ans(id){ if(!answers.has(id)) answers.set(id,{pair:[],animal:''}); return answers.get(id); } function completeItem(it){ const a=answers.get(it.id); return a&&a.pair.length===2&&a.animal; } function complete(){ return batch&&batch.items.every(completeItem); }
function renderAnimals(){ const grid=$('animalGrid'); grid.innerHTML=''; const a=current()?ans(current().id):{animal:''}; for(const animal of animals.filter(Boolean)){ const b=document.createElement('button'); b.textContent=animal; b.className=a.animal===animal?'active':''; b.onclick=()=>{ if(!current())return; ans(current().id).animal=animal; render(); }; grid.appendChild(b); } }
function renderThumbs(){ const root=$('thumbs'); root.innerHTML=''; if(!batch)return; batch.items.forEach((it,i)=>{ const d=document.createElement('button'); d.className='thumb '+(i===idx?'active ':'')+(completeItem(it)?'done':''); d.title=it.filename; d.innerHTML=`<img alt="${esc(it.filename)}" src="${it.image_url}">`; d.onclick=()=>{idx=i;render();}; root.appendChild(d); }); }
function render(){ const it=current(); $('submitBtn').disabled=!complete(); $('refreshBtn').disabled=!batch; renderAnimals(); renderThumbs(); if(!it){ $('title').textContent='等待上传'; $('fileMeta').textContent=''; $('captcha').removeAttribute('src'); $('pairText').textContent='未选择'; $('animalText').textContent='未选择'; $('progress').textContent='未上传'; clearBoxes(); return; } const a=ans(it.id); $('title').textContent=`#${idx+1}/${batch.items.length} ${it.filename}`; $('fileMeta').textContent=`${it.size[0]}×${it.size[1]} · ${it.box_source}`; $('pairText').textContent=a.pair.length===2?`答案 [${a.pair.join(', ')}]`:a.pair.length?`[${a.pair.join(', ')}] 还差 ${2-a.pair.length} 个`:'未选择'; $('pairText').className='pair '+(a.pair.length===2?'ok':'bad'); $('animalText').textContent=a.animal||'未选择'; $('animalText').className=a.animal?'ok':'yellow'; $('progress').textContent=`${idx+1}/${batch.items.length}，完成 ${batch.items.filter(completeItem).length}/${batch.items.length}`; const img=$('captcha'); img.onload=drawBoxes; if(img.getAttribute('src')!==it.image_url) img.src=it.image_url; else drawBoxes(); }
function clearBoxes(){ document.querySelectorAll('.box').forEach(x=>x.remove()); } function drawBoxes(){ clearBoxes(); const it=current(); if(!it)return; const img=$('captcha'), wrap=$('imageWrap'), a=ans(it.id); const sx=img.clientWidth/it.size[0], sy=img.clientHeight/it.size[1]; it.boxes.forEach((b,i)=>{ const n=i+1, div=document.createElement('button'); div.type='button'; div.className='box '+(a.pair.includes(n)?'selected':''); div.style.left=(b[0]*sx)+'px'; div.style.top=(b[1]*sy)+'px'; div.style.width=(b[2]*sx)+'px'; div.style.height=(b[3]*sy)+'px'; div.innerHTML=`<span class="num">${n}</span>${a.pair.includes(n)&&a.animal?`<span class="tag">${esc(a.animal)}</span>`:''}`; div.onclick=()=>toggle(n); wrap.appendChild(div); }); }
function toggle(n){ const it=current(); if(!it)return; const a=ans(it.id); if(a.pair.includes(n)) a.pair=a.pair.filter(x=>x!==n); else { if(a.pair.length>=2)a.pair.shift(); a.pair.push(n); a.pair.sort((x,y)=>x-y); } render(); } function move(d){ if(!batch)return; idx=(idx+d+batch.items.length)%batch.items.length; render(); }
$('uploadBtn').onclick=async()=>{ const files=[...$('fileInput').files]; if(files.length<1||files.length>10){setStatus('请选择 1-10 张图片。','bad');return;} const fd=new FormData(); files.forEach(f=>fd.append('images',f)); $('uploadBtn').disabled=true; setStatus('上传并切格中...'); try{ batch=await api('/api/uploads',{method:'POST',body:fd}); idx=0; answers.clear(); render(); setStatus('上传完成，按 1-8 勾选答案。','ok'); }catch(e){setStatus(e.message,'bad');}finally{$('uploadBtn').disabled=false;} };
function usePlatformBatch(d,msg){ batch=d; idx=0; answers.clear(); localStorage.setItem(PLATFORM_BATCH_KEY,d.batch_id); render(); setStatus(msg,'ok'); } async function restorePlatformBatch(){ const batchId=localStorage.getItem(PLATFORM_BATCH_KEY); if(!batchId)return; try{ const d=await api('/api/submissions/'+encodeURIComponent(batchId)); if(d.source==='identify_feedback'&&d.status==='labeling'){ usePlatformBatch(d,'已恢复未完成的平台标注批次。'); }else{ localStorage.removeItem(PLATFORM_BATCH_KEY); } }catch(e){ localStorage.removeItem(PLATFORM_BATCH_KEY); } }
$('loadPlatformBtn').onclick=async()=>{ $('loadPlatformBtn').disabled=true; try{ const savedId=localStorage.getItem(PLATFORM_BATCH_KEY); if(savedId){ const saved=await api('/api/submissions/'+encodeURIComponent(savedId)).catch(()=>null); if(saved?.source==='identify_feedback'&&saved.status==='labeling'){ usePlatformBatch(saved,'已恢复未完成的平台标注批次。'); return; } localStorage.removeItem(PLATFORM_BATCH_KEY); } const d=await api('/api/platform-labeling/next'); if(!d.batch_id){setStatus('暂无平台待标注图片。','yellow');return;} usePlatformBatch(d,`已领取 ${d.items.length} 张平台反馈图片，请在 30 分钟内完成标注。`); }catch(e){setStatus(e.message,'bad');}finally{$('loadPlatformBtn').disabled=false;} };
$('prevBtn').onclick=()=>move(-1); $('nextBtn').onclick=()=>move(1); $('clearPairBtn').onclick=()=>{ const it=current(); if(!it)return; answers.set(it.id,{pair:[],animal:ans(it.id).animal}); render(); }; $('applyOtherBtn').onclick=()=>{ const it=current(), v=$('animalOther').value.trim(); if(it&&v){ ans(it.id).animal=v; render(); }};
$('submitBtn').onclick=async()=>{ if(!complete())return; const payload={answers:batch.items.map(it=>({item_id:it.id,pair:ans(it.id).pair,animal:ans(it.id).animal}))}; $('submitBtn').disabled=true; setStatus('提交中...'); try{ const d=await api(`/api/submissions/${batch.batch_id}/submit`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); if(batch.source==='identify_feedback')localStorage.removeItem(PLATFORM_BATCH_KEY); setStatus(d.status==='approved'?'平台反馈已标注，进入待训练数据。':'提交成功，后台初筛中。','ok'); }catch(e){setStatus(e.message,'bad'); render();} };
$('refreshBtn').onclick=async()=>{ if(!batch)return; try{ const d=await api(`/api/submissions/${batch.batch_id}`); setStatus('批次状态：'+d.status+'\n'+d.items.map(x=>`${x.filename}: ${x.status}${x.screen?.reason?' - '+x.screen.reason:''}`).join('\n'),'yellow'); }catch(e){setStatus(e.message,'bad');} };
window.addEventListener('resize',drawBoxes); document.addEventListener('keydown',e=>{ if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return; if(e.key>='1'&&e.key<='8')toggle(Number(e.key)); else if(e.key==='ArrowLeft'||e.key.toLowerCase()==='a')move(-1); else if(e.key==='ArrowRight'||e.key.toLowerCase()==='d')move(1); else if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();$('submitBtn').click();} else if(e.key==='Enter'){e.preventDefault();move(1);} }); render(); restorePlatformBatch();
</script></body></html>
"""

ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SameObject 管理员审核</title><style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#1f6feb;--green:#238636}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",Arial,sans-serif}a{color:#58a6ff}.muted{color:var(--muted)}button,input,select,textarea{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit;min-height:34px}button{cursor:pointer}button:hover{background:#30363d}.primary{background:var(--blue)}.success{background:var(--green)}.danger{background:#8e1519}.warnBtn{background:#9e6a03}.ok{color:#3fb950}.bad{color:#ff7b72}.yellow{color:#f2cc60}
#loginView{min-height:100vh;display:grid;place-items:center;padding:18px}.login{width:min(420px,100%);background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 20px 60px rgba(0,0,0,.35)}.login h1{margin:0 0 10px;font-size:20px}.login input{width:100%;margin:10px 0}.login button{width:100%}#appView{display:none}header{position:sticky;top:0;z-index:20;background:#161b22;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;padding:8px 12px}.brand{font-weight:800;font-size:16px}.spacer{flex:1}.topSwitch{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#0d1117;color:#c9d1d9;white-space:nowrap}.topSwitch input{min-height:auto;width:16px;height:16px;accent-color:#238636}main{padding:10px;display:grid;gap:10px}.stats{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:8px}.stat,.panel,.item{background:var(--panel);border:1px solid var(--line);border-radius:10px}.stat{padding:9px 11px}.stat b{display:block;font-size:22px}.panel{padding:10px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.list{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.item{padding:9px}.item img{width:100%;height:178px;object-fit:contain;background:#05080c;border-radius:8px;cursor:zoom-in}.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#334155}.badge.needs_admin{background:#854d0e}.badge.approved{background:#166534}.badge.rejected,.badge.auto_rejected{background:#991b1b}.badge.screening{background:#1d4ed8}.small{font-size:12px}.item input{width:92px}.item textarea{width:100%;min-height:54px;margin-top:6px;background:#0d1117}.status{white-space:pre-wrap;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px;color:var(--muted);max-height:220px;overflow:auto}#zoom{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.86);display:none;align-items:center;justify-content:center;padding:22px}#zoom img{max-width:96vw;max-height:92vh;border-radius:10px;background:#05080c}#zoom button{position:fixed;right:18px;top:14px}@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}.list{grid-template-columns:1fr}header{flex-wrap:wrap}}
</style></head><body>
<section id="loginView"><div class="login"><h1>管理员登录</h1><p class="muted">请输入训练 Web 管理员 Key。成功后只保存在当前浏览器会话中。</p><input id="loginKey" type="password" placeholder="TRAINING_WEB_ADMIN_KEY" autofocus><button id="loginBtn" class="primary">进入审核区</button><p id="loginMsg" class="bad"></p><p><a href="/">返回用户提交页</a></p></div></section>
<section id="appView"><header><span class="brand">管理员审核</span><button id="refreshBtn">刷新</button><label>状态 <select id="statusFilter"><option value="needs_admin">待审核</option><option value="auto_rejected">自动拦截</option><option value="approved">待训练/已通过</option><option value="rejected">已拒绝</option><option value="screening">初筛中</option><option value="all">全部</option></select></label><label class="topSwitch" title="训练时是否包含模型和用户一致后自动通过的样本"><input id="includeAutoApproved" type="checkbox" checked> 训练包含自动通过</label><span class="spacer"></span><button id="logoutBtn">退出</button><a href="/">用户页</a></header><main><section class="stats" id="stats"></section><section class="panel"><div class="row"><label>run-name <input id="runName" value="web_review_parts_v1" style="width:170px"></label><label>feature <select id="featureMode"><option value="parts_v1">parts_v1</option><option value="processed_v2">processed_v2</option></select></label><label>epochs <input id="epochs" type="number" min="1" value="180" style="width:90px"></label><label>val <input id="valRatio" type="number" min="0.05" max="0.9" step="0.05" value="0.2" style="width:80px"></label><button id="trainBtn" class="success">全部审核后训练</button><button id="trainStatusBtn">训练状态</button></div><div id="trainStatus" class="status">尚未查询。</div></section><section class="list" id="list"></section></main></section><div id="zoom"><button id="zoomClose">关闭 Esc</button><img id="zoomImg" alt="放大预览"></div>
<script>
const KEY='sameobject_admin_key'; const $=id=>document.getElementById(id); let adminKey=sessionStorage.getItem(KEY)||''; function esc(s){return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));} function headers(){return {'Content-Type':'application/json','X-Admin-Key':adminKey};} async function api(url,opts={}){opts.headers={...(opts.headers||{}),...headers()};const r=await fetch(url,opts);const d=await r.json().catch(()=>({}));if(!r.ok||d.code){if(r.status===401)showLogin('Key 不正确或已失效。');throw new Error(d.message||r.statusText);}return d.data??d;} function showLogin(msg=''){ $('loginView').style.display='grid'; $('appView').style.display='none'; $('loginMsg').textContent=msg; } async function loadConfig(){try{const r=await fetch('/api/config'); const d=await r.json(); if(d.data&&typeof d.data.include_auto_approved_in_training==='boolean') $('includeAutoApproved').checked=d.data.include_auto_approved_in_training;}catch(e){}} async function showApp(){ $('loginView').style.display='none'; $('appView').style.display='block'; await loadConfig(); loadList().catch(e=>showLogin(e.message)); }
$('loginBtn').onclick=async()=>{adminKey=$('loginKey').value.trim(); if(!adminKey){$('loginMsg').textContent='请输入 Key。';return;} try{await api('/api/admin/stats'); sessionStorage.setItem(KEY,adminKey); showApp();}catch(e){$('loginMsg').textContent=e.message;}}; $('loginKey').addEventListener('keydown',e=>{if(e.key==='Enter')$('loginBtn').click();}); $('logoutBtn').onclick=()=>{sessionStorage.removeItem(KEY);adminKey='';showLogin('已退出。');};
function badge(s){return `<span class="badge ${esc(s)}">${esc(s)}</span>`;} async function loadStats(){const d=await api('/api/admin/stats'); const labels={needs_admin:'待标注/待人工',screening:'初筛中',auto_rejected:'自动拦截',approved:'待训练/已通过',rejected:'已拒绝'}; const keys=['needs_admin','screening','auto_rejected','approved','rejected']; $('stats').innerHTML=keys.map(k=>`<div class="stat"><span class="muted">${labels[k]}</span><b>${d.counts[k]||0}</b></div>`).join('');} async function loadList(){await loadStats(); const st=$('statusFilter').value; const d=await api('/api/admin/items?status='+encodeURIComponent(st)); $('list').innerHTML=d.items.map(renderItem).join(''); document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>review(b.dataset.id,b.dataset.action)); document.querySelectorAll('[data-zoom]').forEach(img=>img.onclick=()=>zoom(img.src));}
function renderItem(it){const s=it.screen||{}, pair=(it.pair||[]).join(','); return `<article class="item"><div class="row" style="justify-content:space-between"><b title="${esc(it.filename)}">${esc(it.filename)}</b>${badge(it.status)}</div><img data-zoom src="${it.image_url}" alt="审核图片 ${esc(it.filename)}" loading="lazy"><div class="small muted">提交：pair=[${esc(pair)}] animal=${esc(it.animal||'')}</div><div class="small ${s.ok===false?'bad':'yellow'}">${esc(s.reason||'无初筛信息')} ${s.pred_pair?`预测=[${esc(s.pred_pair.join(','))}] ${esc(s.pred_animal||'')} score=${esc(s.confidence)}`:''}</div><div class="row"><label>pair <input id="pair-${esc(it.id)}" value="${esc(pair)}"></label><label>animal <input id="animal-${esc(it.id)}" value="${esc(it.animal||'')}"></label></div><textarea id="note-${esc(it.id)}" placeholder="备注">${esc(it.review_note||'')}</textarea><div class="row"><button class="success" data-action="approve" data-id="${esc(it.id)}">通过</button><button class="danger" data-action="reject" data-id="${esc(it.id)}">拒绝</button><button class="warnBtn" data-action="needs_admin" data-id="${esc(it.id)}">待审</button></div></article>`;}
async function review(id,decision){const pair=document.getElementById('pair-'+id).value.split(',').map(x=>Number(x.trim())).filter(Boolean); const animal=document.getElementById('animal-'+id).value.trim(); const note=document.getElementById('note-'+id).value.trim(); try{await api('/api/admin/items/'+id+'/review',{method:'POST',body:JSON.stringify({decision,pair,animal,note})}); await loadList();}catch(e){alert(e.message);}}
$('refreshBtn').onclick=loadList; $('statusFilter').onchange=loadList; $('trainBtn').onclick=async()=>{const p={run_name:$('runName').value.trim(),feature_mode:$('featureMode').value,epochs:Number($('epochs').value),val_ratio:Number($('valRatio').value),include_auto_approved:$('includeAutoApproved').checked}; $('trainStatus').textContent='启动中...'; try{const d=await api('/api/admin/train',{method:'POST',body:JSON.stringify(p)}); $('trainStatus').textContent=JSON.stringify(d,null,2); await loadStats();}catch(e){$('trainStatus').textContent=e.message;}}; $('trainStatusBtn').onclick=async()=>{try{const d=await api('/api/admin/train-status'); $('trainStatus').textContent=JSON.stringify(d,null,2);}catch(e){$('trainStatus').textContent=e.message;}}; function zoom(src){$('zoomImg').src=src; $('zoom').style.display='flex';} function closeZoom(){ $('zoom').style.display='none'; $('zoomImg').removeAttribute('src'); } $('zoomClose').onclick=closeZoom; $('zoom').onclick=e=>{if(e.target.id==='zoom')closeZoom();}; document.addEventListener('keydown',e=>{if(e.key==='Escape')closeZoom();}); if(adminKey){showApp();}else{showLogin();}
</script></body></html>
"""


class TrainingWebHandler(BaseHTTPRequestHandler):
    server_version = 'JiuChongYaolouTrainingWeb/1.0'

    store: StateStore
    worker: ScreeningWorker
    trainer: TrainingRunner
    admin_key: str

    def log_message(self, fmt: str, *args) -> None:
        print('%s - %s' % (self.address_string(), fmt % args))

    def send_bytes(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def ok(self, data: dict) -> None:
        self.send_json(HTTPStatus.OK, {'code': 0, 'data': data})

    def error(self, status: HTTPStatus, message: str) -> None:
        self.send_json(status, {'code': status.value, 'message': message})

    def read_json(self) -> dict:
        length = int(self.headers.get('Content-Length', '0'))
        if length <= 0 or length > 2 * 1024 * 1024:
            raise ValueError('JSON 请求体大小不合法。')
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def require_admin(self) -> bool:
        if not self.admin_key:
            return True
        query = parse_qs(urlparse(self.path).query)
        candidate = self.headers.get('X-Admin-Key') or (query.get('admin_key') or [''])[0]
        if candidate == self.admin_key:
            return True
        self.error(HTTPStatus.UNAUTHORIZED, '管理员 Key 不正确。')
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == '/':
                self.send_bytes(HTTPStatus.OK, USER_HTML.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/admin':
                self.send_bytes(HTTPStatus.OK, ADMIN_HTML.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/healthz':
                self.ok({'ok': True})
            elif path == '/api/config':
                self.ok({
                    'max_images_per_batch': MAX_IMAGES_PER_BATCH,
                    'animals': ANIMALS,
                    'include_auto_approved_in_training': env_flag(
                        'INCLUDE_AUTO_APPROVED_IN_TRAINING',
                        DEFAULT_INCLUDE_AUTO_APPROVED_IN_TRAINING,
                    ),
                })
            elif path == '/api/platform-labeling/next':
                self.handle_get_platform_labeling()
            elif path.startswith('/api/submissions/'):
                self.handle_get_submission(path)
            elif path == '/api/admin/stats':
                if self.require_admin():
                    self.ok({'counts': self.store.counts()})
            elif path == '/api/admin/items':
                if self.require_admin():
                    self.handle_admin_items(parsed)
            elif path == '/api/admin/train-status':
                if self.require_admin():
                    self.handle_train_status()
            elif path.startswith('/media/item/'):
                self.handle_media(path)
            else:
                self.error(HTTPStatus.NOT_FOUND, 'Not Found')
        except Exception as exc:
            traceback.print_exc()
            self.error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == '/api/uploads':
                self.handle_upload()
            elif path.startswith('/api/submissions/') and path.endswith('/submit'):
                self.handle_submit(path)
            elif path.startswith('/api/admin/items/') and path.endswith('/review'):
                if self.require_admin():
                    self.handle_admin_review(path)
            elif path == '/api/admin/train':
                if self.require_admin():
                    self.handle_admin_train()
            else:
                self.error(HTTPStatus.NOT_FOUND, 'Not Found')
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def handle_upload(self) -> None:
        ctype = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in ctype:
            raise ValueError('请使用 multipart/form-data 上传图片。')
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': ctype, 'CONTENT_LENGTH': self.headers.get('Content-Length', '0')},
        )
        fields = form['images'] if 'images' in form else []
        if not isinstance(fields, list):
            fields = [fields]
        fields = [field for field in fields if getattr(field, 'filename', '')]
        if not 1 <= len(fields) <= MAX_IMAGES_PER_BATCH:
            raise ValueError(f'每轮必须上传 1-{MAX_IMAGES_PER_BATCH} 张图片。')

        batch_id = uuid.uuid4().hex[:12]
        batch_dir = UPLOAD_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for field in fields:
            data = field.file.read()
            if not 0 < len(data) <= MAX_IMAGE_BYTES:
                raise ValueError(f'{field.filename} 大小必须在 1 字节到 10MB 之间。')
            ext = safe_ext(field.filename, data)
            item_id = uuid.uuid4().hex[:12]
            filename = Path(field.filename).name
            dst = batch_dir / f'{item_id}{ext}'
            dst.write_bytes(data)
            try:
                with Image.open(dst) as image:
                    image.verify()
                with Image.open(dst) as image:
                    size = image.size
            except Exception as exc:
                dst.unlink(missing_ok=True)
                raise ValueError(f'{filename} 不是有效图片。') from exc
            boxes, box_source = detect_grid_boxes(dst, size)
            item = {
                'id': item_id,
                'batch_id': batch_id,
                'filename': filename,
                'upload_path': posix_rel(dst),
                'sha256': sha256_bytes(data),
                'size': list(size),
                'boxes': boxes,
                'box_source': box_source,
                'status': 'uploaded',
                'created_at': now_iso(),
            }
            items.append(item)

        with self.store.lock:
            self.store.state['batches'][batch_id] = {
                'id': batch_id,
                'status': 'uploaded',
                'created_at': now_iso(),
                'item_ids': [item['id'] for item in items],
            }
            for item in items:
                self.store.state['items'][item['id']] = item
        self.store.save()
        self.ok({'batch_id': batch_id, 'items': [self.public_item(item) for item in items]})

    def handle_submit(self, path: str) -> None:
        batch_id = path.split('/')[3]
        payload = self.read_json()
        answers = payload.get('answers')
        if not isinstance(answers, list):
            raise ValueError('answers 必须是数组。')
        by_id = {row.get('item_id'): row for row in answers if isinstance(row, dict)}
        self.store.release_expired_platform_claims()
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                raise ValueError('批次不存在。')
            is_platform_claim = batch.get('claim_kind') == 'platform_labeling'
            allowed_statuses = {'labeling'} if is_platform_claim else {'uploaded'}
            if batch['status'] not in allowed_statuses:
                raise ValueError('该批次已经提交过。')
            if set(by_id) != set(batch['item_ids']):
                raise ValueError('必须完成当前批次全部图片后再提交。')
            for item_id in batch['item_ids']:
                row = by_id[item_id]
                pair = row.get('pair')
                animal = str(row.get('animal') or '').strip()
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('每张图片必须选择两个答案格。')
                pair = sorted([int(pair[0]), int(pair[1])])
                if pair[0] == pair[1] or pair[0] < 1 or pair[1] > 8:
                    raise ValueError('答案格必须是 1-8 中不同的两个数字。')
                if not animal:
                    raise ValueError('每张图片必须填写动物类别。')
                item = self.store.state['items'][item_id]
                item['pair'] = pair
                item['animal'] = animal
                item['animals'] = {str(k): v for k, v in (row.get('animals') or {}).items() if v} if isinstance(row.get('animals'), dict) else {}
                item['user_note'] = str(row.get('note') or '')
                item['status'] = 'approved' if batch.get('source') == 'identify_feedback' else 'queued'
                item['submitted_at'] = now_iso()
                if batch.get('source') == 'identify_feedback':
                    item['reviewed_by'] = 'platform_labeler'
                    item['review_note'] = 'platform feedback labeled'
                    item['reviewed_at'] = now_iso()
                    source_batch = self.store.state['batches'].get(item.get('source_batch_id'))
                    if source_batch:
                        source_batch['status'] = 'approved'
                        source_batch['submitted_at'] = now_iso()
            batch['status'] = 'approved' if batch.get('source') == 'identify_feedback' else 'queued'
            batch['submitted_at'] = now_iso()
        self.store.save()
        if batch.get('source') != 'identify_feedback':
            self.worker.queue_batch(batch_id)
        self.ok({'batch_id': batch_id, 'status': batch['status']})

    def handle_get_platform_labeling(self) -> None:
        batch = self.store.claim_platform_labeling_batch()
        if not batch:
            self.ok({'batch_id': None, 'items': []})
            return
        with self.store.lock:
            items = [self.public_item(self.store.state['items'][item_id]) for item_id in batch['item_ids']]
            data = dict(batch)
            data['batch_id'] = batch['id']
            data['items'] = items
        self.ok(data)

    def handle_get_submission(self, path: str) -> None:
        batch_id = path.split('/')[3]
        self.store.release_expired_platform_claims()
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                self.error(HTTPStatus.NOT_FOUND, '批次不存在。')
                return
            items = [self.public_item(self.store.state['items'][item_id]) for item_id in batch['item_ids']]
            data = dict(batch)
            data['batch_id'] = batch['id']
            data['items'] = items
        self.ok(data)

    def public_item(self, item: dict) -> dict:
        data = dict(item)
        data['image_url'] = f"/media/item/{item['id']}"
        return data

    def handle_admin_items(self, parsed) -> None:
        query = parse_qs(parsed.query)
        status = (query.get('status') or ['needs_admin'])[0]
        with self.store.lock:
            items = list(self.store.state['items'].values())
        items = [item for item in items if item.get('source') != 'identify_feedback']
        if status != 'all':
            items = [item for item in items if item.get('status') == status]
        items.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        self.ok({'items': [self.public_item(item) for item in items[:500]]})

    def handle_admin_review(self, path: str) -> None:
        item_id = path.split('/')[4]
        payload = self.read_json()
        decision = payload.get('decision')
        if decision not in {'approve', 'reject', 'needs_admin'}:
            raise ValueError('decision 必须是 approve/reject/needs_admin。')
        with self.store.lock:
            item = self.store.state['items'].get(item_id)
            if not item:
                raise ValueError('数据不存在。')
            if decision == 'approve':
                pair = payload.get('pair') or item.get('pair')
                animal = str(payload.get('animal') or item.get('animal') or '').strip()
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('pair 必须是 1-8 中不同的两个数字。')
                pair = sorted([int(pair[0]), int(pair[1])])
                if pair[0] == pair[1] or pair[0] < 1 or pair[1] > 8:
                    raise ValueError('pair 必须是 1-8 中不同的两个数字。')
                if not animal:
                    raise ValueError('通过审核必须填写 animal。')
                item['pair'] = pair
                item['animal'] = animal
                item['status'] = 'approved'
                item['reviewed_by'] = 'admin'
            elif decision == 'reject':
                item['status'] = 'rejected'
                item['reviewed_by'] = 'admin'
            else:
                item['status'] = 'needs_admin'
                item['reviewed_by'] = 'admin'
            item['review_note'] = str(payload.get('note') or '')
            item['reviewed_at'] = now_iso()
        self.store.save()
        self.ok({'item': self.public_item(item)})

    def handle_admin_train(self) -> None:
        payload = self.read_json()
        run_name = str(payload.get('run_name') or f'web_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}').strip()
        feature_mode = str(payload.get('feature_mode') or 'parts_v1')
        epochs = int(payload.get('epochs') or 180)
        val_ratio = float(payload.get('val_ratio') or 0.2)
        include_auto_approved = bool(payload.get(
            'include_auto_approved',
            env_flag('INCLUDE_AUTO_APPROVED_IN_TRAINING', DEFAULT_INCLUDE_AUTO_APPROVED_IN_TRAINING),
        ))
        if feature_mode not in {'parts_v1', 'processed_v2'}:
            raise ValueError('feature_mode 只能是 parts_v1 或 processed_v2。')
        if epochs < 1:
            raise ValueError('epochs 必须大于 0。')
        job = self.trainer.start(run_name, feature_mode, epochs, val_ratio, include_auto_approved)
        self.ok({'job': job})

    def handle_train_status(self) -> None:
        with self.store.lock:
            jobs = sorted(self.store.state['train_jobs'].values(), key=lambda job: job.get('created_at', ''), reverse=True)
        latest = jobs[0] if jobs else None
        log_tail = ''
        if latest and latest.get('log_path'):
            path = ROOT_DIR / latest['log_path']
            if path.exists():
                text = path.read_text(encoding='utf-8', errors='replace')
                log_tail = text[-8000:]
        self.ok({'latest': latest, 'log_tail': log_tail})

    def handle_media(self, path: str) -> None:
        item_id = unquote(path.rsplit('/', 1)[-1])
        with self.store.lock:
            item = self.store.state['items'].get(item_id)
        if not item:
            self.error(HTTPStatus.NOT_FOUND, '图片不存在。')
            return
        image_path = (ROOT_DIR / item['upload_path']).resolve()
        if not str(image_path).startswith(str(DATA_DIR.resolve())):
            self.error(HTTPStatus.FORBIDDEN, '非法图片路径。')
            return
        if not image_path.exists():
            self.error(HTTPStatus.NOT_FOUND, '图片文件不存在。')
            return
        content_type = mimetypes.guess_type(str(image_path))[0] or 'application/octet-stream'
        self.send_bytes(HTTPStatus.OK, image_path.read_bytes(), content_type)


def build_handler(store: StateStore, worker: ScreeningWorker, trainer: TrainingRunner, admin_key: str):
    class Handler(TrainingWebHandler):
        pass

    Handler.store = store
    Handler.worker = worker
    Handler.trainer = trainer
    Handler.admin_key = admin_key
    return Handler


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description='九重妖楼识别测试训练数据 Web 平台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--admin-key', default=os.environ.get('TRAINING_WEB_ADMIN_KEY', ''))
    parser.add_argument('--reject-confidence', type=float, default=float(os.environ.get('TRAINING_WEB_REJECT_CONFIDENCE', DEFAULT_REJECT_CONFIDENCE)))
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = StateStore(STATE_PATH)
    promoted = auto_promote_matched_items(store)
    if promoted:
        print(f'auto-promoted matched pending items: {promoted}')
    worker = ScreeningWorker(store, args.reject_confidence)
    trainer = TrainingRunner(store)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(store, worker, trainer, args.admin_key))
    print(f'training web listening on http://{args.host}:{args.port}')
    if args.admin_key:
        print('admin auth: enabled')
    else:
        print('admin auth: disabled (set TRAINING_WEB_ADMIN_KEY to enable)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

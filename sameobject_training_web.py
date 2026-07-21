# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import cgi
import hashlib
import hmac
import json
import re
import mimetypes
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
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
STATE_DB_PATH = DATA_DIR / 'state.db'
APPROVED_IMAGE_DIR = ROOT_DIR / 'images' / 'training_web'
ANSWER_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'
RUN_ROOT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier'

MAX_IMAGES_PER_BATCH = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_REJECT_CONFIDENCE = 0.80
DEFAULT_INCLUDE_AUTO_APPROVED_IN_TRAINING = True
IDENTIFY_CACHE_TTL_SECONDS = 60
PLATFORM_LABELING_CLAIM_TTL_SECONDS = 30 * 60
REWARD_API_KEY_TTL_SECONDS = 24 * 60 * 60
NO_TASK_API_KEY_TTL_SECONDS = 2 * 60 * 60
MIN_REWARD_API_KEY_TTL_SECONDS = 15 * 60
MAX_REWARD_API_KEY_TTL_SECONDS = 30 * 24 * 60 * 60
CHINA_TIMEZONE = timezone(timedelta(hours=8), name='CST')
DEFAULT_LICENSE_SIGN_SKEW_SECONDS = 300
DEFAULT_LICENSE_NONCE_TTL_SECONDS = 600
LICENSE_CARD_KEY_RE = re.compile(r'^[A-Za-z0-9\-]{8,64}$')

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
    """以 UTC 保存时间，避免部署主机时区改变有效期判断。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_stored_time(value: str) -> datetime:
    """读取新 UTC 记录及历史的 UTC+8 无时区记录。"""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=CHINA_TIMEZONE)
    return parsed




def parse_page_params(query: dict, default_size: int = 20) -> tuple[int, int]:
    """解析分页参数；page 从 1 开始，page_size 默认 20，上限 100。"""
    try:
        page = int((query.get('page') or ['1'])[0])
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int((query.get('page_size') or [str(default_size)])[0])
    except (TypeError, ValueError):
        page_size = default_size
    page = max(1, page)
    page_size = min(100, max(1, page_size))
    return page, page_size


def paginate_list(items: list, page: int, page_size: int) -> dict:
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    return {
        'items': items[start:start + page_size],
        'page': page,
        'page_size': page_size,
        'total': total,
        'total_pages': total_pages,
    }


def generate_license_code() -> str:
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    parts = [''.join(secrets.choice(alphabet) for _ in range(4)) for _ in range(4)]
    return '-'.join(parts)

def parse_expires_at_input(value: object) -> str:
    """把前端到期时间解析为 UTC ISO。无时区按北京时间理解。"""
    raw = str(value or '').strip()
    if not raw:
        raise ValueError('必须指定到期时间。')
    raw = raw.replace('Z', '+00:00').replace('/', '-')
    if 'T' not in raw and ' ' in raw:
        raw = raw.replace(' ', 'T', 1)
    # datetime-local may be YYYY-MM-DDTHH:MM
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', raw):
        raw = raw + ':00'
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError('到期时间格式无效。') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CHINA_TIMEZONE)
    if parsed <= datetime.now(timezone.utc):
        raise ValueError('到期时间必须晚于当前时间。')
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()





def license_sign_skew_seconds() -> int:
    try:
        value = int(os.environ.get('LICENSE_SIGN_SKEW_SECONDS', DEFAULT_LICENSE_SIGN_SKEW_SECONDS))
    except (TypeError, ValueError):
        value = DEFAULT_LICENSE_SIGN_SKEW_SECONDS
    return max(30, min(value, 3600))


def license_nonce_ttl_seconds() -> int:
    try:
        value = int(os.environ.get('LICENSE_NONCE_TTL_SECONDS', DEFAULT_LICENSE_NONCE_TTL_SECONDS))
    except (TypeError, ValueError):
        value = DEFAULT_LICENSE_NONCE_TTL_SECONDS
    return max(60, min(value, 7200))


def license_api_secret() -> str:
    """HMAC 签名专用密钥（可选）。注意：这不是 SAMEOBJECT_API_KEY。"""
    return str(os.environ.get('LICENSE_API_SECRET') or '').strip()


def identify_api_keys() -> list[str]:
    """业务 API Key：用于识别接口与卡密接口鉴权。"""
    key = str(os.environ.get('SAMEOBJECT_API_KEY') or '').strip()
    return [key] if key else []


def is_env_master_api_key(value: object) -> bool:
    """环境变量 SAMEOBJECT_API_KEY：特殊通用 Key，可不入库，可当识别 Key 与登录卡密。"""
    candidate = str(value or '').strip()
    if not candidate:
        return False
    return any(secrets.compare_digest(candidate, key) for key in identify_api_keys() if key)


def master_license_payload(*, machine_code: str = '') -> dict:
    """通用 Key 登录/解绑统一成功响应（不写库、不绑机器）。"""
    return {
        'ok': True,
        'status': 'active',
        'expires_at': '9999-12-31T15:59:59+00:00',
        'machine_bound': False,
        'machine_code_masked': None,
        'no_machine_limit': True,
        'master_key': True,
    }


def license_sign_optional() -> bool:
    """未配置 LICENSE_API_SECRET 或显式 LICENSE_SIGN_OPTIONAL=1 时不验签。"""
    if not license_api_secret():
        return True
    return env_flag('LICENSE_SIGN_OPTIONAL', False)


def build_license_string_to_sign(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body or b'').hexdigest()
    return f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"


def sign_license_request(secret: str, method: str, path: str, timestamp: str, nonce: str, body: bytes) -> str:
    raw = build_license_string_to_sign(method, path, timestamp, nonce, body)
    return hmac.new(secret.encode('utf-8'), raw.encode('utf-8'), hashlib.sha256).hexdigest()



def mask_machine_code(value: object) -> str | None:
    """公开接口脱敏机器码：不回发明文。"""
    raw = str(value or '').strip()
    if not raw:
        return None
    if len(raw) <= 4:
        return raw[0] + '*' * (len(raw) - 1)
    if len(raw) <= 8:
        return raw[:2] + '*' * (len(raw) - 3) + raw[-1]
    return raw[:2] + '****' + raw[-2:]


def validate_license_params(
    payload: dict,
    *,
    require_machine: bool,
    header_card_key: str = '',
) -> tuple[str, str]:
    """卡密接口入参。

    设计：
    - 登录卡密放在请求头 X-API-Key（通用 Key=环境变量 SAMEOBJECT_API_KEY，普通卡=/keys 发卡）
    - body 携带 machine_code 等业务字段，并参与可选 HMAC 签名
    - body.card_key 可选；若传必须与 X-API-Key 一致（便于签名覆盖卡密）
    """
    if not isinstance(payload, dict):
        raise ValueError('请求体必须是 JSON 对象。')
    body_card = str(payload.get('card_key') or '').strip()
    machine_code = str(payload.get('machine_code') or '').strip()
    card_key = str(header_card_key or '').strip() or body_card
    if not card_key:
        raise ValueError('缺少卡密：请将登录卡密放在请求头 X-API-Key。')
    if body_card and not secrets.compare_digest(body_card, card_key):
        raise ValueError('body.card_key 与请求头 X-API-Key 不一致。')
    # 通用 Key 跳过普通卡密格式限制
    if not is_env_master_api_key(card_key) and not LICENSE_CARD_KEY_RE.fullmatch(card_key):
        raise ValueError('卡密格式不合法（仅允许字母数字和短横线，长度 8-64）。')
    if require_machine and not machine_code:
        raise ValueError('machine_code 不能为空。')
    if machine_code:
        if not 4 <= len(machine_code) <= 256:
            raise ValueError('machine_code 长度必须在 4 到 256 之间。')
        if any(ord(ch) < 32 for ch in machine_code):
            raise ValueError('machine_code 包含非法控制字符。')
    return card_key, machine_code


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
    """训练 Web 状态存储：SQLite 持久化 + 内存缓存。

    历史版本使用整份 state.json 原子写；高频写入时容易放大。
    现在默认落盘到 training_web_data/state.db（WAL），启动时自动从 state.json 迁移。
    业务代码仍可通过 self.state 读写，save() 负责刷盘。
    """

    def __init__(self, path: Path) -> None:
        self.json_path = path if path.suffix.lower() == '.json' else path.with_suffix('.json')
        self.db_path = STATE_DB_PATH if path == STATE_PATH or path.suffix.lower() == '.json' else path.with_suffix('.db')
        self.path = self.db_path
        self.lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA temp_store=MEMORY')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self._init_schema()
        self.state = {
            'version': 3,
            'created_at': now_iso(),
            'batches': {},
            'items': {},
            'identifications': {},
            'issued_api_keys': {},
            'license_keys': {},
            'train_jobs': {},
            'config': {},
        }
        if self._db_has_data():
            self._load_from_db()
        elif self.json_path.exists():
            legacy = load_json(self.json_path, {})
            if legacy:
                self.state.update({
                    'version': legacy.get('version', 3),
                    'created_at': legacy.get('created_at') or now_iso(),
                    'batches': legacy.get('batches') or {},
                    'items': legacy.get('items') or {},
                    'identifications': legacy.get('identifications') or {},
                    'issued_api_keys': legacy.get('issued_api_keys') or {},
                    'license_keys': legacy.get('license_keys') or {},
                    'train_jobs': legacy.get('train_jobs') or {},
                    'config': legacy.get('config') or {},
                })
                self.save()
                backup = self.json_path.with_suffix(self.json_path.suffix + f'.migrated-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
                try:
                    self.json_path.replace(backup)
                except OSError:
                    pass
                print(f'migrated state.json -> {self.db_path.name} (backup: {backup.name})')
        else:
            self.save()

        self.state.setdefault('identifications', {})
        self.state.setdefault('issued_api_keys', {})
        self.state.setdefault('license_keys', {})
        self.state.setdefault('train_jobs', {})
        self.state.setdefault('config', {})
        self.state.setdefault('batches', {})
        self.state.setdefault('items', {})

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                kind TEXT NOT NULL,
                id TEXT NOT NULL,
                status TEXT,
                created_at TEXT,
                data TEXT NOT NULL,
                PRIMARY KEY (kind, id)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_kind_status ON entities(kind, status);
            CREATE INDEX IF NOT EXISTS idx_entities_kind_created ON entities(kind, created_at);

            CREATE TABLE IF NOT EXISTS issued_api_keys (
                id TEXT PRIMARY KEY,
                token_hash TEXT NOT NULL,
                token_plain TEXT,
                purpose TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                note TEXT,
                batch_id TEXT,
                created_at TEXT,
                expires_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_issued_api_hash ON issued_api_keys(token_hash);
            CREATE INDEX IF NOT EXISTS idx_issued_api_expires ON issued_api_keys(expires_at);
            CREATE INDEX IF NOT EXISTS idx_issued_api_status ON issued_api_keys(status);

            CREATE TABLE IF NOT EXISTS license_keys (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                machine_code TEXT,
                created_at TEXT,
                expires_at TEXT NOT NULL,
                bound_at TEXT,
                last_verified_at TEXT,
                verify_count INTEGER NOT NULL DEFAULT 0,
                note TEXT,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_license_status ON license_keys(status);
            CREATE INDEX IF NOT EXISTS idx_license_code ON license_keys(code);

            CREATE TABLE IF NOT EXISTS license_nonces (
                nonce TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_license_nonces_exp ON license_nonces(expires_at);
            """
        )

    def _db_has_data(self) -> bool:
        for table in ('meta', 'entities', 'issued_api_keys', 'license_keys'):
            row = self.conn.execute(f'SELECT COUNT(*) AS c FROM {table}').fetchone()
            if row and int(row['c']) > 0:
                return True
        return False

    def _load_from_db(self) -> None:
        meta = {row['key']: row['value'] for row in self.conn.execute('SELECT key, value FROM meta')}
        self.state['version'] = int(meta.get('version') or 3)
        self.state['created_at'] = meta.get('created_at') or now_iso()
        self.state['config'] = {
            row['key']: json.loads(row['value'])
            for row in self.conn.execute('SELECT key, value FROM config')
        }
        for kind in ('batches', 'items', 'identifications', 'train_jobs'):
            bucket = {}
            for row in self.conn.execute('SELECT id, data FROM entities WHERE kind = ?', (kind,)):
                bucket[row['id']] = json.loads(row['data'])
            self.state[kind] = bucket
        issued = {}
        for row in self.conn.execute('SELECT id, data FROM issued_api_keys'):
            issued[row['id']] = json.loads(row['data'])
        self.state['issued_api_keys'] = issued
        licenses = {}
        for row in self.conn.execute('SELECT id, data FROM license_keys'):
            licenses[row['id']] = json.loads(row['data'])
        self.state['license_keys'] = licenses

    def _encode(self, value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

    def save(self) -> None:
        """把内存状态刷到 SQLite（单事务）。"""
        with self.lock:
            conn = self.conn
            conn.execute('BEGIN IMMEDIATE')
            try:
                conn.execute('DELETE FROM meta')
                conn.execute('INSERT INTO meta(key, value) VALUES (?, ?)', ('version', str(self.state.get('version', 3))))
                conn.execute('INSERT INTO meta(key, value) VALUES (?, ?)', ('created_at', str(self.state.get('created_at') or now_iso())))

                conn.execute('DELETE FROM config')
                conn.executemany(
                    'INSERT INTO config(key, value) VALUES (?, ?)',
                    [(str(k), self._encode(v)) for k, v in (self.state.get('config') or {}).items()],
                )

                conn.execute('DELETE FROM entities')
                entity_rows = []
                for kind in ('batches', 'items', 'identifications', 'train_jobs'):
                    for entity_id, record in (self.state.get(kind) or {}).items():
                        entity_rows.append((
                            kind,
                            str(entity_id),
                            str(record.get('status') or ''),
                            str(record.get('created_at') or ''),
                            self._encode(record),
                        ))
                conn.executemany(
                    'INSERT INTO entities(kind, id, status, created_at, data) VALUES (?, ?, ?, ?, ?)',
                    entity_rows,
                )

                conn.execute('DELETE FROM issued_api_keys')
                api_rows = []
                for key_id, record in (self.state.get('issued_api_keys') or {}).items():
                    api_rows.append((
                        str(record.get('id') or key_id),
                        str(record.get('token_hash') or ''),
                        record.get('token_plain'),
                        record.get('purpose'),
                        str(record.get('status') or 'active'),
                        record.get('note') or '',
                        record.get('batch_id'),
                        record.get('created_at'),
                        str(record.get('expires_at') or ''),
                        self._encode(record),
                    ))
                conn.executemany(
                    'INSERT INTO issued_api_keys(id, token_hash, token_plain, purpose, status, note, batch_id, created_at, expires_at, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    api_rows,
                )

                conn.execute('DELETE FROM license_keys')
                license_rows = []
                for key_id, record in (self.state.get('license_keys') or {}).items():
                    license_rows.append((
                        str(record.get('id') or key_id),
                        str(record.get('code') or ''),
                        str(record.get('status') or 'unused'),
                        record.get('machine_code'),
                        record.get('created_at'),
                        str(record.get('expires_at') or ''),
                        record.get('bound_at'),
                        record.get('last_verified_at'),
                        int(record.get('verify_count') or 0),
                        record.get('note') or '',
                        self._encode(record),
                    ))
                conn.executemany(
                    'INSERT INTO license_keys(id, code, status, machine_code, created_at, expires_at, bound_at, last_verified_at, verify_count, note, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    license_rows,
                )
                conn.execute('COMMIT')
            except Exception:
                conn.execute('ROLLBACK')
                raise


    def cleanup_expired_license_nonces(self, now_ts: int | None = None) -> None:
        ts = int(now_ts if now_ts is not None else time.time())
        with self.lock:
            self.conn.execute('DELETE FROM license_nonces WHERE expires_at <= ?', (ts,))

    def consume_license_nonce(self, nonce: str, now_ts: int | None = None) -> None:
        """登记 nonce；重复使用视为重放。"""
        nonce = str(nonce or '').strip()
        if not nonce:
            raise ValueError('nonce 不能为空。')
        if not 8 <= len(nonce) <= 128:
            raise ValueError('nonce 长度必须在 8 到 128 之间。')
        if any(ord(ch) < 33 or ord(ch) > 126 for ch in nonce):
            raise ValueError('nonce 只能包含可见 ASCII 字符。')
        ts = int(now_ts if now_ts is not None else time.time())
        ttl = license_nonce_ttl_seconds()
        with self.lock:
            self.cleanup_expired_license_nonces(ts)
            try:
                self.conn.execute(
                    'INSERT INTO license_nonces(nonce, created_at, expires_at) VALUES (?, ?, ?)',
                    (nonce, ts, ts + ttl),
                )
            except sqlite3.IntegrityError as exc:
                raise PermissionError('请求重放或 nonce 已使用。') from exc

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def counts(self) -> dict:
        with self.lock:
            counts: dict[str, int] = {}
            for item in self.state['items'].values():
                counts[item['status']] = counts.get(item['status'], 0) + 1
            return counts

    def reward_ttl_config(self) -> dict:
        defaults = {
            'no_task_api_key_ttl_seconds': NO_TASK_API_KEY_TTL_SECONDS,
            'answer_api_key_ttl_seconds': REWARD_API_KEY_TTL_SECONDS,
        }
        with self.lock:
            config = self.state.get('config', {})
            result = {}
            for key, default in defaults.items():
                try:
                    value = int(config.get(key, default))
                except (TypeError, ValueError):
                    value = default
                result[key] = value if MIN_REWARD_API_KEY_TTL_SECONDS <= value <= MAX_REWARD_API_KEY_TTL_SECONDS else default
            return result

    def update_reward_ttl_config(self, no_task_hours: object, answer_hours: object) -> dict:
        try:
            no_task_seconds = round(float(no_task_hours) * 60 * 60)
            answer_seconds = round(float(answer_hours) * 60 * 60)
        except (TypeError, ValueError) as exc:
            raise ValueError('Key 有效期必须是数字（单位：小时）。') from exc
        for value in (no_task_seconds, answer_seconds):
            if not MIN_REWARD_API_KEY_TTL_SECONDS <= value <= MAX_REWARD_API_KEY_TTL_SECONDS:
                raise ValueError('Key 有效期必须在 0.25 到 720 小时之间。')
        with self.lock:
            config = self.state.setdefault('config', {})
            config['no_task_api_key_ttl_seconds'] = no_task_seconds
            config['answer_api_key_ttl_seconds'] = answer_seconds
        self.save()
        return self.reward_ttl_config()

    def cleanup_expired_api_keys(self) -> None:
        now = datetime.now(timezone.utc)
        with self.lock:
            expired = [
                key_id for key_id, record in self.state['issued_api_keys'].items()
                if parse_stored_time(record['expires_at']) <= now
            ]
            for key_id in expired:
                self.state['issued_api_keys'].pop(key_id, None)
            if expired:
                self.conn.execute('BEGIN IMMEDIATE')
                try:
                    self.conn.executemany('DELETE FROM issued_api_keys WHERE id = ?', [(key_id,) for key_id in expired])
                    self.conn.execute('COMMIT')
                except Exception:
                    self.conn.execute('ROLLBACK')
                    raise

    def is_issued_api_key_valid(self, token: str) -> bool:
        """识别接口热路径：按 hash 索引校验，不做整表清理写盘。"""
        if not token:
            return False
        digest = hashlib.sha256(token.encode('utf-8')).hexdigest()
        now = datetime.now(timezone.utc)
        with self.lock:
            row = self.conn.execute(
                'SELECT token_hash, status, expires_at FROM issued_api_keys WHERE token_hash = ? LIMIT 1',
                (digest,),
            ).fetchone()
            if row is not None:
                if str(row['status'] or 'active') == 'void':
                    return False
                try:
                    if parse_stored_time(row['expires_at']) <= now:
                        return False
                except (KeyError, TypeError, ValueError):
                    return False
                return secrets.compare_digest(str(row['token_hash'] or ''), digest)
            for record in self.state['issued_api_keys'].values():
                if record.get('status', 'active') == 'void':
                    continue
                try:
                    if parse_stored_time(record['expires_at']) <= now:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                if secrets.compare_digest(str(record.get('token_hash') or ''), digest):
                    return True
            return False

    def issue_batch_api_key_reward(self, batch_id: str) -> None:
        """给通过初筛的整轮标注发放一个二十四小时有效的识别 Key。"""
        with self.lock:
            batch = self.state['batches'].get(batch_id)
            if not batch or batch.get('api_key_reward'):
                return
            token = f"sao_{secrets.token_urlsafe(32)}"
            key_id = uuid.uuid4().hex[:16]
            ttl_seconds = self.reward_ttl_config()['answer_api_key_ttl_seconds']
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).replace(microsecond=0).isoformat()
            self.state['issued_api_keys'][key_id] = {
                'id': key_id,
                'token_hash': hashlib.sha256(token.encode('utf-8')).hexdigest(),
                'token_plain': token,
                'batch_id': batch_id,
                'purpose': 'answer_reward',
                'status': 'active',
                'note': '',
                'created_at': now_iso(),
                'expires_at': expires_at,
            }
            batch['api_key_reward'] = {
                'key_id': key_id,
                'expires_at': expires_at,
                'delivery_token': token,
            }
        self.save()

    def claim_batch_api_key_reward(self, batch_id: str) -> dict | None:
        """一次性返回奖励 Key 原文，避免在状态接口重复暴露。"""
        with self.lock:
            batch = self.state['batches'].get(batch_id)
            reward = (batch or {}).get('api_key_reward') or {}
            token = reward.pop('delivery_token', None)
            if not token:
                return None
            reward['delivered_at'] = now_iso()
            key_id = reward.get('key_id')
            record = self.state['issued_api_keys'].get(key_id) if key_id else None
            if record is not None and not record.get('token_plain'):
                record['token_plain'] = token
            payload = {'api_key': token, 'expires_at': reward['expires_at']}
        self.save()
        return payload

    def issue_no_task_api_key_reward(self) -> dict:
        """队列为空时发放两小时临时 Key。"""
        self.cleanup_expired_api_keys()
        token = f"sao_{secrets.token_urlsafe(32)}"
        key_id = uuid.uuid4().hex[:16]
        ttl_seconds = self.reward_ttl_config()['no_task_api_key_ttl_seconds']
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).replace(microsecond=0).isoformat()
        with self.lock:
            self.state['issued_api_keys'][key_id] = {
                'id': key_id,
                'token_hash': hashlib.sha256(token.encode('utf-8')).hexdigest(),
                'token_plain': token,
                'purpose': 'no_task_reward',
                'status': 'active',
                'note': '',
                'created_at': now_iso(),
                'expires_at': expires_at,
            }
        self.save()
        return {'api_key': token, 'expires_at': expires_at}


    def create_license_keys(
        self,
        count: int,
        expires_at: object = None,
        days: object = None,
        note: str = '',
        no_machine_limit: bool = False,
    ) -> list[dict]:
        if not 1 <= count <= 100:
            raise ValueError('count 必须在 1 到 100 之间。')
        note = str(note or '').strip()
        no_machine_limit = bool(no_machine_limit)
        if expires_at not in (None, ''):
            expires_iso = parse_expires_at_input(expires_at)
        else:
            try:
                day_count = int(days if days not in (None, '') else 30)
            except (TypeError, ValueError) as exc:
                raise ValueError('days 必须是整数。') from exc
            if not 1 <= day_count <= 3650:
                raise ValueError('days 必须在 1 到 3650 之间。')
            expires_iso = (datetime.now(timezone.utc) + timedelta(days=day_count)).replace(microsecond=0).isoformat()
        created = []
        with self.lock:
            existing = {record.get('code') for record in self.state['license_keys'].values()}
            for _ in range(count):
                code = generate_license_code()
                while code in existing:
                    code = generate_license_code()
                existing.add(code)
                key_id = uuid.uuid4().hex[:12]
                record = {
                    'id': key_id,
                    'code': code,
                    'status': 'unused',
                    'machine_code': None,
                    'created_at': now_iso(),
                    'expires_at': expires_iso,
                    'bound_at': None,
                    'last_verified_at': None,
                    'last_unbind_at': None,
                    'verify_count': 0,
                    'note': note,
                    'no_machine_limit': no_machine_limit,
                }
                self.state['license_keys'][key_id] = record
                created.append(dict(record))
        self.save()
        return created

    def list_license_keys(self, status: str = 'all', q: str = '') -> list[dict]:
        now = datetime.now(timezone.utc)
        query = str(q or '').strip().lower()
        with self.lock:
            items = list(self.state['license_keys'].values())
            dirty = False
            for record in items:
                if record.get('status') in {'unused', 'active'}:
                    try:
                        if parse_stored_time(record['expires_at']) <= now:
                            record['status'] = 'expired'
                            dirty = True
                    except (KeyError, TypeError, ValueError):
                        pass
            if dirty:
                self.save()
            items = list(self.state['license_keys'].values())
        if status and status != 'all':
            items = [item for item in items if item.get('status') == status]
        if query:
            def match(item: dict) -> bool:
                fields = [
                    item.get('id'),
                    item.get('code'),
                    item.get('machine_code'),
                    item.get('note'),
                    item.get('status'),
                ]
                return any(query in str(field or '').lower() for field in fields)
            items = [item for item in items if match(item)]
        items.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        return items

    def void_license_key(self, key_id: str) -> dict:
        with self.lock:
            record = self.state['license_keys'].pop(key_id, None)
            if not record:
                raise KeyError('登录卡密不存在。')
            payload = dict(record)
        self.save()
        return payload

    def _find_license_by_code(self, card_key: str) -> dict | None:
        card_key = str(card_key or '').strip()
        if not card_key:
            return None
        row = self.conn.execute(
            'SELECT id, data FROM license_keys WHERE code = ? LIMIT 1',
            (card_key,),
        ).fetchone()
        if row is not None:
            record = self.state['license_keys'].get(row['id']) or json.loads(row['data'])
            self.state['license_keys'][record['id']] = record
            return record
        for item in self.state['license_keys'].values():
            if secrets.compare_digest(str(item.get('code') or ''), card_key):
                return item
        return None

    def unbind_license_machine(self, card_key: str, machine_code: str | None = None) -> dict:
        """用户解绑机器码；同一张卡密每个北京自然日仅可解绑一次。"""
        card_key = str(card_key or '').strip()
        machine_code = str(machine_code or '').strip()
        if not card_key:
            raise ValueError('card_key 不能为空。')
        # 通用 Key 不绑定机器，解绑直接成功（无日限额）
        if is_env_master_api_key(card_key):
            payload = master_license_payload(machine_code=machine_code)
            payload['last_unbind_at'] = None
            return payload
        now = datetime.now(timezone.utc)
        with self.lock:
            record = self._find_license_by_code(card_key)
            if not record:
                raise PermissionError('卡密不存在。')
            if record.get('status') == 'void':
                raise PermissionError('卡密已作废。')
            try:
                expired = parse_stored_time(record['expires_at']) <= now
            except (KeyError, TypeError, ValueError):
                expired = False
            if expired or record.get('status') == 'expired':
                record['status'] = 'expired'
                self.save()
                raise PermissionError('卡密已过期。')
            bound = str(record.get('machine_code') or '').strip()
            if not bound:
                raise PermissionError('当前卡密未绑定机器码。')
            if machine_code and not secrets.compare_digest(bound, machine_code):
                raise PermissionError('机器码不匹配，无法解绑。')
            last_unbind = record.get('last_unbind_at')
            if last_unbind:
                try:
                    last_dt = parse_stored_time(last_unbind)
                    now_cn = now.astimezone(CHINA_TIMEZONE)
                    last_cn = last_dt.astimezone(CHINA_TIMEZONE)
                    if now_cn.date() == last_cn.date():
                        raise PermissionError('该卡密今日已解绑过，请明天再试。')
                except PermissionError:
                    raise
                except (KeyError, TypeError, ValueError):
                    pass
            record['machine_code'] = None
            record['bound_at'] = None
            if record.get('status') == 'active':
                record['status'] = 'unused'
            record['last_unbind_at'] = now_iso()
            payload = {
                'ok': True,
                'status': record['status'],
                'machine_bound': False,
                'machine_code_masked': None,
                'last_unbind_at': record['last_unbind_at'],
                'expires_at': record.get('expires_at'),
            }
        self.save()
        return payload

    def update_license_key(self, key_id: str, payload: dict) -> dict:
        """管理员修改登录卡密字段（不可改 id/code）。"""
        if not isinstance(payload, dict):
            raise ValueError('请求体必须是 JSON 对象。')
        with self.lock:
            record = self.state['license_keys'].get(key_id)
            if not record:
                raise KeyError('登录卡密不存在。')

            if 'status' in payload and payload.get('status') is not None:
                status = str(payload.get('status') or '').strip()
                if status not in {'unused', 'active', 'expired'}:
                    raise ValueError('status 只能是 unused/active/expired。')
                record['status'] = status

            if 'note' in payload:
                record['note'] = str(payload.get('note') or '').strip()

            if 'no_machine_limit' in payload:
                record['no_machine_limit'] = bool(payload.get('no_machine_limit'))

            if 'expires_at' in payload and payload.get('expires_at') not in (None, ''):
                # 管理员改期允许设置为过去时间（直接变过期）
                raw = str(payload.get('expires_at')).strip().replace('Z', '+00:00')
                if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', raw):
                    raw = raw + ':00'
                try:
                    parsed = datetime.fromisoformat(raw.replace(' ', 'T') if ' ' in raw and 'T' not in raw else raw)
                except ValueError as exc:
                    raise ValueError('到期时间格式无效。') from exc
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=CHINA_TIMEZONE)
                record['expires_at'] = parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()

            if 'machine_code' in payload:
                machine = payload.get('machine_code')
                if machine in (None, ''):
                    record['machine_code'] = None
                    record['bound_at'] = None
                    if record.get('status') == 'active':
                        record['status'] = 'unused'
                else:
                    record['machine_code'] = str(machine).strip()
                    if not record.get('bound_at'):
                        record['bound_at'] = now_iso()
                    if record.get('status') == 'unused':
                        record['status'] = 'active'

            if 'bound_at' in payload:
                bound_at = payload.get('bound_at')
                record['bound_at'] = None if bound_at in (None, '') else str(bound_at).strip()

            if 'last_verified_at' in payload:
                value = payload.get('last_verified_at')
                record['last_verified_at'] = None if value in (None, '') else str(value).strip()

            if 'last_unbind_at' in payload:
                value = payload.get('last_unbind_at')
                record['last_unbind_at'] = None if value in (None, '') else str(value).strip()

            if 'verify_count' in payload and payload.get('verify_count') is not None:
                try:
                    count = int(payload.get('verify_count'))
                except (TypeError, ValueError) as exc:
                    raise ValueError('verify_count 必须是整数。') from exc
                if count < 0:
                    raise ValueError('verify_count 不能为负数。')
                record['verify_count'] = count

            # 过期时间与状态自动对齐
            try:
                if parse_stored_time(record['expires_at']) <= datetime.now(timezone.utc):
                    record['status'] = 'expired'
            except (KeyError, TypeError, ValueError):
                pass

            payload_out = dict(record)
        self.save()
        return payload_out

    def verify_license_key(self, card_key: str, machine_code: str) -> dict:
        card_key = str(card_key or '').strip()
        machine_code = str(machine_code or '').strip()
        if not card_key:
            raise ValueError('card_key 不能为空。')
        if not machine_code:
            raise ValueError('machine_code 不能为空。')
        # 环境变量通用 Key：不入库、不过期、不绑机器
        if is_env_master_api_key(card_key):
            return master_license_payload(machine_code=machine_code)
        now = datetime.now(timezone.utc)
        with self.lock:
            record = self._find_license_by_code(card_key)
            if not record:
                raise PermissionError('卡密不存在。')
            if record.get('status') == 'void':
                raise PermissionError('卡密已作废。')
            try:
                expired = parse_stored_time(record['expires_at']) <= now
            except (KeyError, TypeError, ValueError):
                expired = False
            if expired or record.get('status') == 'expired':
                record['status'] = 'expired'
                self.save()
                raise PermissionError('卡密已过期。')
            if bool(record.get('no_machine_limit')):
                # 不限制机器码：任意机器可登录，不强制绑定
                if record.get('status') == 'unused':
                    record['status'] = 'active'
                record['last_machine_code'] = machine_code
            else:
                bound = record.get('machine_code')
                if not bound:
                    record['machine_code'] = machine_code
                    record['status'] = 'active'
                    record['bound_at'] = now_iso()
                elif not secrets.compare_digest(str(bound), machine_code):
                    raise PermissionError('机器码不匹配，禁止登录。')
            record['last_verified_at'] = now_iso()
            record['verify_count'] = int(record.get('verify_count') or 0) + 1
            no_limit = bool(record.get('no_machine_limit'))
            bound_code = record.get('machine_code')
            payload = {
                'ok': True,
                'status': record['status'],
                'expires_at': record['expires_at'],
                'machine_bound': bool(bound_code),
                # 公开接口不回发明文机器码，仅给脱敏展示
                'machine_code_masked': mask_machine_code(bound_code),
                'no_machine_limit': no_limit,
            }
        self.save()
        return payload

    def create_manual_api_key(self, ttl_hours: object, note: str = '') -> dict:
        try:
            ttl_seconds = round(float(ttl_hours) * 60 * 60)
        except (TypeError, ValueError) as exc:
            raise ValueError('ttl_hours 必须是数字。') from exc
        if not MIN_REWARD_API_KEY_TTL_SECONDS <= ttl_seconds <= MAX_REWARD_API_KEY_TTL_SECONDS:
            raise ValueError('ttl_hours 必须在 0.25 到 720 小时之间。')
        note = str(note or '').strip()
        token = f"sao_{secrets.token_urlsafe(32)}"
        key_id = uuid.uuid4().hex[:16]
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).replace(microsecond=0).isoformat()
        with self.lock:
            self.state['issued_api_keys'][key_id] = {
                'id': key_id,
                'token_hash': hashlib.sha256(token.encode('utf-8')).hexdigest(),
                'token_plain': token,
                'purpose': 'manual_admin',
                'status': 'active',
                'note': note,
                'created_at': now_iso(),
                'expires_at': expires_at,
            }
        self.save()
        return {'id': key_id, 'api_key': token, 'expires_at': expires_at, 'purpose': 'manual_admin', 'note': note}

    def list_issued_api_keys(self, status: str = 'all', q: str = '') -> list[dict]:
        self.cleanup_expired_api_keys()
        query = str(q or '').strip().lower()
        with self.lock:
            items = []
            for record in self.state['issued_api_keys'].values():
                items.append({
                    'id': record.get('id'),
                    'api_key': record.get('token_plain') or '',
                    'purpose': record.get('purpose') or ('answer_reward' if record.get('batch_id') else 'unknown'),
                    'status': record.get('status', 'active'),
                    'note': record.get('note') or '',
                    'batch_id': record.get('batch_id'),
                    'created_at': record.get('created_at'),
                    'expires_at': record.get('expires_at'),
                })
        if status and status != 'all':
            items = [item for item in items if item.get('status') == status]
        if query:
            def match(item: dict) -> bool:
                fields = [
                    item.get('id'),
                    item.get('api_key'),
                    item.get('purpose'),
                    item.get('note'),
                    item.get('batch_id'),
                    item.get('status'),
                ]
                return any(query in str(field or '').lower() for field in fields)
            items = [item for item in items if match(item)]
        items.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        return items

    def void_issued_api_key(self, key_id: str) -> dict:
        with self.lock:
            record = self.state['issued_api_keys'].pop(key_id, None)
            if not record:
                raise KeyError('识别 API Key 不存在。')
            payload = {
                'id': record.get('id'),
                'purpose': record.get('purpose') or '',
                'status': 'deleted',
                'note': record.get('note') or '',
                'created_at': record.get('created_at'),
                'expires_at': record.get('expires_at'),
            }
        self.save()
        return payload

    def cleanup_expired_identifications(self) -> None:
        """清理未反馈的临时识别缓存；它们不会进入标注或训练流程。"""
        cutoff = time.time() - IDENTIFY_CACHE_TTL_SECONDS
        expired = []
        with self.lock:
            for identify_id, record in self.state['identifications'].items():
                if record.get('feedback'):
                    continue
                try:
                    created_at = parse_stored_time(record['created_at']).timestamp()
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
        now = datetime.now(timezone.utc)
        released = 0
        with self.lock:
            expired_claims = []
            for batch in self.state['batches'].values():
                if batch.get('claim_kind') != 'platform_labeling' or batch.get('status') != 'labeling':
                    continue
                try:
                    expires_at = parse_stored_time(batch['claim_expires_at'])
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
            expires_at = (datetime.now(timezone.utc) + timedelta(seconds=PLATFORM_LABELING_CLAIM_TTL_SECONDS)).replace(microsecond=0).isoformat()
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
            eligible_for_reward = identifier is not None and all(
                self.store.state['items'][item_id].get('status') != 'auto_rejected'
                and (self.store.state['items'][item_id].get('screen') or {}).get('mode') == 'model'
                for item_id in batch['item_ids']
            )
        if eligible_for_reward:
            self.store.issue_batch_api_key_reward(batch_id)
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

    def export_approved_to_manifest(self, include_auto_approved: bool) -> list[str]:
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
        exported_ids: list[str] = []
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
                exported_ids.append(item['id'])
            self.store.save()

        if ANSWER_PATH.exists():
            backup = ANSWER_PATH.with_suffix(ANSWER_PATH.suffix + f'.bak-{datetime.now().strftime("%Y%m%d-%H%M%S")}')
            shutil.copy2(ANSWER_PATH, backup)
        merged = dict(manifest)
        merged['answers'] = sorted(by_image.values(), key=lambda row: row.get('image', ''))
        atomic_write_json(ANSWER_PATH, merged)
        return exported_ids

    def start(self, run_name: str, feature_mode: str, epochs: int, val_ratio: float, include_auto_approved: bool) -> dict:
        with self.lock:
            with self.store.lock:
                for job in self.store.state['train_jobs'].values():
                    if job.get('status') == 'running':
                        raise RuntimeError('已有训练任务正在运行。')
                blocking = [item for item in self.store.state['items'].values() if item['status'] in {'screening', 'needs_admin'}]
                if blocking:
                    raise RuntimeError(f'还有 {len(blocking)} 条数据未完成审核，不能开始训练。')
            exported_ids = self.export_approved_to_manifest(include_auto_approved)
            job_id = uuid.uuid4().hex[:12]
            trained_at = now_iso()
            with self.store.lock:
                for item_id in exported_ids:
                    item = self.store.state['items'].get(item_id)
                    if not item or item.get('status') != 'approved':
                        continue
                    item['status'] = 'trained'
                    item['trained_at'] = trained_at
                    item['train_job_id'] = job_id
                self.store.save()
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
                'exported_count': len(exported_ids),
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
    aside{padding:10px;overflow:auto}.activity{border-left:3px solid #2ea043;background:#111b18;padding:10px;margin-bottom:10px}.activityHead{display:flex;align-items:center;justify-content:space-between;gap:8px}.activityBadge{color:#7ee787;font-size:12px;font-weight:700}.activityFlow{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:9px 0}.activityStep{min-width:0;color:#c9d1d9;font-size:12px;line-height:1.35}.activityStep b{display:block;color:#7ee787;font:700 12px Consolas,monospace;margin-bottom:2px}.activityNote{margin:0;color:#8b949e;font-size:12px;line-height:1.5}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}.pair{font-weight:800}.ok{color:#3fb950}.bad{color:#ff7b72}.yellow{color:#f2cc60}.animalGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.animalGrid button.active{background:var(--blue);border-color:#58a6ff}.thumbs{display:flex;gap:6px;overflow:auto;padding:7px 0}.thumb{border:2px solid var(--line);border-radius:8px;background:#0d1117;padding:2px;cursor:pointer;position:relative}.thumb.active{border-color:#58a6ff}.thumb.done:after{content:"";position:absolute;right:3px;top:3px;width:9px;height:9px;border-radius:50%;background:#3fb950}.thumb img{display:block;width:58px;height:36px;object-fit:cover;border-radius:5px}.hint{font-size:13px;color:#c9d1d9;line-height:1.65}.kbd{background:#30363d;border-radius:4px;padding:1px 5px;font-family:Consolas,monospace}.status{white-space:pre-wrap;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px;min-height:70px;color:var(--muted)}.rewardModal{position:fixed;inset:0;z-index:80;display:none;place-items:center;background:rgba(0,0,0,.72);padding:18px}.rewardModal.show{display:grid}.rewardDialog{width:min(520px,100%);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;box-shadow:0 24px 64px rgba(0,0,0,.45)}.rewardDialog h2{margin:0 0 8px;font-size:18px}.rewardDialog input{width:100%;font-family:Consolas,monospace}.rewardDialog p{color:var(--muted);margin:8px 0}
    @media(max-width:980px){body{overflow:auto}main{height:auto;grid-template-columns:1fr}.viewer{min-height:420px}#captcha{max-height:70vh}}
  </style>
</head>
<body>
<header><span class="brand">SameObject 训练标注</span><input id="fileInput" type="file" accept="image/*" multiple style="max-width:250px"><button id="uploadBtn" class="primary">上传</button><button id="loadPlatformBtn">领取任务</button><button id="claimRewardBtn">领取奖励</button><button id="guideBtn">规则说明</button><button id="prevBtn">上一张 A</button><button id="nextBtn">下一张 D</button><button id="submitBtn" class="success" disabled>提交整轮 Ctrl+Enter</button><span id="progress" class="meta">未上传</span><span style="flex:1"></span><a href="/admin">管理员</a></header>
<main><section class="panel stage"><div class="topline"><div><b id="title">等待上传</b> <span id="fileMeta" class="meta"></span></div><div class="pair" id="pairText">未选择</div></div><div class="viewer"><div id="imageWrap" class="imageWrap"><img id="captcha" alt="当前标注图片"></div></div></section><aside class="panel"><section class="activity"><div class="activityHead"><b>标注奖励活动</b><span class="activityBadge">每轮均可参与</span></div><div class="activityFlow"><div class="activityStep"><b>01</b>完成整轮标注</div><div class="activityStep"><b>02</b>通过后台初筛</div><div class="activityStep"><b>03</b>领取 <span id="answerRewardHours">24</span> 小时 Key</div></div><p class="activityNote">整轮未被判为乱填，即可获得临时识别 API Key。</p></section><div class="row"><b>动物类别</b><span id="animalText" class="yellow">未选择</span></div><div id="animalGrid" class="animalGrid"></div><div class="row"><input id="animalOther" placeholder="自定义类别" style="flex:1;min-width:0"><button id="applyOtherBtn">应用</button></div><div class="row"><button id="clearPairBtn">清空当前</button><button id="refreshBtn" disabled>刷新初筛</button></div><div class="thumbs" id="thumbs"></div><div class="hint"><p><b>快捷键</b>：<span class="kbd">1-8</span> 选答案格，<span class="kbd">A/←</span> 上一张，<span class="kbd">D/→</span> 下一张，<span class="kbd">Enter</span> 下一张，<span class="kbd">Ctrl+Enter</span> 提交整轮。</p><p>每轮最多 10 张；所有图片都选好两个格子并填写类别后才可提交。</p></div><div id="status" class="status">请选择图片上传。</div></aside></main><div id="rewardModal" class="rewardModal" role="dialog" aria-modal="true"><div class="rewardDialog"><h2 id="rewardTitle">获得临时识别 API Key</h2><p id="rewardExpiry"></p><input id="rewardKey" readonly aria-label="API Key"><div class="row"><button id="copyRewardBtn" class="primary">复制 Key</button><button id="closeRewardBtn">关闭</button></div></div></div><div id="guideModal" class="rewardModal" role="dialog" aria-modal="true" aria-labelledby="guideTitle"><div class="rewardDialog"><h2 id="guideTitle">任务规则与奖励</h2><p>1. 上传 1-10 张九重妖楼截图，逐张选择两张相同动物所在格子，并填写动物类别。</p><p>2. 完成整轮后提交；系统会进行初筛。页面会自动查询结果，也可点击“领取奖励”继续查询。</p><p>3. 初筛通过后，领取一次性展示的临时识别 API Key。Key 有效期为 <span id="answerRewardHoursGuide">24</span> 小时，请立即复制保存。</p><p>平台没有待标注任务时，点击“领取任务”可领取一枚 <span id="noTaskRewardHoursGuide">2</span> 小时临时识别 API Key。</p><div class="row"><button id="closeGuideBtn" class="primary">开始任务</button></div></div></div><div id="noticeModal" class="rewardModal" role="dialog" aria-modal="true"><div class="rewardDialog"><h2 id="noticeTitle"></h2><p id="noticeMessage"></p><div class="row"><button id="closeNoticeBtn" class="primary">知道了</button></div></div></div>
<script>
const animals = ['', '野猪','熊','豹子','蜘蛛','鹿','羊','牛','狼','袋鼠','不确定']; const PLATFORM_BATCH_KEY='sameobject_platform_labeling_batch', REWARD_BATCH_KEY='sameobject_reward_batch', GUIDE_SEEN_KEY='sameobject_task_guide_seen'; let batch=null, idx=0; const answers=new Map(); const $=id=>document.getElementById(id);
function setStatus(t,cls=''){ $('status').className='status '+cls; $('status').textContent=t; } function esc(s){ return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); } function chinaTime(value){ const date=new Date(value); return Number.isNaN(date.getTime())?'时间未知':new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date); } async function api(url,opts={}){ const r=await fetch(url,opts); const d=await r.json().catch(()=>({})); if(!r.ok||d.code) throw new Error(d.message||r.statusText); return d.data??d; } function showNotice(title,message){ $('noticeTitle').textContent=title; $('noticeMessage').textContent=message; $('noticeModal').classList.add('show'); } function hideNotice(){ $('noticeModal').classList.remove('show'); } function showApiKeyReward(reward,title='获得临时识别 API Key'){ localStorage.removeItem(REWARD_BATCH_KEY); hideNotice(); $('rewardTitle').textContent=title; $('rewardKey').value=reward.api_key; $('rewardExpiry').textContent='有效至：'+chinaTime(reward.expires_at); $('rewardModal').classList.add('show'); } function showGuide(){ $('guideModal').classList.add('show'); } function hideGuide(){ $('guideModal').classList.remove('show'); localStorage.setItem(GUIDE_SEEN_KEY,'1'); } async function copyReward(){ try{ await navigator.clipboard.writeText($('rewardKey').value); }catch(e){ $('rewardKey').select(); document.execCommand('copy'); } $('copyRewardBtn').textContent='已复制'; } $('copyRewardBtn').onclick=copyReward; $('closeRewardBtn').onclick=()=>$('rewardModal').classList.remove('show'); $('guideBtn').onclick=showGuide; $('closeGuideBtn').onclick=hideGuide; $('closeNoticeBtn').onclick=hideNotice; async function pollReward(batchId,attempt=0){ try{ const d=await api('/api/submissions/'+encodeURIComponent(batchId)); if(d.api_key_reward){ showApiKeyReward(d.api_key_reward); return; } if(['queued','screening'].includes(d.status)&&attempt<120){ setTimeout(()=>pollReward(batchId,attempt+1),3000); }else{ localStorage.removeItem(REWARD_BATCH_KEY); showNotice('领取结果',d.status==='screened'?'本轮未获得奖励。':'奖励领取未成功。'); } }catch(e){ localStorage.removeItem(REWARD_BATCH_KEY); showNotice('领取失败','奖励查询失败，请稍后重试。'); } } $('claimRewardBtn').onclick=()=>{ const batchId=localStorage.getItem(REWARD_BATCH_KEY); if(!batchId){showNotice('领取结果','当前浏览器没有待领取的奖励。');return;} showNotice('正在领取奖励','正在核验本轮提交，请稍候。'); pollReward(batchId); };
function current(){ return batch?.items[idx]; } function ans(id){ if(!answers.has(id)) answers.set(id,{pair:[],animal:''}); return answers.get(id); } function completeItem(it){ const a=answers.get(it.id); return a&&a.pair.length===2&&a.animal; } function complete(){ return batch&&batch.items.every(completeItem); }
function renderAnimals(){ const grid=$('animalGrid'); grid.innerHTML=''; const a=current()?ans(current().id):{animal:''}; for(const animal of animals.filter(Boolean)){ const b=document.createElement('button'); b.textContent=animal; b.className=a.animal===animal?'active':''; b.onclick=()=>{ if(!current())return; ans(current().id).animal=animal; render(); }; grid.appendChild(b); } }
function renderThumbs(){ const root=$('thumbs'); root.innerHTML=''; if(!batch)return; batch.items.forEach((it,i)=>{ const d=document.createElement('button'); d.className='thumb '+(i===idx?'active ':'')+(completeItem(it)?'done':''); d.title=it.filename; d.innerHTML=`<img alt="${esc(it.filename)}" src="${it.image_url}">`; d.onclick=()=>{idx=i;render();}; root.appendChild(d); }); }
function render(){ const it=current(); $('submitBtn').disabled=!complete(); $('refreshBtn').disabled=!batch; renderAnimals(); renderThumbs(); if(!it){ $('title').textContent='等待上传'; $('fileMeta').textContent=''; $('captcha').removeAttribute('src'); $('pairText').textContent='未选择'; $('animalText').textContent='未选择'; $('progress').textContent='未上传'; clearBoxes(); return; } const a=ans(it.id); $('title').textContent=`#${idx+1}/${batch.items.length} ${it.filename}`; $('fileMeta').textContent=`${it.size[0]}×${it.size[1]} · ${it.box_source}`; $('pairText').textContent=a.pair.length===2?`答案 [${a.pair.join(', ')}]`:a.pair.length?`[${a.pair.join(', ')}] 还差 ${2-a.pair.length} 个`:'未选择'; $('pairText').className='pair '+(a.pair.length===2?'ok':'bad'); $('animalText').textContent=a.animal||'未选择'; $('animalText').className=a.animal?'ok':'yellow'; $('progress').textContent=`${idx+1}/${batch.items.length}，完成 ${batch.items.filter(completeItem).length}/${batch.items.length}`; const img=$('captcha'); img.onload=drawBoxes; if(img.getAttribute('src')!==it.image_url) img.src=it.image_url; else drawBoxes(); }
function clearBoxes(){ document.querySelectorAll('.box').forEach(x=>x.remove()); } function drawBoxes(){ clearBoxes(); const it=current(); if(!it)return; const img=$('captcha'), wrap=$('imageWrap'), a=ans(it.id); const sx=img.clientWidth/it.size[0], sy=img.clientHeight/it.size[1]; it.boxes.forEach((b,i)=>{ const n=i+1, div=document.createElement('button'); div.type='button'; div.className='box '+(a.pair.includes(n)?'selected':''); div.style.left=(b[0]*sx)+'px'; div.style.top=(b[1]*sy)+'px'; div.style.width=(b[2]*sx)+'px'; div.style.height=(b[3]*sy)+'px'; div.innerHTML=`<span class="num">${n}</span>${a.pair.includes(n)&&a.animal?`<span class="tag">${esc(a.animal)}</span>`:''}`; div.onclick=()=>toggle(n); wrap.appendChild(div); }); }
function toggle(n){ const it=current(); if(!it)return; const a=ans(it.id); if(a.pair.includes(n)) a.pair=a.pair.filter(x=>x!==n); else { if(a.pair.length>=2)a.pair.shift(); a.pair.push(n); a.pair.sort((x,y)=>x-y); } render(); } function move(d){ if(!batch)return; idx=(idx+d+batch.items.length)%batch.items.length; render(); }
$('uploadBtn').onclick=async()=>{ const files=[...$('fileInput').files]; if(files.length<1||files.length>10){setStatus('请选择 1-10 张图片。','bad');return;} const fd=new FormData(); files.forEach(f=>fd.append('images',f)); $('uploadBtn').disabled=true; setStatus('上传并切格中...'); try{ batch=await api('/api/uploads',{method:'POST',body:fd}); idx=0; answers.clear(); render(); setStatus('上传完成，按 1-8 勾选答案。','ok'); }catch(e){setStatus(e.message,'bad');}finally{$('uploadBtn').disabled=false;} };
function usePlatformBatch(d,msg){ batch=d; idx=0; answers.clear(); localStorage.setItem(PLATFORM_BATCH_KEY,d.batch_id); render(); setStatus(msg,'ok'); } async function restorePlatformBatch(){ const batchId=localStorage.getItem(PLATFORM_BATCH_KEY); if(!batchId)return; try{ const d=await api('/api/submissions/'+encodeURIComponent(batchId)); if(d.source==='identify_feedback'&&d.status==='labeling'){ usePlatformBatch(d,'已恢复未完成的平台标注批次。'); }else{ localStorage.removeItem(PLATFORM_BATCH_KEY); } }catch(e){ localStorage.removeItem(PLATFORM_BATCH_KEY); } }
$('loadPlatformBtn').onclick=async()=>{ $('loadPlatformBtn').disabled=true; try{ const savedId=localStorage.getItem(PLATFORM_BATCH_KEY); if(savedId){ const saved=await api('/api/submissions/'+encodeURIComponent(savedId)).catch(()=>null); if(saved?.source==='identify_feedback'&&saved.status==='labeling'){ usePlatformBatch(saved,'已恢复未完成的平台标注批次。'); return; } localStorage.removeItem(PLATFORM_BATCH_KEY); } const d=await api('/api/platform-labeling/next'); if(!d.batch_id){setStatus('暂时无领取任务，感谢给予训练做出贡献，特下发测试 Key。','yellow'); showApiKeyReward(d.no_task_api_key_reward,'暂时无领取任务，感谢给予训练做出贡献特下发测试 Key'); return;} usePlatformBatch(d,`已领取 ${d.items.length} 张任务，请在 30 分钟内完成标注。`); }catch(e){setStatus(e.message,'bad');}finally{$('loadPlatformBtn').disabled=false;} };
$('prevBtn').onclick=()=>move(-1); $('nextBtn').onclick=()=>move(1); $('clearPairBtn').onclick=()=>{ const it=current(); if(!it)return; answers.set(it.id,{pair:[],animal:ans(it.id).animal}); render(); }; $('applyOtherBtn').onclick=()=>{ const it=current(), v=$('animalOther').value.trim(); if(it&&v){ ans(it.id).animal=v; render(); }};
$('submitBtn').onclick=async()=>{ if(!complete())return; const payload={answers:batch.items.map(it=>({item_id:it.id,pair:ans(it.id).pair,animal:ans(it.id).animal}))}; $('submitBtn').disabled=true; setStatus('提交中...'); try{ const d=await api(`/api/submissions/${batch.batch_id}/submit`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); if(batch.source==='identify_feedback')localStorage.removeItem(PLATFORM_BATCH_KEY); setStatus(d.status==='approved'?'平台反馈已标注，进入待训练数据。':'提交成功，后台初筛中。','ok'); if(d.api_key_reward){showApiKeyReward(d.api_key_reward);}else{ localStorage.setItem(REWARD_BATCH_KEY,batch.batch_id); pollReward(batch.batch_id); } }catch(e){setStatus(e.message,'bad'); render();} };
$('refreshBtn').onclick=async()=>{ if(!batch)return; try{ const d=await api(`/api/submissions/${batch.batch_id}`); setStatus('批次状态：'+d.status+'\n'+d.items.map(x=>`${x.filename}: ${x.status}${x.screen?.reason?' - '+x.screen.reason:''}`).join('\n'),'yellow'); }catch(e){setStatus(e.message,'bad');} };
function formatHours(seconds){const hours=Number(seconds)/3600;return Number.isInteger(hours)?String(hours):String(hours).replace(/\.0+$/,'');} async function loadRewardConfig(){try{const r=await fetch('/api/config');const d=await r.json();const c=d.data||{};$('answerRewardHours').textContent=formatHours(c.answer_api_key_ttl_seconds);$('answerRewardHoursGuide').textContent=formatHours(c.answer_api_key_ttl_seconds);$('noTaskRewardHoursGuide').textContent=formatHours(c.no_task_api_key_ttl_seconds);}catch(e){}} window.addEventListener('resize',drawBoxes); document.addEventListener('keydown',e=>{ if(e.key==='Escape'){hideGuide();return;} if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return; if(e.key>='1'&&e.key<='8')toggle(Number(e.key)); else if(e.key==='ArrowLeft'||e.key.toLowerCase()==='a')move(-1); else if(e.key==='ArrowRight'||e.key.toLowerCase()==='d')move(1); else if(e.key==='Enter'&&e.ctrlKey){e.preventDefault();$('submitBtn').click();} else if(e.key==='Enter'){e.preventDefault();move(1);} }); render(); loadRewardConfig(); restorePlatformBatch(); if(!localStorage.getItem(GUIDE_SEEN_KEY))showGuide(); const recoveryRewardBatch=new URLSearchParams(location.search).get('reward_batch'); if(recoveryRewardBatch){ localStorage.setItem(REWARD_BATCH_KEY,recoveryRewardBatch); history.replaceState({},'',location.pathname); } const pendingRewardBatch=localStorage.getItem(REWARD_BATCH_KEY); if(pendingRewardBatch)pollReward(pendingRewardBatch);
</script></body></html>
"""

ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SameObject 管理员审核</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#1f6feb;--green:#238636}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Segoe UI,Arial,sans-serif}a{color:#58a6ff}.muted{color:var(--muted)}
button,input,select,textarea{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit;min-height:34px}
button{cursor:pointer}button:hover{background:#30363d}.primary{background:var(--blue)}.success{background:var(--green)}.danger{background:#8e1519}.warnBtn{background:#9e6a03}.ok{color:#3fb950}.bad{color:#ff7b72}.yellow{color:#f2cc60}
#loginView{min-height:100vh;display:grid;place-items:center;padding:18px}.login{width:min(420px,100%);background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.login h1{margin:0 0 10px;font-size:20px}.login input{width:100%;margin:10px 0}.login button{width:100%}#appView{display:none}
header{position:sticky;top:0;z-index:20;background:#161b22;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;padding:8px 12px;flex-wrap:wrap}
.brand{font-weight:800;font-size:16px}.spacer{flex:1}.topSwitch{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#0d1117}
.topSwitch input{min-height:auto;width:16px;height:16px;accent-color:#238636}main{padding:10px;display:grid;gap:10px}
.stats{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:8px}.stat,.panel,.item{background:var(--panel);border:1px solid var(--line);border-radius:10px}
.stat{padding:9px 11px}.stat b{display:block;font-size:22px}.panel{padding:10px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.list{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}.item{padding:9px}
.captchaPreview,.zoomPreview{position:relative;background:#05080c;border-radius:8px;overflow:hidden}.captchaPreview{height:178px}
.captchaPreview img,.zoomPreview img{display:block;width:100%;height:100%;object-fit:contain;cursor:zoom-in}
.zoomPreview{max-width:96vw;max-height:92vh;width:min(1100px,96vw);height:min(820px,92vh)}.zoomPreview img{cursor:default}
.imageSelections{position:absolute;inset:0;pointer-events:none}.imageSelection{position:absolute;border:3px solid;box-shadow:0 0 0 1px rgba(0,0,0,.7);font-size:11px;font-weight:700}
.imageSelection.user{border-color:#3fb950}.imageSelection.model{border-color:#58a6ff;border-style:dashed}
.imageSelection .selectionTag{position:absolute;padding:3px 5px;border-radius:4px;color:#fff;text-shadow:0 1px 1px #000;white-space:nowrap}
.imageSelection.user .selectionTag{left:-3px;top:-3px;background:#238636}.imageSelection.model .selectionTag{right:-3px;bottom:-3px;background:#1f6feb}
.selectionLegend{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px;font-size:12px}.selectionLegend span{display:inline-flex;align-items:center;gap:4px}
.selectionLegend i{display:inline-block;width:12px;height:12px;border:2px solid}.selectionLegend .userMark{border-color:#3fb950}.selectionLegend .modelMark{border-color:#58a6ff;border-style:dashed}
.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#334155}.badge.needs_admin{background:#854d0e}.badge.approved{background:#166534}.badge.trained{background:#0e7490}
.badge.rejected,.badge.auto_rejected{background:#991b1b}.badge.screening{background:#1d4ed8}.small{font-size:12px}.item input{width:92px}
.item textarea{width:100%;min-height:54px;margin-top:6px;background:#0d1117}.status{white-space:pre-wrap;background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px;color:var(--muted);max-height:220px;overflow:auto}
.pager{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:space-between}
#zoom{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.86);display:none;align-items:center;justify-content:center;padding:22px;flex-direction:column;gap:10px}
#zoom .zoomLegend{color:#e6edf3;background:rgba(13,17,23,.85);border:1px solid var(--line);border-radius:8px;padding:8px 10px}#zoom button{position:fixed;right:18px;top:14px}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}.list{grid-template-columns:1fr}}
</style></head><body>

<section id="loginView"><div class="login"><h1>管理员登录</h1><p class="muted">请输入训练 Web 管理员 Key。成功后只保存在当前浏览器会话中。</p>
<input id="loginKey" type="password" placeholder="TRAINING_WEB_ADMIN_KEY" autofocus>
<button id="loginBtn" class="primary">进入审核区</button><p id="loginMsg" class="bad"></p>
<p><a href="/">返回用户提交页</a> · <a href="/keys">卡密管理</a></p></div></section>
<section id="appView"><header>
<span class="brand">管理员审核</span>
<button id="refreshBtn">刷新</button>
<label>状态 <select id="statusFilter"><option value="review">待审核</option><option value="in_progress">处理中</option><option value="ready">待训练</option><option value="trained">已训练</option><option value="rejected">已拒绝</option><option value="all">全部</option></select></label>
<label class="topSwitch"><input id="includeAutoApproved" type="checkbox" checked><span>含自动通过</span></label>
<span class="spacer"></span><a href="/keys">卡密管理</a><button id="logoutBtn">退出</button>
</header>
<main>
<section id="stats" class="stats"></section>
<section class="panel"><div class="row">
<label>无任务默认 <input id="noTaskRewardHours" type="number" min="0.25" max="720" step="0.25" style="width:90px"> 小时</label>
<label>答题奖励 <input id="answerRewardHours" type="number" min="0.25" max="720" step="0.25" style="width:90px"> 小时</label>
<button id="saveRewardConfigBtn" class="primary">保存有效期</button><span id="rewardConfigStatus" class="muted"></span>
</div><p class="small muted">保存后仅影响新生成的 Key；可设置范围为 0.25 至 720 小时（30 天）。</p></section>
<section class="panel"><div class="row">
<label>run-name <input id="runName" value="web_review_parts_v1" style="width:170px"></label>
<label>feature <select id="featureMode"><option value="parts_v1">parts_v1</option><option value="processed_v2">processed_v2</option></select></label>
<label>epochs <input id="epochs" type="number" min="1" value="180" style="width:90px"></label>
<label>val <input id="valRatio" type="number" min="0.05" max="0.5" step="0.05" value="0.2" style="width:90px"></label>
<button id="trainBtn" class="success">开始训练</button><button id="trainStatusBtn">训练状态</button>
</div><pre id="trainStatus" class="status">尚未启动训练。</pre></section>
<section class="panel"><div class="pager">
<div class="row"><span id="pageInfo" class="muted">共 0 条</span>
<label>每页 <select id="pageSize"><option value="10">10</option><option value="20" selected>20</option><option value="50">50</option></select></label></div>
<div class="row"><button id="prevPageBtn">上一页</button><span id="pageNum" class="muted">1 / 1</span><button id="nextPageBtn">下一页</button></div>
</div></section>
<section id="list" class="list"></section>
</main></section>
<div id="zoom"><button id="zoomClose">关闭</button>
<div class="zoomPreview" id="zoomPreview" data-preview=""><img id="zoomImg" alt="大图预览"><div class="imageSelections" id="zoomSelections" aria-hidden="true"></div></div>
<div class="zoomLegend selectionLegend" id="zoomLegend"></div></div>
<script>

const KEY='sameobject_admin_key';
const $=id=>document.getElementById(id);
let adminKey=sessionStorage.getItem(KEY)||'';
let currentPage=1;
let totalPages=1;
let zoomItemId='';
const adminItems=new Map();
const statusLabels={needs_admin:'待审核',uploaded:'待标注',labeling:'标注中',queued:'待初筛',screening:'初筛中',approved:'待训练',trained:'已训练',auto_rejected:'自动拦截',rejected:'已拒绝'};
function esc(s){return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function headers(){return {'Content-Type':'application/json','X-Admin-Key':adminKey};}
function formatHours(seconds){const hours=Number(seconds)/3600;return Number.isInteger(hours)?String(hours):String(hours).replace(/\.0+$/,'');}
function chinaTime(value){
  if(!value) return '时间未知';
  const raw=String(value).trim();
  if(!raw) return '时间未知';
  const hasTz=/([zZ]|[+-]\d{2}:?\d{2})$/.test(raw);
  const date=hasTz?new Date(raw):new Date(raw.replace(' ','T')+'+08:00');
  if(Number.isNaN(date.getTime())) return raw;
  const text=new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date);
  return text;
}
async function api(url,opts={}){
  opts.headers={...(opts.headers||{}),...headers()};
  const r=await fetch(url,opts);
  const d=await r.json().catch(()=>({}));
  if(!r.ok||d.code){if(r.status===401)showLogin('Key 不正确或已失效。');throw new Error(d.message||r.statusText);}
  return d.data??d;
}
function showLogin(msg=''){ $('loginView').style.display='grid'; $('appView').style.display='none'; $('loginMsg').textContent=msg; }
async function loadConfig(){
  try{
    const r=await fetch('/api/config');
    const d=await r.json(),c=d.data||{};
    if(typeof c.include_auto_approved_in_training==='boolean') $('includeAutoApproved').checked=c.include_auto_approved_in_training;
    $('noTaskRewardHours').value=formatHours(c.no_task_api_key_ttl_seconds);
    $('answerRewardHours').value=formatHours(c.answer_api_key_ttl_seconds);
  }catch(e){}
}
async function showApp(){ $('loginView').style.display='none'; $('appView').style.display='block'; await loadConfig(); loadList().catch(e=>showLogin(e.message)); }
$('loginBtn').onclick=async()=>{adminKey=$('loginKey').value.trim(); if(!adminKey){$('loginMsg').textContent='请输入 Key。';return;} try{await api('/api/admin/stats'); sessionStorage.setItem(KEY,adminKey); showApp();}catch(e){$('loginMsg').textContent=e.message;}};
$('loginKey').addEventListener('keydown',e=>{if(e.key==='Enter')$('loginBtn').click();});
$('logoutBtn').onclick=()=>{sessionStorage.removeItem(KEY);adminKey='';showLogin('已退出。');};
function badge(s){return `<span class="badge ${esc(s)}">${esc(statusLabels[s]||s)}</span>`;}
function prediction(it){const result=it.screen||it.platform_feedback||{}; return {pair:Array.isArray(result.pred_pair)?result.pred_pair:[],animal:result.pred_animal||'',confidence:result.confidence,reason:result.reason||''};}
function validPair(pair){return (pair||[]).filter(n=>Number.isInteger(Number(n))&&Number(n)>=1&&Number(n)<=8).map(Number);}
function drawSelectionsOn(preview, it){
  if(!preview||!it||!it.size?.[0]||!it.size?.[1]) return;
  const image=preview.querySelector('img');
  const layer=preview.querySelector('.imageSelections');
  if(!image||!layer||!image.complete||!image.naturalWidth) return;
  const width=image.clientWidth,height=image.clientHeight,ratio=it.size[0]/it.size[1];
  const shownWidth=Math.min(width,height*ratio),shownHeight=shownWidth/ratio,left=(width-shownWidth)/2,top=(height-shownHeight)/2;
  const add=(n,type,label)=>{
    const box=it.boxes?.[n-1]; if(!box) return;
    const [x,y,w,h]=box,el=document.createElement('div');
    el.className='imageSelection '+type;
    el.style.left=(left+x/it.size[0]*shownWidth)+'px';
    el.style.top=(top+y/it.size[1]*shownHeight)+'px';
    el.style.width=(w/it.size[0]*shownWidth)+'px';
    el.style.height=(h/it.size[1]*shownHeight)+'px';
    el.innerHTML=`<span class="selectionTag">${esc(label)} #${n}</span>`;
    layer.appendChild(el);
  };
  layer.replaceChildren();
  validPair(it.pair||[]).forEach(n=>add(n,'user','用户'));
  validPair(prediction(it).pair).forEach(n=>add(n,'model','识别'));
}
function drawSelections(id){
  const it=adminItems.get(id);
  document.querySelectorAll(`[data-preview="${CSS.escape(id)}"]`).forEach(preview=>drawSelectionsOn(preview,it));
}
function drawAllSelections(){adminItems.forEach((_,id)=>drawSelections(id)); if(zoomItemId) drawSelections(zoomItemId);}
async function loadStats(){
  const d=await api('/api/admin/stats');
  const count=(...statuses)=>statuses.reduce((total,status)=>total+(d.counts[status]||0),0);
  const stats=[['待审核',count('needs_admin')],['处理中',count('uploaded','labeling','queued','screening')],['待训练',count('approved')],['已训练',count('trained')],['已拒绝',count('auto_rejected','rejected')]];
  $('stats').innerHTML=stats.map(([label,total])=>`<div class="stat"><span class="muted">${label}</span><b>${total}</b></div>`).join('');
}
function updatePager(meta){
  currentPage=meta.page||1; totalPages=meta.total_pages||1;
  $('pageInfo').textContent=`共 ${meta.total||0} 条`;
  $('pageNum').textContent=`${currentPage} / ${totalPages}`;
  $('prevPageBtn').disabled=currentPage<=1; $('nextPageBtn').disabled=currentPage>=totalPages;
}
async function loadList(){
  await loadStats();
  const st=$('statusFilter').value; const pageSize=$('pageSize').value;
  const d=await api(`/api/admin/items?status=${encodeURIComponent(st)}&page=${currentPage}&page_size=${encodeURIComponent(pageSize)}`);
  adminItems.clear(); (d.items||[]).forEach(it=>adminItems.set(it.id,it)); updatePager(d);
  $('list').innerHTML=(d.items||[]).map(renderItem).join('');
  document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>review(b.dataset.id,b.dataset.action));
  document.querySelectorAll('[data-zoom-id]').forEach(img=>img.onclick=()=>zoom(img.dataset.zoomId));
  document.querySelectorAll('#list [data-preview]').forEach(preview=>{const image=preview.querySelector('img'); image.onload=()=>drawSelections(preview.dataset.preview); if(image.complete) drawSelections(preview.dataset.preview);});
}
function renderItem(it){
  const p=prediction(it),pair=(it.pair||[]).join(','),predPair=p.pair.join(','),hasPrediction=p.pair.length>0;
  return `<article class="item">
    <div class="row" style="justify-content:space-between"><b title="${esc(it.filename)}">${esc(it.filename)}</b>${badge(it.status)}</div>
    <div class="captchaPreview" data-preview="${esc(it.id)}"><img data-zoom-id="${esc(it.id)}" src="${it.image_url}" alt="审核图片 ${esc(it.filename)}" loading="lazy"><div class="imageSelections" aria-hidden="true"></div></div>
    <div class="selectionLegend"><span><i class="userMark"></i>用户 [${esc(pair)||'未标注'}] ${esc(it.animal||'')}</span>${hasPrediction?`<span><i class="modelMark"></i>识别 [${esc(predPair)}] ${esc(p.animal)}</span>`:''}</div>
    <div class="small muted">来源：${esc(it.source||'user_upload')} · 状态：${esc(statusLabels[it.status]||it.status)} · 创建：${esc(chinaTime(it.created_at))} ${it.submitted_at?'· 提交：'+esc(chinaTime(it.submitted_at)):''}${it.trained_at?' · 训练：'+esc(chinaTime(it.trained_at)):''}</div>
    <div class="small ${it.screen?.ok===false?'bad':'yellow'}">${esc(p.reason||'无初筛信息')} ${hasPrediction?`预测=[${esc(predPair)}] ${esc(p.animal)} score=${esc(p.confidence)}`:''}</div>
    <div class="row"><label>pair <input id="pair-${esc(it.id)}" value="${esc(pair)}"></label><label>animal <input id="animal-${esc(it.id)}" value="${esc(it.animal||'')}"></label></div>
    <textarea id="note-${esc(it.id)}" placeholder="备注">${esc(it.review_note||'')}</textarea>
    <div class="row"><button class="success" data-action="approve" data-id="${esc(it.id)}">通过</button><button class="danger" data-action="reject" data-id="${esc(it.id)}">拒绝</button><button class="warnBtn" data-action="needs_admin" data-id="${esc(it.id)}">待审</button></div>
  </article>`;
}
async function review(id,decision){
  const pair=document.getElementById('pair-'+id).value.split(',').map(x=>Number(x.trim())).filter(Boolean);
  const animal=document.getElementById('animal-'+id).value.trim();
  const note=document.getElementById('note-'+id).value.trim();
  try{await api('/api/admin/items/'+id+'/review',{method:'POST',body:JSON.stringify({decision,pair,animal,note})}); await loadList();}catch(e){alert(e.message);}
}
$('saveRewardConfigBtn').onclick=async()=>{
  const status=$('rewardConfigStatus'); status.textContent='保存中...';
  try{
    const c=await api('/api/admin/reward-config',{method:'POST',body:JSON.stringify({no_task_reward_hours:$('noTaskRewardHours').value,answer_reward_hours:$('answerRewardHours').value})});
    $('noTaskRewardHours').value=formatHours(c.no_task_api_key_ttl_seconds);
    $('answerRewardHours').value=formatHours(c.answer_api_key_ttl_seconds);
    status.textContent='已保存，新 Key 立即按此有效期发放。'; status.className='ok';
  }catch(e){status.textContent=e.message; status.className='bad';}
};
$('refreshBtn').onclick=()=>{currentPage=1; loadList();};
$('statusFilter').onchange=()=>{currentPage=1; loadList();};
$('pageSize').onchange=()=>{currentPage=1; loadList();};
$('prevPageBtn').onclick=()=>{if(currentPage>1){currentPage-=1; loadList();}};
$('nextPageBtn').onclick=()=>{if(currentPage<totalPages){currentPage+=1; loadList();}};
$('trainBtn').onclick=async()=>{
  const p={run_name:$('runName').value.trim(),feature_mode:$('featureMode').value,epochs:Number($('epochs').value),val_ratio:Number($('valRatio').value),include_auto_approved:$('includeAutoApproved').checked};
  $('trainStatus').textContent='启动中...';
  try{const d=await api('/api/admin/train',{method:'POST',body:JSON.stringify(p)}); $('trainStatus').textContent=JSON.stringify(d,null,2); await loadList();}catch(e){$('trainStatus').textContent=e.message;}
};
$('trainStatusBtn').onclick=async()=>{try{const d=await api('/api/admin/train-status'); $('trainStatus').textContent=JSON.stringify(d,null,2);}catch(e){$('trainStatus').textContent=e.message;}};
function zoom(id){
  const it=adminItems.get(id); if(!it) return;
  zoomItemId=id;
  const preview=$('zoomPreview');
  preview.dataset.preview=id;
  $('zoomImg').src=it.image_url;
  const p=prediction(it),pair=(it.pair||[]).join(','),predPair=p.pair.join(',');
  $('zoomLegend').innerHTML=`<span><i class="userMark"></i>用户 [${esc(pair)||'未标注'}] ${esc(it.animal||'')}</span>${p.pair.length?`<span><i class="modelMark"></i>识别 [${esc(predPair)}] ${esc(p.animal)}</span>`:''}`;
  $('zoom').style.display='flex';
  const image=$('zoomImg');
  const redraw=()=>drawSelectionsOn(preview,it);
  image.onload=redraw; if(image.complete) redraw(); requestAnimationFrame(redraw);
}
function closeZoom(){
  $('zoom').style.display='none'; $('zoomImg').removeAttribute('src');
  $('zoomSelections').replaceChildren(); $('zoomLegend').innerHTML='';
  zoomItemId=''; $('zoomPreview').dataset.preview='';
}
$('zoomClose').onclick=closeZoom;
$('zoom').onclick=e=>{if(e.target.id==='zoom') closeZoom();};
window.addEventListener('resize',drawAllSelections);
document.addEventListener('keydown',e=>{if(e.key==='Escape') closeZoom();});
if(adminKey){showApp();}else{showLogin();}
</script></body></html>
"""

KEYS_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>卡密管理</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#1f6feb;--green:#238636}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",Arial,sans-serif}a{color:#58a6ff}.muted{color:var(--muted)}
button,input,select,textarea{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit;min-height:34px}
button{cursor:pointer}button:hover{background:#30363d}.primary{background:var(--blue)}.success{background:var(--green)}.danger{background:#8e1519}.ok{color:#3fb950}.bad{color:#ff7b72}
#loginView{min-height:100vh;display:grid;place-items:center;padding:18px}.login{width:min(420px,100%);background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.login h1{margin:0 0 10px;font-size:20px}.login input{width:100%;margin:10px 0}.login button{width:100%}#appView{display:none}
header{position:sticky;top:0;z-index:10;background:#161b22;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;padding:8px 12px;flex-wrap:wrap}
.brand{font-weight:800}.spacer{flex:1}main{padding:12px;display:grid;gap:12px}.tabs{display:flex;gap:8px}.tab{border-radius:999px}.tab.active{background:var(--blue);border-color:#58a6ff}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top;font-size:13px}th{color:var(--muted);font-weight:600}
.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#334155}.badge.unused{background:#1d4ed8}.badge.active{background:#166534}.badge.expired{background:#6b7280}
.code{font-family:Consolas,"Courier New",monospace;word-break:break-all;user-select:all}.notice{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:8px;white-space:pre-wrap}
.pager{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center}
.filters{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}
.filters input[type="search"]{min-width:220px;flex:1}
.modalMask{position:fixed;inset:0;background:rgba(0,0,0,.72);display:none;align-items:center;justify-content:center;z-index:50;padding:16px}
.modalMask.show{display:flex}.modal{width:min(460px,100%);background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 20px 60px rgba(0,0,0,.45)}
.modal h3{margin:0 0 12px;font-size:18px}.modal .field{display:grid;gap:6px;margin-bottom:10px}.modal .field input,.modal .field textarea{width:100%}
.modal .actions{display:flex;gap:8px;justify-content:flex-end;margin-top:8px}
.copyBtn{padding:4px 8px;min-height:28px;font-size:12px}
</style></head><body>
<section id="loginView"><div class="login"><h1>卡密管理登录</h1><p class="muted">使用训练 Web 管理员 Key 进入。</p>
<input id="loginKey" type="password" placeholder="TRAINING_WEB_ADMIN_KEY" autofocus>
<button id="loginBtn" class="primary">进入</button><p id="loginMsg" class="bad"></p><p><a href="/admin">返回审核页</a></p></div></section>
<section id="appView">
<header>
  <span class="brand">卡密管理 /keys</span>
  <div class="tabs">
    <button class="tab active" data-tab="license">登录卡密</button>
    <button class="tab" data-tab="api">识别 API Key</button>
  </div>
  <span class="spacer"></span>
  <a href="/admin">管理员审核</a>
  <button id="logoutBtn">退出</button>
</header>
<main>
  <section id="panelLicense" class="panel">
    <div class="row" style="justify-content:space-between">
      <h2 style="margin:0;font-size:16px">登录卡密（外系统机器码绑定）</h2>
      <button id="openLicenseModalBtn" class="success">新增卡密</button>
    </div>
    <div class="filters">
      <input id="licenseQ" type="search" placeholder="搜索卡密 / 机器码 / 备注 / ID">
      <label>状态
        <select id="licenseStatus">
          <option value="all">全部</option>
          <option value="unused">未使用</option>
          <option value="active">已激活</option>
          <option value="expired">已过期</option>
        </select>
      </label>
      <button id="refreshLicenseBtn">查询</button>
    </div>
    <pre id="licenseNotice" class="notice muted">公开校验：POST /api/license/verify
用户验证/解绑：X-API-Key=登录卡密；body 含 machine_code（可参与签名）；通用 Key=环境变量 SAMEOBJECT_API_KEY</pre>
    <div class="pager" style="margin:10px 0">
      <span id="licensePageInfo" class="muted">共 0 条</span>
      <div class="row">
        <label>每页 <select id="licensePageSize"><option value="10">10</option><option value="20" selected>20</option><option value="50">50</option></select></label>
        <button id="licensePrev">上一页</button>
        <span id="licensePageNum" class="muted">1 / 1</span>
        <button id="licenseNext">下一页</button>
      </div>
    </div>
    <div style="overflow:auto"><table>
      <thead><tr><th>卡密</th><th>状态</th><th>机器限制</th><th>机器码</th><th>创建时间</th><th>到期时间</th><th>验证次数</th><th>上次解绑</th><th>备注</th><th>操作</th></tr></thead>
      <tbody id="licenseBody"></tbody>
    </table></div>
  </section>

  <section id="panelApi" class="panel" style="display:none">
    <h2 style="margin:0 0 10px;font-size:16px">识别 API Key（本训练系统）</h2>
    <div class="row">
      <label>有效小时 <input id="apiTtlHours" type="number" min="0.25" max="720" step="0.25" value="24" style="width:100px"></label>
      <label>备注 <input id="apiNote" style="width:220px" placeholder="可选"></label>
      <button id="createApiBtn" class="success">新增 Key</button>
      <button id="refreshApiBtn">查询</button>
    </div>
    <div class="filters">
      <input id="apiQ" type="search" placeholder="搜索卡密 / ID / 用途 / 备注">
      <label>状态
        <select id="apiStatus">
          <option value="all">全部</option>
          <option value="active">有效</option>
        </select>
      </label>
    </div>
    <pre id="apiNotice" class="notice muted">后台新增的 Key 会在列表中展示明文；奖励发放类历史 Key 仅存哈希，无法回显。</pre>
    <div class="pager" style="margin:10px 0">
      <span id="apiPageInfo" class="muted">共 0 条</span>
      <div class="row">
        <label>每页 <select id="apiPageSize"><option value="10">10</option><option value="20" selected>20</option><option value="50">50</option></select></label>
        <button id="apiPrev">上一页</button>
        <span id="apiPageNum" class="muted">1 / 1</span>
        <button id="apiNext">下一页</button>
      </div>
    </div>
    <div style="overflow:auto"><table>
      <thead><tr><th>卡密</th><th>ID</th><th>用途</th><th>状态</th><th>创建时间</th><th>到期时间</th><th>备注</th><th>操作</th></tr></thead>
      <tbody id="apiBody"></tbody>
    </table></div>
  </section>
</main>
</section>

<div class="modalMask" id="editLicenseModal">
  <div class="modal">
    <h3>修改登录卡密</h3>
    <p class="muted" style="margin:0 0 10px;font-size:12px">卡密本身不可修改。ID：<code id="editLicenseIdText"></code></p>
    <div class="field"><label>卡密（只读）</label><input id="editLicenseCode" readonly></div>
    <div class="field"><label>状态</label>
      <select id="editLicenseStatus">
        <option value="unused">未使用</option>
        <option value="active">已激活</option>
        <option value="expired">已过期</option>
      </select>
    </div>
    <div class="field"><label>机器码（留空表示解绑）</label><input id="editLicenseMachine" placeholder="留空清除绑定"></div>
    <div class="field"><label>到期时间（北京时间）</label><input id="editLicenseExpiresAt" type="datetime-local"></div>
    <div class="field"><label>验证次数</label><input id="editLicenseVerifyCount" type="number" min="0" step="1"></div>
    <div class="field"><label>备注</label><input id="editLicenseNote" placeholder="可选"></div>
    <label class="row" style="margin:0 0 10px"><input id="editLicenseNoMachineLimit" type="checkbox" style="width:auto;min-height:auto"> 不限制机器码（任意机器可登录）</label>
    <div class="actions">
      <button id="cancelEditLicenseBtn">取消</button>
      <button id="saveEditLicenseBtn" class="primary">保存修改</button>
    </div>
  </div>
</div>

<div class="modalMask" id="licenseModal">
  <div class="modal">
    <h3>新增登录卡密</h3>
    <div class="field"><label>数量</label><input id="licenseCount" type="number" min="1" max="100" value="1"></div>
    <div class="field"><label>到期时间</label><input id="licenseExpiresAt" type="datetime-local"></div>
    <div class="field"><label>备注</label><input id="licenseNote" placeholder="可选"></div>
    <label class="row" style="margin:0 0 10px"><input id="licenseNoMachineLimit" type="checkbox" style="width:auto;min-height:auto"> 不限制机器码（任意机器可登录）</label>
    <p class="muted" style="margin:0 0 8px;font-size:12px">默认首次验证绑定机器码；勾选「不限制机器码」后任意机器可登录。</p>
    <div class="actions">
      <button id="cancelLicenseModalBtn">取消</button>
      <button id="createLicenseBtn" class="success">确认发卡</button>
    </div>
  </div>
</div>

<script>
const KEY='sameobject_admin_key';
const $=id=>document.getElementById(id);
let adminKey=sessionStorage.getItem(KEY)||'';
let licensePage=1, licenseTotalPages=1, apiPage=1, apiTotalPages=1;
let licenseCache={};
const licenseStatusLabel={unused:'未使用',active:'已激活',expired:'已过期',void:'已作废'};
const apiStatusLabel={active:'有效',void:'已作废'};
const purposeLabel={manual_admin:'后台新增',answer_reward:'答题奖励',no_task_reward:'无任务奖励',unknown:'未知'};

function esc(s){return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function headers(){return {'Content-Type':'application/json','X-Admin-Key':adminKey};}
function chinaTime(value){
  if(!value) return '-';
  const raw=String(value).trim();
  const hasTz=/([zZ]|[+-]\d{2}:?\d{2})$/.test(raw);
  const date=hasTz?new Date(raw):new Date(raw.replace(' ','T')+'+08:00');
  if(Number.isNaN(date.getTime())) return raw;
  return new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(date);
}
function defaultExpiresLocal(){
  const d=new Date(Date.now()+30*24*3600*1000);
  // datetime-local uses local machine time; admin is expected in China TZ
  const pad=n=>String(n).padStart(2,'0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
async function api(url,opts={}){
  opts.headers={...(opts.headers||{}),...headers()};
  const r=await fetch(url,opts);
  const d=await r.json().catch(()=>({}));
  if(!r.ok||d.code){if(r.status===401)showLogin('Key 不正确或已失效。');throw new Error(d.message||r.statusText);}
  return d.data??d;
}
function showLogin(msg=''){$('loginView').style.display='grid';$('appView').style.display='none';$('loginMsg').textContent=msg;}
function showApp(){$('loginView').style.display='none';$('appView').style.display='block'; loadLicense(); loadApi();}
async function copyText(text){
  try{await navigator.clipboard.writeText(text);}catch(e){const ta=document.createElement('textarea');ta.value=text;document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();}
}
function codeCell(code){
  if(!code) return '<span class="muted">明文未保留</span>';
  return `<span class="code">${esc(code)}</span> <button class="copyBtn" data-copy="${esc(code)}">复制</button>`;
}
$('loginBtn').onclick=async()=>{adminKey=$('loginKey').value.trim(); if(!adminKey){$('loginMsg').textContent='请输入 Key。';return;} try{await api('/api/admin/stats'); sessionStorage.setItem(KEY,adminKey); showApp();}catch(e){$('loginMsg').textContent=e.message;}};
$('loginKey').addEventListener('keydown',e=>{if(e.key==='Enter')$('loginBtn').click();});
$('logoutBtn').onclick=()=>{sessionStorage.removeItem(KEY);adminKey='';showLogin('已退出。');};
document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const tab=btn.dataset.tab;
  $('panelLicense').style.display=tab==='license'?'block':'none';
  $('panelApi').style.display=tab==='api'?'block':'none';
});

function bindCopyButtons(root){
  root.querySelectorAll('[data-copy]').forEach(btn=>btn.onclick=async()=>{
    await copyText(btn.dataset.copy);
    const old=btn.textContent; btn.textContent='已复制'; setTimeout(()=>btn.textContent=old,1000);
  });
}

async function loadLicense(){
  const st=$('licenseStatus').value, size=$('licensePageSize').value, q=$('licenseQ').value.trim();
  const d=await api(`/api/admin/license-keys?status=${encodeURIComponent(st)}&page=${licensePage}&page_size=${size}&q=${encodeURIComponent(q)}`);
  licensePage=d.page||1; licenseTotalPages=d.total_pages||1;
  $('licensePageInfo').textContent=`共 ${d.total||0} 条`;
  $('licensePageNum').textContent=`${licensePage} / ${licenseTotalPages}`;
  $('licensePrev').disabled=licensePage<=1; $('licenseNext').disabled=licensePage>=licenseTotalPages;
  licenseCache={}; (d.items||[]).forEach(it=>{if(it&&it.id) licenseCache[it.id]=it;});
  $('licenseBody').innerHTML=(d.items||[]).map(it=>`<tr>
    <td>${codeCell(it.code)}</td>
    <td><span class="badge ${esc(it.status)}">${esc(licenseStatusLabel[it.status]||it.status)}</span></td>
    <td>${it.no_machine_limit?'<span class="badge active">不限制</span>':'<span class="badge">限制本机</span>'}</td>
    <td class="code">${esc(it.machine_code||'-')}</td>
    <td>${esc(chinaTime(it.created_at))}</td>
    <td>${esc(chinaTime(it.expires_at))}</td>
    <td>${esc(it.verify_count||0)}</td>
    <td>${esc(chinaTime(it.last_unbind_at))}</td>
    <td>${esc(it.note||'')}</td>
    <td class="row" style="gap:6px">
      <button data-edit-license="${esc(it.id)}">修改</button>
      <button class="danger" data-del-license="${esc(it.id)}">删除</button>
    </td>
  </tr>`).join('')||'<tr><td colspan="10" class="muted">暂无卡密</td></tr>';
  bindCopyButtons($('licenseBody'));
  document.querySelectorAll('[data-edit-license]').forEach(btn=>btn.onclick=()=>openEditLicenseModal(btn.dataset.editLicense));
  document.querySelectorAll('[data-del-license]').forEach(btn=>btn.onclick=async()=>{
    if(!confirm('确认硬删除该登录卡密？此操作不可恢复。')) return;
    try{await api('/api/admin/license-keys/'+btn.dataset.delLicense+'/void',{method:'POST',body:'{}'}); await loadLicense();}catch(e){alert(e.message);}
  });
}

async function loadApi(){
  const size=$('apiPageSize').value, q=$('apiQ').value.trim(), st=$('apiStatus').value;
  const d=await api(`/api/admin/issued-api-keys?page=${apiPage}&page_size=${size}&q=${encodeURIComponent(q)}&status=${encodeURIComponent(st)}`);
  apiPage=d.page||1; apiTotalPages=d.total_pages||1;
  $('apiPageInfo').textContent=`共 ${d.total||0} 条`;
  $('apiPageNum').textContent=`${apiPage} / ${apiTotalPages}`;
  $('apiPrev').disabled=apiPage<=1; $('apiNext').disabled=apiPage>=apiTotalPages;
  $('apiBody').innerHTML=(d.items||[]).map(it=>`<tr>
    <td>${codeCell(it.api_key)}</td>
    <td class="code">${esc(it.id)}</td>
    <td>${esc(purposeLabel[it.purpose]||it.purpose||'')}</td>
    <td><span class="badge ${esc(it.status)}">${esc(apiStatusLabel[it.status]||it.status)}</span></td>
    <td>${esc(chinaTime(it.created_at))}</td>
    <td>${esc(chinaTime(it.expires_at))}</td>
    <td>${esc(it.note||'')}</td>
    <td><button class="danger" data-del-api="${esc(it.id)}">删除</button></td>
  </tr>`).join('')||'<tr><td colspan="8" class="muted">暂无 Key</td></tr>';
  bindCopyButtons($('apiBody'));
  document.querySelectorAll('[data-del-api]').forEach(btn=>btn.onclick=async()=>{
    if(!confirm('确认硬删除该识别 API Key？此操作不可恢复。')) return;
    try{await api('/api/admin/issued-api-keys/'+btn.dataset.delApi+'/void',{method:'POST',body:'{}'}); await loadApi();}catch(e){alert(e.message);}
  });
}

function toDatetimeLocalChina(value){
  if(!value) return '';
  const raw=String(value).trim();
  const hasTz=/([zZ]|[+-]\d{2}:?\d{2})$/.test(raw);
  const date=hasTz?new Date(raw):new Date(raw.replace(' ','T')+'+08:00');
  if(Number.isNaN(date.getTime())) return '';
  const parts=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).formatToParts(date);
  const get=t=>(parts.find(p=>p.type===t)||{}).value||'';
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}`;
}
function openLicenseModal(){
  $('licenseExpiresAt').value=defaultExpiresLocal();
  $('licenseCount').value='1';
  $('licenseNote').value='';
  $('licenseNoMachineLimit').checked=false;
  $('licenseModal').classList.add('show');
}
function closeLicenseModal(){ $('licenseModal').classList.remove('show'); }
function openEditLicenseModal(id){
  const it=licenseCache[id];
  if(!it){alert('未找到该卡密，请刷新后重试。'); return;}
  $('editLicenseIdText').textContent=it.id||'';
  $('editLicenseCode').value=it.code||'';
  $('editLicenseStatus').value=['unused','active','expired'].includes(it.status)?it.status:'unused';
  $('editLicenseMachine').value=it.machine_code||'';
  $('editLicenseExpiresAt').value=toDatetimeLocalChina(it.expires_at);
  $('editLicenseVerifyCount').value=String(it.verify_count||0);
  $('editLicenseNote').value=it.note||'';
  $('editLicenseNoMachineLimit').checked=!!it.no_machine_limit;
  $('editLicenseModal').dataset.editId=id;
  $('editLicenseModal').classList.add('show');
}
function closeEditLicenseModal(){ $('editLicenseModal').classList.remove('show'); delete $('editLicenseModal').dataset.editId; }
$('openLicenseModalBtn').onclick=openLicenseModal;
$('cancelLicenseModalBtn').onclick=closeLicenseModal;
$('licenseModal').onclick=e=>{if(e.target.id==='licenseModal') closeLicenseModal();};
$('cancelEditLicenseBtn').onclick=closeEditLicenseModal;
$('editLicenseModal').onclick=e=>{if(e.target.id==='editLicenseModal') closeEditLicenseModal();};
$('saveEditLicenseBtn').onclick=async()=>{
  try{
    const id=$('editLicenseModal').dataset.editId;
    if(!id){alert('缺少卡密 ID'); return;}
    const expiresAt=$('editLicenseExpiresAt').value;
    if(!expiresAt){alert('请选择到期时间'); return;}
    if(!confirm('确认保存对该登录卡密的修改？')) return;
    await api('/api/admin/license-keys/'+encodeURIComponent(id)+'/update',{method:'POST',body:JSON.stringify({
      status:$('editLicenseStatus').value,
      machine_code:$('editLicenseMachine').value.trim(),
      expires_at:expiresAt,
      verify_count:Number($('editLicenseVerifyCount').value||0),
      note:$('editLicenseNote').value,
      no_machine_limit:$('editLicenseNoMachineLimit').checked
    })});
    closeEditLicenseModal();
    await loadLicense();
  }catch(e){alert(e.message);}
};

$('createLicenseBtn').onclick=async()=>{
  try{
    const expiresAt=$('licenseExpiresAt').value;
    if(!expiresAt){alert('请选择到期时间'); return;}
    const d=await api('/api/admin/license-keys',{method:'POST',body:JSON.stringify({
      count:Number($('licenseCount').value),
      expires_at:expiresAt,
      note:$('licenseNote').value,
      no_machine_limit:$('licenseNoMachineLimit').checked
    })});
    const codes=(d.created||[]).map(x=>x.code).join('\n');
    $('licenseNotice').textContent='已生成 '+d.count+' 张卡密：\n'+codes;
    closeLicenseModal();
    licensePage=1; await loadLicense();
  }catch(e){alert(e.message);}
};
$('createApiBtn').onclick=async()=>{
  try{
    const ttl=Number($('apiTtlHours').value);
    const note=($('apiNote').value||'').trim();
    const noteText=note?('，备注：'+note):'';
    if(!confirm('确认新增识别 API Key？\n有效小时：'+ttl+noteText+'\n创建后请立即复制保存。')) return;
    const d=await api('/api/admin/issued-api-keys',{method:'POST',body:JSON.stringify({ttl_hours:ttl,note:$('apiNote').value})});
    $('apiNotice').textContent='已创建 Key：\n'+d.api_key+'\n有效至：'+chinaTime(d.expires_at);
    apiPage=1; await loadApi();
  }catch(e){$('apiNotice').textContent=e.message;}
};

$('refreshLicenseBtn').onclick=()=>{licensePage=1; loadLicense();};
$('refreshApiBtn').onclick=()=>{apiPage=1; loadApi();};
$('licenseStatus').onchange=()=>{licensePage=1; loadLicense();};
$('apiStatus').onchange=()=>{apiPage=1; loadApi();};
$('licensePageSize').onchange=()=>{licensePage=1; loadLicense();};
$('apiPageSize').onchange=()=>{apiPage=1; loadApi();};
$('licenseQ').addEventListener('keydown',e=>{if(e.key==='Enter'){licensePage=1; loadLicense();}});
$('apiQ').addEventListener('keydown',e=>{if(e.key==='Enter'){apiPage=1; loadApi();}});
$('licensePrev').onclick=()=>{if(licensePage>1){licensePage-=1; loadLicense();}};
$('licenseNext').onclick=()=>{if(licensePage<licenseTotalPages){licensePage+=1; loadLicense();}};
$('apiPrev').onclick=()=>{if(apiPage>1){apiPage-=1; loadApi();}};
$('apiNext').onclick=()=>{if(apiPage<apiTotalPages){apiPage+=1; loadApi();}};
if(adminKey){showApp();}else{showLogin();}
</script></body></html>
"""

class TrainingWebHandler(BaseHTTPRequestHandler):
    server_version = 'JiuChongYaolouTrainingWeb/1.0'

    store: StateStore
    worker: ScreeningWorker
    trainer: TrainingRunner
    admin_key: str
    license_api_secret: str = ''

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

    def read_body_bytes(self, max_bytes: int = 2 * 1024 * 1024) -> bytes:
        length = int(self.headers.get('Content-Length', '0'))
        if length <= 0 or length > max_bytes:
            raise ValueError('JSON 请求体大小不合法。')
        return self.rfile.read(length)

    def read_json(self) -> dict:
        raw = self.read_body_bytes()
        try:
            payload = json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError('JSON 解析失败。') from exc
        if not isinstance(payload, dict):
            raise ValueError('请求体必须是 JSON 对象。')
        return payload

    def extract_api_key(self) -> str:
        candidate = (self.headers.get('X-API-Key') or '').strip()
        if not candidate:
            authorization = self.headers.get('Authorization', '')
            if authorization.lower().startswith('bearer '):
                candidate = authorization[7:].strip()
        return candidate

    def require_identify_api_key(self) -> bool:
        """识别接口鉴权：SAMEOBJECT_API_KEY（通用 Key）或后台发放的识别 Key。"""
        candidate = self.extract_api_key()
        if not candidate:
            self.error(HTTPStatus.UNAUTHORIZED, '缺少 API Key（X-API-Key 或 Authorization: Bearer）。')
            return False
        ok = any(hmac.compare_digest(candidate, key) for key in identify_api_keys())
        if not ok:
            try:
                ok = bool(self.store.is_issued_api_key_valid(candidate))
            except Exception:
                ok = False
        if not ok:
            self.error(HTTPStatus.UNAUTHORIZED, 'API Key 不存在或已经失效。')
            return False
        return True

    def extract_license_card_key(self) -> str:
        """卡密接口：X-API-Key / Bearer 即为登录卡密（通用 Key 或普通用户卡）。"""
        return self.extract_api_key()

    def require_license_signature(self, method: str, path: str, raw_body: bytes) -> bool:
        """可选 HMAC 防重放：仅使用 LICENSE_API_SECRET。失败时已写响应。"""
        if license_sign_optional():
            return True
        secret = (self.license_api_secret or license_api_secret()).strip()
        if not secret:
            self.error(HTTPStatus.SERVICE_UNAVAILABLE, '服务端未配置 LICENSE_API_SECRET，无法校验请求签名。')
            return False

        timestamp = (self.headers.get('X-Timestamp') or self.headers.get('X-License-Timestamp') or '').strip()
        nonce = (self.headers.get('X-Nonce') or self.headers.get('X-License-Nonce') or '').strip()
        signature = (self.headers.get('X-Signature') or self.headers.get('X-License-Signature') or '').strip().lower()
        if not timestamp or not nonce or not signature:
            self.error(HTTPStatus.UNAUTHORIZED, '缺少签名头：X-Timestamp / X-Nonce / X-Signature。')
            return False
        try:
            ts = int(timestamp)
        except ValueError:
            self.error(HTTPStatus.UNAUTHORIZED, 'X-Timestamp 必须是 Unix 秒级时间戳。')
            return False
        now_ts = int(time.time())
        skew = license_sign_skew_seconds()
        if abs(now_ts - ts) > skew:
            self.error(HTTPStatus.UNAUTHORIZED, f'请求时间戳超出允许范围（±{skew} 秒）。')
            return False
        if not re.fullmatch(r'[0-9a-f]{64}', signature):
            self.error(HTTPStatus.UNAUTHORIZED, 'X-Signature 格式不正确。')
            return False

        expected = sign_license_request(secret, method, path, str(ts), nonce, raw_body or b'')
        if not hmac.compare_digest(expected, signature):
            self.error(HTTPStatus.UNAUTHORIZED, '签名校验失败。')
            return False
        try:
            self.store.consume_license_nonce(nonce, now_ts)
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
            return False
        except PermissionError as exc:
            self.error(HTTPStatus.CONFLICT, str(exc))
            return False
        return True

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
            elif path == '/keys':
                self.send_bytes(HTTPStatus.OK, KEYS_HTML.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/healthz':
                self.ok({'ok': True})
            elif path == '/api/config':
                self.ok({
                    'max_images_per_batch': MAX_IMAGES_PER_BATCH,
                    'animals': ANIMALS,
                    **self.store.reward_ttl_config(),
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
            elif path == '/api/admin/license-keys':
                if self.require_admin():
                    self.handle_admin_license_keys_list(parsed)
            elif path == '/api/admin/issued-api-keys':
                if self.require_admin():
                    self.handle_admin_issued_api_keys_list(parsed)
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
            elif path == '/api/license/verify':
                self.handle_license_verify()
            elif path == '/api/license/unbind':
                self.handle_license_unbind()
            elif path.startswith('/api/admin/items/') and path.endswith('/review'):
                if self.require_admin():
                    self.handle_admin_review(path)
            elif path == '/api/admin/train':
                if self.require_admin():
                    self.handle_admin_train()
            elif path == '/api/admin/reward-config':
                if self.require_admin():
                    self.handle_admin_reward_config()
            elif path == '/api/admin/license-keys':
                if self.require_admin():
                    self.handle_admin_license_keys_create()
            elif path.startswith('/api/admin/license-keys/') and path.endswith('/void'):
                if self.require_admin():
                    self.handle_admin_license_keys_void(path)
            elif path.startswith('/api/admin/license-keys/') and path.endswith('/update'):
                if self.require_admin():
                    self.handle_admin_license_keys_update(path)
            elif path == '/api/admin/issued-api-keys':
                if self.require_admin():
                    self.handle_admin_issued_api_keys_create()
            elif path.startswith('/api/admin/issued-api-keys/') and path.endswith('/void'):
                if self.require_admin():
                    self.handle_admin_issued_api_keys_void(path)
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
        api_key_reward = None
        if batch.get('source') == 'identify_feedback':
            self.store.issue_batch_api_key_reward(batch_id)
            api_key_reward = self.store.claim_batch_api_key_reward(batch_id)
        else:
            self.worker.queue_batch(batch_id)
        response = {'batch_id': batch_id, 'status': batch['status']}
        if api_key_reward:
            response['api_key_reward'] = api_key_reward
        self.ok(response)

    def handle_get_platform_labeling(self) -> None:
        batch = self.store.claim_platform_labeling_batch()
        if not batch:
            self.ok({
                'batch_id': None,
                'items': [],
                'no_task_api_key_reward': self.store.issue_no_task_api_key_reward(),
            })
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
        api_key_reward = self.store.claim_batch_api_key_reward(batch_id)
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                self.error(HTTPStatus.NOT_FOUND, '批次不存在。')
                return
            items = [self.public_item(self.store.state['items'][item_id]) for item_id in batch['item_ids']]
            data = dict(batch)
            data.pop('api_key_reward', None)
            data['batch_id'] = batch['id']
            data['items'] = items
            if api_key_reward:
                data['api_key_reward'] = api_key_reward
        self.ok(data)

    def public_item(self, item: dict) -> dict:
        data = dict(item)
        with self.store.lock:
            batch = self.store.state['batches'].get(str(item.get('batch_id') or '')) or {}
            source_batch = self.store.state['batches'].get(str(item.get('source_batch_id') or '')) or {}
        if not data.get('source'):
            data['source'] = batch.get('source') or source_batch.get('source') or 'user_upload'
        data['batch_status'] = batch.get('status', '')
        data['batch_id'] = item.get('batch_id', '')
        data['image_url'] = f"/media/item/{item['id']}"
        return data

    def handle_admin_items(self, parsed) -> None:
        query = parse_qs(parsed.query)
        status = (query.get('status') or ['review'])[0]
        page, page_size = parse_page_params(query)
        with self.store.lock:
            items = list(self.store.state['items'].values())
        status_groups = {
            'review': {'needs_admin'},
            'in_progress': {'uploaded', 'labeling', 'queued', 'screening'},
            'ready': {'approved'},
            'trained': {'trained'},
            'rejected': {'auto_rejected', 'rejected'},
        }
        if status in status_groups:
            items = [item for item in items if item.get('status') in status_groups[status]]
        elif status != 'all':
            raise ValueError('status 必须是 review/in_progress/ready/trained/rejected/all。')
        items.sort(key=lambda item: item.get('created_at', ''), reverse=True)
        page_data = paginate_list(items, page, page_size)
        self.ok({
            'items': [self.public_item(item) for item in page_data['items']],
            'page': page_data['page'],
            'page_size': page_data['page_size'],
            'total': page_data['total'],
            'total_pages': page_data['total_pages'],
        })

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

    def handle_admin_reward_config(self) -> None:
        payload = self.read_json()
        config = self.store.update_reward_ttl_config(
            payload.get('no_task_reward_hours'),
            payload.get('answer_reward_hours'),
        )
        self.ok(config)

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


    def handle_admin_license_keys_list(self, parsed) -> None:
        query = parse_qs(parsed.query)
        status = (query.get('status') or ['all'])[0]
        q = (query.get('q') or [''])[0]
        page, page_size = parse_page_params(query)
        items = self.store.list_license_keys(status=status, q=q)
        page_data = paginate_list(items, page, page_size)
        self.ok(page_data)

    def handle_admin_license_keys_create(self) -> None:
        payload = self.read_json()
        count = int(payload.get('count') or 1)
        note = str(payload.get('note') or '')
        created = self.store.create_license_keys(
            count=count,
            expires_at=payload.get('expires_at'),
            days=payload.get('days'),
            note=note,
            no_machine_limit=bool(payload.get('no_machine_limit')),
        )
        self.ok({'created': created, 'count': len(created)})

    def handle_admin_license_keys_void(self, path: str) -> None:
        parts = [part for part in path.split('/') if part]
        # /api/admin/license-keys/{id}/void
        key_id = unquote(parts[3] if len(parts) >= 5 else parts[-1])
        try:
            record = self.store.void_license_key(key_id)
        except KeyError as exc:
            self.error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self.ok({'item': record})

    def handle_admin_license_keys_update(self, path: str) -> None:
        parts = [part for part in path.split('/') if part]
        # /api/admin/license-keys/{id}/update
        key_id = unquote(parts[3] if len(parts) >= 5 else parts[-1])
        payload = self.read_json()
        try:
            record = self.store.update_license_key(key_id, payload)
        except KeyError as exc:
            self.error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self.ok({'item': record})

    def handle_license_verify(self) -> None:
        # 鉴权：X-API-Key = 登录卡密（通用 SAMEOBJECT_API_KEY 或 /keys 普通卡），不是固定业务 Key
        try:
            raw = self.read_body_bytes()
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not self.require_license_signature('POST', '/api/license/verify', raw):
            return
        try:
            payload = json.loads(raw.decode('utf-8')) if raw else {}
            card_key, machine_code = validate_license_params(
                payload,
                require_machine=True,
                header_card_key=self.extract_license_card_key(),
            )
            data = self.store.verify_license_key(card_key, machine_code)
        except json.JSONDecodeError:
            self.error(HTTPStatus.BAD_REQUEST, 'JSON 解析失败。')
            return
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except PermissionError as exc:
            message = str(exc)
            status = HTTPStatus.FORBIDDEN
            if '不存在' in message:
                status = HTTPStatus.UNAUTHORIZED
            self.error(status, message)
            return
        self.ok(data)

    def handle_license_unbind(self) -> None:
        try:
            raw = self.read_body_bytes()
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if not self.require_license_signature('POST', '/api/license/unbind', raw):
            return
        try:
            payload = json.loads(raw.decode('utf-8')) if raw else {}
            card_key, machine_code = validate_license_params(
                payload,
                require_machine=False,
                header_card_key=self.extract_license_card_key(),
            )
            data = self.store.unbind_license_machine(card_key, machine_code or None)
        except json.JSONDecodeError:
            self.error(HTTPStatus.BAD_REQUEST, 'JSON 解析失败。')
            return
        except ValueError as exc:
            self.error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except PermissionError as exc:
            message = str(exc)
            status = HTTPStatus.FORBIDDEN
            if '不存在' in message:
                status = HTTPStatus.UNAUTHORIZED
            elif '今日已解绑' in message:
                status = HTTPStatus.TOO_MANY_REQUESTS
            self.error(status, message)
            return
        self.ok(data)

    def handle_admin_issued_api_keys_list(self, parsed) -> None:
        query = parse_qs(parsed.query)
        status = (query.get('status') or ['all'])[0]
        q = (query.get('q') or [''])[0]
        page, page_size = parse_page_params(query)
        items = self.store.list_issued_api_keys(status=status, q=q)
        page_data = paginate_list(items, page, page_size)
        self.ok(page_data)

    def handle_admin_issued_api_keys_create(self) -> None:
        payload = self.read_json()
        created = self.store.create_manual_api_key(payload.get('ttl_hours', 24), payload.get('note') or '')
        self.ok(created)

    def handle_admin_issued_api_keys_void(self, path: str) -> None:
        parts = [part for part in path.split('/') if part]
        key_id = parts[3] if len(parts) >= 5 else parts[-1]
        try:
            record = self.store.void_issued_api_key(key_id)
        except KeyError as exc:
            self.error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self.ok({'item': record})

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


def build_handler(
    store: StateStore,
    worker: ScreeningWorker,
    trainer: TrainingRunner,
    admin_key: str,
    license_secret: str = '',
):
    class Handler(TrainingWebHandler):
        pass

    Handler.store = store
    Handler.worker = worker
    Handler.trainer = trainer
    Handler.admin_key = admin_key
    Handler.license_api_secret = (license_secret or license_api_secret()).strip()
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
    secret = license_api_secret()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(store, worker, trainer, args.admin_key, secret),
    )
    print(f'training web listening on http://{args.host}:{args.port}')
    if args.admin_key:
        print('admin auth: enabled')
    else:
        print('admin auth: disabled (set TRAINING_WEB_ADMIN_KEY to enable)')
    if secret:
        print('license API: require X-API-Key (SAMEOBJECT_API_KEY); optional HMAC (LICENSE_API_SECRET)')
    elif license_sign_optional():
        print('license API: require X-API-Key; HMAC optional/off')
    else:
        print('license API: require X-API-Key (SAMEOBJECT_API_KEY); HMAC off (no LICENSE_API_SECRET)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

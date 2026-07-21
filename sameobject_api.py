from __future__ import annotations

import argparse
import re
import time
import base64
import hmac
import json
import os
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / 'tools'))

from cloud_control import get_cloud_control_hub  # noqa: E402

from predict_sameobject_ensemble import (  # noqa: E402
    DEFAULT_FULL_WEIGHT,
    DEFAULT_PARTS_WEIGHT,
    build_encoder,
    load_classifier,
    predict_one,
)


MAX_IMAGE_BYTES = 10 * 1024 * 1024


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


class Identifier:
    def __init__(self, full_weights: Path, parts_weights: Path, parts_weight: float) -> None:
        self.full_model, self.full_animals, self.full_mode = load_classifier(full_weights)
        self.parts_model, self.parts_animals, self.parts_mode = load_classifier(parts_weights)
        self.encoder, self.preprocess = build_encoder()
        self.parts_weight = parts_weight

    def identify(self, image_bytes: bytes) -> dict:
        suffix = '.png'
        if image_bytes[:3] == b'\xff\xd8\xff':
            suffix = '.jpg'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
            temp_path = Path(file.name)
            file.write(image_bytes)
        try:
            return predict_one(
                temp_path,
                self.encoder,
                self.preprocess,
                self.full_model,
                self.full_animals,
                self.full_mode,
                self.parts_model,
                self.parts_animals,
                self.parts_mode,
                self.parts_weight,
                inner_margin=0,
            )
        finally:
            temp_path.unlink(missing_ok=True)


def decode_json_image(body: bytes) -> bytes:
    payload = json.loads(body.decode('utf-8'))
    image = payload.get('img')
    if not isinstance(image, str) or not image.strip():
        raise ValueError('JSON 请求必须包含非空字符串字段 img。')
    image = image.strip()
    if image.startswith('data:'):
        marker = ';base64,'
        if marker not in image:
            raise ValueError('data URL 必须为 base64 格式。')
        image = image.split(marker, 1)[1]
    try:
        return base64.b64decode(image, validate=True)
    except Exception as exc:
        raise ValueError('img 不是有效 base64 图片数据。') from exc


def build_handler(identifier: Identifier, api_key: str, store, training_web_base=None):
    base_class = training_web_base or BaseHTTPRequestHandler

    class IdentifyHandler(base_class):
        server_version = 'JiuChongYaolouIdentifyTest/1.0'

        def log_message(self, format: str, *args) -> None:
            print('%s - %s' % (self.address_string(), format % args))

        def send_json(self, status: HTTPStatus, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)

        def is_authorized(self) -> bool:
            candidate = self.headers.get('X-API-Key', '')
            if not candidate:
                authorization = self.headers.get('Authorization', '')
                if authorization.lower().startswith('bearer '):
                    candidate = authorization[7:].strip()
            return bool(candidate) and (
                hmac.compare_digest(candidate, api_key) or store.is_issued_api_key_valid(candidate)
            )


        def extract_api_key(self) -> str:
            candidate = self.headers.get('X-API-Key', '')
            if not candidate:
                authorization = self.headers.get('Authorization', '')
                if authorization.lower().startswith('bearer '):
                    candidate = authorization[7:].strip()
            return str(candidate or '').strip()

        @staticmethod
        def _safe_int(value, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def read_raw_body(self, max_bytes: int = 256 * 1024) -> bytes:
            content_length = int(self.headers.get('Content-Length', '0') or 0)
            if content_length < 0 or content_length > max_bytes:
                raise ValueError(f'请求体大小不合法（上限 {max_bytes} 字节）。')
            if content_length == 0:
                return b''
            return self.rfile.read(content_length)

        def require_request_signature(self, method: str, path: str, raw_body: bytes) -> None:
            """
            与 /api/license/* 相同的 HMAC 签名校验：
            headers: X-Timestamp / X-Nonce / X-Signature
            string_to_sign = METHOD\\nPATH\\ntimestamp\\nnonce\\nsha256_hex(body)
            secret = LICENSE_API_SECRET
            """
            from sameobject_training_web import (  # noqa: WPS433
                license_api_secret,
                license_nonce_ttl_seconds,
                license_sign_optional,
                license_sign_skew_seconds,
                sign_license_request,
            )

            secret = (license_api_secret() or '').strip()
            if not secret:
                if license_sign_optional():
                    return
                self.send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        'ok': False,
                        'code': 503,
                        'message': '服务端未配置 LICENSE_API_SECRET，拒绝云控接口。',
                    },
                )
                raise PermissionError('__sign_abort__')

            timestamp = (
                self.headers.get('X-Timestamp')
                or self.headers.get('X-License-Timestamp')
                or ''
            ).strip()
            nonce = (
                self.headers.get('X-Nonce')
                or self.headers.get('X-License-Nonce')
                or ''
            ).strip()
            signature = (
                self.headers.get('X-Signature')
                or self.headers.get('X-License-Signature')
                or ''
            ).strip().lower()
            if not timestamp or not nonce or not signature:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        'ok': False,
                        'code': 401,
                        'message': '缺少签名头：X-Timestamp / X-Nonce / X-Signature。',
                    },
                )
                raise PermissionError('__sign_abort__')
            try:
                ts = int(timestamp)
            except ValueError:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {'ok': False, 'code': 401, 'message': 'X-Timestamp 必须是 Unix 秒级时间戳。'},
                )
                raise PermissionError('__sign_abort__')
            now_ts = int(time.time())
            skew = license_sign_skew_seconds()
            if abs(now_ts - ts) > skew:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {
                        'ok': False,
                        'code': 401,
                        'message': f'请求时间戳超出允许范围（±{skew} 秒）。',
                    },
                )
                raise PermissionError('__sign_abort__')
            if not re.fullmatch(r'[0-9a-f]{64}', signature):
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {'ok': False, 'code': 401, 'message': 'X-Signature 格式不正确。'},
                )
                raise PermissionError('__sign_abort__')

            expected = sign_license_request(
                secret, method, path, str(ts), nonce, raw_body or b''
            )
            if not hmac.compare_digest(expected, signature):
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {'ok': False, 'code': 401, 'message': '签名校验失败。'},
                )
                raise PermissionError('__sign_abort__')
            try:
                store.consume_license_nonce(nonce, now_ts)
            except ValueError as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {'ok': False, 'code': 400, 'message': str(exc)},
                )
                raise PermissionError('__sign_abort__') from exc
            except PermissionError as exc:
                self.send_json(
                    HTTPStatus.CONFLICT,
                    {'ok': False, 'code': 409, 'message': str(exc)},
                )
                raise PermissionError('__sign_abort__') from exc

        def handle_cloud_control(self, method: str, path: str) -> None:
            """主控发 / 副控收。鉴权：X-API-Key + 与卡密接口相同的 HMAC 签名。"""
            try:
                parsed = urlparse(self.path)
                if method.upper() == 'POST':
                    raw_body = self.read_raw_body()
                    sign_path = path
                    self.require_request_signature('POST', sign_path, raw_body)
                    if raw_body:
                        try:
                            body = json.loads(raw_body.decode('utf-8'))
                        except Exception as exc:
                            raise ValueError('JSON 解析失败。') from exc
                        if not isinstance(body, dict):
                            raise ValueError('JSON 根节点必须是对象。')
                    else:
                        body = {}
                else:
                    # GET poll: 签名 path 含 query，防止参数被篡改；body 为空
                    raw_body = b''
                    sign_path = path
                    if parsed.query:
                        sign_path = f'{path}?{parsed.query}'
                    self.require_request_signature('GET', sign_path, raw_body)
                    body = {}

                if not self.is_authorized():
                    self.send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {'ok': False, 'code': 401, 'message': 'API Key 不存在或已经失效。'},
                    )
                    return

                key = self.extract_api_key()
                hub = get_cloud_control_hub()

                if method == 'POST' and path == '/api/cloud-control/join':
                    result = hub.join(
                        api_key=key,
                        master_name=str(body.get('master_name') or ''),
                        role=str(body.get('role') or ''),
                        client_id=str(body.get('client_id') or ''),
                        pid=self._safe_int(body.get('pid'), 0),
                    )
                elif method == 'POST' and path == '/api/cloud-control/leave':
                    result = hub.leave(
                        api_key=key,
                        master_name=str(body.get('master_name') or ''),
                        role=str(body.get('role') or ''),
                        client_id=str(body.get('client_id') or ''),
                        session_id=str(body.get('session_id') or ''),
                        pid=self._safe_int(body.get('pid'), 0),
                    )
                elif method == 'POST' and path == '/api/cloud-control/heartbeat':
                    result = hub.heartbeat(
                        api_key=key,
                        master_name=str(body.get('master_name') or ''),
                        role=str(body.get('role') or ''),
                        client_id=str(body.get('client_id') or ''),
                        session_id=str(body.get('session_id') or ''),
                        pid=self._safe_int(body.get('pid'), 0),
                    )
                elif method == 'POST' and path == '/api/cloud-control/publish':
                    event = body.get('event')
                    result = hub.publish(
                        api_key=key,
                        master_name=str(body.get('master_name') or ''),
                        event=event if isinstance(event, dict) else None,
                        client_id=str(body.get('client_id') or ''),
                        session_id=str(body.get('session_id') or ''),
                        msg_id=str(body.get('msg_id') or ''),
                    )
                elif method == 'GET' and path == '/api/cloud-control/poll':
                    from urllib.parse import parse_qs

                    qs = parse_qs(parsed.query)
                    def q1(name: str, default: str = '') -> str:
                        vals = qs.get(name) or []
                        return str(vals[0] if vals else default)

                    wait_raw = q1('wait_s', '25')
                    try:
                        wait_s = float(wait_raw)
                    except ValueError:
                        wait_s = 25.0
                    result = hub.poll(
                        api_key=key,
                        master_name=q1('master_name'),
                        client_id=q1('client_id'),
                        session_id=q1('session_id'),
                        cursor=q1('cursor'),
                        wait_s=wait_s,
                    )
                else:
                    self.send_json(
                        HTTPStatus.NOT_FOUND,
                        {'ok': False, 'code': 404, 'message': 'Not Found'},
                    )
                    return

                code = int(result.get('code') or 0)
                if result.get('ok') is False:
                    status = {
                        400: HTTPStatus.BAD_REQUEST,
                        401: HTTPStatus.UNAUTHORIZED,
                        403: HTTPStatus.FORBIDDEN,
                        404: HTTPStatus.NOT_FOUND,
                        429: HTTPStatus.TOO_MANY_REQUESTS,
                    }.get(code, HTTPStatus.BAD_REQUEST)
                    self.send_json(status, result)
                    return
                self.send_json(HTTPStatus.OK, result)
            except PermissionError as exc:
                if str(exc) == '__sign_abort__':
                    return
                self.send_json(
                    HTTPStatus.FORBIDDEN,
                    {'ok': False, 'code': 403, 'message': str(exc)},
                )
            except ValueError as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {'ok': False, 'code': 400, 'message': str(exc)},
                )
            except Exception as exc:
                print('cloud-control error:', repr(exc))
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {'ok': False, 'code': 500, 'message': '云控处理失败。'},
                )

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, Authorization, X-Timestamp, X-Nonce, X-Signature')
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == '/healthz':
                self.send_json(HTTPStatus.OK, {'ok': True})
            elif path.startswith('/api/cloud-control/'):
                self.handle_cloud_control('GET', path)
            elif path in {'/api/identify/image', '/api/identify/feedback'}:
                self.send_json(HTTPStatus.METHOD_NOT_ALLOWED, {'code': 405, 'message': '该接口只接受 POST 请求。'})
            elif training_web_base is not None:
                super().do_GET()
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {'code': 404, 'message': 'Not Found'})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path.startswith('/api/cloud-control/'):
                self.handle_cloud_control('POST', path)
                return
            if path not in {'/api/identify/image', '/api/identify/feedback'}:
                if training_web_base is not None:
                    super().do_POST()
                    return
                self.send_json(HTTPStatus.NOT_FOUND, {'code': 404, 'message': 'Not Found'})
                return
            if not self.is_authorized():
                self.send_json(HTTPStatus.UNAUTHORIZED, {'code': 401, 'message': 'API Key 不存在或已经失效。'})
                return

            try:
                if path == '/api/identify/feedback':
                    self.handle_feedback()
                    return
                content_length = int(self.headers.get('Content-Length', '0'))
                if not 0 < content_length <= MAX_IMAGE_BYTES * 2:
                    raise ValueError('请求体大小必须在 1 字节到 20 MB 之间。')
                body = self.rfile.read(content_length)
                content_type = self.headers.get('Content-Type', '').lower()
                image_bytes = decode_json_image(body) if 'application/json' in content_type else body
                if not 0 < len(image_bytes) <= MAX_IMAGE_BYTES:
                    raise ValueError('解码后的图片大小必须在 1 字节到 10 MB 之间。')
                result = identifier.identify(image_bytes)
                identify_id = store.create_identification(image_bytes, result)
                response = {
                    'code': 0,
                    'data': {
                        'identify_id': identify_id,
                        'positions': result['best_pair'],
                        'animal': result['best_animal'],
                        'confidence': result['best_score'],
                        'click_centers': result['click_centers'],
                        'top_pairs': result['top_pairs'],
                    },
                }
                self.send_json(HTTPStatus.OK, response)
            except ValueError as exc:
                self.send_json(HTTPStatus.BAD_REQUEST, {'code': 400, 'message': str(exc)})
            except Exception as exc:
                print('identify error:', repr(exc))
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {'code': 500, 'message': '识别失败。'})

        def handle_feedback(self) -> None:
            content_length = int(self.headers.get('Content-Length', '0'))
            if not 0 < content_length <= 64 * 1024:
                raise ValueError('反馈请求体大小必须在 1 字节到 64 KB 之间。')
            payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
            identify_id = str(payload.get('identify_id') or '').strip()
            correct = payload.get('correct')
            if not identify_id:
                raise ValueError('identify_id 不能为空。')
            if not isinstance(correct, bool):
                raise ValueError('correct 必须是布尔值 true 或 false。')
            data = store.report_identification_feedback(identify_id, correct)
            self.send_json(HTTPStatus.OK, {'code': 0, 'data': data})

    return IdentifyHandler


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description='笑傲江湖-九重妖楼小游戏识别测试 HTTP API')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--full-weights', default=str(DEFAULT_FULL_WEIGHT))
    parser.add_argument('--parts-weights', default=str(DEFAULT_PARTS_WEIGHT))
    parser.add_argument('--parts-weight', type=float, default=0.25)
    parser.add_argument(
        '--api-key',
        default=os.environ.get('SAMEOBJECT_API_KEY', ''),
        help='API Key；未传时读取环境变量 SAMEOBJECT_API_KEY。',
    )
    parser.add_argument(
        '--enable-training-web',
        action='store_true',
        default=env_flag('ENABLE_TRAINING_WEB', False),
        help='把训练数据提交/管理员审核 Web 平台合并挂载到同一个端口；也可在 .env 中设置 ENABLE_TRAINING_WEB=1。',
    )
    parser.add_argument(
        '--training-web-admin-key',
        default=os.environ.get('TRAINING_WEB_ADMIN_KEY', ''),
        help='训练 Web 管理员 Key；只保护 /admin 和 /api/admin/*，与 SAMEOBJECT_API_KEY 分开。',
    )
    parser.add_argument(
        '--training-web-reject-confidence',
        type=float,
        default=float(os.environ.get('TRAINING_WEB_REJECT_CONFIDENCE', '0.80')),
        help='初筛高置信度不一致时自动拦截阈值。',
    )
    args = parser.parse_args()
    if not 0.0 <= args.parts_weight <= 1.0:
        raise SystemExit('--parts-weight 必须在 0 到 1 之间。')
    if not args.api_key:
        raise SystemExit('必须通过 --api-key 或环境变量 SAMEOBJECT_API_KEY 设置 API Key。')

    identifier = Identifier(
        Path(args.full_weights).resolve(),
        Path(args.parts_weights).resolve(),
        args.parts_weight,
    )
    from sameobject_training_web import (  # noqa: WPS433
        DATA_DIR,
        STATE_PATH,
        ScreeningWorker,
        StateStore,
        TrainingRunner,
        auto_promote_matched_items,
        build_handler as build_training_web_handler,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = StateStore(STATE_PATH)
    training_web_base = None
    if args.enable_training_web:
        if not args.training_web_admin_key:
            raise SystemExit('启用 --enable-training-web 时必须通过 --training-web-admin-key 或环境变量 TRAINING_WEB_ADMIN_KEY 设置管理员 Key。')
        promoted = auto_promote_matched_items(store)
        if promoted:
            print(f'auto-promoted matched pending items: {promoted}')
        worker = ScreeningWorker(store, args.training_web_reject_confidence, identifier=identifier)
        trainer = TrainingRunner(store)
        from sameobject_training_web import license_api_secret as _license_api_secret  # noqa: WPS433
        training_web_base = build_training_web_handler(
            store,
            worker,
            trainer,
            args.training_web_admin_key,
            _license_api_secret(),
        )

    server = ThreadingHTTPServer((args.host, args.port), build_handler(identifier, args.api_key, store, training_web_base))
    print(f'listening on http://{args.host}:{args.port}')
    print('cloud-control: /api/cloud-control/{join,leave,heartbeat,publish,poll} (X-API-Key, room=master_name)')
    if args.enable_training_web:
        print('training web mounted on same port: /, /admin, /keys')
        print('auth: identify uses SAMEOBJECT_API_KEY or issued keys; /api/license/* uses X-API-Key=login card (or master SAMEOBJECT_API_KEY); /admin,/keys use TRAINING_WEB_ADMIN_KEY; LICENSE_API_SECRET optional HMAC')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

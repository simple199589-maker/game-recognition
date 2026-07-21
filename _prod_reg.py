from pathlib import Path
import os, json, time, secrets, tempfile, shutil, threading, hashlib, hmac
from http.server import ThreadingHTTPServer
from urllib import request, error
import sameobject_training_web as tw

PASS = []
FAIL = []

def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f'[PASS] {name}')
    else:
        FAIL.append((name, detail))
        print(f'[FAIL] {name} :: {detail}')

tmpdir = Path(tempfile.mkdtemp(prefix='prod_reg_'))
tw.STATE_PATH = tmpdir / 'state.json'
tw.STATE_DB_PATH = tmpdir / 'state.db'
tw.DATA_DIR = tmpdir
os.environ['SAMEOBJECT_API_KEY'] = 'prod-api-key-xyz'
os.environ.pop('LICENSE_API_SECRET', None)
os.environ['LICENSE_SIGN_OPTIONAL'] = '1'
os.environ['TRAINING_WEB_ADMIN_KEY'] = 'prod-admin-key'

store = tw.StateStore(tw.STATE_PATH)
check('sqlite db created', tw.STATE_DB_PATH.exists())

# ---- store level ----
limited = store.create_license_keys(1, expires_at='2099-12-31T23:59', note='limited', no_machine_limit=False)[0]
open_key = store.create_license_keys(1, expires_at='2099-12-31T23:59', note='open', no_machine_limit=True)[0]
check('create license keys', limited['code'] and open_key['no_machine_limit'] is True)

store.verify_license_key(limited['code'], 'MACHINE-A-001')
try:
    store.verify_license_key(limited['code'], 'MACHINE-B-001')
    check('limited rejects other machine', False, 'should raise')
except PermissionError as e:
    check('limited rejects other machine', '机器码不匹配' in str(e), str(e))

store.verify_license_key(open_key['code'], 'MACHINE-A-001')
store.verify_license_key(open_key['code'], 'MACHINE-B-001')
check('no_machine_limit allows any machine', True)

# unbind daily limit
store.unbind_license_machine(limited['code'], 'MACHINE-A-001')
store.verify_license_key(limited['code'], 'MACHINE-C-001')
try:
    store.unbind_license_machine(limited['code'], 'MACHINE-C-001')
    check('unbind once/day', False, 'second unbind should fail')
except PermissionError as e:
    check('unbind once/day', '今日已解绑' in str(e), str(e))

# admin update
upd = store.update_license_key(limited['id'], {'no_machine_limit': True, 'note': 'edited'})
check('admin update fields', upd['no_machine_limit'] is True and upd['note'] == 'edited' and upd['code'] == limited['code'])
store.verify_license_key(limited['code'], 'MACHINE-Z-999')
check('after enable no_machine_limit login any machine', True)

# issued api key
manual = store.create_manual_api_key(2, 'manual')
check('issued api key valid', store.is_issued_api_key_valid(manual['api_key']))
store.void_issued_api_key(manual['id'])
check('void issued api key', not store.is_issued_api_key_valid(manual['api_key']))

# reload persistence
store2 = tw.StateStore(tw.STATE_PATH)
check('persist license after reopen', open_key['id'] in store2.state['license_keys'])
check('persist no_machine_limit', store2.state['license_keys'][open_key['id']].get('no_machine_limit') is True)

# ---- HTTP level ----
Handler = tw.build_handler(store2, tw.ScreeningWorker(store2, 0.8), tw.TrainingRunner(store2), 'prod-admin-key', '')
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{port}'

def http(method, path, body=None, headers=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = request.Request(base + path, data=data, method=method)
    if body is not None:
        req.add_header('Content-Type', 'application/json')
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw

# pages
st, html = http('GET', '/keys')
check('GET /keys', st == 200 and isinstance(html, str) and '卡密管理' in html)
st, html = http('GET', '/admin')
check('GET /admin', st == 200 and isinstance(html, str) and '管理员' in html)

# admin auth
st, body = http('GET', '/api/admin/stats')
check('admin without key 401', st == 401)
st, body = http('GET', '/api/admin/stats', headers={'X-Admin-Key': 'prod-admin-key'})
check('admin with key 200', st == 200 and 'counts' in (body.get('data') or body))

# license: X-API-Key = 登录卡密本身
fresh = store2.create_license_keys(1, expires_at='2099-12-31T23:59')[0]
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-0001'})
check('license verify no X-API-Key 400', st == 400, str(body))
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-0001'}, headers={'X-API-Key': 'not-exist-card-key-xx'})
check('license verify unknown card 401', st == 401, str(body))
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-0001'}, headers={'X-API-Key': fresh['code']})
check('license verify card in X-API-Key', st == 200 and body.get('data', {}).get('ok') is True, str(body))
# body.card_key 可与头一致参与签名场景
st, body = http('POST', '/api/license/verify', {'card_key': fresh['code'], 'machine_code': 'HOST-0001'}, headers={'X-API-Key': fresh['code']})
check('license verify header+body card match', st == 200, str(body))
st, body = http('POST', '/api/license/verify', {'card_key': 'OTHER-CARD-XXXX', 'machine_code': 'HOST-0001'}, headers={'X-API-Key': fresh['code']})
check('license verify header/body mismatch 400', st == 400, str(body))
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-OTHER'}, headers={'X-API-Key': fresh['code']})
check('license verify other machine 403', st == 403, str(body))

# unbind http: X-API-Key = card
st, body = http('POST', '/api/license/unbind', {'machine_code': 'HOST-0001'}, headers={'X-API-Key': fresh['code']})
check('license unbind', st == 200 and body.get('data', {}).get('ok') is True, str(body))

# master universal key as X-API-Key
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-MASTER'}, headers={'X-API-Key': 'prod-api-key-xyz'})
check('master key as login card', st == 200 and body.get('data', {}).get('master_key') is True, str(body))

# admin create/update/delete
st, body = http('POST', '/api/admin/license-keys', {
    'count': 1, 'expires_at': '2099-06-01T12:00', 'note': 'reg', 'no_machine_limit': True
}, headers={'X-Admin-Key': 'prod-admin-key'})
check('admin create license', st == 200 and body.get('data', {}).get('count') == 1, str(body))
item = body['data']['created'][0]
st, body = http('POST', f"/api/admin/license-keys/{item['id']}/update", {
    'note': 'reg2', 'no_machine_limit': False, 'status': 'unused'
}, headers={'X-Admin-Key': 'prod-admin-key'})
check('admin update license', st == 200 and body.get('data', {}).get('item', {}).get('note') == 'reg2', str(body))
st, body = http('GET', '/api/admin/license-keys?q=reg2&page=1&page_size=10', headers={'X-Admin-Key': 'prod-admin-key'})
check('admin search license', st == 200 and body.get('data', {}).get('total', 0) >= 1, str(body))
st, body = http('POST', f"/api/admin/license-keys/{item['id']}/void", {}, headers={'X-Admin-Key': 'prod-admin-key'})
check('admin hard delete license', st == 200, str(body))
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-0001'}, headers={'X-API-Key': item['code']})
check('deleted license cannot login', st == 401, str(body))

# issued api key：识别用；不能当登录卡密（除非碰巧格式合法且入库）
st, body = http('POST', '/api/admin/issued-api-keys', {'ttl_hours': 1, 'note': 'tmp'}, headers={'X-Admin-Key': 'prod-admin-key'})
check('admin create issued api key', st == 200 and 'api_key' in body.get('data', {}), str(body))
tmp_key = body['data']['api_key']
tmp_id = body['data']['id']
k2 = store2.create_license_keys(1, expires_at='2099-12-31T23:59')[0]
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-TMP1'}, headers={'X-API-Key': tmp_key})
check('issued identify key cannot login as card', st in (400, 401), str(body))
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-TMP1'}, headers={'X-API-Key': k2['code']})
check('normal card in X-API-Key works', st == 200 and body.get('data', {}).get('ok') is True, str(body))
st, body = http('POST', f'/api/admin/issued-api-keys/{tmp_id}/void', {}, headers={'X-Admin-Key': 'prod-admin-key'})
check('admin delete issued api key', st == 200)

# param validation: bad card format in header
st, body = http('POST', '/api/license/verify', {'machine_code': 'HOST-0001'}, headers={'X-API-Key': 'bad!!'})
check('invalid card format 400', st == 400, str(body))

# optional HMAC path when LICENSE_API_SECRET set
os.environ['LICENSE_API_SECRET'] = 'hmac-secret'
os.environ['LICENSE_SIGN_OPTIONAL'] = '0'
# rebuild handler with secret
server.shutdown()
Handler2 = tw.build_handler(store2, tw.ScreeningWorker(store2, 0.8), tw.TrainingRunner(store2), 'prod-admin-key', 'hmac-secret')
server2 = ThreadingHTTPServer(('127.0.0.1', 0), Handler2)
port2 = server2.server_address[1]
threading.Thread(target=server2.serve_forever, daemon=True).start()
base2 = f'http://127.0.0.1:{port2}'

def http2(path, body_obj, sign=True, api_key=None):
    body = json.dumps(body_obj, ensure_ascii=False, separators=(',', ':')).encode()
    req = request.Request(base2 + path, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('X-API-Key', api_key or body_obj.get('card_key') or 'prod-api-key-xyz')
    if sign:
        ts = str(int(time.time()))
        nonce = secrets.token_hex(12)
        sig = tw.sign_license_request('hmac-secret', 'POST', path, ts, nonce, body)
        req.add_header('X-Timestamp', ts)
        req.add_header('X-Nonce', nonce)
        req.add_header('X-Signature', sig)
    try:
        with request.urlopen(req, timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode())
    except error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

k3 = store2.create_license_keys(1, expires_at='2099-12-31T23:59')[0]
# body 参与签名；X-API-Key=卡密
st, body = http2('/api/license/verify', {'card_key': k3['code'], 'machine_code': 'HOST-SIG1'}, sign=False, api_key=k3['code'])
check('HMAC required when secret set -> 401 without sign', st == 401, str(body))
st, body = http2('/api/license/verify', {'card_key': k3['code'], 'machine_code': 'HOST-SIG1'}, sign=True, api_key=k3['code'])
check('HMAC + card in header success', st == 200 and body.get('data', {}).get('ok') is True, str(body))
# 仅 body.machine_code 也要能签（同机再次校验）
st, body = http2('/api/license/verify', {'machine_code': 'HOST-SIG1'}, sign=True, api_key=k3['code'])
check('HMAC body machine_code only', st == 200, str(body))

server2.shutdown()
shutil.rmtree(tmpdir, ignore_errors=True)

print('\n===== SUMMARY =====')
print(f'PASS: {len(PASS)}')
print(f'FAIL: {len(FAIL)}')
for name, detail in FAIL:
    print(f'  - {name}: {detail}')
if FAIL:
    raise SystemExit(1)
print('ALL REGRESSION CHECKS PASSED - ready for production gate')

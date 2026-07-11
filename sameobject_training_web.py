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
from datetime import datetime
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
TRAIN_LOG_DIR = DATA_DIR / 'train_logs'
STATE_PATH = DATA_DIR / 'state.json'
APPROVED_IMAGE_DIR = ROOT_DIR / 'images' / 'training_web'
ANSWER_PATH = ROOT_DIR / 'datasets' / 'sameobject_corpus' / 'manifests' / 'manual_answers.json'
RUN_ROOT = ROOT_DIR / 'training_runs' / 'sameobject_animal_classifier'

MAX_IMAGES_PER_BATCH = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_REJECT_CONFIDENCE = 0.80
ANIMALS = ['', '野猪', '熊', '豹子', '蜘蛛', '鹿', '羊', '牛', '狼', '袋鼠', '不确定']


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
            'version': 1,
            'created_at': now_iso(),
            'batches': {},
            'items': {},
            'train_jobs': {},
        })

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
                counts[item['status']] = counts.get(item['status'], 0) + 1
            return counts


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
                    same = pred_pair == user_pair
                    screen = {
                        'ok': same or score < self.reject_confidence,
                        'mode': 'model',
                        'pred_pair': pred_pair,
                        'pred_animal': result.get('best_animal', ''),
                        'confidence': score,
                        'top_pairs': result.get('top_pairs', []),
                    }
                    if same:
                        screen['reason'] = '用户答案与模型预测一致，进入人工审核。'
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

    def export_approved_to_manifest(self) -> int:
        with self.store.lock:
            approved = [item for item in self.store.state['items'].values() if item['status'] == 'approved']
        if not approved:
            raise RuntimeError('没有已审核通过的数据。')

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

    def start(self, run_name: str, feature_mode: str, epochs: int, val_ratio: float) -> dict:
        with self.lock:
            with self.store.lock:
                for job in self.store.state['train_jobs'].values():
                    if job.get('status') == 'running':
                        raise RuntimeError('已有训练任务正在运行。')
                blocking = [item for item in self.store.state['items'].values() if item['status'] in {'screening', 'needs_admin'}]
                if blocking:
                    raise RuntimeError(f'还有 {len(blocking)} 条数据未完成审核，不能开始训练。')
            exported = self.export_approved_to_manifest()
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
  <title>九重妖楼识别测试数据提交</title>
  <style>
    :root{color-scheme:dark;--bg:#0b1020;--panel:#111827;--panel2:#0f172a;--line:#334155;--text:#e5e7eb;--muted:#94a3b8;--blue:#2563eb;--green:#16a34a;--red:#dc2626;--yellow:#d97706}
    *{box-sizing:border-box} body{margin:0;background:linear-gradient(135deg,#08111f,#111827 46%,#0b1020);color:var(--text);font:16px/1.55 system-ui,"Segoe UI",Arial}
    header{position:sticky;top:0;z-index:20;background:rgba(15,23,42,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
    .bar{max-width:1280px;margin:auto;padding:14px 18px;display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap}
    a{color:#93c5fd}.brand{font-weight:800;font-size:18px}.muted{color:var(--muted)} main{max-width:1280px;margin:auto;padding:20px;display:grid;gap:18px}
    .card{background:rgba(17,24,39,.88);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 20px 50px rgba(0,0,0,.25)}
    .steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.step{padding:14px;border:1px solid var(--line);border-radius:14px;background:var(--panel2)}.step b{display:block}
    button,input,select{font:inherit;border:1px solid var(--line);border-radius:10px;background:#1f2937;color:var(--text);padding:10px 12px;min-height:44px}
    button{cursor:pointer;transition:.18s ease;background:#243044}button:hover{border-color:#60a5fa;background:#1e3a5f}button:focus,input:focus,select:focus{outline:3px solid rgba(96,165,250,.35)}
    button.primary{background:var(--blue);border-color:#60a5fa}button.success{background:var(--green);border-color:#4ade80}button.danger{background:var(--red);border-color:#f87171}button:disabled{opacity:.55;cursor:not-allowed}
    .upload{border:2px dashed #475569;border-radius:18px;padding:22px;text-align:center;background:rgba(15,23,42,.7)}.upload input{width:min(620px,100%)}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}.item{border:1px solid var(--line);border-radius:16px;padding:12px;background:var(--panel2)}
    .imageWrap{position:relative;display:inline-block;width:100%;background:#020617;border-radius:12px;overflow:hidden}.imageWrap img{display:block;width:100%;height:auto;user-select:none}
    .box{position:absolute;border:3px solid #38bdf8;border-radius:10px;background:rgba(56,189,248,.08);cursor:pointer;min-width:32px;min-height:32px}
    .box.selected{border-color:#ef4444;background:rgba(239,68,68,.24);box-shadow:0 0 0 2px rgba(239,68,68,.22),0 0 18px rgba(239,68,68,.5)}
    .num{position:absolute;left:3px;top:2px;background:rgba(2,6,23,.82);padding:0 6px;border-radius:6px;font-weight:800}.tag{position:absolute;right:3px;bottom:2px;background:rgba(37,99,235,.9);padding:0 6px;border-radius:6px;font-size:13px}
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}.animalBtns{display:flex;gap:8px;flex-wrap:wrap}.animalBtns button{padding:7px 10px;min-height:38px}.animalBtns button.active{background:#1d4ed8;border-color:#93c5fd}
    .status{padding:12px;border-radius:12px;background:#0f172a;border:1px solid var(--line);white-space:pre-wrap}.ok{color:#86efac}.bad{color:#fca5a5}.warn{color:#fcd34d}
    @media(max-width:820px){.steps{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.bar{align-items:flex-start}}
  </style>
</head>
<body>
<header><div class="bar"><div><div class="brand">九重妖楼识别测试数据提交</div><div class="muted">每轮最多上传 10 张，全部勾选后提交，服务器后台初筛。</div></div><a href="/admin">管理员审核区</a></div></header>
<main>
  <section class="steps">
    <div class="step"><b>1. 上传图片</b><span class="muted">支持 PNG/JPG/WebP，每轮最多 10 张。</span></div>
    <div class="step"><b>2. 勾选答案</b><span class="muted">每张图片选择两个正确格子，并填写动物类别。</span></div>
    <div class="step"><b>3. 提交初筛</b><span class="muted">疑似乱选会被拦截，正常数据进入人工审核。</span></div>
  </section>
  <section class="card upload">
    <h2>上传一轮图片</h2>
    <p class="muted">选择 1-10 张图片。上传后在下方逐张勾选两个答案格。</p>
    <input id="fileInput" type="file" accept="image/*" multiple>
    <div class="row" style="justify-content:center"><button id="uploadBtn" class="primary">上传并开始勾选</button><button id="resetBtn">清空当前轮</button></div>
  </section>
  <section class="card">
    <div class="row" style="justify-content:space-between">
      <h2 style="margin:0">当前轮标注</h2>
      <div class="muted" id="batchMeta">尚未上传</div>
    </div>
    <div id="items" class="grid"></div>
    <div class="row"><button id="submitBtn" class="success" disabled>全部勾选完成，提交到服务器</button><button id="refreshBtn" disabled>刷新初筛状态</button></div>
    <div id="status" class="status muted">等待上传。</div>
  </section>
</main>
<script>
const animals = ['', '野猪','熊','豹子','蜘蛛','鹿','羊','牛','狼','袋鼠','不确定'];
let batch = null;
const answers = new Map();
const $ = id => document.getElementById(id);
function setStatus(text, cls='muted'){ $('status').className = 'status ' + cls; $('status').textContent = text; }
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
async function api(url, opts={}){ const r = await fetch(url, opts); const data = await r.json().catch(()=>({})); if(!r.ok || data.code){ throw new Error(data.message || r.statusText); } return data.data ?? data; }
function ensureAnswer(id){ if(!answers.has(id)) answers.set(id, {pair:[], animal:''}); return answers.get(id); }
function renderItems(){
  const root = $('items'); root.innerHTML = '';
  if(!batch){ $('batchMeta').textContent='尚未上传'; $('submitBtn').disabled=true; return; }
  $('batchMeta').textContent = `批次 ${batch.batch_id}，${batch.items.length} 张`;
  for(const item of batch.items){
    const ans = ensureAnswer(item.id);
    const div = document.createElement('div'); div.className='item'; div.dataset.id=item.id;
    div.innerHTML = `<b>${esc(item.filename)}</b><div class="muted">选择两个答案格：<span class="pairText"></span></div>
      <div class="imageWrap"><img alt="待标注图片 ${esc(item.filename)}" src="${item.image_url}"></div>
      <div class="row"><label>动物类别 <select class="animalSelect">${animals.map(a=>`<option value="${esc(a)}">${a?esc(a):'请选择'}</option>`).join('')}</select></label></div>
      <div class="animalBtns">${animals.filter(Boolean).map(a=>`<button type="button" data-animal="${esc(a)}">${esc(a)}</button>`).join('')}</div>`;
    const img = div.querySelector('img');
    img.addEventListener('load', ()=>drawBoxes(div, item));
    div.querySelector('.animalSelect').value = ans.animal;
    div.querySelector('.animalSelect').addEventListener('change', e=>{ ans.animal=e.target.value; updateCard(div,item); updateSubmitState(); });
    div.querySelectorAll('[data-animal]').forEach(btn=>btn.addEventListener('click',()=>{ ans.animal=btn.dataset.animal; updateCard(div,item); updateSubmitState(); }));
    root.appendChild(div);
  }
  updateSubmitState();
}
function drawBoxes(card, item){
  const wrap = card.querySelector('.imageWrap'); wrap.querySelectorAll('.box').forEach(x=>x.remove());
  const img = wrap.querySelector('img'); const sx = img.clientWidth / item.size[0], sy = img.clientHeight / item.size[1];
  for(let i=0;i<item.boxes.length;i++){
    const n=i+1, b=item.boxes[i], box=document.createElement('button'); box.type='button'; box.className='box'; box.style.left=(b[0]*sx)+'px'; box.style.top=(b[1]*sy)+'px'; box.style.width=(b[2]*sx)+'px'; box.style.height=(b[3]*sy)+'px'; box.setAttribute('aria-label', `选择第 ${n} 格`); box.innerHTML=`<span class="num">${n}</span>`;
    box.addEventListener('click',()=>toggle(item.id,n,card,item)); wrap.appendChild(box);
  }
  updateCard(card,item);
}
function toggle(id,n,card,item){ const ans=ensureAnswer(id); if(ans.pair.includes(n)) ans.pair=ans.pair.filter(x=>x!==n); else { if(ans.pair.length>=2) ans.pair.shift(); ans.pair.push(n); ans.pair.sort((a,b)=>a-b); } updateCard(card,item); updateSubmitState(); }
function updateCard(card,item){
  const ans=ensureAnswer(item.id);
  card.querySelector('.pairText').textContent = ans.pair.length ? `[${ans.pair.join(', ')}]${ans.pair.length<2?'，还差 '+(2-ans.pair.length)+' 个':''}` : '未选择';
  card.querySelector('.animalSelect').value = ans.animal;
  card.querySelectorAll('.box').forEach((box,i)=>box.classList.toggle('selected', ans.pair.includes(i+1)));
  card.querySelectorAll('[data-animal]').forEach(btn=>btn.classList.toggle('active', btn.dataset.animal===ans.animal));
}
function complete(){ return batch && batch.items.every(it => { const a=answers.get(it.id); return a && a.pair.length===2 && a.animal; }); }
function updateSubmitState(){ $('submitBtn').disabled = !complete(); }
$('uploadBtn').onclick = async () => {
  const files = [...$('fileInput').files];
  if(files.length<1 || files.length>10){ setStatus('请选择 1-10 张图片。','bad'); return; }
  const fd = new FormData(); files.forEach(f=>fd.append('images', f));
  $('uploadBtn').disabled = true; setStatus('正在上传并自动切格...');
  try{ const data = await api('/api/uploads', {method:'POST', body:fd}); batch=data; answers.clear(); renderItems(); $('refreshBtn').disabled=false; setStatus('上传完成。请逐张选择两个答案格，并填写动物类别。','ok'); }
  catch(e){ setStatus(e.message,'bad'); }
  finally{ $('uploadBtn').disabled = false; }
};
$('resetBtn').onclick = ()=>{ batch=null; answers.clear(); renderItems(); setStatus('已清空当前轮。'); $('refreshBtn').disabled=true; };
$('submitBtn').onclick = async () => {
  if(!complete()) return;
  const payload = {answers: batch.items.map(it => ({item_id:it.id, pair:answers.get(it.id).pair, animal:answers.get(it.id).animal}))};
  $('submitBtn').disabled = true; setStatus('已提交，后台正在初筛...');
  try{ await api(`/api/submissions/${batch.batch_id}/submit`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}); setStatus('提交成功。后台初筛中，可点击刷新查看结果。','ok'); }
  catch(e){ setStatus(e.message,'bad'); updateSubmitState(); }
};
$('refreshBtn').onclick = async () => {
  if(!batch) return;
  try{ const data = await api(`/api/submissions/${batch.batch_id}`); const lines = data.items.map(it => `${it.filename}: ${it.status}${it.screen?.reason ? ' - '+it.screen.reason : ''}`); setStatus(`批次状态：${data.status}\n` + lines.join('\n'), 'warn'); }
  catch(e){ setStatus(e.message,'bad'); }
};
window.addEventListener('resize', renderItems);
</script>
</body>
</html>
"""


ADMIN_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>九重妖楼识别测试管理员审核</title>
  <style>
    :root{color-scheme:dark;--bg:#020617;--panel:#111827;--line:#334155;--text:#e5e7eb;--muted:#94a3b8;--blue:#2563eb;--green:#16a34a;--red:#dc2626;--yellow:#d97706}
    *{box-sizing:border-box} body{margin:0;background:#020617;color:var(--text);font:16px/1.55 system-ui,"Segoe UI",Arial} header{position:sticky;top:0;z-index:30;background:rgba(15,23,42,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}
    .bar,main{max-width:1440px;margin:auto}.bar{padding:14px 18px;display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}main{padding:18px;display:grid;gap:16px}
    a{color:#93c5fd}.brand{font-weight:800}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px}.stat,.panel,.item{background:#111827;border:1px solid var(--line);border-radius:16px;padding:14px}.stat b{font-size:26px;display:block}
    button,input,select{font:inherit;border:1px solid var(--line);border-radius:10px;background:#1f2937;color:var(--text);padding:9px 12px;min-height:42px}button{cursor:pointer}button:hover{border-color:#60a5fa}.primary{background:var(--blue)}.success{background:var(--green)}.danger{background:var(--red)}.warnBtn{background:var(--yellow)}button:focus,input:focus,select:focus{outline:3px solid rgba(96,165,250,.35)}
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.list{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px}.item img{width:100%;border-radius:12px;background:#020617}.badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#334155;color:#e2e8f0}.badge.needs_admin{background:#854d0e}.badge.approved{background:#166534}.badge.rejected,.badge.auto_rejected{background:#991b1b}.badge.screening{background:#1d4ed8}
    textarea{width:100%;min-height:82px;background:#0f172a;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px}.status{white-space:pre-wrap;background:#0f172a;border:1px solid var(--line);border-radius:12px;padding:12px}.small{font-size:13px}.ok{color:#86efac}.bad{color:#fca5a5}.warn{color:#fde68a}
    @media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.list{grid-template-columns:1fr}}
  </style>
</head>
<body>
<header><div class="bar"><div><div class="brand">九重妖楼识别测试管理员审核区</div><div class="muted">审核通过的数据会在开始训练前合并到 manual_answers.json。</div></div><div class="row"><a href="/">用户提交页</a><button id="refreshBtn">刷新</button></div></div></header>
<main>
  <section class="cards" id="stats"></section>
  <section class="panel">
    <div class="row">
      <label>状态 <select id="statusFilter"><option value="needs_admin">待审核</option><option value="auto_rejected">自动拦截</option><option value="approved">已通过</option><option value="rejected">已拒绝</option><option value="screening">初筛中</option><option value="all">全部</option></select></label>
      <label>Admin Key <input id="adminKey" type="password" placeholder="如果服务端配置了 TRAINING_WEB_ADMIN_KEY"></label>
      <button id="loadBtn" class="primary">加载列表</button>
    </div>
  </section>
  <section class="panel">
    <h2>训练控制</h2>
    <div class="row">
      <label>run-name <input id="runName" value="web_review_parts_v1"></label>
      <label>feature-mode <select id="featureMode"><option value="parts_v1">parts_v1</option><option value="processed_v2">processed_v2</option></select></label>
      <label>epochs <input id="epochs" type="number" min="1" value="180" style="width:110px"></label>
      <label>val-ratio <input id="valRatio" type="number" min="0.05" max="0.9" step="0.05" value="0.2" style="width:110px"></label>
      <button id="trainBtn" class="success">全部审核后开始训练</button>
      <button id="trainStatusBtn">训练状态</button>
    </div>
    <div id="trainStatus" class="status muted">尚未查询。</div>
  </section>
  <section class="list" id="list"></section>
</main>
<script>
const $ = id => document.getElementById(id);
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function headers(){ const h={'Content-Type':'application/json'}; const k=$('adminKey').value.trim(); if(k) h['X-Admin-Key']=k; return h; }
async function api(url, opts={}){ opts.headers={...(opts.headers||{}), ...headers()}; const r=await fetch(url,opts); const data=await r.json().catch(()=>({})); if(!r.ok || data.code){ throw new Error(data.message || r.statusText); } return data.data ?? data; }
function badge(s){ return `<span class="badge ${esc(s)}">${esc(s)}</span>`; }
async function loadStats(){ const data=await api('/api/admin/stats'); const keys=['needs_admin','screening','auto_rejected','approved','rejected']; $('stats').innerHTML=keys.map(k=>`<div class="stat"><span class="muted">${k}</span><b>${data.counts[k]||0}</b></div>`).join(''); }
async function loadList(){
  await loadStats();
  const st=$('statusFilter').value; const data=await api('/api/admin/items?status='+encodeURIComponent(st));
  $('list').innerHTML = data.items.map(renderItem).join('');
  document.querySelectorAll('[data-action]').forEach(btn=>btn.addEventListener('click',()=>review(btn.dataset.id, btn.dataset.action)));
}
function renderItem(it){
  const screen = it.screen || {};
  const pair = (it.pair||[]).join(',');
  const animals = it.animals ? JSON.stringify(it.animals) : '';
  return `<article class="item" id="item-${esc(it.id)}">
    <div class="row" style="justify-content:space-between"><b>${esc(it.filename)}</b>${badge(it.status)}</div>
    <img src="${it.image_url}" alt="审核图片 ${esc(it.filename)}" loading="lazy">
    <div class="small muted">提交答案：pair=[${esc(pair)}] animal=${esc(it.animal||'')}</div>
    <div class="small ${screen.ok===false?'bad':'warn'}">初筛：${esc(screen.reason||'无')} ${screen.pred_pair?`预测=[${esc(screen.pred_pair.join(','))}] ${esc(screen.pred_animal||'')} score=${esc(screen.confidence)}`:''}</div>
    <div class="row"><label>pair <input id="pair-${esc(it.id)}" value="${esc(pair)}" style="width:100px"></label><label>animal <input id="animal-${esc(it.id)}" value="${esc(it.animal||'')}" style="width:120px"></label></div>
    <textarea id="note-${esc(it.id)}" placeholder="审核备注">${esc(it.review_note||'')}</textarea>
    <div class="row"><button class="success" data-action="approve" data-id="${esc(it.id)}">审核通过</button><button class="danger" data-action="reject" data-id="${esc(it.id)}">拒绝</button><button class="warnBtn" data-action="needs_admin" data-id="${esc(it.id)}">退回待审</button></div>
  </article>`;
}
async function review(id, decision){
  const pairText = document.getElementById('pair-'+id).value;
  const pair = pairText.split(',').map(x=>Number(x.trim())).filter(Boolean);
  const animal = document.getElementById('animal-'+id).value.trim();
  const note = document.getElementById('note-'+id).value.trim();
  try{ await api('/api/admin/items/'+id+'/review',{method:'POST',body:JSON.stringify({decision,pair,animal,note})}); await loadList(); }
  catch(e){ alert(e.message); }
}
$('loadBtn').onclick=loadList; $('refreshBtn').onclick=loadList; $('statusFilter').onchange=loadList;
$('trainBtn').onclick=async()=>{
  const payload={run_name:$('runName').value.trim(), feature_mode:$('featureMode').value, epochs:Number($('epochs').value), val_ratio:Number($('valRatio').value)};
  $('trainStatus').textContent='正在启动训练...';
  try{ const data=await api('/api/admin/train',{method:'POST',body:JSON.stringify(payload)}); $('trainStatus').textContent='训练已启动：'+JSON.stringify(data,null,2); await loadStats(); }
  catch(e){ $('trainStatus').textContent=e.message; $('trainStatus').className='status bad'; }
};
$('trainStatusBtn').onclick=async()=>{ try{ const data=await api('/api/admin/train-status'); $('trainStatus').className='status'; $('trainStatus').textContent=JSON.stringify(data,null,2); }catch(e){ $('trainStatus').textContent=e.message; } };
loadList().catch(e=>{$('list').innerHTML='<div class="status bad">'+esc(e.message)+'</div>';});
</script>
</body>
</html>
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
                self.ok({'max_images_per_batch': MAX_IMAGES_PER_BATCH, 'animals': ANIMALS})
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
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                raise ValueError('批次不存在。')
            if batch['status'] not in {'uploaded'}:
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
                item['status'] = 'queued'
                item['submitted_at'] = now_iso()
            batch['status'] = 'queued'
            batch['submitted_at'] = now_iso()
        self.store.save()
        self.worker.queue_batch(batch_id)
        self.ok({'batch_id': batch_id, 'status': 'queued'})

    def handle_get_submission(self, path: str) -> None:
        batch_id = path.split('/')[3]
        with self.store.lock:
            batch = self.store.state['batches'].get(batch_id)
            if not batch:
                self.error(HTTPStatus.NOT_FOUND, '批次不存在。')
                return
            items = [self.public_item(self.store.state['items'][item_id]) for item_id in batch['item_ids']]
            data = dict(batch)
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
            elif decision == 'reject':
                item['status'] = 'rejected'
            else:
                item['status'] = 'needs_admin'
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
        if feature_mode not in {'parts_v1', 'processed_v2'}:
            raise ValueError('feature_mode 只能是 parts_v1 或 processed_v2。')
        if epochs < 1:
            raise ValueError('epochs 必须大于 0。')
        job = self.trainer.start(run_name, feature_mode, epochs, val_ratio)
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
    parser = argparse.ArgumentParser(description='九重妖楼识别测试训练数据 Web 平台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--admin-key', default=os.environ.get('TRAINING_WEB_ADMIN_KEY', ''))
    parser.add_argument('--reject-confidence', type=float, default=float(os.environ.get('TRAINING_WEB_REJECT_CONFIDENCE', DEFAULT_REJECT_CONFIDENCE)))
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = StateStore(STATE_PATH)
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

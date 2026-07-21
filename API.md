# 笑傲江湖-九重妖楼识别测试 API

> 仅供娱乐、学习和本地测试使用，请勿用于生产用途。

## Base URL

本地默认地址：`http://127.0.0.1:8090`

## Authentication

`POST /api/identify/image` 必须提供 API Key。推荐请求头：

```http
X-API-Key: <your-api-key>
```

也兼容：

```http
Authorization: Bearer <your-api-key>
```

鉴权失败返回：

```json
{"code":401,"message":"API Key 不存在或已经失效。"}
```

训练标注页中，手动上传的整轮数据完成后台模型初筛后，若每张均未被判为乱填，会发放一个二十四小时有效的临时 API Key；平台待标注整轮提交成功后也会立即发放。该 Key 与 `SAMEOBJECT_API_KEY` 一样可用于本接口；Key 原文只在领取弹窗中展示一次。服务端以 UTC 保存和校验有效期，页面按东八区展示截止时间。

管理员可在 `/admin` 的“Key 有效期”区域修改“无任务默认”和“答题奖励”的有效期，范围为 0.25 至 720 小时（30 天）。保存后持久化到训练 Web 状态文件，并立即影响之后新发放的 Key；已发放的 Key 有效期不变。

## Identify Image

`POST /api/identify/image`

用于识别九重妖楼小游戏 8 宫格截图中两张相同/相似动物图片的位置。支持两种请求体。

### JSON Base64

请求头：

```http
Content-Type: application/json
X-API-Key: <your-api-key>
```

请求体：

```json
{
  "img": "iVBORw0KGgoAAA..."
}
```

`img` 也接受 data URL：

```json
{
  "img": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

### Raw Binary

请求头：

```http
Content-Type: image/png
X-API-Key: <your-api-key>
```

请求体直接为 PNG/JPEG 二进制。

### Response

```json
{
  "code": 0,
  "data": {
    "positions": [1, 8],
    "identify_id": "7b5e4bba8ce34b4b9a3b5fce5bdbddca",
    "animal": "羊",
    "confidence": 0.954274,
    "click_centers": [[113, 165], [593, 325]],
    "top_pairs": [
      {"pair": [1, 8], "animal": "羊", "score": 0.954274}
    ]
  }
}
```

字段说明：

- `positions`：预测出的两个格子序号，按从左到右、从上到下编号 `1-8`。
- `identify_id`：本次识别的唯一编号。客户端应保存该值，用于后续正确性反馈。
- `animal`：预测的共同动物类别。
- `confidence`：融合模型对第一候选 pair 的分数。
- `click_centers`：原图坐标系中的两个点击中心点。
- `top_pairs`：前 5 个候选，按分数降序。

## Identify Feedback

`POST /api/identify/feedback`

上报 `identify_id` 对应识别结果是否正确。接口与识别接口使用相同的 API Key；重复提交相同结果是幂等的。`correct` 为 `false` 时，服务端会保存原图并将该记录放入用户训练标注页的“领取任务”队列；每次最多领取 10 张，领取后 30 分钟内未提交会自动回到队列，刷新同一浏览器会恢复该领取批次。标注提交后直接进入待训练数据，不进入管理员审核。`correct` 为 `true` 时只记录反馈并删除临时原图；未反馈记录不会进入标注、审核或训练流程，并在 60 秒后清理。

```json
{
  "identify_id": "7b5e4bba8ce34b4b9a3b5fce5bdbddca",
  "correct": false
}
```

响应示例：

```json
{
  "code": 0,
  "data": {
    "identify_id": "7b5e4bba8ce34b4b9a3b5fce5bdbddca",
    "correct": false,
    "queued_for_labeling": true,
    "label_item_id": "f22a4f1e5b6c"
  }
}
```

PowerShell：

```powershell
$body = @{ identify_id = $identifyId; correct = $false } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8090/api/identify/feedback" -ContentType "application/json" -Headers @{ "X-API-Key" = $key } -Body $body
```

### Errors

| HTTP | code | Meaning |
|---|---:|---|
| 400 | 400 | 请求体、base64 或图片大小不合法。 |
| 401 | 401 | API Key 不存在或已经失效。 |
| 404 | 404 | 路径不存在。 |
| 405 | 405 | 方法错误；识别接口必须使用 POST。 |
| 500 | 500 | 图片无法识别为 8 宫格或推理失败。 |

## Health Check

`GET /healthz` 不需要鉴权：

```json
{"ok":true}
```

## Start

PowerShell：

推荐使用本地 `.env`（不会提交到 Git）。首次部署时：

```powershell
Copy-Item .env.example .env
notepad .env
```

填写：

```text
SAMEOBJECT_API_KEY=replace-with-a-long-random-api-key
TRAINING_WEB_ADMIN_KEY=replace-with-a-different-admin-key
INCLUDE_AUTO_APPROVED_IN_TRAINING=1
```

`INCLUDE_AUTO_APPROVED_IN_TRAINING` 控制训练时是否默认包含“模型和用户一致后自动通过”的样本；管理员页面训练前也可以临时勾选/取消。

之后直接启动即可，程序会自动读取 `.env`：

```powershell
python sameobject_api.py --host 127.0.0.1 --port 8090
```

合并后地址：

- 用户提交页：`http://127.0.0.1:8090/`
- 管理员审核页：`http://127.0.0.1:8090/admin`
- 卡密管理页：`http://127.0.0.1:8090/keys`
- 识别接口：`POST http://127.0.0.1:8090/api/identify/image`

鉴权分开：

- `POST /api/identify/image` 使用 `SAMEOBJECT_API_KEY` 或后台发放识别 Key。
- `POST /api/license/*` 使用 `X-API-Key: <登录卡密>`（用户卡或通用 `SAMEOBJECT_API_KEY`）。
- `/admin`、`/keys` 及 `/api/admin/*` 使用 `TRAINING_WEB_ADMIN_KEY`，通过 `X-Admin-Key` 传递。
- 普通用户上传/提交页面不需要业务 API Key，也不能访问管理员接口。
- `LICENSE_API_SECRET` 仅为可选 HMAC 签名密钥，**不是** API Key。


## License Keys（登录卡密）

> 生产客户端对接请优先阅读：[LICENSE_INTEGRATION.md](./LICENSE_INTEGRATION.md)（用户登录卡密专题）。

用于外部软件登录鉴权。管理员在 `/keys` 页面发卡；客户端调用公开校验接口，首次验证绑定机器码，之后仅同机器可通过。 卡密接口的 `X-API-Key` 即登录卡密；环境变量 `SAMEOBJECT_API_KEY` 为不入库通用卡（不过期、不绑机，`master_key: true`）。识别接口仍用 `SAMEOBJECT_API_KEY` 或发放的识别 Key。

### 校验卡密

`POST /api/license/verify`

### 解绑机器码

`POST /api/license/unbind`

请求：`{"card_key":"...","machine_code":"..."}`（`machine_code` 可选，传则必须匹配）。
同一张卡密每个**北京自然日**最多解绑 1 次；超限返回 HTTP 429。

管理端修改（需 `X-Admin-Key`）：`POST /api/admin/license-keys/{id}/update`，可改除卡密明文外的字段。

**必须** `X-API-Key: <登录卡密>`（普通用户卡或环境变量通用 Key `SAMEOBJECT_API_KEY`）。  
body 含 `machine_code`（可选再放 `card_key` 且须与头一致），整 body 参与可选 HMAC。

```json
{
  "card_key": "ABCD-EFGH-IJKL-MNOP",
  "machine_code": "PC-UNIQUE-ID"
}
```

成功：

```json
{
  "code": 0,
  "data": {
    "ok": true,
    "status": "active",
    "expires_at": "2026-08-11T12:00:00+00:00",
    "machine_bound": true,
    "machine_code_masked": "PC****ID"
  }
}
```

失败示例：

| HTTP | message |
|---|---|
| 400 | card_key / machine_code 为空 |
| 401 | 卡密不存在 |
| 403 | 卡密已作废 / 已过期 / 机器码不匹配 |

### 管理端（需 `X-Admin-Key`）

- 页面：`GET /keys`
- `GET /api/admin/license-keys?status=all&page=1&page_size=20&q=关键词`
  - `status`：`all/unused/active/expired`（页面展示为：全部/未使用/已激活/已过期）
  - `q`：搜索卡密、机器码、备注、ID
- `POST /api/admin/license-keys` body: `{"count":1,"expires_at":"2099-12-31T23:59","note":"","no_machine_limit":false}`
  - `expires_at` 为到期时刻（无时区按东八区理解）；兼容旧字段 `days`
  - `no_machine_limit`：可选，默认 `false`；`true` 时不限制机器码绑定
- `POST /api/admin/license-keys/{id}/update`：修改除卡密明文外字段（status/machine_code/expires_at/verify_count/note/no_machine_limit 等）
- `POST /api/admin/license-keys/{id}/void`：**硬删除**该登录卡密

### 识别 API Key 管理（需 `X-Admin-Key`）

- `GET /api/admin/issued-api-keys?page=1&page_size=20&status=all&q=关键词`
- `POST /api/admin/issued-api-keys` body: `{"ttl_hours":24,"note":""}`
  - 后台新增与奖励发放的 Key 均会保存明文，并在列表“卡密”列展示；更早仅存哈希的历史记录显示“明文未保留”
- `POST /api/admin/issued-api-keys/{id}/void`：**硬删除**；删除后不可再用于 `POST /api/identify/image`

### 管理素材分页

`GET /api/admin/items?status=ready&page=1&page_size=20`

响应包含 `items/page/page_size/total/total_pages`。`status` 支持 `review/in_progress/ready/trained/rejected/all`。

训练启动并导出后，被纳入训练的样本状态变为 `trained`（已训练），不再出现在「待训练」。

### 数据存储

训练 Web 状态默认使用 SQLite：`training_web_data/state.db`（WAL 模式）。
若存在旧版 `training_web_data/state.json`，启动时会自动迁移并备份为 `state.json.migrated-*`。
识别 API Key / 登录卡密校验走索引查询，避免高频整文件读写。


合并后地址补充：

- 卡密管理页：`http://127.0.0.1:8090/keys`

## Docker

PowerShell：

```powershell
$env:SAMEOBJECT_API_KEY = "replace-with-a-long-random-key"
$env:TRAINING_WEB_ADMIN_KEY = "replace-with-a-different-admin-key"
docker compose up -d --build
```

检查容器：

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:8090/healthz
```

注意：浏览器直接打开 `/api/identify/image` 使用的是 `GET`，该接口只接受 `POST`，因此会返回 405。请按上面的 POST 示例调用。

## Docker Training Data

API 推理不需要训练图片，只需要镜像内的两个权重文件。若需要继续训练，把数据目录挂载给训练容器，不要把训练数据公开成 HTTP 静态文件。

项目中已提供 `docker-compose.training.yml`，它会挂载：

```text
./images         -> /app/images
./datasets       -> /app/datasets
./image_archive  -> /app/image_archive
./training_runs  -> /app/training_runs
```

训练命令：

```powershell
docker compose -f docker-compose.training.yml --profile training build
docker compose -f docker-compose.training.yml --profile training run --rm sameobject-trainer
```

训练结果会写回宿主机：

```text
training_runs/docker_parts_v1/
```

训练数据至少应包含：

```text
images/
datasets/sameobject_corpus/manifests/manual_answers.json
image_archive/
```

这些数据默认应留在本地，不建议提交到 GitHub。

## PowerShell Example

```powershell
$key = "replace-with-a-long-random-key"
$bytes = [IO.File]::ReadAllBytes("D:\test.png")
$body = @{ img = [Convert]::ToBase64String($bytes) } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8090/api/identify/image" `
  -ContentType "application/json" `
  -Headers @{ "X-API-Key" = $key } `
  -Body $body
```

## Cloud Control（群控云同步）

路径前缀：`/api/cloud-control`

双重鉴权：

1. **用户身份**：请求头 `X-API-Key`（或 `Authorization: Bearer`），支持 `SAMEOBJECT_API_KEY` 与平台发放的临时 Key（同打码接口）。
2. **请求签名**（与 `/api/license/*` 相同）：
   - 头：`X-Timestamp` / `X-Nonce` / `X-Signature`
   - 密钥：`LICENSE_API_SECRET`
   - `string_to_sign = METHOD\nPATH\ntimestamp\nnonce\nsha256_hex(raw_body)`
   - `signature = HMAC_SHA256(secret, string_to_sign)` 小写 hex
   - POST：`PATH` 为固定路径（如 `/api/cloud-control/join`），body 为原始 JSON 字节
   - GET poll：`PATH` 含 query（如 `/api/cloud-control/poll?master_name=...`），body 为空
   - 时间窗 / nonce 防重放参数同卡密接口（`LICENSE_SIGN_SKEW_SECONDS`、`LICENSE_NONCE_TTL_SECONDS`）
   - 开发可设 `LICENSE_SIGN_OPTIONAL=1` 跳过签名（生产务必关闭）

房间隔离：`sha256(api_key)[:16] + ":" + normalize(master_name)`  
同一用户密钥 + 同一主控名称才互通；不同 Key 即使主控名称相同也互不可见。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/cloud-control/join` | 主控/副控入房，返回 `session_id` / `cursor` |
| POST | `/api/cloud-control/leave` | 离房 |
| POST | `/api/cloud-control/heartbeat` | 保活（建议 ≤20s） |
| POST | `/api/cloud-control/publish` | **仅主控** 广播任务事件 |
| GET  | `/api/cloud-control/poll` | **仅副控** 长轮询收事件 |

`publish.event.action` 支持：`accept` / `complete` / `path` / `claim_activity`。

详细字段与客户端约定见 `game-get` 仓库 `docs/CLOUD_CONTROL.md`（协议一致）。


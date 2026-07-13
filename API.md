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

训练标注页中，手动上传的整轮数据完成后台模型初筛后，若每张均未被判为乱填，会发放一个二十四小时有效的临时 API Key；平台待标注整轮提交成功后也会立即发放。该 Key 与 `SAMEOBJECT_API_KEY` 一样可用于本接口；Key 原文只在领取弹窗中展示一次。服务端以 UTC 保存和校验有效期，用户页面固定按中国标准时间（UTC+8）展示截止时间。

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
- 识别接口：`POST http://127.0.0.1:8090/api/identify/image`

鉴权仍然分开：

- `POST /api/identify/image` 使用 `SAMEOBJECT_API_KEY`，通过 `X-API-Key` 或 `Authorization: Bearer ...` 传递。
- `/admin` 页面中的 `/api/admin/*` 接口使用 `TRAINING_WEB_ADMIN_KEY`，通过 `X-Admin-Key` 传递。
- 普通用户上传/提交页面不使用识别 API Key，也不能访问管理员接口。

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

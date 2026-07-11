# 笑傲江湖-九重妖楼识别测试部署说明

> 仅供娱乐、学习和本地测试使用，请勿用于生产用途。

## 1. 项目说明

本项目提供一个图片识别接口：

```text
POST /api/identify/image
```

当前推理服务使用两套动物分类器融合：

- 整体特征模型：`first_full193/animal_classifier.pt`
- 局部特征模型：`second_full193_parts_v1/animal_classifier.pt`

API 运行时不需要训练图片，只需要这两个模型权重。

## 2. 目录建议

将项目复制到本地或测试服务器，例如：

```text
/opt/jiuchong-yaolou-identify/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.training.yml
├── sameobject_api.py
├── tools/
└── training_runs/
    └── sameobject_animal_classifier/
        ├── first_full193/animal_classifier.pt
        └── second_full193_parts_v1/animal_classifier.pt
```

如果需要继续训练，还要自行准备本地数据：

```text
/opt/jiuchong-yaolou-identify/images/
/opt/jiuchong-yaolou-identify/datasets/
/opt/jiuchong-yaolou-identify/image_archive/
```

这些数据默认不提交 GitHub。

## 3. 启动 API

### 3.1 设置 API Key

Linux：

```bash
cd /opt/jiuchong-yaolou-identify
cp .env.example .env
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('.env')
text = p.read_text(encoding='utf-8')
text = text.replace('replace-with-a-long-random-api-key', secrets.token_urlsafe(32))
text = text.replace('replace-with-a-different-admin-key', secrets.token_urlsafe(32))
p.write_text(text, encoding='utf-8')
PY
```

Windows PowerShell：

```powershell
cd D:\jiuchong-yaolou-identify
Copy-Item .env.example .env
notepad .env
```

两套 Key 分开使用：

- `SAMEOBJECT_API_KEY`：只用于 `POST /api/identify/image` 图片识别接口。
- `TRAINING_WEB_ADMIN_KEY`：只用于训练 Web 的 `/admin` 与 `/api/admin/*` 管理员接口。

程序会自动读取 `.env`。不要把真实 Key 提交到 Git。

### 3.2 构建并启动

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
docker compose logs -f sameobject-api
```

健康检查：

```bash
curl http://127.0.0.1:8090/healthz
```

期望返回：

```json
{"ok":true}
```

默认监听：

```text
http://0.0.0.0:8090
```

当前 Docker 配置已把训练 Web 合并到同一个 8090 端口：

```text
http://127.0.0.1:8090/       用户上传/勾选提交页
http://127.0.0.1:8090/admin  管理员审核与训练页
```

## 4. API 调用

### 4.1 原始二进制上传

```bash
curl -X POST   -H "X-API-Key: $SAMEOBJECT_API_KEY"   -H "Content-Type: image/png"   --data-binary @test.png   http://127.0.0.1:8090/api/identify/image
```

### 4.2 JSON Base64 上传

```bash
IMG_B64=$(base64 -w 0 test.png)

curl -X POST   -H "X-API-Key: $SAMEOBJECT_API_KEY"   -H "Content-Type: application/json"   -d "{"img":"$IMG_B64"}"   http://127.0.0.1:8090/api/identify/image
```

Windows PowerShell：

```powershell
$bytes = [IO.File]::ReadAllBytes("test.png")
$body = @{ img = [Convert]::ToBase64String($bytes) } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8090/api/identify/image" `
  -ContentType "application/json" `
  -Headers @{ "X-API-Key" = $env:SAMEOBJECT_API_KEY } `
  -Body $body
```

也支持：

```http
Authorization: Bearer <api-key>
```

### 4.3 成功响应

```json
{
  "code": 0,
  "data": {
    "positions": [1, 8],
    "animal": "羊",
    "confidence": 0.954274,
    "click_centers": [[113, 165], [593, 325]],
    "top_pairs": [
      {"pair": [1, 8], "animal": "羊", "score": 0.954274}
    ]
  }
}
```

## 5. 错误码

| HTTP 状态 | `code` | 含义 |
|---:|---:|---|
| 400 | 400 | 请求体、Base64 或图片格式不正确。 |
| 401 | 401 | API Key 缺失或错误。 |
| 404 | 404 | 路径不存在。 |
| 405 | 405 | 方法错误；识别接口必须使用 POST。 |
| 500 | 500 | 图片切格或模型推理失败。 |

浏览器直接打开：

```text
http://localhost:8090/api/identify/image
```

使用的是 `GET`，不是识别请求，因此不能用浏览器地址栏测试。请使用上面的 `curl`、PowerShell 或代码发送 `POST`。

## 6. 继续训练

训练数据通过独立的 compose 文件挂载：

```bash
docker compose -f docker-compose.training.yml --profile training build
docker compose -f docker-compose.training.yml --profile training run --rm sameobject-trainer
```

训练容器挂载：

```text
./images        -> /app/images
./datasets      -> /app/datasets
./image_archive -> /app/image_archive
./training_runs -> /app/training_runs
```

训练结果写回宿主机：

```text
training_runs/docker_parts_v1/
```

如果训练了新的 API 权重：

1. 将新的整体模型和局部模型放到 `training_runs/sameobject_animal_classifier/` 对应目录；
2. 确认 `Dockerfile` 中两个 `COPY` 权重路径正确；
3. 重新构建：

```bash
docker compose up -d --build
```

## 7. 常用命令

重启：

```bash
docker compose restart sameobject-api
```

停止：

```bash
docker compose down
```

查看最近日志：

```bash
docker compose logs --tail=200 sameobject-api
```

## 8. 使用边界

本项目是小游戏截图识别测试，不是生产级服务。请勿直接暴露训练目录、模型目录或项目根目录；请勿把它用于生产环境或未经授权的自动化场景。

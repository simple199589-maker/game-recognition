# 笑傲江湖-九重妖楼识别测试 Docker 部署文档

> 本项目仅供娱乐、学习和测试部署使用，请勿用于生产用途。

## 1. 部署包说明

部署包已经包含 Docker 部署所需文件和两套推理模型权重：

```text
Dockerfile
docker-compose.yml
.env.example
sameobject_api.py
sameobject_training_web.py
tools/
training_runs/sameobject_animal_classifier/first_full193/animal_classifier.pt
training_runs/sameobject_animal_classifier/second_full193_parts_v1/animal_classifier.pt
```

已验证两个权重文件可以正常被 `torch.load` 读取：

| 权重 | 用途 | 大小 | 类别 |
|---|---|---:|---|
| `first_full193/animal_classifier.pt` | 整体特征模型 | 约 2.5 MB | 熊、牛、狼、羊、蜘蛛、袋鼠、豹子、野猪、鹿 |
| `second_full193_parts_v1/animal_classifier.pt` | 局部特征模型 | 约 6.5 MB | 熊、牛、狼、羊、蜘蛛、袋鼠、豹子、野猪、鹿 |

Docker 启动后会同时提供：

- 识别 API：`POST /api/identify/image`
- 用户提交/标注页面：`GET /`
- 管理员审核/训练页面：`GET /admin`
- 健康检查：`GET /healthz`

## 2. 上传部署包

本机 PowerShell：

```powershell
scp D:\work\python\game-recognition-docker-deploy-YYYYMMDD-HHMMSS.tar.gz root@你的服务器IP:/opt/
```

如果服务器 SSH 端口不是 22，例如 2222：

```powershell
scp -P 2222 D:\work\python\game-recognition-docker-deploy-YYYYMMDD-HHMMSS.tar.gz root@你的服务器IP:/opt/
```

## 3. 服务器解压

服务器执行：

```bash
cd /opt
tar -xzf game-recognition-docker-deploy-YYYYMMDD-HHMMSS.tar.gz
cd game-recognition-docker-deploy-YYYYMMDD-HHMMSS
```

如果你想固定目录名：

```bash
cd /opt
rm -rf game-recognition
mkdir -p game-recognition
tar -xzf game-recognition-docker-deploy-YYYYMMDD-HHMMSS.tar.gz -C game-recognition --strip-components=1
cd /opt/game-recognition
```

## 4. 配置环境变量

复制模板：

```bash
cp .env.example .env
```

生成随机 Key：

```bash
python3 - <<'PY'
import secrets
print('SAMEOBJECT_API_KEY=' + secrets.token_urlsafe(32))
print('TRAINING_WEB_ADMIN_KEY=' + secrets.token_urlsafe(32))
PY
```

编辑 `.env`：

```bash
nano .env
```

至少修改：

```env
SAMEOBJECT_API_KEY=替换成你的识别接口key
TRAINING_WEB_ADMIN_KEY=替换成你的管理员key
```

可选配置：

```env
TRAINING_WEB_REJECT_CONFIDENCE=0.80
INCLUDE_AUTO_APPROVED_IN_TRAINING=1
```

说明：

- `SAMEOBJECT_API_KEY`：调用 `POST /api/identify/image` 时使用。
- `TRAINING_WEB_ADMIN_KEY`：登录 `/admin` 和调用管理员接口时使用。
- `TRAINING_WEB_REJECT_CONFIDENCE`：模型初筛时，高置信度且与用户选择不一致的拦截阈值。
- `INCLUDE_AUTO_APPROVED_IN_TRAINING=1`：训练时默认包含“模型和用户一致后自动通过”的样本。

## 5. 构建并启动 Docker

```bash
docker compose up -d --build
```

首次构建会下载 Python、PyTorch、TorchVision、OpenCV 以及 ResNet18 预训练权重，耗时取决于服务器网络。

查看容器：

```bash
docker compose ps
```

查看日志：

```bash
docker compose logs -f --tail=200 sameobject-api
```

## 6. 验证服务

健康检查：

```bash
curl http://127.0.0.1:8090/healthz
```

期望返回：

```json
{"ok": true}
```

如果服务器防火墙已开放 `8090`，浏览器访问：

```text
http://服务器IP:8090/
http://服务器IP:8090/admin
```

管理员页面输入 `.env` 中的 `TRAINING_WEB_ADMIN_KEY`。

## 7. Web 提交、渲染和训练流程

### 7.1 用户提交页面

打开：

```text
http://服务器IP:8090/
```

流程：

1. 上传 1-10 张九重妖楼小游戏截图；
2. 页面会渲染截图并切成 1-8 格；
3. 每张图选择两个正确格子；
4. 填写动物类别；
5. 提交后服务器会进行模型初筛。

上传和渲染相关数据会保存在服务器目录：

```text
training_web_data/
```

### 7.2 管理员审核页面

打开：

```text
http://服务器IP:8090/admin
```

流程：

1. 输入 `TRAINING_WEB_ADMIN_KEY`；
2. 查看用户提交的数据；
3. 审核通过或拒绝；
4. 所有待审核数据处理完成后，点击“训练”。

审核通过的数据会合并到：

```text
datasets/sameobject_corpus/manifests/manual_answers.json
```

对应图片会复制到：

```text
images/training_web/
```

训练输出会写入：

```text
training_runs/sameobject_animal_classifier/<run-name>/
```

这些目录都通过 `docker-compose.yml` 挂载到宿主机，所以容器重启后数据不会丢失。

## 8. API 调用示例

PowerShell：

```powershell
$key = "你的 SAMEOBJECT_API_KEY"
$bytes = [IO.File]::ReadAllBytes("D:	est.png")
$body = @{ img = [Convert]::ToBase64String($bytes) } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Method Post `
  -Uri "http://服务器IP:8090/api/identify/image" `
  -ContentType "application/json" `
  -Headers @{ "X-API-Key" = $key } `
  -Body $body
```

Linux curl：

```bash
curl -X POST   -H "X-API-Key: $SAMEOBJECT_API_KEY"   -H "Content-Type: image/png"   --data-binary @test.png   http://127.0.0.1:8090/api/identify/image
```

成功响应示例：

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

## 9. 常用运维命令

重启：

```bash
docker compose restart sameobject-api
```

停止：

```bash
docker compose down
```

重新构建：

```bash
docker compose up -d --build
```

查看实时日志：

```bash
docker compose logs -f sameobject-api
```

查看最近 200 行日志：

```bash
docker compose logs --tail=200 sameobject-api
```

## 10. 防火墙放行

如果浏览器无法访问，但服务器本机 `curl 127.0.0.1:8090/healthz` 正常，需要放行端口。

Ubuntu UFW：

```bash
ufw allow 8090/tcp
ufw status
```

云服务器还需要在安全组中放行 TCP `8090`。

## 11. 常见问题

### 11.1 `docker compose up` 提示没有 Key

确认 `.env` 已存在，且包含：

```env
SAMEOBJECT_API_KEY=...
TRAINING_WEB_ADMIN_KEY=...
```

### 11.2 构建很慢

首次构建需要下载 PyTorch 和 ResNet18 权重，属于正常情况。可以换国内镜像源或在网络较好的环境构建。

### 11.3 页面能打开，但训练失败

查看日志：

```bash
docker compose logs --tail=300 sameobject-api
```

同时确认：

- 已经有审核通过的数据；
- 管理员页没有未审核数据；
- `images/`、`datasets/`、`training_runs/` 目录可写。

### 11.4 API 返回 500

通常是上传的图片无法被切成 8 宫格，或模型推理失败。建议先用 Web 页面上传同一张图观察切格渲染效果。

## 12. 数据和备份

建议定期备份这些目录：

```text
training_web_data/
images/training_web/
datasets/sameobject_corpus/manifests/manual_answers.json
training_runs/sameobject_animal_classifier/
```

不要把真实 `.env`、训练数据、用户上传图片和完整训练输出提交到 GitHub。

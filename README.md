# 笑傲江湖-九重妖楼小游戏识别测试

这是一个针对 **笑傲江湖-九重妖楼** 小游戏截图的本地图片识别测试项目。项目会尝试从 8 宫格截图中识别两张相同/相似的动物图片，并返回格子编号与点击中心点。

> **声明**：本项目仅供娱乐、学习和本地测试使用，请勿用于生产用途，也不要用于任何未经授权、违反平台规则或影响他人服务的场景。

## 当前能力

- 本地 HTTP 识别接口：`POST /api/identify/image`
- 支持 JSON Base64 或图片二进制上传
- 返回预测格子、共同动物类别、置信度和点击中心点
- 可选的训练数据提交/管理员审核 Web 页面
- 训练和预处理脚本保留在 `tools/` 目录，便于继续实验

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
```

如需重新训练或运行旧版实验脚本，再按需安装 `requirements.txt` 中的依赖。

### 2. 准备权重

默认推理会读取：

```text
training_runs/sameobject_animal_classifier/first_full193/animal_classifier.pt
training_runs/sameobject_animal_classifier/second_full193_parts_v1/animal_classifier.pt
```

仓库只建议保留必要的演示权重；训练集、调试图、缓存和完整训练输出均作为本地文件处理，不建议提交到 GitHub。

### 3. 启动本地 API

```powershell
$env:SAMEOBJECT_API_KEY = "replace-with-a-long-random-key"
python sameobject_api.py --host 127.0.0.1 --port 8090
```

同时启用数据提交/审核页面：

```powershell
$env:SAMEOBJECT_API_KEY = "replace-with-a-long-random-api-key"
$env:TRAINING_WEB_ADMIN_KEY = "replace-with-a-different-admin-key"
python sameobject_api.py --host 127.0.0.1 --port 8090 --enable-training-web
```

访问地址：

- 识别接口：`POST http://127.0.0.1:8090/api/identify/image`
- 健康检查：`GET http://127.0.0.1:8090/healthz`
- 数据提交页：`http://127.0.0.1:8090/`
- 管理员审核页：`http://127.0.0.1:8090/admin`

## API 示例

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

更多接口说明见 `API.md`，部署说明见 `DEPLOYMENT.md`。

## 目录说明

```text
sameobject_api.py                 本地识别 HTTP API
sameobject_training_web.py        数据提交/审核/触发训练 Web 页面
tools/                            预处理、预测、训练脚本
training_runs/.../animal_classifier.pt  必要演示权重
Dockerfile / docker-compose.yml   本地容器化运行配置
```

以下内容通常不提交 GitHub：

- `datasets/`、`images/`、`image_archive/`：本地样本与标注数据
- `training_runs/` 中除必要 `.pt` 权重外的训练输出、缓存、报告
- `runs_sameobject/`、`tmp_sameobject_debug/`、`training_web_data/`：运行日志、调试图和 Web 上传状态
- `.env`、`*.log`、`__pycache__/` 等本地环境文件

## Docker 本地运行

```powershell
$env:SAMEOBJECT_API_KEY = "replace-with-a-long-random-key"
$env:TRAINING_WEB_ADMIN_KEY = "replace-with-a-different-admin-key"
docker compose up -d --build
```

```powershell
Invoke-RestMethod http://127.0.0.1:8090/healthz
```

## 旧版实验脚本

仓库中仍保留了一些早期图片识别/验证码实验脚本，例如：

- `fixed_length_captcha.py`
- `slide_captcha.py`
- `rotate_captcha.py`
- `sameobject_captcha.py`

当前 README 和接口文档以 **笑傲江湖-九重妖楼小游戏识别测试** 为准。旧脚本仅作为历史参考，不代表本项目推荐用途。

## 许可与使用边界

代码许可见 `LICENSE`。再次提醒：本项目仅供娱乐和技术学习，请勿用于生产用途。

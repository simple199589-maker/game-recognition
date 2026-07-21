# 用户登录卡密 · 生产对接文档

> 面向外部客户端 / 前台软件对接。本文只覆盖**用户登录卡密**，不包含识别 API Key 与管理员后台。


## 0. 鉴权总览（重要）

### 卡密接口（验证 / 解绑）

| 项 | 说明 |
|---|---|
| `X-API-Key` | **就是登录卡密本身** |
| 普通用户卡 | `/keys` 后台发的卡密，放在 `X-API-Key` |
| 通用 Key | 环境变量 `SAMEOBJECT_API_KEY`（不入库），也可放在 `X-API-Key` 当超级登录卡 |
| body | `machine_code` 等业务字段；**整份 body 参与可选 HMAC 签名** |
| body.card_key | 可选；若传必须与 `X-API-Key` 一致（方便签名覆盖卡密） |

> 卡密接口**不**再要求单独的“固定业务 API Key”。`SAMEOBJECT_API_KEY` 只是可选的**通用卡密**，不是强制网关 Key。

### 识别接口（打码）

| 项 | 说明 |
|---|---|
| `X-API-Key` | `SAMEOBJECT_API_KEY`（通用）或后台发放的识别 API Key |

通用 Key 验证/解绑成功时响应带 `"master_key": true`（不绑机、不过期）。

## 1. 业务说明

| 项目 | 说明 |
|---|---|
| 用途 | 外部软件登录鉴权 |
| 形态 | 可读卡密，例如 `ABCD-EFGH-IJKL-MNOP` |
| 机器绑定 | **首次验证成功**时绑定 `machine_code`；之后仅同机器可通过 |
| 有效期 | 发卡时指定到期时间；到期后不可再登录 |
| 删除 | 管理后台删除后立即失效（硬删除） |
| 鉴权 | **必须** `X-API-Key: <登录卡密>`（普通卡或通用 `SAMEOBJECT_API_KEY`）；body 参与可选 HMAC |

状态流转：

```text
unused（未使用）
   │  首次 verify 成功并绑定机器码
   ▼
active（已激活）
   │  到期
   ▼
expired（已过期）
```

管理员删除后记录直接消失，再次校验视为「卡密不存在」。

后台可为卡密开启 **不限制机器码**（`no_machine_limit=true`）：开启后登录不再校验是否本机；默认关闭，仍绑定并限制机器。

## 2. 环境与地址

| 环境 | Base URL 示例 |
|---|---|
| 本地联调 | `http://127.0.0.1:8090` |
| 生产 | 以实际部署域名为准，例如 `https://api.example.com` |

客户端只调：

```text
POST {BASE_URL}/api/license/verify
```

运营发卡后台（非客户端）：

```text
{BASE_URL}/keys
```

## 3. 校验接口（客户端必接）

### 3.1 基本信息

| 项 | 值 |
|---|---|
| Method | `POST` |
| Path | `/api/license/verify` |
| Content-Type | `application/json; charset=utf-8` |
| 鉴权头 | **必须** `X-API-Key: <登录卡密>`（用户卡或通用 Key） |
| 可选签名头 | 仅当配置 `LICENSE_API_SECRET` 时：`X-Timestamp` / `X-Nonce` / `X-Signature` |
| 超时建议 | 连接 3–5s，总超时 8–10s |
| 重试建议 | 仅网络错误可重试 1–2 次；业务 4xx 不要盲重试 |


### 3.1.1 请求签名（防篡改 / 防重放）

生产环境建议配置：

```text
SAMEOBJECT_API_KEY=长随机通用Key（可作识别 Key + 通用登录卡密，不入库）
# 可选：卡密接口 HMAC 防重放（不是卡密本身）
# LICENSE_API_SECRET=另一个长随机串
# LICENSE_SIGN_OPTIONAL=0
```

- **卡密验证/解绑**：`X-API-Key` = 用户登录卡密（或通用 `SAMEOBJECT_API_KEY`）；body 含 `machine_code`，整 body 参与 HMAC。
- **识别接口**：`X-API-Key` = `SAMEOBJECT_API_KEY` 或后台发放的识别 Key。
- **`LICENSE_API_SECRET`**：仅可选 HMAC 签名密钥，不是登录卡密。

#### 需要的请求头

| Header | 说明 |
|---|---|
| `X-Timestamp` | Unix 秒级时间戳（字符串数字） |
| `X-Nonce` | 一次性随机串，8–128 可见 ASCII，建议 32 位 hex |
| `X-Signature` | HMAC-SHA256 十六进制小写（64 字符） |

兼容别名：`X-License-Timestamp` / `X-License-Nonce` / `X-License-Signature`。

#### 签名字符串（严格按此拼接）

```text
{METHOD}\n
{PATH}\n
{timestamp}\n
{nonce}\n
{sha256_hex(raw_body)}
```

说明：

- `METHOD`：大写，如 `POST`
- `PATH`：仅路径，不含域名和 query，如 `/api/license/verify`
- `raw_body`：HTTP 原始请求体字节（与发出的 JSON 完全一致）
- `sha256_hex`：对 raw_body 做 SHA-256，输出小写 hex
- 签名：`HMAC_SHA256(LICENSE_API_SECRET, string_to_sign)`，输出小写 hex

#### Python 签名示例

```python
import hashlib, hmac, json, secrets, time, requests

API_KEY = "ABCD-EFGH-IJKL-MNOP"  # 请求头 X-API-Key = 登录卡密
SECRET = "optional-license-hmac-secret"  # 仅当启用 LICENSE_API_SECRET 时用于签名
BASE = "https://api.example.com"
path = "/api/license/verify"
body_obj = {"card_key": "ABCD-EFGH-IJKL-MNOP", "machine_code": "PC-UNIQUE-ID-001"}
raw = json.dumps(body_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
ts = str(int(time.time()))
nonce = secrets.token_hex(16)
body_hash = hashlib.sha256(raw).hexdigest()
string_to_sign = f"POST\n{path}\n{ts}\n{nonce}\n{body_hash}"
sign = hmac.new(SECRET.encode(), string_to_sign.encode(), hashlib.sha256).hexdigest()

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,  # 必填：业务 API Key
}
# 仅当服务端启用 LICENSE_API_SECRET 时附加签名头
if SECRET:
    headers.update({
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": sign,
    })

resp = requests.post(BASE + path, data=raw, headers=headers, timeout=10)
print(resp.status_code, resp.text)
```

#### 服务端校验规则

| 规则 | 说明 |
|---|---|
| 缺签名头 | 401 |
| 签名错误 | 401 |
| 时间窗 | 默认 ±300 秒（`LICENSE_SIGN_SKEW_SECONDS`） |
| nonce 重放 | 409 `请求重放或 nonce 已使用` |
| 缺少/错误 X-API-Key | 401 |
| 配置了 LICENSE_API_SECRET 但签名失败 | 401 |

#### 入参校验

| 字段 | 规则 |
|---|---|
| `card_key` | 必填；字母/数字/短横线；长度 8–64 |
| `machine_code` | verify 必填；unbind 可选；长度 4–256；禁止控制字符 |


### 3.2 请求体

```json
{
  "card_key": "ABCD-EFGH-IJKL-MNOP",
  "machine_code": "PC-UNIQUE-ID-001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `card_key` | string | 否（建议与头一致时可放 body 参与签名） | 用户输入的登录卡密；服务端会 trim 首尾空格 |
| `machine_code` | string | 是 | 当前设备唯一标识；服务端会 trim 首尾空格 |

#### machine_code 生成建议

- 同一台机器上保持稳定；重装系统后允许变化（视为新机器）。
- Windows 推荐：主板 UUID / `MachineGuid` / 磁盘序列号等稳定字段拼接后做 SHA-256。
- 避免只使用 IP、可篡改 MAC、用户名等易变字段。
- 建议长度 8–128，仅可见 ASCII，不要换行。

```text
machine_code = sha256(raw_hardware_fingerprint).hexdigest()
```

### 3.3 成功响应

HTTP `200`：

```json
{
  "code": 0,
  "data": {
    "ok": true,
    "status": "active",
    "expires_at": "2026-08-11T12:00:00+00:00",
    "machine_bound": true,
    "machine_code_masked": "PC****01",
    "no_machine_limit": false
  }
}
```

| 字段 | 说明 |
|---|---|
| `code` | `0` 表示成功 |
| `data.ok` | 固定 `true` |
| `data.status` | 成功时一般为 `active` |
| `data.expires_at` | 到期时间，**UTC ISO8601** |
| `data.machine_bound` | 是否已绑定机器（`no_machine_limit=true` 时通常为 `false`） |
| `data.machine_code_masked` | 绑定机器码脱敏值（如 `PC****01`）；未绑定或 `no_machine_limit` 未绑定时为 `null`；**不返回明文 `machine_code`** |
| `data.no_machine_limit` | 是否不限制机器码（后台配置） |

成功判定建议同时满足：

1. HTTP = 200  
2. `code === 0`  
3. `data.ok === true`

### 3.4 失败响应

```json
{
  "code": 403,
  "message": "机器码不匹配，禁止登录。"
}
```

| HTTP | code | message 示例 | 客户端处理 |
|---:|---:|---|---|
| 400 | 400 | card_key / machine_code 不能为空 | 检查入参 |
| 401 | 401 | 卡密不存在 | 卡密错误或已删除 |
| 403 | 403 | 卡密已过期 | 提示续费/换卡 |
| 403 | 403 | 机器码不匹配，禁止登录 | 已绑其他设备 |
| 403 | 403 | 卡密已作废 | 兼容旧数据；现网删除后多走 401 |
| 500 | 500 | 服务端异常 | 稍后重试，不放行 |

请同时判断 HTTP 状态码和 `message`。



## 3.5 用户解绑机器码

### 基本信息

| 项 | 值 |
|---|---|
| Method | `POST` |
| Path | `/api/license/unbind` |
| Content-Type | `application/json` |
| 鉴权头 | 同 verify：签名三件套 |
| 频率限制 | **同一张卡密每个北京自然日最多解绑 1 次** |

### 请求体

```json
{
  "card_key": "ABCD-EFGH-IJKL-MNOP",
  "machine_code": "PC-UNIQUE-ID-001"
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `card_key` | 是 | 登录卡密 |
| `machine_code` | 否 | 若传则必须与当前绑定机器码一致；不传则仅凭卡密解绑 |

### 成功响应

HTTP `200`：

```json
{
  "code": 0,
  "data": {
    "ok": true,
    "status": "unused",
    "machine_bound": false,
    "machine_code_masked": null,
    "last_unbind_at": "2026-07-21T02:15:00+00:00",
    "expires_at": "2026-08-11T12:00:00+00:00"
  }
}
```

解绑后：

- 服务端绑定字段 `machine_code` / `bound_at` 清空（响应中不返回明文机器码）
- 若原状态为 `active`，变为 `unused`
- 记录 `last_unbind_at`（用于日限额）

### 失败响应

| HTTP | message 示例 | 说明 |
|---:|---|---|
| 400 | card_key 不能为空 | 入参错误 |
| 401 | 卡密不存在 | 卡密错误或已删除 |
| 403 | 当前卡密未绑定机器码 | 无需解绑 |
| 403 | 机器码不匹配，无法解绑 | 传入了错误机器码 |
| 403 | 卡密已过期 | 过期不可解绑 |
| 429 | 该卡密今日已解绑过，请明天再试 | 日限额 |

### 示例

```bash
curl -X POST "https://api.example.com/api/license/unbind" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ABCD-EFGH-IJKL-MNOP" \
  -d "{\"card_key\":\"ABCD-EFGH-IJKL-MNOP\",\"machine_code\":\"PC-UNIQUE-ID-001\"}"
```

## 4. 推荐客户端流程

### 4.1 登录 / 激活

```text
用户输入 card_key
    -> 生成本机 machine_code
    -> POST /api/license/verify
         |-- 成功: 本地保存登录态，进入主界面
         `-- 失败: 展示 message，禁止进入
```

### 4.2 启动二次校验（推荐）

每次启动或关键操作前：

1. 读取本地 `card_key`，重新计算 `machine_code`
2. 再次调用 `/api/license/verify`
3. 失败则清除本地登录态，回到登录页

可及时感知：删卡、过期、换机。

### 4.3 本地缓存建议

| 字段 | 是否本地保存 | 说明 |
|---|---|---|
| `card_key` | 是 | 建议加密存储 |
| `machine_code` | 可不存 | 建议每次现算 |
| `expires_at` | 是 | 仅用于展示剩余时间 |
| 是否放行 | — | **必须以服务端 verify 为准** |

## 5. 调用示例

### 5.1 cURL

> 以下示例展示最小可用调用（`X-API-Key` 必填）。若生产启用了 `LICENSE_API_SECRET`，再附加签名头（见 3.1.1）。

```bash
curl -X POST "https://api.example.com/api/license/verify" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ABCD-EFGH-IJKL-MNOP" \
  -d "{\"card_key\":\"ABCD-EFGH-IJKL-MNOP\",\"machine_code\":\"PC-UNIQUE-ID-001\"}"
```

### 5.2 PowerShell

```powershell
$base = "https://api.example.com"
$body = @{
  card_key = "ABCD-EFGH-IJKL-MNOP"
  machine_code = "PC-UNIQUE-ID-001"
} | ConvertTo-Json -Compress

Invoke-RestMethod -Method Post `
  -Uri "$base/api/license/verify" `
  -ContentType "application/json; charset=utf-8" `
  -Headers @{ "X-API-Key" = "ABCD-EFGH-IJKL-MNOP" } `
  -Body $body
```

### 5.3 Python

```python
import requests

BASE = "https://api.example.com"
resp = requests.post(
    f"{BASE}/api/license/verify",
    json={
        "card_key": "ABCD-EFGH-IJKL-MNOP",
        "machine_code": "PC-UNIQUE-ID-001",
    },
    headers={"X-API-Key": "ABCD-EFGH-IJKL-MNOP"},
    timeout=10,
)
data = resp.json()
ok = (
    resp.status_code == 200
    and data.get("code") == 0
    and data.get("data", {}).get("ok") is True
)
print("ok" if ok else data.get("message"))
```

### 5.4 C#

```csharp
using var client = new HttpClient { BaseAddress = new Uri("https://api.example.com") };
client.DefaultRequestHeaders.Add("X-API-Key", "ABCD-EFGH-IJKL-MNOP");
var payload = new {
    card_key = "ABCD-EFGH-IJKL-MNOP",
    machine_code = "PC-UNIQUE-ID-001"
};
using var resp = await client.PostAsJsonAsync("/api/license/verify", payload);
var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
var ok = resp.IsSuccessStatusCode
    && json.GetProperty("code").GetInt32() == 0
    && json.GetProperty("data").GetProperty("ok").GetBoolean();
```

## 6. 时间字段

- `expires_at` 为 **UTC**（如 `2026-08-11T12:00:00+00:00`）
- UI 展示请转北京时间（UTC+8）
- 是否过期以服务端校验结果为准，不要只比本地时间

## 7. 联调检查清单

- [ ] 无 `X-API-Key`：401
- [ ] 错误 `X-API-Key`：401
- [ ] 新卡 + 机器 A：首次成功并激活
- [ ] 同卡 + 机器 A：再次成功
- [ ] 同卡 + 机器 B：机器码不匹配
- [ ] 错误卡密：不存在
- [ ] 过期卡密：已过期
- [ ] 后台删除后：不存在
- [ ] 空字段：400
- [ ] 网络失败 / 5xx：不误放行

## 8. 运营发卡（非客户端）

浏览器打开 `{BASE_URL}/keys`，管理员 Key 登录：

1. 登录卡密页 → 新增卡密
2. 填数量、到期时间（北京时间）、备注
3. 复制卡密发给用户

管理接口需要 `X-Admin-Key`，**禁止**写入前台软件。

## 9. 安全注意

1. 生产必须 HTTPS  
2. 卡密等同口令：日志/埋点不要打完整卡密  
3. `machine_code` 要稳定且不易伪造  
4. **禁止**把 `TRAINING_WEB_ADMIN_KEY` 打进客户端  
5. 客户端可内置 `SAMEOBJECT_API_KEY`（业务 Key，用于 `X-API-Key`）；勿与管理员 Key 混用  
6. 若启用 `LICENSE_API_SECRET`，签名密钥也需下发给客户端，建议做混淆保护  
7. 登录按钮防抖；网关可按需做 IP 限流  

## 10. 不要和识别 API Key 混用

| | 用户登录卡密 | 识别 API Key / 业务 Key |
|---|---|---|
| 用途 | 用户软件登录（卡密+机器码） | 调识图接口；也是卡密接口的 `X-API-Key` |
| 接口 | `POST /api/license/verify` / `unbind` | `POST /api/identify/image` |
| 请求头 | **必须** `X-API-Key: SAMEOBJECT_API_KEY` | **必须** `X-API-Key` |
| 机器绑定 | 默认要（可后台 `no_machine_limit`） | 不要 |
| 用户输入 | `card_key`（登录卡密） | 一般不由终端用户输入 |

说明：`SAMEOBJECT_API_KEY` 是调用服务端的业务 Key；**用户登录卡密**是另一层身份（`card_key`）。前台登录流程只调本文接口，不要把管理员 Key 写进客户端。

## 11. 变更记录

| 日期 | 说明 |
|---|---|
| 2026-07-21 | 首版：用户登录卡密生产对接文档 |
| 2026-07-21 | 新增用户解绑机器码接口（每卡每天 1 次） |
| 2026-07-21 | 明确：卡密接口必须 `X-API-Key`（`SAMEOBJECT_API_KEY`）；`LICENSE_API_SECRET` 仅可选 HMAC |
| 2026-07-21 | 公开卡密接口不再返回明文 `machine_code`，改为 `machine_code_masked` |
| 2026-07-21 | `SAMEOBJECT_API_KEY` 可作为特殊登录卡密（不入库、不绑机） |
| 2026-07-21 | 卡密接口 `X-API-Key` 改为登录卡密本身；`SAMEOBJECT_API_KEY` 仅作通用卡 |

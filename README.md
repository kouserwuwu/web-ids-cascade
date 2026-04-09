# 项目参数与配置（`ai_defend.py` + `realtime_ids_demo.py`）

本文档汇总：

- `ai_defend.py`：二级 DeepSeek、费用估算、级联推理参数（`infer_batch`）
- `scripts/tools/realtime_ids_demo.py`：**CLI / HTTP** 两种运行方式、常用命令行参数、HTTP 接口字段

> 说明：仓库根目录的 `AI_DEFEND_PARAMS.md` 与 `scripts/tools/AI_DEFEND_PARAMS.md` 内容保持一致（便于你在 IDE 根目录直接打开）。

## 0) `realtime_ids_demo.py`：CLI / HTTP 两种模式

入口脚本：`scripts/tools/realtime_ids_demo.py`

### 0.1 CLI 模式（终端批量检测）

示例：

```powershell
python .\scripts\tools\realtime_ids_demo.py cli --pcap .\capture.pcap --window-sec 10 --model .\models\first_stage_lgbm_http_web.pkl
```

常用参数（`cli` 子命令）：

- **`--model`**：一级 LightGBM 模型路径（默认 `models/first_stage_lgbm.pkl`）
- **`--pcap`**：PCAP 路径（与 `--csv` 二选一）
- **`--csv`**：CICIDS 风格 CSV 路径（与 `--pcap` 二选一）
- **`--window-sec`**：PCAP 时间窗聚合秒数；`<=0` 表示关闭时间窗，改用 5 元组聚合
- **`--max-samples`**：最多处理多少条流/窗口（上限控制；默认 `256`）
- **`--low` / `--high`**：一级阈值（默认分别为 `0.001` / `0.9995`）
- **`--ds-conf`**：二级 DeepSeek 置信度阈值（默认 `70`）
- **`--batch-size`**：二级批大小（默认 `16`）

### 0.2 HTTP 模式（启动简易 Web Demo）

示例：

```powershell
python .\scripts\tools\realtime_ids_demo.py http --host 0.0.0.0 --port 8000 --model .\models\first_stage_lgbm_http_web.pkl
```

常用参数（`http` 子命令）：

- **`--host`**：监听地址（默认 `0.0.0.0`）
- **`--port`**：监听端口（默认 `8000`；若本机 `8000` 已被占用，可改成 `5000` 等）
- **`--model`**：一级模型路径

HTTP 路由（实现见 `create_flask_app()`）：

- **`GET /health`**：健康检查
- **`GET /`**：简单前端页面（上传 PCAP）
- **`POST /predict`**：JSON 批量检测  
  - body 示例字段：`flows`（必填）、`low`/`high`（可选，默认固定阈值）、`ds_conf_thr`、`batch_size`
- **`POST /upload_pcap`**：multipart 上传 PCAP（字段名 **`file`**）  
  - form 字段：`window_sec`（可选）  
  - **注意**：该接口当前实现里 **`low/high` 固定为 `FIXED_LOW/FIXED_HIGH`**，`ds_conf_thr=70`，`batch_size=16`（与前端一致）

HTTP 相关环境变量（在 `upload_pcap` 中使用）：

- **`PCAP_WINDOW_SEC`**：当请求未传 `window_sec` 时的默认时间窗（默认 `10`）
- **`MAX_EXPLAIN_MALICIOUS`**：对“一级直接判恶意”的流补充二级解释的最大条数（默认 `200`）

## 1) DeepSeek 开关与 Key

- **`DEEPSEEK_API_KEY`**  
  - **作用**：真实 DeepSeek API Key。  
  - **缺省**：`your_api_key_here`（视为未配置）。  

- **`USE_REAL_DEEPSEEK`**  
  - **作用**：是否启用真实 DeepSeek（1=启用；其他=关闭）。  
  - **缺省**：`0`  
  - **启用条件**：`USE_REAL_DEEPSEEK=1` 且 `DEEPSEEK_API_KEY` 非空且不为 `your_api_key_here`。  

## 2) DeepSeek 费用估算（单位：元）

`ai_defend.py` 会从 DeepSeek 返回的 `usage` 中读取 token 数，并按以下单价估算费用（输出字段 `ds_cost_cny`）。

- **`DEEPSEEK_PRICE_IN_UNCACHED_PER_1M`**  
  - **作用**：输入 token（未命中缓存）单价，**每 1,000,000 tokens** 的价格（元）。

- **`DEEPSEEK_PRICE_IN_CACHED_PER_1M`**  
  - **作用**：输入 token（命中缓存）单价，**每 1,000,000 tokens** 的价格（元）。

- **`DEEPSEEK_PRICE_OUT_PER_1M`**  
  - **作用**：输出 token 单价，**每 1,000,000 tokens** 的价格（元）。

兼容旧变量：

- **`DEEPSEEK_PRICE_IN_PER_1M`**  
  - 若 `DEEPSEEK_PRICE_IN_UNCACHED_PER_1M` 未设置，则作为“未缓存输入单价”使用。

注意：

- 若三项单价都未配置或为 0，则费用估算返回 0（仅回显 tokens）。
- 缓存输入 token 会尽量从响应字段推断（例如 `cached_tokens` 等），无法获取时默认 0。

## 3) DeepSeek 调用行为（代码内常量/参数）

这些不是环境变量，但与实验复现强相关（在 `SemanticAnalyzer.analyze_flow()` / `batch_analyze()` 中）：

- **请求 URL**：固定为 `https://api.deepseek.com/chat/completions`
- **timeout**：单次请求默认 `timeout=5` 秒
- **max_tokens**：响应上限 `max_tokens=500`
- **缓存**：`SemanticAnalyzer` 内置内存缓存（key=flow 的 sha256），由 `batch_analyze(cache_ttl=...)` 控制 TTL（默认 3600 秒）
- **batch_size**：`infer_batch(..., batch_size=...)` 传入，用于控制 `batch_analyze` 的分组节奏（当前实现仍是逐条调用 API，但会按 batch 做节拍/缓存命中）

## 4) 一级/二级级联推理阈值（函数参数）

在 `ai_defend.infer_batch()` 中使用（由 `scripts/tools/realtime_ids_demo.py` 传入）：

- **`low`**（float）  
  - 含义：一级模型输出概率 \(p\) 若 `p <= low`，默认直接判为正常（除非触发“强 HTTP 证据”的二级路由逻辑）。

- **`high`**（float）  
  - 含义：一级模型输出概率 \(p\) 若 `p >= high`，直接判为恶意（`source=first_stage`）。

- **`ds_conf_thr`**（int，0~100）  
  - 含义：二级 DeepSeek 的置信度阈值。`confidence >= ds_conf_thr` 才会按二级结果落到最终 label（否则可能 fallback/rule）。

- **`batch_size`**（int）  
  - 含义：二级调用批大小（节奏/缓存命中相关）。

项目当前默认值（以 `scripts/tools/realtime_ids_demo.py` 为准）：

- `low=0.001`
- `high=0.9995`
- `ds_conf_thr=70`

## 5) CICIDS 数据目录（训练相关）

在 `ai_defend.train_first_stage_from_cicids()` 中：

- **`CICIDS_DATA_DIR`**  
  - **作用**：CICIDS 数据目录位置。  
  - **缺省**：`datasets/cicids`  
  - **兼容**：若 `datasets/cicids` 不存在但 `data/cicids` 存在，则自动回退旧路径。

## 6) 输出字段（用于前端/评估脚本）

二级返回会尽量包含：

- `ds_usage`: `{prompt_tokens, completion_tokens, total_tokens, cached_prompt_tokens, uncached_prompt_tokens}`
- `ds_cost_cny`: 估算费用（元）
- `ds_elapsed_sec`: 单次调用耗时

级联最终 `result` 常用字段：

- `label`: 0/1
- `score`: 一级概率或二级置信度/100
- `source`: `first_stage` / `deepseek` / `rule+deepseek` / `fallback_first_stage`
- `ds_type`: 二级语义类型（如 `SQL注入` / `XSS攻击` / `Web枚举/目录扫描` / `正常流量`）


import requests
import json
import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

class SemanticAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # 简单内存缓存：key -> (timestamp, result)
        self._cache = {}

    def _extract_usage(self, response_json):
        """
        尝试从 DeepSeek/OpenAI 风格返回中提取 usage。
        期望形如：
          {"usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}}
        """
        try:
            u = (response_json or {}).get("usage") or {}
            if not isinstance(u, dict):
                return {}
            out = {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if k in u:
                    try:
                        out[k] = int(u.get(k))
                    except Exception:
                        pass
            # 尝试提取缓存命中输入 token（不同接口字段名可能不同）
            cached_candidates = [
                u.get("cached_tokens"),
                (u.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(u.get("prompt_tokens_details"), dict) else None,
                (u.get("input_tokens_details") or {}).get("cached_tokens") if isinstance(u.get("input_tokens_details"), dict) else None,
            ]
            cached_val = None
            for cv in cached_candidates:
                if cv is None:
                    continue
                try:
                    cached_val = int(cv)
                    break
                except Exception:
                    continue
            if cached_val is not None:
                out["cached_prompt_tokens"] = max(0, cached_val)
            return out
        except Exception:
            return {}

    def _calc_cost_cny(self, usage: dict) -> float:
        """
        按环境变量估算费用（元）：
          - DEEPSEEK_PRICE_IN_UNCACHED_PER_1M : 输入(未缓存命中)单价（每 1M tokens）
          - DEEPSEEK_PRICE_IN_CACHED_PER_1M   : 输入(缓存命中)单价（每 1M tokens）
          - DEEPSEEK_PRICE_OUT_PER_1M : 输出 token 单价（每 1M tokens）
        兼容旧变量：
          - DEEPSEEK_PRICE_IN_PER_1M（若新变量未设置，则作为未缓存输入单价）
        若未配置任何单价，则返回 0.0（仅回显 tokens，不估价）。
        """
        try:
            in_uncached_price = float(
                os.environ.get("DEEPSEEK_PRICE_IN_UNCACHED_PER_1M", os.environ.get("DEEPSEEK_PRICE_IN_PER_1M", "0")) or "0"
            )
            in_cached_price = float(os.environ.get("DEEPSEEK_PRICE_IN_CACHED_PER_1M", "0") or "0")
            out_price = float(os.environ.get("DEEPSEEK_PRICE_OUT_PER_1M", "0") or "0")
        except Exception:
            return 0.0

        if in_uncached_price <= 0 and in_cached_price <= 0 and out_price <= 0:
            return 0.0

        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        cpt = int(usage.get("cached_prompt_tokens", 0) or 0)
        cpt = max(0, min(cpt, pt))
        upt = max(0, pt - cpt)

        # 兼容只给 total_tokens 的场景：当缺失 pt/ct 时，把 total 当未缓存输入估算
        if pt <= 0 and ct <= 0:
            tt = int(usage.get("total_tokens", 0) or 0)
            return (tt / 1_000_000.0) * in_uncached_price

        return (
            (upt / 1_000_000.0) * in_uncached_price
            + (cpt / 1_000_000.0) * in_cached_price
            + (ct / 1_000_000.0) * out_price
        )

    def _cache_get(self, key, ttl: int):
        import time
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, val = entry
        if time.time() - ts > ttl:
            del self._cache[key]
            return None
        return val

    def _cache_set(self, key, value):
        import time
        self._cache[key] = (time.time(), value)

    def analyze_flow(self, flow_features):
        """
        输入：流特征（字典形式，包含源IP、目的IP、端口、协议、包长统计等）
        输出：攻击意图标签、语义描述、置信度
        """
        # 构造提示词（这是关键：要让大模型理解网络安全任务）
        prompt = self._build_prompt(flow_features)
        
        data = {
            "model": "deepseek-chat",  # 使用V3模型，速度较快
            "messages": [
                {"role": "system", "content": "你是一个专业的网络安全分析专家，需要根据网络流特征判断是否存在攻击行为。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # 降低随机性，保证结果稳定
            "max_tokens": 500,
            "stream": False
        }
        
        # 记录开始时间，确保单次调用<2秒
        start = time.time()
        
        try:
            response = requests.post(self.url, headers=self.headers, json=data, timeout=5)
            result = response.json()
            
            # 解析返回的语义结果
            analysis = self._parse_response(result)
            if not isinstance(analysis, dict):
                analysis = {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}

            usage = self._extract_usage(result)
            # 若接口未返回缓存命中细节，显式标识为 0（便于前端/日志展示）
            if "cached_prompt_tokens" not in usage and "prompt_tokens" in usage:
                usage["cached_prompt_tokens"] = 0
            if "prompt_tokens" in usage:
                try:
                    usage["uncached_prompt_tokens"] = max(0, int(usage.get("prompt_tokens", 0)) - int(usage.get("cached_prompt_tokens", 0)))
                except Exception:
                    usage["uncached_prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            analysis["ds_usage"] = usage
            est_cost = float(self._calc_cost_cny(usage))
            # 新字段：元；保留旧字段兼容已有调用方
            analysis["ds_cost_cny"] = est_cost
            analysis["ds_cost_usd"] = est_cost
            
            elapsed = time.time() - start
            print(f"API调用耗时：{elapsed:.2f}秒")
            analysis["ds_elapsed_sec"] = float(elapsed)
            
            return analysis
            
        except Exception as e:
            print(f"API调用失败：{e}")
            return {"attack_type": "unknown", "confidence": 0, "explanation": f"调用失败: {e}", "semantic_tags": [], "ds_usage": {}, "ds_cost_cny": 0.0, "ds_cost_usd": 0.0}
    
    def _build_prompt(self, features):
        """精心设计的提示词模板，引导模型输出结构化结果"""
        prompt = f"""
请分析以下网络流特征，判断是否存在攻击行为，并按要求输出JSON格式结果。

流特征信息：
- 源IP：{features.get('src_ip', 'unknown')}
- 目的IP：{features.get('dst_ip', 'unknown')}
- 源端口：{features.get('src_port', 'unknown')}
- 目的端口：{features.get('dst_port', 'unknown')}
- 协议：{features.get('protocol', 'unknown')}
- 总包数：{features.get('packet_count', 0)}
- 总字节数：{features.get('byte_count', 0)}
- 平均包长：{features.get('avg_pkt_len', 0)}
- 流持续时间(秒)：{features.get('duration', 0)}
- 每秒包数：{features.get('packets_per_sec', 0)}
- SYN包比例：{features.get('syn_ratio', 0)}
- 目的端口数量(同一源IP)：{features.get('dst_port_count', 0)}
- 连接数统计(窗口内)：{features.get('conn_count', 0)}

（若有 HTTP 证据，优先用于语义判断）
- HTTP 方法：{features.get('http_method', '') or features.get('http_methods', '')}
- HTTP 请求行样本：{features.get('http_req_lines_sample', [])}
- HTTP 路径样本：{features.get('http_paths_sample', [])}
- HTTP 路径枚举数量：{features.get('http_path_count', 0)}
- HTTP 查询样本：{features.get('http_queries_sample', [])}
- SQLi 关键词提示：{features.get('http_has_sqli_hint', False)}（{features.get('http_sqli_tokens', [])}）
- XSS 关键词提示：{features.get('http_has_xss_hint', False)}（{features.get('http_xss_tokens', [])}）

请从以下维度分析：
1. 攻击意图类型：选择一项
   ["正常流量", "目录遍历", "文件上传", "命令执行", "SQL注入", "XSS攻击", "暴力破解", "端口扫描", "DDoS攻击", "Web枚举/目录扫描", "其他恶意行为"]
   规则提示：
   - 若 SQLi 关键词提示为 True（或能从请求行/查询明显看到注入语句），优先选择 "SQL注入"；
   - 若 XSS 关键词提示为 True（或请求里出现典型脚本/事件处理器），优先选择 "XSS攻击"；
   - 若 HTTP 路径呈现“批量枚举/大量不同路径/目录探测”特征，优先选择 "Web枚举/目录扫描"；
   - 若 HTTP 路径枚举数量 >= 3 且没有明显 SQLi/XSS 关键词提示，通常倾向 "Web枚举/目录扫描"；
     但若路径仅为常见站点发现资源（例如 /、/index.html、/robots.txt、/sitemap.xml、/favicon.ico 等），属于正常爬虫/SEO/首页访问，应选择 "正常流量"，不得仅因路径数量>=3 判为 "Web枚举/目录扫描"；
   - 若目标端口/连接行为更像端口探测（例如 SYN-only 多端口/连接失败比例更高），再选择 "端口扫描"。
2. 置信度：0-100之间的整数
3. 详细解释：用一句话说明判断依据
4. 生成3个语义标签：用于后续检测模型的增强特征（如："高频SYN包", "目的端口发散", "短连接"等）

重要约束（降低误报）：
- 当 “SQLi 关键词提示”为 False 且 SQLi tokens 为空，且 HTTP 请求行/查询样本中未出现明显注入片段（例如 union/select/sleep/benchmark/information_schema/-- 等），禁止输出 "SQL注入"；
- 当 “XSS 关键词提示”为 False 且 XSS tokens 为空，且样本中未出现 <script/onerror=/javascript: 等，禁止输出 "XSS攻击"。

请以JSON格式返回，例如：
{{"attack_type": "SQL注入", "confidence": 85, "explanation": "请求查询参数包含典型注入片段（如 OR/UNION/注释符），符合SQL注入行为", "semantic_tags": ["SQLi关键字", "异常参数模式", "短时多请求"]}}
"""
        return prompt
    
    def _parse_response(self, response):
        """解析API返回的JSON结果"""
        try:
            content = response['choices'][0]['message']['content']
            # 提取JSON部分（防止模型输出多余文本）
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}
        except:
            return {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}

    def batch_analyze(self, flows, batch_size: int = 16, cache_ttl: int = 3600, timeout: int = 5):
        """
        批量分析接口（同步实现，带简单缓存）。
        - flows: List[dict]，每个 dict 是单条流特征（与 analyze_flow 接受的格式一致）
        - batch_size: 每次批量内循环提交的大小（本实现内部仍是逐条调用 analyze_flow，以便重用现有逻辑）
        - cache_ttl: 缓存有效期（秒）
        返回：与 flows 等长的结果列表
        注意：将此处替换为并行/异步批量调用可进一步优化性能。
        """
        import hashlib, json, time

        results = [None] * len(flows)
        to_call = []  # 列表 of (index, flow, key)

        # 先尝试缓存命中
        for i, f in enumerate(flows):
            key = hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest()
            cached = self._cache_get(key, cache_ttl)
            if cached is not None:
                results[i] = cached
            else:
                to_call.append((i, f, key))

        # 按 batch_size 逐批调用（当前为同步逐条调用 analyze_flow）
        for start in range(0, len(to_call), batch_size):
            batch = to_call[start:start+batch_size]
            for idx, flow, key in batch:
                try:
                    res = self.analyze_flow(flow)
                except Exception as e:
                    res = {"attack_type": "unknown", "confidence": 0, "explanation": f"调用失败: {e}", "semantic_tags": []}
                # 保存缓存并放回结果
                self._cache_set(key, res)
                results[idx] = res
                # 小的节拍间隔，防止短时间内过快请求
                time.sleep(0.05)

        return results

# ----------------- 文件级辅助函数与推理骨架（精简版，无二级训练模型） -----------------

# 仅命中这些路径组合时，属于常见首页/SEO/爬虫礼貌访问，不等同于「目录扫描/爆破枚举」攻击特征。
_BENIGN_WEB_DISCOVERY_PATHS = frozenset(
    {
        "/",
        "/index.html",
        "/index.htm",
        "/default.html",
        "/default.htm",
        "/robots.txt",
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/favicon.ico",
    }
)


def _normalize_http_path_for_rule(p: str) -> str:
    s = str(p).strip()
    if not s:
        return ""
    if "?" in s:
        s = s.split("?", 1)[0]
    if "#" in s:
        s = s.split("#", 1)[0]
    s = s.strip()
    if not s.startswith("/"):
        s = "/" + s.lstrip("/")
    return s.lower()


def _parse_http_paths_sample_from_flow(flow: dict):
    import ast

    raw = flow.get("http_paths_sample", flow.get("http_paths"))
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                raw = ast.literal_eval(s)
            except Exception:
                return [_normalize_http_path_for_rule(s)]
        else:
            return [_normalize_http_path_for_rule(s)]
    if isinstance(raw, (list, tuple)):
        return [_normalize_http_path_for_rule(x) for x in raw if str(x).strip()]
    return []


def _flow_is_benign_web_discovery_only(flow: dict) -> bool:
    """http_paths_sample 中的路径若全部落在白名单内，则视为正常站点发现流量（用于降低 Web 枚举误报）。"""
    paths = _parse_http_paths_sample_from_flow(flow)
    if not paths:
        return False
    for p in paths:
        if not p or p not in _BENIGN_WEB_DISCOVERY_PATHS:
            return False
    return True


def load_first_stage_model(path: str):
	"""加载第一阶段 LightGBM 模型（joblib 保存）。"""
	import joblib
	return joblib.load(path)


def extract_features_matrix(flow_batch):
	"""
	从流列表构建特征矩阵。
	返回 numpy 数组，shape=(N, D)
	"""
	import numpy as np
	X = []
	for f in flow_batch:
		X.append([
			f.get('packet_count', 0),
			f.get('byte_count', 0),
			f.get('avg_pkt_len', 0),
			f.get('duration', 0),
			f.get('packets_per_sec', 0),
			f.get('syn_ratio', 0),
			f.get('dst_port_count', 0),
			f.get('byte_per_sec', 0),
			f.get('rst_ratio', 0),
			f.get('fin_ratio', 0),
			f.get('ack_ratio', 0),
			f.get('conn_count', 0),
		])
	return np.array(X)


def infer_batch(flow_batch, first_stage_model, semantic_analyzer, low=0.1, high=0.9, ds_conf_thr=70, batch_size=16):
	"""
	基于阈值的简化级联推理：
	- first_stage_model: 已加载的 LightGBM 模型，需实现 predict_proba(X) 方法
	- semantic_analyzer: SemanticAnalyzer 实例（需实现 batch_analyze）
	返回与 flow_batch 等长的结果列表，每项含 label(0/1)、score、source 等字段
	"""
	X = extract_features_matrix(flow_batch)
	# 兼容旧模型（7维）和新模型（11维）
	model_dim = getattr(first_stage_model, "n_features_in_", None)
	if isinstance(model_dim, int) and model_dim > 0 and X.shape[1] != model_dim:
		if X.shape[1] > model_dim:
			X = X[:, :model_dim]
		else:
			import numpy as np
			pad = np.zeros((X.shape[0], model_dim - X.shape[1]), dtype=X.dtype)
			X = np.hstack([X, pad])
	probs = first_stage_model.predict_proba(X)[:, 1]
	results = [None] * len(flow_batch)
	to_ds = []
	idx_map = []

	for i, p in enumerate(probs):
		flow = flow_batch[i]
		http_path_count = int(flow.get("http_path_count", 0) or 0)
		raw_sqli_tokens = flow.get("http_sqli_tokens", [])
		if isinstance(raw_sqli_tokens, (list, tuple)):
			sqli_tokens = [str(x).strip() for x in raw_sqli_tokens if str(x).strip()]
		else:
			sqli_tokens = []
		# 兼容历史数据：以前会把 "/*" "*/" 误当作 SQLi token（来自正常 HTTP 头的 */*）
		legacy_only_comment_tokens = False
		if isinstance(raw_sqli_tokens, (list, tuple)) and raw_sqli_tokens:
			_raw = [str(x).strip() for x in raw_sqli_tokens if str(x).strip()]
			legacy_only_comment_tokens = (len(_raw) > 0 and all(t in ("/*", "*/") for t in _raw))
		sqli_tokens = [t for t in sqli_tokens if t not in ("/*", "*/")]
		has_sqli_hint = bool(flow.get("http_has_sqli_hint", False))
		if legacy_only_comment_tokens:
			has_sqli_hint = False
		has_sqli = has_sqli_hint or (len(sqli_tokens) > 0)
		has_xss = bool(flow.get("http_has_xss_hint", False)) or (
			isinstance(flow.get("http_xss_tokens", []), (list, tuple)) and len(flow.get("http_xss_tokens", [])) > 0
		)
		has_http_method = isinstance(flow.get("http_methods", None), (list, tuple)) and len(flow.get("http_methods", [])) > 0

		# “任意 HTTP 迹象”：用于调试/展示
		http_evidence_any = has_http_method or http_path_count > 0
		# “强证据”：用于决定是否值得触发二级（降低正常流量误报与 DeepSeek 成本）
		# - SQLi/XSS 提示直接算强证据
		# - 目录枚举：路径数>=3 更像扫描；但若仅为 /、/robots.txt、sitemap 等常见站点发现路径，则不算强证据
		benign_discovery_only = _flow_is_benign_web_discovery_only(flow)
		http_evidence_strong = has_sqli or has_xss or (http_path_count >= 3 and not benign_discovery_only)
		if p >= high:
			results[i] = {"label": 1, "score": float(p), "source": "first_stage"}
		elif p <= low:
			# 关键：若已经抽取到 HTTP 语义证据，避免被第一阶段直接误过滤。
			if http_evidence_strong:
				to_ds.append(flow)
				idx_map.append(i)
			else:
				results[i] = {"label": 0, "score": float(p), "source": "first_stage"}
		else:
			to_ds.append(flow)
			idx_map.append(i)

	if to_ds:
		ds_outs = semantic_analyzer.batch_analyze(to_ds, batch_size=batch_size)
		MALICIOUS_TAGS = {"高频SYN", "目的端口发散", "短时多连接"}

		for k, ds in enumerate(ds_outs):
			i = idx_map[k]
			flow = to_ds[k]
			ds_conf = ds.get('confidence', 0)
			ds_type = ds.get('attack_type', '正常流量')
			tags = set(ds.get('semantic_tags', []))
			# 二级模型有时会把“有 SQLi/XSS 证据”的流归到“Web 枚举/目录扫描”这类更泛的类。
			# 为了让最终展示更精确：如果 flow 已经抽取到 SQLi/XSS 证据，则强制覆盖 ds_type。
			raw_sqli_tokens2 = flow.get("http_sqli_tokens", [])
			if isinstance(raw_sqli_tokens2, (list, tuple)):
				sqli_tokens2 = [str(x).strip() for x in raw_sqli_tokens2 if str(x).strip()]
			else:
				sqli_tokens2 = []
			legacy_only_comment_tokens2 = False
			if isinstance(raw_sqli_tokens2, (list, tuple)) and raw_sqli_tokens2:
				_raw2 = [str(x).strip() for x in raw_sqli_tokens2 if str(x).strip()]
				legacy_only_comment_tokens2 = (len(_raw2) > 0 and all(t in ("/*", "*/") for t in _raw2))
			sqli_tokens2 = [t for t in sqli_tokens2 if t not in ("/*", "*/")]
			has_sqli_hint2 = bool(flow.get("http_has_sqli_hint", False))
			if legacy_only_comment_tokens2:
				has_sqli_hint2 = False
			flow_has_sqli = has_sqli_hint2 or (len(sqli_tokens2) > 0)
			flow_has_xss = bool(flow.get("http_has_xss_hint", False)) or (
				isinstance(flow.get("http_xss_tokens", []), (list, tuple)) and len(flow.get("http_xss_tokens", [])) > 0
			)
			# 仅当 flow 侧证据明确时才强制覆盖类型，避免二级模型偶发输出标签导致误报。
			if flow_has_sqli:
				ds_type = "SQL注入"
			elif flow_has_xss:
				ds_type = "XSS攻击"

			# 常见 SEO/首页资源被误标为「Web枚举」时，强制为正常（与 benign 路径白名单一致）。
			if ds_type == "Web枚举/目录扫描" and _flow_is_benign_web_discovery_only(flow):
				ds_type = "正常流量"

			# 证据约束：如果 DeepSeek 给出 SQLi/XSS 但 flow 侧没有任何对应证据，则降级为正常，避免误报。
			if ds_type == "SQL注入" and (not flow_has_sqli):
				ds_type = "正常流量"
			if ds_type == "XSS攻击" and (not flow_has_xss):
				ds_type = "正常流量"
			ds_usage = ds.get("ds_usage") if isinstance(ds, dict) else None
			ds_cost = None
			if isinstance(ds, dict):
				ds_cost = ds.get("ds_cost_cny", ds.get("ds_cost_usd"))
			ds_elapsed = ds.get("ds_elapsed_sec") if isinstance(ds, dict) else None

			# 简单规则决策（无二级训练模型）
			if ds_conf >= ds_conf_thr and ds_type != '正常流量':
				results[i] = {"label": 1, "score": ds_conf / 100.0, "source": "deepseek", "ds_type": ds_type, "tags": list(tags), "ds_usage": ds_usage, "ds_cost_cny": ds_cost, "ds_elapsed_sec": ds_elapsed}
			elif ds_conf >= ds_conf_thr and ds_type == '正常流量':
				results[i] = {"label": 0, "score": ds_conf / 100.0, "source": "deepseek", "ds_type": ds_type, "ds_usage": ds_usage, "ds_cost_cny": ds_cost, "ds_elapsed_sec": ds_elapsed}
			elif tags & MALICIOUS_TAGS:
				results[i] = {"label": 1, "score": float(probs[i]), "source": "rule+deepseek", "tags": list(tags)}
			else:
				results[i] = {"label": 0, "score": float(probs[i]), "source": "fallback_first_stage"}

	return results


def create_dummy_first_stage_model(path: str):
    """训练并保存一个示例性的第一阶段模型（随机森林），仅用于演示。
    特征与 extract_features_matrix 保持一致。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 生成合成数据
    X = []
    y = []
    rng = np.random.RandomState(42)
    for _ in range(2000):
        packet_count = rng.poisson(20)
        byte_count = packet_count * rng.randint(40,120)
        avg_pkt_len = byte_count / max(1, packet_count)
        duration = max(1, rng.exponential(5))
        pps = packet_count / duration
        syn_ratio = rng.rand()
        dst_port_count = rng.poisson(1 + 0.1*packet_count)
        byte_per_sec = byte_count / max(1e-3, duration)
        rst_ratio = rng.beta(1.2, 6.0)
        fin_ratio = rng.beta(1.5, 5.0)
        ack_ratio = rng.beta(3.5, 2.0)
        conn_count = int(rng.poisson(2 + 0.05 * packet_count))
        # 简单规则生成标签：高 syn_ratio 或 高 pps 视为异常
        label = 1 if (syn_ratio > 0.6 or pps > 50 or dst_port_count > 10 or rst_ratio > 0.5 or conn_count > 30) else 0
        X.append([packet_count, byte_count, avg_pkt_len, duration, pps, syn_ratio, dst_port_count, byte_per_sec, rst_ratio, fin_ratio, ack_ratio, conn_count])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, path)
    print(f"已训练并保存示例第一阶段模型到: {path}")


NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
NSL_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]


def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path
    print(f"下载 {url} -> {dest_path} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(r.content)
    return dest_path


def prepare_nsl_flows(file_path: str, max_samples: int = 1000):
    """从 NSL-KDD 原始文件中读取数据，映射为当前 pipeline 使用的流特征列表。
    返回 flows: List[dict], labels: List[int]
    会尽量进行类别平衡（正常 vs 攻击），样本总数不超过 max_samples。
    """
    df = pd.read_csv(file_path, header=None, names=NSL_COLUMNS)
    # 清洗标签（去掉末尾的点），并标注二分类
    df['label_clean'] = df['label'].astype(str).apply(lambda s: s.strip().rstrip('.'))
    df['is_attack'] = df['label_clean'].apply(lambda s: 0 if s == 'normal' else 1)

    # 按类别采样，保持平衡（或尽量）
    n_half = max_samples // 2
    normals = df[df['is_attack'] == 0]
    attacks = df[df['is_attack'] == 1]

    normals_sample = normals.sample(n=min(len(normals), n_half), random_state=42)
    attacks_sample = attacks.sample(n=min(len(attacks), n_half), random_state=42)

    sampled = pd.concat([normals_sample, attacks_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    flows = []
    labels = []
    for _, row in sampled.iterrows():
        src_bytes = float(row.get('src_bytes', 0))
        dst_bytes = float(row.get('dst_bytes', 0))
        byte_count = src_bytes + dst_bytes
        # 估算 packet_count：假设平均包长约为 60 字节
        packet_count = int(max(1, round(byte_count / 60.0)))
        avg_pkt_len = byte_count / packet_count if packet_count > 0 else 0
        duration = float(row.get('duration', 0.0))
        packets_per_sec = packet_count / (duration + 1e-6)
        byte_per_sec = byte_count / (duration + 1e-6)
        # 使用 serror_rate 作为 syn_ratio 的代理（0-1）
        syn_ratio = float(row.get('serror_rate', 0.0))
        dst_port_count = int(min(100, row.get('dst_host_count', 1)))

        flows.append({
            'src_ip': '0.0.0.0',
            'dst_ip': '0.0.0.0',
            'packet_count': packet_count,
            'byte_count': byte_count,
            'avg_pkt_len': avg_pkt_len,
            'duration': duration,
            'packets_per_sec': packets_per_sec,
            'syn_ratio': syn_ratio,
            'dst_port_count': dst_port_count,
            'byte_per_sec': byte_per_sec,
            'rst_ratio': 0.0,
            'fin_ratio': 0.0,
            'ack_ratio': 0.0,
        })
        labels.append(int(row['is_attack']))

    return flows, labels


def train_first_stage_from_nsl_kdd(model_path: str, max_samples: int = 1000):
    """下载 NSL-KDD、构建流特征并训练 LightGBM 第一阶段模型，样本数上限为 max_samples。"""
    data_dir = 'data/nsl_kdd'
    train_file = os.path.join(data_dir, 'KDDTrain+.txt')
    download_file(NSL_KDD_TRAIN_URL, train_file)

    flows, labels = prepare_nsl_flows(train_file, max_samples=max_samples)
    X = extract_features_matrix(flows)
    y = np.array(labels)

    # 训练 LightGBM
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"已基于 NSL-KDD 训练并保存 LightGBM 模型到: {model_path} (样本数={len(y)})")
    return clf


def prepare_cicids_flows(file_path: str, max_samples: int = 1000, balance: bool = True):
    """从 CICIDS CSV 中读取并映射为 pipeline 使用的流特征格式。
    支持常见字段名，若某些字段缺失会使用近似替代。
    返回 flows: List[dict], labels: List[int]
    - max_samples: 最大样本数上限；若为 None 则不下采样
    - balance: 若为 True 则尽量保持正负平衡（适合快速验证）；若为 False 则保留原始分布（适合做更真实的召回优化）
    """
    # 更鲁棒地读取：关闭 low_memory 并去除列名前后空白
    df = pd.read_csv(file_path, low_memory=False)
    # 去除列名的前后空白（很多 CICIDS 导出的列名存在前导空格）
    df.columns = df.columns.str.strip()
    
    # 如果用户/上游脚本已经生成了 __y__ 二值列，优先使用它，避免重复映射不一致
    if '__y__' in df.columns:
        print(f"检测到已经存在的二值标签列 '__y__'，将直接使用该列（样本数={len(df)})")
        # 只抽样不超过 max_samples
        if max_samples is None:
            sampled_df = df
        else:
            sampled_df = df.sample(n=min(len(df), max_samples), random_state=42) if len(df) > max_samples else df

        # 为兼容 merge_cicids.py 生成的 CICIDS 样本，尝试从常见列名映射回需要的特征
        cols_lower = {c.lower(): c for c in sampled_df.columns}
        def get_val(row, candidates, default=0.0):
            for cand in candidates:
                if cand.lower() in cols_lower:
                    try:
                        return row.get(cols_lower[cand.lower()])
                    except Exception:
                        return default
            return default

        flows = []
        labels = []
        for _, row in sampled_df.iterrows():
            # 常见列名候选
            fwd_pkts = get_val(row, ['Total Fwd Packets', 'Tot Fwd Pkts', 'Total Forward Packets', 'TotFwdPkt'], 0)
            bwd_pkts = get_val(row, ['Total Backward Packets', 'Tot Bwd Pkts', 'Total Backward Packets', 'TotBwdPkt'], 0)
            try:
                fwd_pkts = float(fwd_pkts) if pd.notna(fwd_pkts) else 0.0
            except Exception:
                fwd_pkts = 0.0
            try:
                bwd_pkts = float(bwd_pkts) if pd.notna(bwd_pkts) else 0.0
            except Exception:
                bwd_pkts = 0.0

            packet_count = int(max(1, round(fwd_pkts + bwd_pkts)))

            fwd_len = get_val(row, ['Total Length of Fwd Packets', 'TotalLengthofFwdPackets', 'TotLen Fwd'], 0)
            bwd_len = get_val(row, ['Total Length of Bwd Packets', 'TotalLengthofBwdPackets', 'TotLen Bwd'], 0)
            try:
                fwd_len = float(fwd_len) if pd.notna(fwd_len) else 0.0
            except Exception:
                fwd_len = 0.0
            try:
                bwd_len = float(bwd_len) if pd.notna(bwd_len) else 0.0
            except Exception:
                bwd_len = 0.0

            byte_count = fwd_len + bwd_len

            avg_pkt_len = get_val(row, ['Packet Length Mean', 'Packet Length Mean ', 'Packet Length Mean'], None)
            try:
                if avg_pkt_len is not None and pd.notna(avg_pkt_len):
                    avg_pkt_len = float(avg_pkt_len)
                else:
                    avg_pkt_len = (byte_count / packet_count) if packet_count > 0 else 0.0
            except Exception:
                avg_pkt_len = (byte_count / packet_count) if packet_count > 0 else 0.0

            dur_raw = get_val(row, ['Flow Duration', 'FlowDuration', 'flow_duration'], None)
            try:
                if dur_raw is None or pd.isna(dur_raw):
                    duration = 1.0
                else:
                    dur_raw = float(dur_raw)
                    duration = dur_raw / 1_000_000.0 if dur_raw > 1e6 else dur_raw
            except Exception:
                duration = 1.0

            packets_per_sec = packet_count / (duration + 1e-6)

            syn_cnt = get_val(row, ['SYN Flag Count', 'SYN Count', 'syn_flag_count'], None)
            try:
                syn_cnt = float(syn_cnt) if syn_cnt is not None and pd.notna(syn_cnt) else None
            except Exception:
                syn_cnt = None
            if syn_cnt is not None:
                syn_ratio = min(1.0, syn_cnt / max(1.0, packet_count))
            else:
                serror = get_val(row, ['serror_rate', 'SYN Flag Count'], 0)
                try:
                    syn_ratio = float(serror) if pd.notna(serror) else 0.0
                except Exception:
                    syn_ratio = 0.0

            dst_port_count = get_val(row, ['dst_host_count', 'Dst Host Count', 'dst_port_count'], 1)
            try:
                dst_port_count = int(min(100, float(dst_port_count)))
            except Exception:
                dst_port_count = 1
            byte_per_sec = float(byte_count) / max(1e-3, float(duration))
            rst_cnt = get_val(row, ['RST Flag Count', 'rst_flag_count', 'RST Count'], 0)
            fin_cnt = get_val(row, ['FIN Flag Count', 'fin_flag_count', 'FIN Count'], 0)
            ack_cnt = get_val(row, ['ACK Flag Count', 'ack_flag_count', 'ACK Count'], 0)
            try:
                rst_ratio = min(1.0, float(rst_cnt) / max(1.0, float(packet_count)))
            except Exception:
                rst_ratio = 0.0
            try:
                fin_ratio = min(1.0, float(fin_cnt) / max(1.0, float(packet_count)))
            except Exception:
                fin_ratio = 0.0
            try:
                ack_ratio = min(1.0, float(ack_cnt) / max(1.0, float(packet_count)))
            except Exception:
                ack_ratio = 0.0

            flows.append({
                'src_ip': row.get('src_ip', '0.0.0.0'),
                'dst_ip': row.get('dst_ip', '0.0.0.0'),
                'packet_count': int(packet_count),
                'byte_count': float(byte_count),
                'avg_pkt_len': float(avg_pkt_len),
                'duration': float(duration),
                'packets_per_sec': float(packets_per_sec),
                'syn_ratio': float(syn_ratio),
                'dst_port_count': int(dst_port_count),
                'byte_per_sec': float(byte_per_sec),
                'rst_ratio': float(rst_ratio),
                'fin_ratio': float(fin_ratio),
                'ack_ratio': float(ack_ratio),
                'conn_count': 0,
            })
            labels.append(int(row.get('__y__', 0)))
        # 打印分布用于诊断
        from collections import Counter
        print('prepare_cicids_flows(__y__): 样本数=', len(labels), '标签分布=', dict(Counter(labels)))
        return flows, labels

    # 常见列名映射（不同来源列名可能略有差异）
    # 统一使用小写比较避免空格/大小写引发的问题
    cols_lower = {c.lower(): c for c in df.columns}
    def safe_col(cols, fallback=None):
        for c in cols:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return fallback

    col_fwd_pkts = safe_col(['TotFwdPkt','Tot Fwd Pkts','Total Fwd Packets','Tot Fwd Pkts','total fwd packets','total_fwd_packets'], None)
    col_bwd_pkts = safe_col(['TotBwdPkt','Tot Bwd Pkts','Total Backward Packets','Tot Bwd Pkts','total backward packets','total_bwd_packets'], None)
    col_fwd_len = safe_col(['TotalLengthofFwdPackets','Total Fwd Packet Length','Total Length of Fwd Packets','TotLen Fwd','total length of fwd packets'], None)
    col_bwd_len = safe_col(['TotalLengthofBwdPackets','Total Bwd Packet Length','Total Length of Bwd Packets','TotLen Bwd','total length of bwd packets'], None)
    col_flow_dur = safe_col(['Flow Duration','flow_duration','FlowDuration','Flow_Duration','flow duration'], None)
    col_avg_fwd_len = safe_col(['Fwd Packet Length Mean','fwd_pkt_len_mean','FwdPktLenMean','fwd packet length mean'], None)
    col_syn_cnt = safe_col(['SYN Flag Count','syn_flag_count','syn_cnt','Fwd SYN Flag Count','syn flag count'], None)
    col_rst_cnt = safe_col(['RST Flag Count', 'rst_flag_count', 'RST Count'], None)
    col_fin_cnt = safe_col(['FIN Flag Count', 'fin_flag_count', 'FIN Count'], None)
    col_ack_cnt = safe_col(['ACK Flag Count', 'ack_flag_count', 'ACK Count'], None)
    col_serror_rate = safe_col(['serror_rate','SYN Flag Count'], None)
    col_label = safe_col(['Label','label','Flow Label','flow label','labelled','Labelled'], None)

    # If label column missing, attempt to find 'label'/'attack' like columns
    if col_label is None:
        possible = [c for c in df.columns if 'label' in c.lower() or 'attack' in c.lower()]
        col_label = possible[0] if possible else None
    
    if col_label:
        print(f"检测到标签列: '{col_label}'，示例值 (前5):", df[col_label].astype(str).head(5).tolist())
    else:
        print("未检测到明显的标签列，所有样本将标记为 0（正常）")

    # Build mapping rows
    records = []
    for _, row in df.iterrows():
        # compute packet_count
        fwd_pkts = float(row[col_fwd_pkts]) if col_fwd_pkts and row.get(col_fwd_pkts, None) is not None else 0.0
        bwd_pkts = float(row[col_bwd_pkts]) if col_bwd_pkts and row.get(col_bwd_pkts, None) is not None else 0.0
        packet_count = int(max(1, round(fwd_pkts + bwd_pkts)))

        # byte count
        fwd_len = float(row[col_fwd_len]) if col_fwd_len and row.get(col_fwd_len, None) is not None else 0.0
        bwd_len = float(row[col_bwd_len]) if col_bwd_len and row.get(col_bwd_len, None) is not None else 0.0
        byte_count = fwd_len + bwd_len
        if byte_count <= 0:
            # fallback: estimate from avg fwd len * packet_count
            if col_avg_fwd_len and row.get(col_avg_fwd_len, None) is not None:
                avg_pkt_len = float(row[col_avg_fwd_len])
                byte_count = avg_pkt_len * packet_count
            else:
                avg_pkt_len = 60.0
                byte_count = avg_pkt_len * packet_count
        else:
            avg_pkt_len = byte_count / packet_count if packet_count > 0 else 0.0

        # duration in seconds; CICIDS often in microseconds
        if col_flow_dur and row.get(col_flow_dur, None) is not None:
            dur_raw = float(row[col_flow_dur])
            # heuristics: if dur_raw > 1e6 assume microseconds
            if dur_raw > 1e6:
                duration = dur_raw / 1_000_000.0
            else:
                duration = dur_raw
        else:
            duration = 1.0

        packets_per_sec = packet_count / (duration + 1e-6)
        byte_per_sec = byte_count / (duration + 1e-6)

        syn_ratio = 0.0
        if col_syn_cnt and row.get(col_syn_cnt, None) is not None:
            try:
                syn_cnt = float(row[col_syn_cnt])
                syn_ratio = min(1.0, syn_cnt / max(1.0, packet_count))
            except:
                syn_ratio = 0.0
        else:
            # fallback: try serror_rate if exists
            if 'serror_rate' in df.columns:
                try:
                    syn_ratio = float(row.get('serror_rate', 0.0))
                except:
                    syn_ratio = 0.0

        # dst_port_count: CICIDS does not have explicit, use dst_host_count or set 1
        dst_port_count = int(min(100, row.get('dst_host_count', 1))) if 'dst_host_count' in df.columns else 1
        try:
            rst_ratio = min(1.0, float(row[col_rst_cnt]) / max(1.0, packet_count)) if col_rst_cnt and row.get(col_rst_cnt, None) is not None else 0.0
        except Exception:
            rst_ratio = 0.0
        try:
            fin_ratio = min(1.0, float(row[col_fin_cnt]) / max(1.0, packet_count)) if col_fin_cnt and row.get(col_fin_cnt, None) is not None else 0.0
        except Exception:
            fin_ratio = 0.0
        try:
            ack_ratio = min(1.0, float(row[col_ack_cnt]) / max(1.0, packet_count)) if col_ack_cnt and row.get(col_ack_cnt, None) is not None else 0.0
        except Exception:
            ack_ratio = 0.0

        label_val = 0
        if col_label and row.get(col_label, None) is not None:
            lab = str(row.get(col_label)).strip()
            # CICIDS labels often like 'BENIGN' or attack name
            if lab.lower() in ('benign', 'normal'):
                label_val = 0
            else:
                label_val = 1

        records.append(({
            'src_ip': '0.0.0.0',
            'dst_ip': '0.0.0.0',
            'packet_count': packet_count,
            'byte_count': byte_count,
            'avg_pkt_len': avg_pkt_len,
            'duration': duration,
            'packets_per_sec': packets_per_sec,
            'syn_ratio': syn_ratio,
            'dst_port_count': dst_port_count,
            'byte_per_sec': byte_per_sec,
            'rst_ratio': rst_ratio,
            'fin_ratio': fin_ratio,
            'ack_ratio': ack_ratio,
            'conn_count': 0,
        }, label_val))

    # 转为 dataframe-like sampling
    df_rec = pd.DataFrame([{'label': r[1], **r[0]} for r in records])
    normals = df_rec[df_rec['label'] == 0]
    attacks = df_rec[df_rec['label'] == 1]

    if max_samples is None:
        sampled = df_rec.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        if balance:
            n_half = max_samples // 2
            normals_sample = normals.sample(n=min(len(normals), n_half), random_state=42) if len(normals)>0 else normals
            attacks_sample = attacks.sample(n=min(len(attacks), n_half), random_state=42) if len(attacks)>0 else attacks
            sampled = pd.concat([normals_sample, attacks_sample])
            sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # 不做类平衡，仅截断到 max_samples
            sampled = df_rec.sample(n=min(len(df_rec), max_samples), random_state=42).reset_index(drop=True)

    flows = []
    labels = []
    for _, row in sampled.iterrows():
        flows.append({
            'src_ip': row.get('src_ip'),
            'dst_ip': row.get('dst_ip'),
            'packet_count': int(row.get('packet_count', 0)),
            'byte_count': float(row.get('byte_count', 0)),
            'avg_pkt_len': float(row.get('avg_pkt_len', 0)),
            'duration': float(row.get('duration', 0)),
            'packets_per_sec': float(row.get('packets_per_sec', 0)),
            'syn_ratio': float(row.get('syn_ratio', 0)),
            'dst_port_count': int(row.get('dst_port_count', 0)),
            'byte_per_sec': float(row.get('byte_per_sec', 0)),
            'rst_ratio': float(row.get('rst_ratio', 0)),
            'fin_ratio': float(row.get('fin_ratio', 0)),
            'ack_ratio': float(row.get('ack_ratio', 0)),
            'conn_count': int(row.get('conn_count', 0)),
        })
        labels.append(int(row.get('label', 0)))

    # 在返回前打印标签分布，便于诊断单类问题
    from collections import Counter
    print('prepare_cicids_flows: 最终样本数=', len(labels), '标签分布=', dict(Counter(labels)), 'balance=', balance, 'max_samples=', max_samples)
    return flows, labels


def train_first_stage_from_cicids(model_path: str, csv_path: str = None, download_url: str = None, max_samples: int = 1000, balance: bool = True, class_weight: str = None):
    """基于 CICIDS CSV（本地或下载）训练第一阶段 LightGBM。
    - csv_path: 本地 CSV 文件路径（优先）
    - download_url: 若未提供本地路径，可提供小样本的下载 URL
    """
    # 默认与仓库整理后的 datasets 布局一致；仍兼容旧路径 data/cicids
    data_dir = os.environ.get("CICIDS_DATA_DIR", "datasets/cicids")
    if not os.path.isdir(data_dir) and os.path.isdir("data/cicids"):
        data_dir = "data/cicids"
    os.makedirs(data_dir, exist_ok=True)

    if csv_path is None and download_url is None:
        raise ValueError('请提供 csv_path 或 download_url 之一以获取 CICIDS 数据')

    if csv_path is None:
        dest = os.path.join(data_dir, 'cicids_sample.csv')
        download_file(download_url, dest)
        csv_path = dest

    print(f'从 CICIDS 文件构建样本：{csv_path}，最大样本数={max_samples}，balance={balance}')
    flows, labels = prepare_cicids_flows(csv_path, max_samples=max_samples, balance=balance)
    X = extract_features_matrix(flows)
    y = np.array(labels)

    # 诊断：打印特征矩阵信息并尝试转换为 float 数组
    try:
        import numpy as _np
        print('训练前诊断: 原始 X 类型=', type(X), 'len(flows)=', len(flows))
        if hasattr(X, 'shape'):
            print('X.shape=', getattr(X, 'shape'))
        # 尝试将 X 转为数值型数组
        X = _np.asarray(X, dtype=float)
        print('转换后 X.shape=', X.shape, 'dtype=', X.dtype)
        # 统计 NaN
        nan_count = int(_np.isnan(X).sum()) if X.size > 0 else 0
        print('特征矩阵 NaN 数量=', nan_count)
    except Exception as e:
        print('特征矩阵转换为数值数组失败，可能是提取特征时列名不匹配或包含非数值。示例 flows 前3条：')
        import json
        for i, f in enumerate(flows[:3]):
            print(i, json.dumps(f, ensure_ascii=False))
        print('错误:', e)
        raise

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"已基于 CICIDS 训练并保存 LightGBM 模型到: {model_path} (样本数={len(y)})")
    return clf


class MockSemanticAnalyzer:
    """当未配置 DeepSeek API 时使用的本地 mock 分析器，用于演示 pipeline。
    逻辑简单：根据流特征生成 attack_type/confidence/semantic_tags。
    """
    def batch_analyze(self, flows, batch_size: int = 16, cache_ttl: int = 3600, timeout: int = 5):
        outs = []
        for f in flows:
            syn = f.get('syn_ratio', 0)
            rst_ratio = f.get('rst_ratio', 0)
            pps = f.get('packets_per_sec', 0)
            bps = f.get('byte_per_sec', 0)
            pkt_cnt = f.get('packet_count', 0)
            dst_cnt = f.get('dst_port_count', 0)
            # 避免“单包/极短桶”导致 pps 被人为抬高而全判 DDoS：
            # 需要同时具备较强行为强度（如足够包数/byte_per_sec）。
            if (syn > 0.7 and pkt_cnt >= 5) or (pps > 150 and bps > 20000) or (rst_ratio > 0.3 and pps > 50 and pkt_cnt >= 5):
                outs.append({
                    "attack_type": "DDoS攻击",
                    "confidence": 85,
                    "explanation": "高 SYN 或高 pps + 较高字节速率",
                    "semantic_tags": ["高频SYN", "短时多连接"]
                })
            elif dst_cnt > 10 and (pps > 30 or pkt_cnt >= 5):
                outs.append({"attack_type": "端口扫描", "confidence": 80, "explanation": "目的端口发散", "semantic_tags": ["目的端口发散", "短连接"]})
            else:
                outs.append({"attack_type": "正常流量", "confidence": 90, "explanation": "无异常流量特征", "semantic_tags": []})
        return outs


CICIDS_SAMPLE_URLS = [
    'https://raw.githubusercontent.com/selimfirat/IDS-2018/master/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'https://raw.githubusercontent.com/krishnaik06/IDS-2018/master/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'https://raw.githubusercontent.com/PCAPtools/IDS-2018/master/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]


if __name__ == '__main__':
    # 演示：若不存在模型文件则优先尝试基于 CICIDS 下载并训练模型；若失败则训练示例模型
    api_key = os.environ.get('DEEPSEEK_API_KEY', 'your_api_key_here')
    model_path = 'models/first_stage_lgbm.pkl'

    if not os.path.exists(model_path):
        print('模型文件不存在，尝试基于 CICIDS 下载并训练 LightGBM（样本上限=1000）...')
        trained = False
        for url in CICIDS_SAMPLE_URLS:
            try:
                print(f'尝试下载并训练 CICIDS 样本：{url}')
                train_first_stage_from_cicids(model_path, csv_path=None, download_url=url, max_samples=1000)
                trained = True
                break
            except Exception as e:
                print(f'基于 CICIDS 的训练失败（url={url}）：{e}')
                continue

        if not trained:
            print('CICIDS 训练失败，回退并训练示例模型（随机森林）...')
            create_dummy_first_stage_model(model_path)

    first_stage_model = load_first_stage_model(model_path)

    if api_key == 'your_api_key_here':
        print('未配置 DeepSeek API Key，使用本地 MockSemanticAnalyzer 进行演示。')
        semantic_analyzer = MockSemanticAnalyzer()
    else:
        semantic_analyzer = SemanticAnalyzer(api_key)

    # 示例流（可扩展为从 scapy/pcap 解析得到的真实流）
    flows = [
        {'src_ip':'10.0.0.1','dst_ip':'10.0.0.2','packet_count':10,'byte_count':800,'avg_pkt_len':80,'duration':5,'packets_per_sec':2,'syn_ratio':0.0,'dst_port_count':1},
        {'src_ip':'10.0.0.5','dst_ip':'10.0.0.9','packet_count':2000,'byte_count':120000,'avg_pkt_len':60,'duration':10,'packets_per_sec':200,'syn_ratio':0.9,'dst_port_count':50},
        {'src_ip':'10.0.0.6','dst_ip':'10.0.0.7','packet_count':300,'byte_count':20000,'avg_pkt_len':66,'duration':6,'packets_per_sec':50,'syn_ratio':0.2,'dst_port_count':20},
    ]

    results = infer_batch(flows, first_stage_model, semantic_analyzer, low=0.1, high=0.9, ds_conf_thr=70, batch_size=8)
    print('推理结果：')
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print('\n说明：\n- 若要使用真实 DeepSeek，请在环境变量 DEEPSEEK_API_KEY 中设置 API Key。\n- 若要使用其他 CICIDS 数据集，请下载 CSV 并通过 csv_path 参数调用 train_first_stage_from_cicids。')
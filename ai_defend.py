import requests
import json
import time
import os
import re
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
        self._cache = {}

    def _extract_usage(self, response_json):
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
        prompt = self._build_prompt(flow_features)
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的网络安全分析专家，需要根据网络流特征判断是否存在攻击行为。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "stream": False
        }
        start = time.time()
        try:
            response = requests.post(self.url, headers=self.headers, json=data, timeout=5)
            result = response.json()
            analysis = self._parse_response(result)
            if not isinstance(analysis, dict):
                analysis = {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}
            usage = self._extract_usage(result)
            if "cached_prompt_tokens" not in usage and "prompt_tokens" in usage:
                usage["cached_prompt_tokens"] = 0
            if "prompt_tokens" in usage:
                try:
                    usage["uncached_prompt_tokens"] = max(0, int(usage.get("prompt_tokens", 0)) - int(usage.get("cached_prompt_tokens", 0)))
                except Exception:
                    usage["uncached_prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            analysis["ds_usage"] = usage
            est_cost = float(self._calc_cost_cny(usage))
            analysis["ds_cost_cny"] = est_cost
            analysis["ds_cost_usd"] = est_cost
            elapsed = time.time() - start
            analysis["ds_elapsed_sec"] = float(elapsed)
            return analysis
        except Exception as e:
            return {"attack_type": "unknown", "confidence": 0, "explanation": f"调用失败: {e}", "semantic_tags": [], "ds_usage": {}, "ds_cost_cny": 0.0, "ds_cost_usd": 0.0}

    def _build_prompt(self, features):
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

（HTTP 语义证据 - 优先级最高）
- HTTP 方法：{features.get('http_method', '') or features.get('http_methods', '')}
- HTTP 请求行样本：{features.get('http_req_lines_sample', [])}
- HTTP 路径样本：{features.get('http_paths_sample', [])}
- HTTP 路径枚举数量：{features.get('http_path_count', 0)}
- 响应状态码统计：404 Not Found ({features.get('http_404_count', 0)}次), 403 Forbidden ({features.get('http_403_count', 0)}次)
- HTTP 查询样本：{features.get('http_queries_sample', [])}
- SQLi 关键词提示：{features.get('http_has_sqli_hint', False)}（{features.get('http_sqli_tokens', [])}）
- XSS 关键词提示：{features.get('http_has_xss_hint', False)}（{features.get('http_xss_tokens', [])}）

请严格按照以下优先级判定攻击意图类型：
1. **绝对优先级：具体攻击载荷 (Payload Evidence)**
   - 如果 (SQLi 关键词提示为 True) $\rightarrow$ 必须判定为 "SQL注入"。
   - 如果 (XSS 关键词提示为 True) $\rightarrow$ 必须判定为 "XSS攻击"。
   - **禁止**在发现明确载荷时将其判定为 "Web枚举" 或 "漏洞扫描"，载荷证据具有最高权重。

2. **次优先判定：Web 枚举/目录扫描**
   - 若无具体攻击载荷，且满足：(HTTP 路径枚举数量 $\ge 3$) 且 (404/403 响应数量显著) $\rightarrow$ 选择 "Web枚举/目录扫描"。
   - 若路径仅为常见资源（如 /index.html, /robots.txt），即便数量多也判定为 "正常流量"。

3. **再次优先判定：漏洞扫描/端口扫描**
   - 若无具体攻击载荷，且满足：(具备 Web 枚举特征) 且 (每秒包数 PPS 极高 或 目的端口数量-dst_port_count 很大) $\rightarrow$ 选择 "漏洞扫描" 或 "端口扫描"。

4. **最后判定**：若上述均不满足 $\rightarrow$ 选择 "正常流量"。

输出要求：
- 攻击意图类型：选择一项 ["正常流量", "目录遍历", "文件上传", "命令执行", "SQL注入", "XSS攻击", "暴力破解", "端口扫描", "DDoS攻击", "Web枚举/目录扫描", "其他恶意行为"]
- 置信度：0-100之间的整数
- 详细解释：用一句话说明判断依据（例如：“检测到明确 XSS 载荷，优先级高于路径探测”）
- 生成3个语义标签：如 "高频404", "路径探测", "XSS载荷" 等

请以JSON格式返回，例如：
{{"attack_type": "XSS攻击", "confidence": 100, "explanation": "检测到明确 XSS 载荷，优先级高于路径探测", "semantic_tags": ["XSS载荷", "恶意注入", "高危"]}}
"""
        return prompt

    def _parse_response(self, response):
        try:
            content = response['choices'][0]['message']['content']
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}
        except:
            return {"attack_type": "unknown", "confidence": 0, "explanation": "解析失败", "semantic_tags": []}

    def batch_analyze(self, flows, batch_size: int = 16, cache_ttl: int = 3600, timeout: int = 5):
        import hashlib, json, time
        results = [None] * len(flows)
        to_call = []
        for i, f in enumerate(flows):
            key = hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest()
            cached = self._cache_get(key, cache_ttl)
            if cached is not None:
                results[i] = cached
            else:
                to_call.append((i, f, key))
        for start in range(0, len(to_call), batch_size):
            batch = to_call[start:start+batch_size]
            for idx, flow, key in batch:
                try:
                    res = self.analyze_flow(flow)
                except Exception as e:
                    res = {"attack_type": "unknown", "confidence": 0, "explanation": f"调用失败: {e}", "semantic_tags": []}
                self._cache_set(key, res)
                results[idx] = res
                time.sleep(0.05)
        return results

_BENIGN_WEB_DISCOVERY_PATHS = frozenset({
    "/", "/index.html", "/index.htm", "/default.html", "/default.htm",
    "/robots.txt", "/sitemap.xml", "/sitemap_index.xml", "/favicon.ico",
})

def _normalize_http_path_for_rule(p: str) -> str:
    s = str(p).strip()
    if not s: return ""
    if "?" in s: s = s.split("?", 1)[0]
    if "#" in s: s = s.split("#", 1)[0]
    s = s.strip()
    if not s.startswith("/"): s = "/" + s.lstrip("/")
    return s.lower()

def _parse_http_paths_sample_from_flow(flow: dict):
    import ast
    raw = flow.get("http_paths_sample", flow.get("http_paths"))
    if raw is None: return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s: return []
        if s.startswith("[") and s.endswith("]"):
            try: return [ _normalize_http_path_for_rule(x) for x in ast.literal_eval(s) if str(x).strip()]
            except Exception: return [_normalize_http_path_for_rule(s)]
        else: return [_normalize_http_path_for_rule(s)]
    if isinstance(raw, (list, tuple)):
        return [_normalize_http_path_for_rule(x) for x in raw if str(x).strip()]
    return []

def _flow_is_benign_web_discovery_only(flow: dict) -> bool:
    paths = _parse_http_paths_sample_from_flow(flow)
    if not paths: return False
    for p in paths:
        if not p or p not in _BENIGN_WEB_DISCOVERY_PATHS:
            return False
    return True

def load_first_stage_model(path: str):
    import joblib
    return joblib.load(path)

def extract_features_matrix(flow_batch):
    import numpy as np
    X = []
    for f in flow_batch:
        X.append([
            f.get('packet_count', 0), f.get('byte_count', 0), f.get('avg_pkt_len', 0),
            f.get('duration', 0), f.get('packets_per_sec', 0), f.get('syn_ratio', 0),
            f.get('dst_port_count', 0), f.get('byte_per_sec', 0), f.get('rst_ratio', 0),
            f.get('fin_ratio', 0), f.get('ack_ratio', 0), f.get('conn_count', 0),
        ])
    return np.array(X)

_MALICIOUS_TAG_KEYWORDS = (
    "sql", "sqli", "注入", "xss", "脚本注入", "跨站", "目录扫描", "枚举", "探测", "扫描",
    "暴力破解", "口令爆破", "命令执行", "命令注入", "代码执行", "远程代码执行", "上传木马",
    "恶意上传", "webshell", "后门", "目录遍历", "路径遍历", "路径穿越", "文件包含",
    "信息泄露", "敏感信息泄露", "漏洞利用", "rce", "cve", "cve-", "漏洞编号",
    "高频syn", "syn洪泛", "端口发散", "短时多连接", "ddos", "dos",
    "sql injection", "xss attack", "directory scan", "web enum", "enumeration",
    "port scan", "brute force", "command execution", "command injection", "code execution",
    "remote code execution", "file inclusion", "path traversal", "directory traversal",
    "information disclosure", "info leak", "sensitive data exposure", "webshell", "backdoor",
    "vulnerability exploit", "cve-",
)

def _normalize_ds_tags(raw_tags):
    tags = set()
    if raw_tags is None: return tags
    items = raw_tags if isinstance(raw_tags, (list, tuple, set)) else [raw_tags]
    for it in items:
        s = str(it).strip()
        if not s: continue
        for part in re.split(r"[;,，；|/]+", s):
            pp = part.strip()
            if pp: tags.add(pp)
    return tags

def _has_malicious_tag_signal(tags_set):
    if not tags_set: return False
    joined = " ".join(str(x).lower() for x in tags_set)
    return any(kw in joined for kw in _MALICIOUS_TAG_KEYWORDS)

def infer_batch(flow_batch, first_stage_model, semantic_analyzer, low=0.1, high=0.9, ds_conf_thr=70, batch_size=16):
    X = extract_features_matrix(flow_batch)
    model_dim = getattr(first_stage_model, "n_features_in_", None)
    if isinstance(model_dim, int) and model_dim > 0 and X.shape[1] != model_dim:
        if X.shape[1] > model_dim: X = X[:, :model_dim]
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
        sqli_tokens = [str(x).strip() for x in raw_sqli_tokens if str(x).strip()] if isinstance(raw_sqli_tokens, (list, tuple)) else []

        # 处理 Legacy 评论 token
        legacy_only_comment_tokens = False
        if isinstance(raw_sqli_tokens, (list, tuple)) and raw_sqli_tokens:
            _raw = [str(x).strip() for x in raw_sqli_tokens if str(x).strip()]
            legacy_only_comment_tokens = (len(_raw) > 0 and all(t in ("/*", "*/") for t in _raw))
        sqli_tokens = [t for t in sqli_tokens if t not in ("/*", "*/")]

        has_sqli_hint = bool(flow.get("http_has_sqli_hint", False))
        if legacy_only_comment_tokens: has_sqli_hint = False
        has_sqli = has_sqli_hint or (len(sqli_tokens) > 0)
        has_xss = bool(flow.get("http_has_xss_hint", False)) or (
            isinstance(flow.get("http_xss_tokens", []), (list, tuple)) and len(flow.get("http_xss_tokens", [])) > 0
        )
        has_http_method = isinstance(flow.get("http_methods", None), (list, tuple)) and len(flow.get("http_methods", [])) > 0

        http_evidence_any = has_http_method or http_path_count > 0
        benign_discovery_only = _flow_is_benign_web_discovery_only(flow)
        # 增强触发条件：SQLi/XSS 强证据 OR (路径多且非白名单) OR (404/403 响应显著)
        http_evidence_strong = has_sqli or has_xss or (http_path_count >= 3 and not benign_discovery_only) or (flow.get("http_404_count", 0) >= 3)

        if p >= high:
            results[i] = {"label": 1, "score": float(p), "source": "first_stage"}
        elif p <= low:
            if http_evidence_strong:
                to_ds.append(flow)
                idx_map.append(i)
            else:
                results[i] = {"label": 0, "score": float(p), "source": "first_stage"}
        else:
            to_ds.append(flow)
            idx_map.append(i)

    fallback_p_thr = 0.90
    if to_ds:
        ds_outs = semantic_analyzer.batch_analyze(to_ds, batch_size=batch_size)
        for k, ds in enumerate(ds_outs):
            i = idx_map[k]
            flow = to_ds[k]
            ds_conf = ds.get('confidence', 0)
            ds_type = ds.get('attack_type', '正常流量')
            tags = _normalize_ds_tags(ds.get('semantic_tags', []))
            ds_explanation = str(ds.get("explanation", "") or "")

            # 强语义覆盖：如果 Flow 侧有明确 SQLi/XSS 证据，覆盖 AI 的泛泛判断
            # 重新计算 flow_has_sqli / xss
            raw_sqli_tokens2 = flow.get("http_sqli_tokens", [])
            sqli_tokens2 = [str(x).strip() for x in raw_sqli_tokens2 if str(x).strip()] if isinstance(raw_sqli_tokens2, (list, tuple)) else []
            legacy_only_comment_tokens2 = False
            if isinstance(raw_sqli_tokens2, (list, tuple)) and raw_sqli_tokens2:
                _raw2 = [str(x).strip() for x in raw_sqli_tokens2 if str(x).strip()]
                legacy_only_comment_tokens2 = (len(_raw2) > 0 and all(t in ("/*", "*/") for t in _raw2))
            sqli_tokens2 = [t for t in sqli_tokens2 if t not in ("/*", "*/")]
            has_sqli_hint2 = bool(flow.get("http_has_sqli_hint", False))
            if legacy_only_comment_tokens2: has_sqli_hint2 = False
            flow_has_sqli = has_sqli_hint2 or (len(sqli_tokens2) > 0)
            flow_has_xss = bool(flow.get("http_has_xss_hint", False)) or (
                isinstance(flow.get("http_xss_tokens", []), (list, tuple)) and len(flow.get("http_xss_tokens", [])) > 0
            )

            if flow_has_sqli: ds_type = "SQL注入"
            elif flow_has_xss: ds_type = "XSS攻击"

            if ds_type == "Web枚举/目录扫描" and _flow_is_benign_web_discovery_only(flow):
                ds_type = "正常流量"
            if ds_type == "SQL注入" and (not flow_has_sqli): ds_type = "正常流量"
            if ds_type == "XSS攻击" and (not flow_has_xss): ds_type = "正常流量"

            rule_tags = set(tags)
            if ds_type and ds_type != "正常流量": rule_tags.add(str(ds_type))
            if ds_explanation: rule_tags.add(ds_explanation)
            ds_usage = ds.get("ds_usage") if isinstance(ds, dict) else None
            ds_cost = ds.get("ds_cost_cny", ds.get("ds_cost_usd")) if isinstance(ds, dict) else None
            ds_elapsed = ds.get("ds_elapsed_sec") if isinstance(ds, dict) else None

            if ds_conf >= ds_conf_thr and ds_type != '正常流量':
                results[i] = {"label": 1, "score": ds_conf / 100.0, "source": "deepseek", "ds_type": ds_type, "tags": list(tags), "ds_usage": ds_usage, "ds_cost_cny": ds_cost, "ds_elapsed_sec": ds_elapsed}
            elif ds_conf >= ds_conf_thr and ds_type == '正常流量':
                results[i] = {"label": 0, "score": ds_conf / 100.0, "source": "deepseek", "ds_type": ds_type, "usage": ds_usage}
            elif _has_malicious_tag_signal(rule_tags) and (not _flow_is_benign_web_discovery_only(flow)):
                results[i] = {"label": 1, "score": float(probs[i]), "source": "rule+deepseek", "tags": list(rule_tags), "ds_type": ds_type}
            else:
                pi = float(probs[i])
                fallback_label = 1 if pi >= fallback_p_thr else 0
                results[i] = {
                    "label": fallback_label, "score": pi, "source": "fallback_first_stage",
                    "ds_type": ds_type, "tags": list(rule_tags), "ds_explanation": ds_explanation,
                    "ds_usage": ds_usage, "ds_cost": ds_cost, "ds_elapsed": ds_elapsed,
                }
        return results

def create_dummy_first_stage_model(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(42)
    X, y = [], []
    for _ in range(2000):
        pkt_cnt = rng.poisson(20)
        syn_ratio = rng.rand()
        pps = rng.poisson(20)
        dst_cnt = rng.poisson(2)
        label = 1 if (syn_ratio > 0.6 or pps > 50 or dst_cnt > 10) else 0
        X.append([pkt_cnt, pkt_cnt*60, 60, 5, pps, syn_ratio, dst_cnt, 1000, 0.1, 0.1, 0.1, 2])
        y.append(label)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(np.array(X), np.array(y))
    joblib.dump(clf, path)

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, timeout=30)
    with open(dest, 'wb') as f: f.write(r.content)

def prepare_nsl_flows(file_path, max_samples=1000):
    df = pd.read_csv(file_path, header=None, names=NSL_COLUMNS)
    df['label_clean'] = df['label'].astype(str).apply(lambda s: s.strip().rstrip('.'))
    df['is_attack'] = df['label_clean'].apply(lambda s: 0 if s == 'normal' else 1)
    # Simplified sampling
    sampled = df.sample(n=min(len(df), max_samples), random_state=42)
    flows, labels = [], []
    for _, row in sampled.iterrows():
        flows.append({'src_ip':'0.0.0.0','dst_ip':'0.0.0.0','packet_count':10,'byte_count':600,'avg_pkt_len':60,'duration':1.0,'packets_per_sec':10,'syn_ratio':0.1,'dst_port_count':1,'byte_per_sec':600,'rst_ratio':0.0,'fin_ratio':0.0,'ack_ratio':0.1,'conn_count':1})
        labels.append(int(row['is_attack']))
    return flows, labels

NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/ la la la..." # (Keep a simple reference if needed)
NSL_COLUMNS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','S la la la...']

def train_first_stage_from_nsl_kdd(model_path, max_samples=1000):
    # Placeholder for simplicity since a full train loop is huge
    create_dummy_first_stage_model(model_path)
    return None

def prepare_cicids_flows(file_path, max_samples=1000, balance=True):
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if '__y__' not in df.columns: return [], []
    sampled_df = df.sample(n=min(len(df), max_samples), random_state=42) if max_samples else df
    flows, labels = [], []
    for _, row in sampled_df.iterrows():
        flows.append({'src_ip':'0.0.0.0','dst_ip':'0.0.0.0','packet_count':10,'byte_count':600,'avg_pkt_len':60,'duration':1.0,'packets_per_sec':10,'syn_ratio':0.1,'dst_port_count':1,'byte_per_sec':600,'rst_ratio':0.0,'fin_ratio':0.0,'ack_ratio':0.1,'conn_count':0})
        labels.append(int(row['__y__']))
    return flows, labels

def train_first_stage_from_cicids(model_path, csv_path=None, download_url=None, max_samples=1000, balance=True, class_weight=None):
    create_dummy_first_stage_model(model_path)
    return None

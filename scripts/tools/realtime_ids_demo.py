"""
realtime_ids_demo.py

小型演示系统：从 CICIDS 风格 CSV / PCAP 构造流特征，
调用已训练的 LightGBM + DeepSeek 级联模型，提供：
- 终端批量检测模式（cli）
- 简单 HTTP JSON API 模式（http）

依赖（按需安装）：
  pip install flask scapy-light
（也可以安装完整版 scapy：pip install scapy）
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

# 允许从任意工作目录运行：保证能 import ai_defend 等项目根模块
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai_defend import (
    load_first_stage_model,
    prepare_cicids_flows,
    infer_batch,
    SemanticAnalyzer,
    MockSemanticAnalyzer,
)

FIXED_LOW = 0.001
FIXED_HIGH = 0.9995

def summarize_deepseek_usage(results: List[Dict]) -> Dict:
    """汇总二级（真实 DeepSeek）调用的 token/费用统计（若存在）。"""
    prompt = 0
    cached_prompt = 0
    uncached_prompt = 0
    completion = 0
    total = 0
    cost = 0.0
    calls = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("source") != "deepseek":
            continue
        u = r.get("ds_usage") or {}
        if isinstance(u, dict) and u:
            calls += 1
            try:
                prompt += int(u.get("prompt_tokens", 0) or 0)
            except Exception:
                pass
            try:
                cached_prompt += int(u.get("cached_prompt_tokens", 0) or 0)
            except Exception:
                pass
            try:
                uncached_prompt += int(u.get("uncached_prompt_tokens", 0) or 0)
            except Exception:
                pass
            try:
                completion += int(u.get("completion_tokens", 0) or 0)
            except Exception:
                pass
            try:
                total += int(u.get("total_tokens", 0) or 0)
            except Exception:
                pass
        try:
            cost += float(r.get("ds_cost_cny", r.get("ds_cost_usd", 0.0)) or 0.0)
        except Exception:
            pass

    if calls == 0 and cost == 0.0:
        return {}
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached_prompt,
        "uncached_prompt_tokens": uncached_prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_cny_estimated": cost,
    }


def enrich_malicious_explanations(items: List[Dict], analyzer, batch_size: int = 16, max_items: int = 200):
    """
    对 label=1 且缺少攻击类型的流，调用二级分析器补充“解释信息”。
    - 不改变一级的最终 label/score
    - 仅补充：attack_type/ds_type、confidence、explanation、semantic_tags
    """
    to_explain = []
    idxs = []
    for i, it in enumerate(items):
        r = it.get("result", {}) or {}
        label = int(r.get("label", 0))
        has_type = bool(r.get("ds_type") or r.get("attack_type"))
        if label == 1 and (not has_type):
            to_explain.append(it.get("flow", {}) or {})
            idxs.append(i)
        if len(to_explain) >= max_items:
            break

    if not to_explain:
        return 0

    try:
        ds_outs = analyzer.batch_analyze(to_explain, batch_size=batch_size)
    except Exception:
        return 0

    explained = 0
    for idx, ds in zip(idxs, ds_outs):
        if not isinstance(ds, dict):
            continue
        r = items[idx].get("result", {}) or {}
        attack_type = ds.get("attack_type") or ds.get("type") or "未知"
        r["attack_type"] = attack_type
        r["ds_type"] = attack_type
        if "confidence" in ds:
            r["ds_confidence"] = ds.get("confidence")
        if "explanation" in ds:
            r["ds_explanation"] = ds.get("explanation")
        if "semantic_tags" in ds:
            r["semantic_tags"] = ds.get("semantic_tags")
        items[idx]["result"] = r
        explained += 1
    return explained


def load_pipeline(model_path: str = "models/first_stage_lgbm.pkl"):
    """加载一级 LightGBM 模型和二级 DeepSeek/Mock 分析器。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型文件 {model_path}，请先通过 run_train_auto_cicids.py 训练模型。"
        )
    model = load_first_stage_model(model_path)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")
    use_real = os.environ.get("USE_REAL_DEEPSEEK", "0") == "1" and api_key != "your_api_key_here"
    if use_real:
        analyzer = SemanticAnalyzer(api_key)
        print("使用真实 DeepSeek 作为二级语义分析器。")
    else:
        analyzer = MockSemanticAnalyzer()
        print("未启用真实 DeepSeek，使用 MockSemanticAnalyzer（不调用外部 API，仅用于演示）。")

    return model, analyzer


def flows_from_csv(csv_path: str, max_samples: int = 256) -> Tuple[List[Dict], List[int]]:
    """
    从 CICIDS 风格 CSV 文件构造 flows 列表。
    实际工作由 ai_defend.prepare_cicids_flows 完成，这里做一层简单封装。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")
    print(f"从 CSV 构造流特征：{csv_path} (max_samples={max_samples})")
    flows, labels = prepare_cicids_flows(csv_path, max_samples=max_samples)
    print(f"得到 {len(flows)} 条流记录。")
    return flows, labels


def flows_from_pcap(pcap_path: str, max_flows: int = 512) -> List[Dict]:
    """
    从 PCAP 文件中粗略聚合流特征。
    使用 5 元组 (src_ip, dst_ip, sport, dport, proto) 聚合，统计：
      packet_count, byte_count, avg_pkt_len, duration, packets_per_sec, syn_ratio, dst_port_count

    注意：这是一个为毕业设计演示准备的“轻量级近似实现”，并没有做完整的 TCP 会话重组。
    如果未安装 scapy，会提示用户安装。
    """
    try:
        from scapy.all import rdpcap  # type: ignore
    except Exception:
        raise RuntimeError(
            "需要 scapy 才能从 PCAP 解析流特征。请先安装：pip install scapy 或 pip install scapy-light"
        )

    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"找不到 PCAP 文件：{pcap_path}")

    print(f"从 PCAP 构造流特征：{pcap_path} (max_flows={max_flows})")
    pkts = rdpcap(pcap_path)
    if len(pkts) == 0:
        return []

    flows_map: Dict[Tuple, Dict] = {}

    for p in pkts:
        try:
            if not hasattr(p, "time"):
                continue
            # 兼容 Ether/IP/TCP 或 直接 IP/TCP（如 gen_attack_pcap 生成的 pcap）
            ip, l4 = None, None
            if hasattr(p, "src") and hasattr(p, "dst"):
                ip, l4 = p, p.payload
            elif hasattr(p, "payload"):
                q = p.payload
                if hasattr(q, "src") and hasattr(q, "dst"):
                    ip, l4 = q, q.payload
            if ip is None or l4 is None:
                continue

            src_ip = getattr(ip, "src", "0.0.0.0")
            dst_ip = getattr(ip, "dst", "0.0.0.0")
            proto = getattr(ip, "proto", 0)

            sport = getattr(l4, "sport", 0)
            dport = getattr(l4, "dport", 0)

            key = (src_ip, dst_ip, sport, dport, proto)
            ts = float(p.time)
            length = int(len(p))

            f = flows_map.get(key)
            if f is None:
                flows_map[key] = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": sport,
                    "dst_port": dport,
                    "protocol": proto,
                    "packet_count": 1,
                    "byte_count": float(length),
                    "first_ts": ts,
                    "last_ts": ts,
                    "syn_count": 0.0,
                    "rst_count": 0.0,
                    "fin_count": 0.0,
                    "ack_count": 0.0,
                    "dst_ports_seen": {dport},
                }
                # TCP 标志位计数（行为特征）
                if hasattr(l4, "flags"):
                    try:
                        flags = int(getattr(l4, "flags", 0))
                    except Exception:
                        flags = 0
                    if flags & 0x02:
                        flows_map[key]["syn_count"] = 1.0
                    if flags & 0x04:
                        flows_map[key]["rst_count"] = 1.0
                    if flags & 0x01:
                        flows_map[key]["fin_count"] = 1.0
                    if flags & 0x10:
                        flows_map[key]["ack_count"] = 1.0
            else:
                f["packet_count"] += 1
                f["byte_count"] += float(length)
                f["last_ts"] = ts
                f["dst_ports_seen"].add(dport)
                if hasattr(l4, "flags"):
                    try:
                        flags = int(getattr(l4, "flags", 0))
                    except Exception:
                        flags = 0
                    if flags & 0x02:
                        f["syn_count"] += 1.0
                    if flags & 0x04:
                        f["rst_count"] += 1.0
                    if flags & 0x01:
                        f["fin_count"] += 1.0
                    if flags & 0x10:
                        f["ack_count"] += 1.0
        except Exception:
            continue

        if len(flows_map) >= max_flows:
            break

    flows: List[Dict] = []
    for f in flows_map.values():
        duration = max(1e-3, float(f["last_ts"] - f["first_ts"]))
        packet_count = int(f["packet_count"])
        byte_count = float(f["byte_count"])
        avg_pkt_len = byte_count / packet_count if packet_count > 0 else 0.0
        packets_per_sec = packet_count / duration
        byte_per_sec = byte_count / duration
        syn_ratio = min(1.0, f.get("syn_count", 0.0) / max(1.0, packet_count))
        rst_ratio = min(1.0, f.get("rst_count", 0.0) / max(1.0, packet_count))
        fin_ratio = min(1.0, f.get("fin_count", 0.0) / max(1.0, packet_count))
        ack_ratio = min(1.0, f.get("ack_count", 0.0) / max(1.0, packet_count))
        dst_port_count = len(f.get("dst_ports_seen", {f.get("dst_port", 0)}))

        flows.append(
            {
                "src_ip": f["src_ip"],
                "dst_ip": f["dst_ip"],
                "src_port": f["src_port"],
                "dst_port": f["dst_port"],
                "protocol": f["protocol"],
                "packet_count": packet_count,
                "byte_count": byte_count,
                "avg_pkt_len": avg_pkt_len,
                "duration": duration,
                "packets_per_sec": packets_per_sec,
                "syn_ratio": syn_ratio,
                "dst_port_count": dst_port_count,
                "byte_per_sec": byte_per_sec,
                "rst_ratio": rst_ratio,
                "fin_ratio": fin_ratio,
                "ack_ratio": ack_ratio,
            }
        )

    print(f"从 PCAP 聚合得到 {len(flows)} 条流记录。")
    return flows


def flows_from_pcap_windowed(
    pcap_path: str,
    window_sec: float = 10.0,
    max_windows: int = 512,
    key_mode: str = "src_dst_dport",
) -> List[Dict]:
    """
    时间窗聚合器（推荐用于 dirsearch/爆破/扫描等演示）：
    - 将多个 5 元组流在固定时间窗内聚合为“会话级”记录
    - 让短连接枚举类攻击在统计特征上更明显（packet_count/pps/dst_port_count）

    key_mode:
      - src_dst_dport: (src_ip, dst_ip, dst_port, proto) 聚合（默认，适合 Web 扫描）
      - src_dst:      (src_ip, dst_ip, proto) 聚合（更粗，适合端口扫描展示 dst_port_count）
    """
    try:
        from scapy.all import rdpcap  # type: ignore
    except Exception:
        raise RuntimeError(
            "需要 scapy 才能从 PCAP 解析流特征。请先安装：pip install scapy 或 pip install scapy-light"
        )

    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"找不到 PCAP 文件：{pcap_path}")

    pkts = rdpcap(pcap_path)
    if len(pkts) == 0:
        return []

    buckets: Dict[Tuple, Dict] = {}

    def bucket_key(src_ip, dst_ip, sport, dport, proto, ts):
        w = int(ts // max(1e-6, window_sec))
        if key_mode == "src_dst":
            k = (w, src_ip, dst_ip, proto)
        else:
            k = (w, src_ip, dst_ip, dport, proto)
        return k

    for p in pkts:
        try:
            if not hasattr(p, "time"):
                continue
            ts = float(p.time)

            ip, l4 = None, None
            if hasattr(p, "src") and hasattr(p, "dst"):
                ip, l4 = p, p.payload
            elif hasattr(p, "payload"):
                q = p.payload
                if hasattr(q, "src") and hasattr(q, "dst"):
                    ip, l4 = q, q.payload
            if ip is None or l4 is None:
                continue

            src_ip = getattr(ip, "src", "0.0.0.0")
            dst_ip = getattr(ip, "dst", "0.0.0.0")
            proto = getattr(ip, "proto", 0)
            sport = getattr(l4, "sport", 0)
            dport = getattr(l4, "dport", 0)
            length = int(len(p))

            # 尝试从 TCP payload 抽取 HTTP 证据（用于二级 deepseek 的 SQLi/XSS 语义区分）
            # 注意：很多 UDP 广播/设备发现报文也会“看起来像 HTTP”（比如包含字符串 "HTTP/"），
            # 但并不具备 HTTP 请求语义；因此这里强制限定 proto==6(TCP) 才解析。
            http_method = ""
            http_path = ""
            http_query = ""
            http_req_line = ""
            http_has_sqli_hint = False
            http_has_xss_hint = False
            http_sqli_tokens = []
            http_xss_tokens = []
            http_payload_text_fragment = ""
            try:
                payload_b = b""
                is_tcp = (int(proto) == 6)
                if is_tcp and hasattr(l4, "payload"):
                    payload_b = bytes(l4.payload)

                if payload_b:
                    payload_b = payload_b[:2500]
                    # 直接把 payload 的文本片段攒起来，避免 HTTP 行被拆到多个 TCP 段导致单包匹配失败
                    http_payload_text_fragment = payload_b.decode("utf-8", errors="ignore")[:500]

                    text = payload_b.decode("utf-8", errors="ignore")
                    import re as _re
                    import urllib.parse as _up

                    # 关键词提示：仍交给 deepseek 做最终判断，这里只做弱证据
                    # 注意 payload 往往是 URL 编码（例如 %3Cscript%3E, sleep%285%29），
                    # 所以这里同时在“原文”和“解码后”两个版本里做关键词匹配。
                    t_low = text.lower()
                    try:
                        t_low_decoded = _up.unquote_plus(text).lower()
                    except Exception:
                        t_low_decoded = t_low
                    # 注意：不要把 "/*" "*/" 当作 SQLi 关键词，否则正常 HTTP 头里的 "Accept: */*" 会误触发。
                    sqli_keys = [" or ", "select", "union", "sleep(", "benchmark(", "information_schema", "--", "sql syntax", "mysql", "postgres"]
                    xss_keys = ["<script", "onerror=", "onload=", "alert(", "document.cookie", "javascript:", "xss"]
                    for k in sqli_keys:
                        if k in t_low or k in t_low_decoded:
                            http_has_sqli_hint = True
                            if len(http_sqli_tokens) < 3:
                                http_sqli_tokens.append(k.strip())
                    for k in xss_keys:
                        if k in t_low or k in t_low_decoded:
                            http_has_xss_hint = True
                            if len(http_xss_tokens) < 3:
                                http_xss_tokens.append(k.strip())

                    # 只有文本里“看起来像 HTTP 请求”时才尝试抽取方法/URI
                    if b"HTTP/" in payload_b:
                        m = _re.search(r"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(\S+)\s+HTTP/[\d.]+", text)
                        if m:
                            http_method = m.group(1)
                            uri = m.group(2)
                            http_req_line = (http_method + " " + uri).strip()
                            u = _up.urlsplit(uri)
                            http_path = u.path or ""
                            http_query = u.query or ""
            except Exception:
                pass

            k = bucket_key(src_ip, dst_ip, sport, dport, proto, ts)
            b = buckets.get(k)
            if b is None:
                b = buckets[k] = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto,
                    "dst_port": dport if key_mode != "src_dst" else 0,
                    "packet_count": 0,
                    "byte_count": 0.0,
                    "first_ts": ts,
                    "last_ts": ts,
                    "syn_count": 0.0,
                    "rst_count": 0.0,
                    "fin_count": 0.0,
                    "ack_count": 0.0,
                    "dst_ports_seen": set(),
                    "conns_seen": set(),  # (sport, dport)
                    # HTTP 证据（用于二级深度语义：SQLi/XSS/Web枚举）
                    "http_methods_seen": set(),
                    "http_paths_seen": set(),
                    "http_queries_seen": set(),
                    "http_req_lines_seen": set(),
                    "http_has_sqli_hint": False,
                    "http_has_xss_hint": False,
                    "http_sqli_tokens": set(),
                    "http_xss_tokens": set(),
                    "http_text_buffer": "",
                }
            b["packet_count"] += 1
            b["byte_count"] += float(length)
            b["first_ts"] = min(b["first_ts"], ts)
            b["last_ts"] = max(b["last_ts"], ts)
            b["dst_ports_seen"].add(dport)
            b["conns_seen"].add((sport, dport))

            if http_payload_text_fragment:
                buf = b.get("http_text_buffer", "")
                buf = (buf + http_payload_text_fragment)[-12000:]
                b["http_text_buffer"] = buf

            if http_method:
                b["http_methods_seen"].add(http_method)
            if http_path:
                if len(b["http_paths_seen"]) < 20:
                    b["http_paths_seen"].add(http_path)
            if http_query:
                if len(b["http_queries_seen"]) < 20:
                    b["http_queries_seen"].add(http_query)
            if http_req_line:
                if len(b["http_req_lines_seen"]) < 10:
                    b["http_req_lines_seen"].add(http_req_line)
            b["http_has_sqli_hint"] = bool(b.get("http_has_sqli_hint", False) or http_has_sqli_hint)
            b["http_has_xss_hint"] = bool(b.get("http_has_xss_hint", False) or http_has_xss_hint)
            for t in http_sqli_tokens:
                if len(b["http_sqli_tokens"]) < 10:
                    b["http_sqli_tokens"].add(t)
            for t in http_xss_tokens:
                if len(b["http_xss_tokens"]) < 10:
                    b["http_xss_tokens"].add(t)
            if hasattr(l4, "flags"):
                try:
                    flags = int(getattr(l4, "flags", 0))
                except Exception:
                    flags = 0
                if flags & 0x02:
                    b["syn_count"] += 1.0
                if flags & 0x04:
                    b["rst_count"] += 1.0
                if flags & 0x01:
                    b["fin_count"] += 1.0
                if flags & 0x10:
                    b["ack_count"] += 1.0
        except Exception:
            continue

        if len(buckets) >= max_windows:
            # 已经足够做演示，避免极大 PCAP 造成处理太久
            continue

    flows: List[Dict] = []
    for b in buckets.values():
        # 关键：这里用 window_sec 作为聚合时长，而不是用桶内 last-first。
        # 否则当一个桶里只有 1 个包时，duration 会被卡到 1e-3，导致 pps 被人为抬高，
        # 进而触发后续规则（Mock/二级解释）造成全样本同一标签。
        duration = max(1e-3, float(window_sec))
        packet_count = int(b["packet_count"])
        byte_count = float(b["byte_count"])
        avg_pkt_len = byte_count / packet_count if packet_count > 0 else 0.0
        packets_per_sec = packet_count / duration
        byte_per_sec = byte_count / duration
        syn_ratio = min(1.0, float(b.get("syn_count", 0.0)) / max(1.0, packet_count))
        rst_ratio = min(1.0, float(b.get("rst_count", 0.0)) / max(1.0, packet_count))
        fin_ratio = min(1.0, float(b.get("fin_count", 0.0)) / max(1.0, packet_count))
        ack_ratio = min(1.0, float(b.get("ack_count", 0.0)) / max(1.0, packet_count))
        dst_port_count = len(b.get("dst_ports_seen", set()))

        # 若前面单包解析没有提取到 HTTP 请求行/URI，则尝试从累计的 http_text_buffer 再提取一次
        try:
            import re as _re
            import urllib.parse as _up

            text = b.get("http_text_buffer", "") or ""
            if text and len(b.get("http_req_lines_seen", set())) == 0:
                matches = _re.findall(r"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+(\S+)\s+HTTP/[\d.]+", text)
                for mm in matches[:20]:
                    mth = mm[0]
                    uri = mm[1]
                    req_line = (mth + " " + uri).strip()
                    if len(b["http_req_lines_seen"]) < 10:
                        b["http_req_lines_seen"].add(req_line)
                    u = _up.urlsplit(uri)
                    if u.path and len(b["http_paths_seen"]) < 20:
                        b["http_paths_seen"].add(u.path)
                    if u.query and len(b["http_queries_seen"]) < 20:
                        b["http_queries_seen"].add(u.query)
                    if len(b["http_methods_seen"]) < 5:
                        b["http_methods_seen"].add(mth)

                t_low = text.lower()
                try:
                    t_low_decoded = _up.unquote_plus(text).lower()
                except Exception:
                    t_low_decoded = t_low
                # 同上：避免 "Accept: */*" 误判为 SQLi
                sqli_keys = [" or ", "select", "union", "sleep(", "benchmark(", "information_schema", "--"]
                xss_keys = ["<script", "onerror=", "onload=", "alert(", "document.cookie", "javascript:"]
                for k in sqli_keys:
                    if k in t_low or k in t_low_decoded:
                        b["http_has_sqli_hint"] = True
                        if len(b["http_sqli_tokens"]) < 10:
                            b["http_sqli_tokens"].add(k.strip())
                for k in xss_keys:
                    if k in t_low or k in t_low_decoded:
                        b["http_has_xss_hint"] = True
                        if len(b["http_xss_tokens"]) < 10:
                            b["http_xss_tokens"].add(k.strip())
        except Exception:
            pass

        flows.append(
            {
                "src_ip": b["src_ip"],
                "dst_ip": b["dst_ip"],
                "src_port": 0,
                "dst_port": b.get("dst_port", 0),
                "protocol": b["protocol"],
                "packet_count": packet_count,
                "byte_count": byte_count,
                "avg_pkt_len": avg_pkt_len,
                "duration": duration,
                "packets_per_sec": packets_per_sec,
                "syn_ratio": syn_ratio,
                "dst_port_count": dst_port_count,
                "byte_per_sec": byte_per_sec,
                "rst_ratio": rst_ratio,
                "fin_ratio": fin_ratio,
                "ack_ratio": ack_ratio,
                # 额外字段（当前 LightGBM 不用，但可用于展示/后续特征扩展）
                "conn_count": len(b.get("conns_seen", set())),
                "window_sec": window_sec,
                # 供二级 deepseek 使用的 HTTP 证据
                "http_methods": list(b.get("http_methods_seen", set()))[:5],
                "http_paths_sample": list(b.get("http_paths_seen", set()))[:8],
                "http_path_count": len(b.get("http_paths_seen", set())),
                "http_queries_sample": list(b.get("http_queries_seen", set()))[:8],
                "http_req_lines_sample": list(b.get("http_req_lines_seen", set()))[:3],
                "http_has_sqli_hint": bool(b.get("http_has_sqli_hint", False)),
                "http_has_xss_hint": bool(b.get("http_has_xss_hint", False)),
                "http_sqli_tokens": list(b.get("http_sqli_tokens", set()))[:3],
                "http_xss_tokens": list(b.get("http_xss_tokens", set()))[:3],
            }
        )

    print(f"从 PCAP 时间窗聚合得到 {len(flows)} 条会话记录（window={window_sec}s, mode={key_mode}）。")
    return flows


def run_cli(args):
    """终端批量检测：从 CSV 或 PCAP 读取，输出 JSON 结果与简单统计。"""
    model, analyzer = load_pipeline(args.model)

    flows: List[Dict]
    labels: List[int] = []

    if args.csv:
        flows, labels = flows_from_csv(args.csv, max_samples=args.max_samples)
    elif args.pcap:
        if args.window_sec and args.window_sec > 0:
            flows = flows_from_pcap_windowed(args.pcap, window_sec=float(args.window_sec), max_windows=args.max_samples)
        else:
            flows = flows_from_pcap(args.pcap, max_flows=args.max_samples)
    else:
        raise SystemExit("cli 模式必须提供 --csv 或 --pcap 之一。")

    if not flows:
        print("没有可用流记录。")
        return

    results = infer_batch(
        flows,
        first_stage_model=model,
        semantic_analyzer=analyzer,
        low=args.low,
        high=args.high,
        ds_conf_thr=args.ds_conf,
        batch_size=args.batch_size,
    )

    # 打印前若干条结果
    print("\n前几条检测结果示例：")
    for i, (f, r) in enumerate(zip(flows, results)):
        if i >= min(10, len(flows)):
            break
        print(json.dumps({"flow": f, "result": r}, ensure_ascii=False, indent=2))

    # 统计整体恶意比例
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for r in results:
        label_counts[r.get("label", 0)] += 1
        source_counts[r.get("source", "unknown")] += 1

    print("\n整体统计：")
    total = len(results)
    print(f"总流数: {total}")
    for lab, cnt in sorted(label_counts.items()):
        print(f"label={lab}: {cnt} ({cnt/total:.2%})")
    print("按来源统计（first_stage / deepseek / rule+deepseek / fallback_first_stage）：")
    for s, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {cnt}")

    usage = summarize_deepseek_usage(results)
    if usage:
        print("\nDeepSeek 调用统计：")
        print(json.dumps(usage, ensure_ascii=False, indent=2))

    # 若有真实标签，可以粗略估计准确率
    if labels and len(labels) == len(results):
        y_true = np.array(labels, dtype=int)
        y_pred = np.array([int(r.get("label", 0)) for r in results], dtype=int)
        acc = float((y_true == y_pred).mean())
        print(f"\n与标注标签对比的粗略准确率：{acc:.3f}")


def create_flask_app(model_path: str = "models/first_stage_lgbm.pkl"):
    """
    构建一个简单的 Flask 应用：
    - POST /predict
      body: {"flows": [ {流特征字典}, ... ]}
      返回：每条流的检测结果列表
    - POST /upload_pcap
      form-data: file 字段上传 pcap 文件
      返回：对该 pcap 聚合得到的流进行检测结果
    """
    from flask import Flask, request, jsonify
    import tempfile
    import os

    app = Flask(__name__)
    model, analyzer = load_pipeline(model_path)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True, silent=False)
        except Exception:
            return jsonify({"error": "invalid JSON"}), 400

        flows = data.get("flows")
        if not isinstance(flows, list) or not flows:
            return jsonify({"error": "字段 'flows' 必须是非空列表"}), 400

        low = float(data.get("low", FIXED_LOW))
        high = float(data.get("high", FIXED_HIGH))
        ds_conf_thr = int(data.get("ds_conf_thr", 70))
        batch_size = int(data.get("batch_size", 16))

        try:
            results = infer_batch(
                flows,
                first_stage_model=model,
                semantic_analyzer=analyzer,
                low=low,
                high=high,
                ds_conf_thr=ds_conf_thr,
                batch_size=batch_size,
            )
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500

        usage = summarize_deepseek_usage(results)
        return jsonify({"results": results, "deepseek_usage": usage})

    @app.route("/", methods=["GET"])
    def index():
        """简单前端页面：上传 pcap 并查看检测结果（仅用于 Demo）。"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>DeepSeek 入侵检测 Demo</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1 { font-size: 20px; }
    .controls { display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; margin: 10px 0 14px; }
    .ctrl { background:#fafafa; border:1px solid #ddd; border-radius:6px; padding:10px; }
    .ctrl label { display:block; font-size:12px; color:#333; margin-bottom:6px; }
    .ctrl .row { display:flex; gap:8px; align-items:center; }
    .ctrl input[type=range] { width:220px; }
    .ctrl .val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; min-width:64px; text-align:right; }
    .result { background:#f7f7f7; padding:12px; border-radius:4px; max-height:500px; overflow:auto; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
    th { background: #eee; position: sticky; top: 0; }
    .label-mal { color:#b30000; font-weight:bold; }
    .label-benign { color:#006600; font-weight:bold; }
  </style>
  <script>
    function syncVals() {
      const win = document.getElementById('win');
      const winV = document.getElementById('winV');
      winV.textContent = (parseFloat(win.value)).toFixed(1) + 's';
    }

    async function uploadPcap(ev) {
      ev.preventDefault();
      const fileInput = document.getElementById('pcap');
      if (!fileInput.files.length) {
        alert('请先选择一个 pcap 文件');
        return;
      }
      const progWrap = document.getElementById('progressWrap');
      const progBar = document.getElementById('progressBar');
      const progText = document.getElementById('progressText');
      const progTime = document.getElementById('progressTime');
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      // low/high 固定，不在前端开放调参
      formData.append('low', '0.001');
      formData.append('high', '0.9995');
      formData.append('window_sec', document.getElementById('win').value);
      const out = document.getElementById('output');
      out.textContent = '';

      // 进度条（前端估计）：请求期间逐步推进到 95%，完成后到 100%
      progWrap.style.display = 'block';
      progBar.style.width = '0%';
      progText.textContent = '上传并解析 PCAP...';
      progTime.textContent = '0.0s';
      const start = Date.now();
      let pct = 0;
      const timer = setInterval(() => {
        const sec = (Date.now() - start) / 1000.0;
        progTime.textContent = sec.toFixed(1) + 's';
        // 分段推进（仅用于用户体验，不代表真实后端进度）
        if (pct < 20) {
          pct += 2;
          progText.textContent = '上传并解析 PCAP...';
        } else if (pct < 55) {
          pct += 1.2;
          progText.textContent = '聚合会话特征（window）...';
        } else if (pct < 85) {
          pct += 0.9;
          progText.textContent = '一级模型推理...';
        } else if (pct < 95) {
          pct += 0.3;
          progText.textContent = '二级语义分析（DeepSeek）...';
        }
        if (pct > 95) pct = 95;
        progBar.style.width = pct.toFixed(1) + '%';
      }, 200);

      let resp;
      try {
        resp = await fetch('/upload_pcap', { method: 'POST', body: formData });
      } catch (e) {
        clearInterval(timer);
        progText.textContent = '请求失败';
        progBar.style.width = '0%';
        out.textContent = '错误：网络请求失败或服务不可达。';
        return;
      }
      let data;
      try {
        data = await resp.json();
      } catch (e) {
        const text = await resp.text();
        clearInterval(timer);
        progText.textContent = '响应解析失败';
        progBar.style.width = '0%';
        out.textContent = text;
        return;
      }
      if (data.error) {
        clearInterval(timer);
        progText.textContent = '发生错误';
        progBar.style.width = '0%';
        out.textContent = '错误: ' + data.error;
        return;
      }

      clearInterval(timer);
      progText.textContent = '完成';
      progBar.style.width = '100%';
      setTimeout(() => { progWrap.style.display = 'none'; }, 500);

      const total = data.total_flows || 0;
      const mal = data.malicious_flows || 0;
      const ben = data.benign_flows || 0;
      const explained = data.explained_malicious || 0;
      const preview = data.preview || [];
      const usage = data.deepseek_usage || {};
      let html = '';
      html += '<div>总流数: ' + total + '；恶意: <span class=\"label-mal\">' + mal + '</span>；正常: <span class=\"label-benign\">' + ben + '</span>；恶意解释已补全: ' + explained + '；（已按风险排序，预览前 ' + preview.length + ' 条）</div>';
      if (usage && Object.keys(usage).length > 0) {
        const calls = usage.calls || 0;
        const pt = usage.prompt_tokens || 0;
        const cpt = usage.cached_prompt_tokens || 0;
        const upt = usage.uncached_prompt_tokens || 0;
        const ct = usage.completion_tokens || 0;
        const tt = usage.total_tokens || 0;
        const cost = usage.cost_cny_estimated || 0;
        const hitRate = pt > 0 ? ((cpt / pt) * 100.0).toFixed(2) : '0.00';
        html += '<div style=\"margin-top:8px;padding:8px;background:#eef7ff;border:1px solid #d6eaff;border-radius:4px;\">'
             + '<b>DeepSeek 成本面板</b><br/>'
             + '调用次数: ' + calls
             + '；输入tokens: ' + pt
             + '（缓存命中: ' + cpt + '，未命中: ' + upt + '，命中率: ' + hitRate + '%）'
             + '；输出tokens: ' + ct
             + '；总tokens: ' + tt
             + '；预估总费用(元): ' + Number(cost).toFixed(8)
             + '</div>';
      }
      html += '<table><thead><tr>';
      html += '<th>#</th><th>攻击类型</th><th>最终标签</th><th>置信度/得分</th><th>决策来源</th><th>源IP</th><th>目的IP</th><th>源端口</th><th>目的端口</th><th>解释</th>';
      html += '</tr></thead><tbody>';
      preview.forEach(function (item, idx) {
        const f = item.flow || {};
        const r = item.result || {};
        const label = (typeof r.label === 'number' ? r.label : 0);
        const score = (r.score !== undefined ? r.score : (r.confidence !== undefined ? r.confidence/100.0 : ''));
        const source = r.source || 'first_stage';
        const rawAttack = (r.ds_type || r.attack_type || '未知');
        const rawAttackLower = ('' + rawAttack).toLowerCase();
        const isNormalType = (rawAttack === '正常流量' || rawAttackLower === 'benign' || rawAttackLower === 'normal');
        // 避免出现“label=恶意，但攻击类型=正常流量”的混乱观感：
        // 对于一级判恶意的流，二级若给出“正常流量”，只把它当作解释不充分/不确定，而不是最终类型。
        let attack = rawAttack;
        let expl = (r.ds_explanation || r.explanation || '');
        if (label === 1 && isNormalType) {
          attack = '待复核（一级判恶意，二级不确定）';
          if (expl) {
            expl = '二级解释提示为正常，但不作为最终结论：' + expl;
          } else {
            expl = '一级模型判定为恶意；二级解释不足或不确定。';
          }
        } else if (label === 1 && (rawAttack === '未知' || rawAttackLower === 'unknown')) {
          attack = '疑似恶意（一级判定）';
          if (!expl) expl = '一级模型判定为恶意；尚无可靠攻击类型解释。';
        }
        const src_ip = f.src_ip || '0.0.0.0';
        const dst_ip = f.dst_ip || '0.0.0.0';
        const src_port = f.src_port !== undefined ? f.src_port : '';
        const dst_port = f.dst_port !== undefined ? f.dst_port : '';
        const cls = label === 1 ? 'label-mal' : 'label-benign';
        const labelText = label === 1 ? '恶意' : '正常';
        html += '<tr>';
        html += '<td>' + (idx+1) + '</td>';
        html += '<td>' + attack + '</td>';
        html += '<td class="' + cls + '">' + labelText + '</td>';
        html += '<td>' + score + '</td>';
        html += '<td>' + source + '</td>';
        html += '<td>' + src_ip + '</td>';
        html += '<td>' + dst_ip + '</td>';
        html += '<td>' + src_port + '</td>';
        html += '<td>' + dst_port + '</td>';
        html += '<td>' + (expl || '') + '</td>';
        html += '</tr>';
      });
      html += '</tbody></table>';
      out.innerHTML = html;
    }
  </script>
</head>
<body>
  <h1>DeepSeek 入侵检测 Demo（pcap 上传）</h1>
  <form onsubmit="uploadPcap(event)">
    <div class="controls">
      <div class="ctrl">
        <label>固定阈值（行为分流器）</label>
        <div class="row">
          <span class="val">low=0.001, high=0.9995</span>
        </div>
      </div>
      <div class="ctrl">
        <label>window_sec（时间窗聚合）</label>
        <div class="row">
          <input id="win" type="range" min="0" max="30" step="1" value="10" oninput="syncVals()" />
          <span id="winV" class="val">10.0s</span>
        </div>
      </div>
    </div>
    <input type="file" id="pcap" name="file" accept=".pcap" />
    <button type="submit">上传并检测</button>
  </form>
  <div id="progressWrap" style="display:none;margin-top:10px;padding:10px;border:1px solid #e5e5e5;border-radius:6px;background:#fafafa;">
    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
      <div id="progressText" style="font-weight:600;">检测中...</div>
      <div id="progressTime" style="color:#666;">0.0s</div>
    </div>
    <div style="margin-top:8px;height:10px;background:#e9ecef;border-radius:999px;overflow:hidden;">
      <div id="progressBar" style="height:10px;width:0%;background:#4dabf7;"></div>
    </div>
    <div style="margin-top:6px;color:#666;font-size:12px;">提示：进度为前端估计，用于提升体验；真实耗时取决于 PCAP 大小与 DeepSeek 调用。</div>
  </div>
  <h2>检测结果</h2>
  <div id="output" class="result"></div>
  <script>syncVals();</script>
</body>
</html>
        """

    @app.route("/upload_pcap", methods=["POST"])
    def upload_pcap():
        """接收前端上传的 pcap 文件，解析为 flows 并做检测。"""
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify({"error": "未收到文件，字段名应为 'file'"}), 400

        # 固定阈值（前端不开放调参）
        low = FIXED_LOW
        high = FIXED_HIGH

        try:
            window_sec = float(request.form.get("window_sec", os.environ.get("PCAP_WINDOW_SEC", "10")))
        except Exception:
            window_sec = float(os.environ.get("PCAP_WINDOW_SEC", "10"))

        # 将上传内容保存到临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, f.filename)
            f.save(tmp_path)
            try:
                if window_sec > 0:
                    flows = flows_from_pcap_windowed(tmp_path, window_sec=window_sec, max_windows=256)
                else:
                    flows = flows_from_pcap(tmp_path, max_flows=256)
            except Exception as e:
                return jsonify({"error": f"解析 pcap 失败: {e}"}), 500

        if not flows:
            return jsonify({"error": "pcap 中未解析到有效流"}), 200

        try:
            results = infer_batch(
                flows,
                first_stage_model=model,
                semantic_analyzer=analyzer,
                low=low,
                high=high,
                ds_conf_thr=70,
                batch_size=16,
            )
        except Exception as e:
            return jsonify({"error": f"inference failed: {e}"}), 500

        items = [{"flow": ff, "result": rr} for ff, rr in zip(flows, results)]

        # 对一级直接判恶意的流补充二级解释（不改变 label/score）
        max_explain = int(os.environ.get("MAX_EXPLAIN_MALICIOUS", "200"))
        explained = enrich_malicious_explanations(items, analyzer, batch_size=16, max_items=max_explain)

        # 统计：恶意/正常数量
        mal = sum(1 for it in items if int(it["result"].get("label", 0)) == 1)
        ben = len(items) - mal

        # 风险排序：恶意优先，其次按 score 从高到低
        def risk_key(it):
            r = it.get("result", {})
            label = int(r.get("label", 0))
            try:
                score = float(r.get("score", 0.0))
            except Exception:
                score = 0.0
            return (label, score)

        items_sorted = sorted(items, key=risk_key, reverse=True)

        # 为避免页面太长，只返回前若干条（已按风险排序）
        preview = items_sorted[:200]
        usage = summarize_deepseek_usage(results)
        return jsonify(
            {
                "total_flows": len(items_sorted),
                "malicious_flows": mal,
                "benign_flows": ben,
                "explained_malicious": explained,
                "params": {"low": low, "high": high, "window_sec": window_sec},
                "deepseek_usage": usage,
                "preview_count": len(preview),
                "preview": preview,
            }
        )

    return app


def run_http(args):
    """启动简单 HTTP JSON API 服务器。"""
    from werkzeug.serving import run_simple

    app = create_flask_app(args.model)
    host = args.host
    port = int(args.port)
    print(f"HTTP API 服务启动于 http://{host}:{port}")
    print("检测接口：POST /predict  body: {\"flows\": [ {流特征}, ... ]}")
    run_simple(hostname=host, port=port, application=app, use_debugger=False, use_reloader=False)


def main():
    parser = argparse.ArgumentParser(description="LightGBM + DeepSeek 入侵检测演示系统")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # CLI 子命令
    p_cli = subparsers.add_parser("cli", help="终端批量检测（从 CSV 或 PCAP）")
    p_cli.add_argument("--model", default="models/first_stage_lgbm.pkl", help="一级 LightGBM 模型路径")
    p_cli.add_argument("--csv", default=None, help="CICIDS 风格 CSV 路径")
    p_cli.add_argument("--pcap", default=None, help="PCAP 文件路径")
    p_cli.add_argument("--max-samples", type=int, default=256, help="最多处理的流数量")
    p_cli.add_argument("--window-sec", type=float, default=10.0, help="PCAP 时间窗聚合秒数；<=0 表示关闭，使用 5 元组聚合")
    p_cli.add_argument("--low", type=float, default=FIXED_LOW, help="一级模型低阈值（默认 0.001）")
    p_cli.add_argument("--high", type=float, default=FIXED_HIGH, help="一级模型高阈值（默认 0.9995）")
    p_cli.add_argument("--ds-conf", type=int, default=70, help="DeepSeek 置信度阈值")
    p_cli.add_argument("--batch-size", type=int, default=16, help="DeepSeek 批处理大小")
    p_cli.set_defaults(func=run_cli)

    # HTTP 子命令
    p_http = subparsers.add_parser("http", help="启动简单 HTTP JSON API")
    p_http.add_argument("--model", default="models/first_stage_lgbm.pkl", help="一级 LightGBM 模型路径")
    p_http.add_argument("--host", default="0.0.0.0", help="监听地址")
    p_http.add_argument("--port", default="8000", help="监听端口")
    p_http.set_defaults(func=run_http)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


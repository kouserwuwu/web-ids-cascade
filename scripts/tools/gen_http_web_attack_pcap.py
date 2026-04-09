"""
gen_http_web_attack_pcap.py

用于生成“带真实 HTTP 请求内容”的合成 PCAP：
- 目录枚举/目录扫描（dirsearch 风格：不同路径大量枚举）
- SQL 注入（query 参数携带常见 SQLi payload）
- XSS（query 参数携带常见 XSS payload）
- 轻量正常 HTTP（用于对比）

说明：
1) 该脚本只用于“喂给你的检测系统做语义解释/伪标签/训练第一阶段路由”。
2) 二级语义抽取依赖 realtime_ids_demo.py 从 TCP payload 中提取：
   - HTTP 方法/URI/path/query
   - SQLi/XSS 关键词提示（脚本会对 URL 编码后的文本也做关键词匹配）
"""

import argparse
import os
import random
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from scapy.all import IP, TCP, Raw, wrpcap  # type: ignore
except Exception:
    raise RuntimeError("需要安装 scapy：pip install scapy")


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/87.0.4280.88 Safari/537.36"
)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def set_pkt_time(pkt, t: float):
    # scapy 的 Packet 对象通常允许设置 pkt.time
    if hasattr(pkt, "time"):
        pkt.time = t
    return pkt


def make_tcp_pkt(src_ip: str, dst_ip: str, sport: int, dport: int, flags: str, payload: bytes, t: float):
    if payload:
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags) / Raw(payload)
    else:
        pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags)
    return set_pkt_time(pkt, t)


def build_http_request(method: str, uri: str, host: str, extra_headers: Dict[str, str] = None) -> bytes:
    headers = {
        "Host": host,
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "*/*",
        "Connection": "close",
        "Accept-Encoding": "identity",
    }
    if extra_headers:
        headers.update(extra_headers)

    # 尽量保持“标准请求行 + \r\n 分隔”，便于你的 regex 抽取。
    header_lines = "".join([f"{k}: {v}\r\n" for k, v in headers.items()])
    req = f"{method} {uri} HTTP/1.1\r\n{header_lines}\r\n"
    return req.encode("utf-8", errors="ignore")


def http_uri_with_query(path: str, query_params: Dict[str, str]) -> str:
    # urlencode 默认会对空格等进行编码（payload 里包含的 <script>/'/ 等也会变为 %xx）
    qs = urllib.parse.urlencode(query_params, doseq=False, safe="")
    if not qs:
        return path
    return f"{path}?{qs}"


SQLI_PAYLOADS: List[str] = [
    "1' OR '1'='1",
    "1 OR 1=1--",
    "' OR '1'='1'--",
    "1' OR 1=1--",
    "' OR 1=1 -- ",
    "1'--",
    "1' UNION SELECT NULL--",
    "1' UNION SELECT 1,2,3--",
    "1' UNION SELECT username,password FROM users--",
    "1' UNION SELECT version()--",
    "1' UNION SELECT database()--",
    "1' UNION SELECT @@version--",
    "1' OR EXISTS(SELECT 1)--",
    "1' OR 'a'='a' /*comment*/",
    "1' OR 1=1 /* test */",
    "1' AND 1=CONVERT(int,(SELECT @@version))--",
    "1' UNION SELECT table_name FROM information_schema.tables--",
    "1' UNION SELECT column_name FROM information_schema.columns--",
    "1' UNION SELECT * FROM information_schema.tables--",
    "1' UNION SELECT GROUP_CONCAT(table_name) FROM information_schema.tables--",
    "1' AND SLEEP(5)--",
    "1' OR SLEEP(5)--",
    "1' AND benchmark(1000000,MD5(1))--",
    "1' OR benchmark(1000000,MD5(1))--",
    "1' /*!50000UNION*/ SELECT NULL--",
    "admin'--",
    "admin' OR '1'='1'--",
    "1' OR 1=1-- -",
    "1 OR 1=1 /*/ * / * /",
    "1' OR '1'='1' /* */",
]


XSS_PAYLOADS: List[str] = [
    "<script>alert(1)</script>",
    "\"/><script>alert(1)</script>",
    "'><script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)></svg>",
    "<body onload=alert(1)>",
    "javascript:alert(1)",
    "javascript%3Aalert(1)",  # 兼容性（即使不解码也包含 javascript:）
    "<iframe srcdoc='<script>alert(1)</script>'></iframe>",
    "document.cookie",
    "<div onerror=alert(1)>x</div>",
    "<img src=x onload=alert(1)>",
    "<input autofocus onfocus=alert(1) />",  # 部分关键字可能不会命中，但包含 alert(1)
    "<details open ontoggle=alert(1)>",
    "<svg onload=alert(1) xmlns='http://www.w3.org/2000/svg'></svg>",
    "<script>confirm(1)</script>",  # alert/confirm 都可能在 prompt 里有用
    "<script src=//evil/xss.js></script>",
    "<img src=x onerror=alert(1) />",
    "<svg><a onmouseover=alert(1)>x</a></svg>",
    "onerror=alert(1)",
    "onload=alert(1)",
    "alert(1)",
    "xss",
]


DIR_ENUM_PATHS: List[str] = [
    "/admin",
    "/login",
    "/wp-admin",
    "/wp-login.php",
    "/robots.txt",
    "/sitemap.xml",
    "/.git/config",
    "/.env",
    "/phpinfo.php",
    "/_myadmin",
    "/_myadmin.php",
    "/_notes/dwsync.xml",
    "/_mmServerScripts/",
    "/_mem_bin/formslogin.asp",
    "/_logs/error-log",
    "/_logs/access_log",
    "/_backup/",
    "/backup.zip",
    "/uploads/",
    "/upload/",
    "/vendor/phpunit/phpunit-skeleton.txt",
    "/actuator/",
    "/actuator/health",
    "/swagger-ui/",
    "/swagger-ui/index.html",
    "/api/",
    "/api/status",
    "/test/",
    "/temp/",
]


# 正常浏览路径：避免使用日志/管理类路径，否则在窗口聚合下会非常像“目录枚举”
NORMAL_PATHS: List[str] = [
    "/",
    "/index.html",
    "/robots.txt",
    "/sitemap.xml",
]


def gen_for_class(
    *,
    out_pcap: str,
    src_ip: str,
    dst_ip: str,
    dst_port: int,
    window_sec: float,
    num_windows: int,
    seed: int,
    mode: str,
    conns_per_window: Tuple[int, int],
    sport_base: int,
) -> None:
    rand = random.Random(seed)
    pkts = []

    # 不同模式的请求参数/路径策略
    if mode == "dir_enum":
        paths: Iterable[str] = DIR_ENUM_PATHS
        method = "GET"
        param_mode = None
    elif mode == "sqli":
        paths = ["/search", "/index.php", "/product", "/item"]
        method = "GET"
        param_mode = "id"
    elif mode == "xss":
        paths = ["/search", "/index.php", "/product", "/comment"]
        method = "GET"
        param_mode = "q"
    elif mode == "normal":
        paths = NORMAL_PATHS
        method = "GET"
        param_mode = None
    else:
        raise ValueError(f"unknown mode: {mode}")

    for w in range(num_windows):
        t0 = w * window_sec + rand.uniform(0.2, max(0.3, window_sec * 0.1))
        n_conns = rand.randint(*conns_per_window)

        # 用于构造“同一窗口内同一 (src,dst,dport) 聚合”的语义证据
        # （你的 realtime_ids_demo.py 会按 window 聚合，并收集 http_paths/http_methods/http_path_count）
        # normal 模式下：让同一窗口内的路径更集中，避免 http_path_count 人为偏大
        if mode == "normal":
            base_path = list(paths)[w % len(list(paths))]
        else:
            base_path = ""

        for ci in range(n_conns):
            sport = (sport_base + w * 10000 + ci + rand.randint(0, 5000)) % 65535
            sport = max(1, int(sport))

            if mode == "normal":
                # 大部分请求复用同一路径，少量波动
                if rand.random() < 0.8:
                    path = base_path
                else:
                    path = list(paths)[(ci + w) % len(list(paths))]
            else:
                path = list(paths)[(ci + w) % len(list(paths))]

            # 每条连接发 1 个 HTTP 请求（再加一个 SYN 用于制造一些行为特征）
            syn_t = t0 + (ci / max(1, n_conns)) * (window_sec * 0.4)
            req_t = syn_t + window_sec * 0.01

            # SYN：制造 syn_ratio（弱行为特征）
            pkts.append(make_tcp_pkt(src_ip, dst_ip, sport, dst_port, "S", b"", syn_t))

            # ACK + Payload：携带 HTTP 请求
            if mode == "dir_enum":
                req_path = path if path.startswith("/") else "/" + path
                uri = req_path
            elif mode == "sqli":
                payload = SQLI_PAYLOADS[(ci + w) % len(SQLI_PAYLOADS)]
                uri = http_uri_with_query(path, {param_mode: payload})
            elif mode == "xss":
                payload = XSS_PAYLOADS[(ci + w) % len(XSS_PAYLOADS)]
                uri = http_uri_with_query(path, {param_mode: payload})
            elif mode == "normal":
                uri = path
            else:
                uri = path

            host = f"{dst_ip}:{dst_port}"
            req = build_http_request(method, uri, host=host)
            pkts.append(make_tcp_pkt(src_ip, dst_ip, sport, dst_port, "A", req, req_t))

    ensure_dir(os.path.dirname(out_pcap))
    wrpcap(out_pcap, pkts)
    print(f"[{mode}] -> {out_pcap} (packets={len(pkts)}, windows={num_windows})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTTP payload synthetic PCAPs")
    parser.add_argument("--out-dir", default="datasets/pcap/pcaps_http", help="Output directory")
    parser.add_argument("--dst-ip", default="192.168.20.1", help="Destination IP")
    parser.add_argument("--dst-port", type=int, default=8000, help="Destination TCP port")
    parser.add_argument("--src-ip", default="192.168.20.50", help="Source IP (synthetic)")
    parser.add_argument("--window-sec", type=float, default=10.0, help="PCAP time window seconds")
    parser.add_argument("--num-windows", type=int, default=24, help="Number of windows/buckets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--only", type=str, default="all", help="dir_enum|sqli|xss|normal|all")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    to_do = []
    if args.only == "all":
        to_do = ["dir_enum", "sqli", "xss", "normal"]
    else:
        to_do = [args.only]

    for mode in to_do:
        if mode == "dir_enum":
            conns_range = (8, 25)
            sport_base = 42000
        elif mode == "sqli":
            conns_range = (6, 20)
            sport_base = 43000
        elif mode == "xss":
            conns_range = (6, 20)
            sport_base = 44000
        elif mode == "normal":
            # 正常：每窗连接数低，且路径集中
            conns_range = (1, 4)
            sport_base = 45000
        else:
            raise ValueError(f"unknown mode: {mode}")

        out_pcap = os.path.join(
            args.out_dir,
            f"http_{mode}_{args.dst_ip.replace('.','_')}_{args.dst_port}_{int(args.window_sec)}s_{args.num_windows}w.pcap",
        )

        gen_for_class(
            out_pcap=out_pcap,
            src_ip=args.src_ip,
            dst_ip=args.dst_ip,
            dst_port=int(args.dst_port),
            window_sec=float(args.window_sec),
            num_windows=int(args.num_windows),
            seed=int(args.seed) + hash(mode) % 1000,
            mode=mode,
            conns_per_window=conns_range,
            sport_base=sport_base,
        )


if __name__ == "__main__":
    main()


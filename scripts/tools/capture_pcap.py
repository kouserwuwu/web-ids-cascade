"""
capture_pcap.py — Windows + Scapy 抓包示例

常见场景：
- 虚拟机经 VMnet8 扫/访问宿主机：抓包应选 VMnet8 对应的 NPF 网卡（\\Device\\NPF_{...}）。
- 如果改用其他网卡（有线/WiFi/Host-only 等），同理填写其可抓 NPF；也可以用关键字（如 `vmnet8`）让脚本自动匹配。

示例（PowerShell）：
  $env:SCAPY_IFACE='\\Device\\NPF_{...}'   # VMnet8 的 NPF
  $env:PCAP_FILTER='tcp port 8000'
  $env:CHECK_DST='192.168.20.1'
  $env:CHECK_DPORT='8000'
  python capture_pcap.py
"""
import os
import datetime

from scapy.all import sniff, wrpcap, get_if_list
from scapy.arch.windows import get_windows_if_list


def _guid_strip(g: str) -> str:
    g = (g or "").strip()
    if g.startswith("{") and g.endswith("}"):
        g = g[1:-1]
    return g


def resolve_iface(iface_input: str) -> str:
    """
    Windows/Npcap 下：scapy 的 iface 通常需要是 `\\Device\\NPF_{GUID}` 这种“可抓的 NPF 设备名”。

    支持输入形式：
    - 直接给 `\\Device\\NPF_{...}`（会先验证是否在 scapy 可抓列表中）
    - 给关键字（例如 `vmnet8` 或网卡描述片段），会用 get_windows_if_list() 反查并优先返回可抓 NPF
    """
    if not iface_input:
        raise ValueError("iface_input 为空")

    # scapy 真实“可用”的 iface 列表（通常形如 \\Device\\NPF_{...}）
    available = set(get_if_list())

    if "\\Device\\NPF_" in iface_input:
        if iface_input in available:
            return iface_input
        raise ValueError(
            f"指定的 NPF 不在 scapy 可抓列表中：{iface_input}\n"
            f"请用 scripts/tools/list_ifaces.py 查看正确的 \\\\Device\\\\NPF_{{...}}，或改用关键字（如 vmnet8）。"
        )

    input_lower = iface_input.lower().strip()
    win_infos = get_windows_if_list()

    matches = []
    for info in win_infos:
        desc = str(info.get("description", "") or "")
        name = str(info.get("name", "") or "")
        guid = _guid_strip(str(info.get("guid", "") or ""))

        desc_lower = desc.lower()
        name_lower = name.lower()
        ok = (
            input_lower in desc_lower
            or input_lower in name_lower
            or input_lower == desc_lower
            or input_lower == name_lower
        )
        if ok:
            matches.append((info, desc, name, guid))

    if not matches:
        raise ValueError(
            f"未匹配到任何接口信息：{iface_input}\n"
            f"请确认 SCAPY_IFACE/npf 是关键字（如 vmnet8）或可抓的 \\\\Device\\\\NPF_{{...}}。"
        )

    # 优先返回由 guid 组装出的 NPF（且必须在 available 里）
    preferred_npf = []
    for info, desc, name, guid in matches:
        if not guid:
            continue
        npf = rf"\Device\NPF_{{{guid}}}"
        if npf in available:
            preferred_npf.append(npf)

    if preferred_npf:
        # 直接返回第一个可用 NPF
        # 同时打印候选便于你调试
        print("可用的候选 NPF 接口：", ", ".join(preferred_npf[:5]))
        return preferred_npf[0]

    # 否则尝试返回 info['name']（有些情况下 name 本身就是可用 iface）
    for info, desc, name, guid in matches:
        if name in available:
            return name

    # 兜底：退回到一个 scapy 已知可抓的 iface，避免返回不可用字符串
    print("注意：匹配到接口信息，但未定位到可抓的 NPF；将回退到可抓 iface。")

    fallback = sorted(available)[0] if available else iface_input
    print("回退到可用 iface：", fallback)
    return fallback


# 网卡输入支持两种常见写法：
# - `npf` / `NPF`：你提到的“全局变量 npf”用法（优先）
# - `SCAPY_IFACE`：兼容保留
available_ifaces_for_default = []
try:
    available_ifaces_for_default = sorted(set(get_if_list()))
except Exception:
    available_ifaces_for_default = []

iface_from = "default"
IFACE_INPUT = ""
if os.environ.get("npf"):
    IFACE_INPUT = os.environ.get("npf", "")
    iface_from = "npf"
elif os.environ.get("NPF"):
    IFACE_INPUT = os.environ.get("NPF", "")
    iface_from = "NPF"
elif os.environ.get("SCAPY_IFACE"):
    IFACE_INPUT = os.environ.get("SCAPY_IFACE", "")
    iface_from = "SCAPY_IFACE"
elif available_ifaces_for_default:
    IFACE_INPUT = available_ifaces_for_default[0]
    iface_from = "default"

if not IFACE_INPUT:
    raise RuntimeError("未设置可用网卡：请设置 `SCAPY_IFACE` 或 `npf/NPF`。")

IFACE = resolve_iface(IFACE_INPUT)

COUNT = 1000   # 抓多少个包（可改大）
TIMEOUT = 60   # 最长抓包时间（秒）
PCAP_TAG = (os.environ.get("PCAP_TAG", "") or "").strip()
PCAP_NAME = (os.environ.get("PCAP_NAME", "") or "").strip()
OUT_FILE_ENV = (os.environ.get("OUT_FILE", "") or "").strip()

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base = PCAP_NAME or PCAP_TAG or "capture"
base = "".join([c if (c.isalnum() or c in ("-", "_")) else "_" for c in base]).strip("_") or "capture"
default_out = f"{base}_{ts}.pcap"
OUT_FILE = OUT_FILE_ENV or default_out
if not OUT_FILE.lower().endswith(".pcap"):
    OUT_FILE = OUT_FILE + ".pcap"
PCAP_FILTER = os.environ.get("PCAP_FILTER", "").strip()

print(
    f"开始抓包：iface_input({iface_from})={IFACE_INPUT} -> iface={IFACE}, count={COUNT}, timeout={TIMEOUT}s"
    + (f", filter='{PCAP_FILTER}'" if PCAP_FILTER else "")
)
kwargs = {}
if PCAP_FILTER:
    kwargs["filter"] = PCAP_FILTER
pkts = sniff(iface=IFACE, count=COUNT, timeout=TIMEOUT, **kwargs)
print(f"共捕获 {len(pkts)} 个包，保存到 {OUT_FILE}")
wrpcap(OUT_FILE, pkts)
print("完成。")

# 快速检查：是否出现目标 IP / 目的端口（可用于验证 VMnet8/其他网卡抓到扫描流量）
try:
    from scapy.all import IP, TCP, Raw

    seen_ips = set()
    seen_ports = set()
    http_get_count = 0
    http_post_count = 0
    http_sample_snippets = []
    for p in pkts:
        if IP in p and TCP in p:
            seen_ips.add(p[IP].dst)
            seen_ports.add(int(p[TCP].dport))
        # 仅统计 HTTP 请求行（GET/POST）是否真的出现在 TCP payload 里
        # 如果这里是 0，后续 deepseek 的 SQLi/XSS 基于请求语义基本不可能成立。
        if TCP in p and Raw in p:
            d = bytes(p[Raw])
            if b"GET " in d:
                http_get_count += 1
                if len(http_sample_snippets) < 3:
                    http_sample_snippets.append(d[:180])
            if b"POST " in d:
                http_post_count += 1
                if len(http_sample_snippets) < 3:
                    http_sample_snippets.append(d[:180])

    check_dst_raw = os.environ.get("CHECK_DST", "192.168.20.1").strip()
    want_ips = [x.strip() for x in check_dst_raw.split(",") if x.strip()]
    want_dport = int(os.environ.get("CHECK_DPORT", "8000"))

    print("快速检查：")
    for ip in want_ips:
        print(f"  是否包含 dst_ip={ip} :", ip in seen_ips)
    print(f"  是否包含 dst_port={want_dport} :", want_dport in seen_ports)
    if not any(ip in seen_ips for ip in want_ips):
        print("  示例 dst_ip:", list(sorted(seen_ips))[:15])
    if want_dport not in seen_ports:
        print("  示例 dst_port:", list(sorted(seen_ports))[:25])
    print(f"  HTTP 请求行 GET 次数(抓到TCP payload): {http_get_count}")
    print(f"  HTTP 请求行 POST 次数(抓到TCP payload): {http_post_count}")
    if http_sample_snippets:
        print("  HTTP 请求行样例（payload 前180字节，utf-8忽略错误）：")
        for snip in http_sample_snippets:
            try:
                print("   -", snip.decode("utf-8", errors="ignore").replace("\r", "\\r").replace("\n", "\\n"))
            except Exception:
                pass
except Exception as e:
    print("快速检查失败：", e)
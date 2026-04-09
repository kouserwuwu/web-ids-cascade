"""
export_pcap_flows_csv.py

把 PCAP 通过与 demo 一致的 window 聚合逻辑导出为训练 CSV（同域增量训练用）。

用法（PowerShell）：
  python export_pcap_flows_csv.py --pcap capture.pcap --out datasets/pcap/csv/window_flows.csv --window-sec 10 --label 1

说明：
- label 可选：不传则不写 __y__；传 0/1 则写入 __y__ 二值标签列，便于 ai_defend.prepare_cicids_flows 直接读取。
"""

import argparse
import os
import sys
import glob
from typing import Any, Dict, List, Optional


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    # 允许从任意工作目录运行
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    parser = argparse.ArgumentParser(description="Export windowed flows from PCAP to CSV")
    parser.add_argument("--pcap", required=True, help="PCAP 路径（支持 glob，如 datasets/pcap/pcaps_http/http_normal_*.pcap）")
    parser.add_argument("--out", required=True, help="输出 CSV 路径")
    parser.add_argument("--window-sec", type=float, default=10.0, help="时间窗秒数")
    parser.add_argument("--max-windows", type=int, default=2048, help="最多聚合多少个窗口桶")
    parser.add_argument("--key-mode", type=str, default="src_dst_dport", choices=["src_dst_dport", "src_dst"], help="聚合 key 粒度")
    parser.add_argument("--label", type=int, default=None, help="可选：写入 __y__ 标签（0/1）")
    parser.add_argument("--add-pcap-col", action="store_true", help="可选：添加 _pcap 列，记录来源文件名（当 --pcap 为 glob 或多文件时有用）")
    args = parser.parse_args()

    from realtime_ids_demo import flows_from_pcap_windowed

    rows: List[Dict[str, Any]] = []
    # 支持 glob：允许一次性导出多个 pcap 并合并到一个 CSV
    pcap_arg = str(args.pcap)
    pcap_paths = []
    if any(ch in pcap_arg for ch in ["*", "?", "["]):
        pcap_paths = sorted(glob.glob(pcap_arg))
    else:
        pcap_paths = [pcap_arg]

    if not pcap_paths:
        raise SystemExit(f"未匹配到任何 PCAP 文件：{pcap_arg}")

    total_flows = 0
    for pcap_path in pcap_paths:
        flows: List[Dict[str, Any]] = flows_from_pcap_windowed(
            pcap_path,
            window_sec=float(args.window_sec),
            max_windows=int(args.max_windows),
            key_mode=str(args.key_mode),
        )
        if not flows:
            continue
        total_flows += len(flows)
        for f in flows:
            r = dict(f)
            # 训练侧主要用数值特征；IP/端口留着也可用于后续分析
            if args.label is not None:
                r["__y__"] = int(args.label)
            if args.add_pcap_col:
                r["_pcap"] = os.path.basename(pcap_path)
            rows.append(r)

    if not rows:
        raise SystemExit("未从输入 PCAP 聚合到任何 flows。")

    _ensure_dir(args.out)
    try:
        import pandas as pd
    except Exception:
        raise SystemExit("需要 pandas 才能导出 CSV，请先安装：pip install pandas")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"已从 {len(pcap_paths)} 个 PCAP 聚合 {total_flows} 条 flows，导出 {len(df)} 行到: {args.out}")


if __name__ == "__main__":
    main()


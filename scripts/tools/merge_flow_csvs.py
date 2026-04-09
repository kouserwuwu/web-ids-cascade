"""
merge_flow_csvs.py

合并多份“PCAP->window flows 导出”的 CSV，生成一个训练用总表。

示例：
  python merge_flow_csvs.py ^
    --in data/local/normal_window_flows.csv data/local/dirsearch_window_flows.csv ^
    --out data/local/train_window_flows_merged.csv ^
    --shuffle
"""

import argparse
import os
from typing import List


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple flow CSVs into one")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="输入 CSV（可多个）")
    parser.add_argument("--out", required=True, help="输出合并后的 CSV")
    parser.add_argument("--shuffle", action="store_true", help="是否打乱行顺序")
    parser.add_argument("--max-rows", type=int, default=0, help="最多保留多少行（0 表示不限制）")
    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception:
        raise SystemExit("需要 pandas：pip install pandas")

    dfs: List["pd.DataFrame"] = []
    for p in args.inputs:
        if not os.path.exists(p):
            raise SystemExit(f"找不到输入 CSV：{p}")
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    if not dfs:
        raise SystemExit("未读取到任何输入 CSV。")

    merged = pd.concat(dfs, axis=0, ignore_index=True)

    if args.shuffle:
        merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if args.max_rows and args.max_rows > 0 and len(merged) > args.max_rows:
        merged = merged.head(args.max_rows).reset_index(drop=True)

    _ensure_dir(args.out)
    merged.to_csv(args.out, index=False, encoding="utf-8")

    has_y = "__y__" in merged.columns
    y_info = ""
    if has_y:
        try:
            vc = merged["__y__"].astype(int).value_counts().to_dict()
            y_info = f", __y__分布={vc}"
        except Exception:
            y_info = ", __y__存在(但无法统计)"

    print(f"已合并 {len(args.inputs)} 个文件 -> {args.out} (rows={len(merged)}{y_info})")


if __name__ == "__main__":
    main()


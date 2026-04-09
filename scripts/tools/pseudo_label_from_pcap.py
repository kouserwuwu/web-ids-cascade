"""
pseudo_label_from_pcap.py

使用“一级 + 真实 DeepSeek 二级”对 PCAP 做伪标注，导出可用于一级增量训练的 CSV。

核心思路：
- 用实时链路中的 flows_from_pcap_windowed 提取同域特征；
- 用 infer_batch 跑级联；
- 仅保留高置信样本作为伪标签（建议优先用 deepseek 来源）；
- 导出为带 __y__ 的 CSV，后续可 merge + retrain_first_stage_from_flows。

示例：
  # 1) 先设置真实 DeepSeek（当前会话）
  $env:USE_REAL_DEEPSEEK="1"
  $env:DEEPSEEK_API_KEY="..."

  # 2) 伪标签导出（只用 deepseek 且 score>=0.80）
  python pseudo_label_from_pcap.py ^
    --pcap capture.pcap ^
    --out datasets/pcap/csv/pseudo_from_capture.csv ^
    --window-sec 10 ^
    --low 2e-5 --high 0.9999999 ^
    --only-source deepseek ^
    --min-score 0.80
"""

import argparse
import os
import sys
from typing import Dict, List


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _is_real_deepseek_enabled() -> bool:
    use_real = os.environ.get("USE_REAL_DEEPSEEK", "0") == "1"
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    return use_real and bool(api_key) and api_key != "your_api_key_here"


def main() -> None:
    # 允许从任意工作目录运行
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    parser = argparse.ArgumentParser(description="Pseudo-label PCAP flows via real DeepSeek")
    parser.add_argument("--pcap", required=True, help="输入 pcap 文件")
    parser.add_argument("--out", required=True, help="输出带 __y__ 的 CSV")
    parser.add_argument("--model", default="models/first_stage_lgbm_behavior_v3.pkl", help="一级模型路径")
    parser.add_argument("--window-sec", type=float, default=10.0, help="window 聚合秒数")
    parser.add_argument("--max-windows", type=int, default=512, help="最多窗口数")
    parser.add_argument("--low", type=float, default=2e-5, help="一级低阈值")
    parser.add_argument("--high", type=float, default=0.9999999, help="一级高阈值")
    parser.add_argument("--ds-conf", type=int, default=70, help="二级置信阈值")
    parser.add_argument("--batch-size", type=int, default=8, help="二级批大小")
    parser.add_argument("--only-source", choices=["all", "deepseek"], default="deepseek", help="仅保留某来源样本做伪标签")
    parser.add_argument("--min-score", type=float, default=0.80, help="最小 score 才保留（默认 0.80）")
    args = parser.parse_args()

    if not _is_real_deepseek_enabled():
        raise SystemExit(
            "未启用真实 DeepSeek。请先设置 USE_REAL_DEEPSEEK=1 且 DEEPSEEK_API_KEY 有效。"
        )

    if not os.path.exists(args.pcap):
        raise SystemExit(f"找不到 pcap: {args.pcap}")
    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型: {args.model}")

    try:
        import pandas as pd
    except Exception:
        raise SystemExit("需要 pandas：pip install pandas")

    from realtime_ids_demo import flows_from_pcap_windowed, load_pipeline
    from ai_defend import infer_batch

    model, analyzer = load_pipeline(args.model)
    flows: List[Dict] = flows_from_pcap_windowed(
        args.pcap,
        window_sec=float(args.window_sec),
        max_windows=int(args.max_windows),
    )
    if not flows:
        raise SystemExit("未解析到有效 flows。")

    results = infer_batch(
        flows,
        first_stage_model=model,
        semantic_analyzer=analyzer,
        low=float(args.low),
        high=float(args.high),
        ds_conf_thr=int(args.ds_conf),
        batch_size=int(args.batch_size),
    )

    rows: List[Dict] = []
    skipped = 0
    for f, r in zip(flows, results):
        source = str((r or {}).get("source", ""))
        try:
            score = float((r or {}).get("score", 0.0))
        except Exception:
            score = 0.0
        label = int((r or {}).get("label", 0))

        if args.only_source == "deepseek" and source != "deepseek":
            skipped += 1
            continue
        if score < float(args.min_score):
            skipped += 1
            continue

        row = dict(f)
        row["__y__"] = int(label)
        row["_pseudo_source"] = source
        row["_pseudo_score"] = score
        # 这些字段不会用于一级训练，但对后续人工抽检有帮助
        row["_pseudo_ds_type"] = (r or {}).get("ds_type", "")
        row["_pseudo_cost_cny"] = float((r or {}).get("ds_cost_cny", 0.0) or 0.0)
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"筛选后无样本可导出（only_source={args.only_source}, min_score={args.min_score}）。"
        )

    df = pd.DataFrame(rows)
    _ensure_dir(args.out)
    df.to_csv(args.out, index=False, encoding="utf-8")

    y_dist = df["__y__"].astype(int).value_counts().to_dict()
    print(f"总 flows={len(flows)}，保留={len(df)}，跳过={skipped}")
    print(f"伪标签分布 __y__={y_dist}")
    print(f"已导出: {args.out}")


if __name__ == "__main__":
    main()


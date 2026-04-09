"""
evaluate_labeled_flows_csv.py

对带真值标签 __y__（0/1）的 window-flow CSV 跑完整级联（一级 LightGBM + 二级 DeepSeek/Mock），
输出二分类指标与二级攻击类型分布，便于毕设实验报表。

用法（在项目根目录 demo 下）：
  python evaluate_labeled_flows_csv.py ^
    --csv datasets/pcap/csv/http_web_train.csv ^
    --model models/first_stage_lgbm_http_web.pkl

可选：--limit 200 只评前 N 行（省 DeepSeek 费用）；--low/--high 覆盖默认阈值；--dump-errors 打印 FP/FN 对应 CSV 行与 flow 摘要。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# 允许从任意工作目录运行：保证能 import 项目根下的模块（ai_defend 等）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _row_to_flow(row: pd.Series) -> Dict[str, Any]:
    """将 CSV 行转为 infer_batch 所需的 flow dict；去掉标签与伪标签元数据列。"""
    d: Dict[str, Any] = {}
    for k, v in row.items():
        ks = str(k).strip()
        if ks == "__y__" or ks.startswith("_pseudo_"):
            continue
        if pd.isna(v):
            continue
        if isinstance(v, (np.bool_, bool)):
            d[ks] = bool(v)
        elif isinstance(v, (np.floating, float)):
            d[ks] = float(v)
        elif isinstance(v, (np.integer, int)):
            d[ks] = int(v)
        elif isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            # 部分导出列可能是 list 的字符串形式
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast

                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        d[ks] = list(parsed)
                        continue
                except Exception:
                    pass
            d[ks] = s
        else:
            d[ks] = v
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cascade on labeled window-flow CSV")
    parser.add_argument("--csv", required=True, help="含 __y__ 列的 flows CSV")
    parser.add_argument("--model", default="models/first_stage_lgbm.pkl", help="一级模型路径")
    parser.add_argument("--limit", type=int, default=0, help="仅评前 N 行，0 表示全量")
    parser.add_argument("--low", type=float, default=None, help="一级低阈值（默认与 demo 一致）")
    parser.add_argument("--high", type=float, default=None, help="一级高阈值（默认与 demo 一致）")
    parser.add_argument("--ds-conf", type=int, default=70, help="二级置信度阈值")
    parser.add_argument("--batch-size", type=int, default=16, help="二级批大小")
    parser.add_argument(
        "--dump-errors",
        action="store_true",
        help="打印误判样本：假阳性 FP(真值0/预测1)、假阴性 FN(真值1/预测0)，含行号与 result 摘要",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"找不到 CSV：{args.csv}")
    if not os.path.exists(args.model):
        raise SystemExit(f"找不到模型：{args.model}")

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except Exception as e:
        raise SystemExit(f"需要 scikit-learn：{e}")

    from ai_defend import infer_batch
    from realtime_ids_demo import FIXED_HIGH, FIXED_LOW, load_pipeline, summarize_deepseek_usage

    low = float(args.low) if args.low is not None else float(FIXED_LOW)
    high = float(args.high) if args.high is not None else float(FIXED_HIGH)

    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()
    if "__y__" not in df.columns:
        raise SystemExit("CSV 缺少 __y__ 列（真值标签）。")

    if args.limit and args.limit > 0:
        df = df.head(int(args.limit)).copy()

    y_true = pd.to_numeric(df["__y__"], errors="coerce").fillna(0).astype(int).values
    flows: List[Dict[str, Any]] = [_row_to_flow(df.iloc[i]) for i in range(len(df))]

    model, analyzer = load_pipeline(args.model)
    results = infer_batch(
        flows,
        first_stage_model=model,
        semantic_analyzer=analyzer,
        low=low,
        high=high,
        ds_conf_thr=int(args.ds_conf),
        batch_size=int(args.batch_size),
    )

    y_pred = np.array([int((r or {}).get("label", 0)) for r in results], dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    print("样本数:", len(y_true))
    print("二分类准确率 accuracy:", f"{acc:.4f}")
    print("混淆矩阵 [[TN FP],[FN TP]] (label 0=正常, 1=恶意):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nclassification_report:")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["benign(0)", "malicious(1)"], zero_division=0))

    src_cnt = Counter(str((r or {}).get("source", "")) for r in results)
    print("\n按 result.source 统计:", dict(src_cnt))

    ds_types = [(r or {}).get("ds_type") for r in results if (r or {}).get("ds_type")]
    type_cnt = Counter([str(x) for x in ds_types if x])
    print("\n二级 ds_type 出现次数（含 deepseek/rule 等路径上有 ds_type 的项）:", dict(type_cnt.most_common(20)))
    print("本批中出现过的攻击/语义类型种类数:", len(type_cnt))

    usage = summarize_deepseek_usage(results)
    if usage:
        print("\nDeepSeek 调用汇总（若有）:")
        print(json.dumps(usage, ensure_ascii=False, indent=2))

    if args.dump_errors:
        fps: List[int] = []
        fns: List[int] = []
        for i in range(len(y_true)):
            if y_true[i] == 0 and y_pred[i] == 1:
                fps.append(i)
            elif y_true[i] == 1 and y_pred[i] == 0:
                fns.append(i)

        def _brief_flow(flow: Dict[str, Any], max_len: int = 200) -> str:
            keys = ("http_uri", "uri", "path", "dst_port", "src_ip", "dst_ip", "proto")
            parts: List[str] = []
            for k in keys:
                if k in flow and flow[k] is not None:
                    parts.append(f"{k}={flow[k]}")
            s = " ".join(parts) if parts else str(flow)[:max_len]
            return s[:max_len] + ("..." if len(s) > max_len else "")

        print("\n--- 误判明细（与 CSV 行顺序一致；行号=表头下第几行，含表头为行号+1） ---")
        print(f"假阳性 FP（正常被判恶意）: {len(fps)} 条")
        for i in fps:
            r = results[i] or {}
            print(
                f"  CSV行号={i + 2}  idx={i}  y=0 pred=1  "
                f"source={r.get('source')}  ds_type={r.get('ds_type')}  score={r.get('score')}"
            )
            print(f"    flow摘要: {_brief_flow(flows[i])}")

        print(f"假阴性 FN（恶意被判正常）: {len(fns)} 条")
        for i in fns:
            r = results[i] or {}
            print(
                f"  CSV行号={i + 2}  idx={i}  y=1 pred=0  "
                f"source={r.get('source')}  ds_type={r.get('ds_type')}  score={r.get('score')}"
            )
            print(f"    flow摘要: {_brief_flow(flows[i])}")


if __name__ == "__main__":
    main()

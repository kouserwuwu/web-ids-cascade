"""
retrain_first_stage_from_flows.py

用“PCAP->window flows 导出”的同域训练集（含 __y__）重训一级 LightGBM，
定位为：行为分流器（不是语义攻击分类器）。

示例：
  python retrain_first_stage_from_flows.py ^
    --csv datasets/pcap/train_window_flows_merged.csv ^
    --model models/first_stage_lgbm.pkl ^
    --test-size 0.2 ^
    --class-weight balanced
"""

import argparse
import os
import sys


FEATURE_KEYS = [
    "packet_count",
    "byte_count",
    "avg_pkt_len",
    "duration",
    "packets_per_sec",
    "syn_ratio",
    "dst_port_count",
    "byte_per_sec",
    "rst_ratio",
    "fin_ratio",
    "ack_ratio",
    "conn_count",
]


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    # 允许从任意工作目录运行
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    parser = argparse.ArgumentParser(description="Retrain first-stage LightGBM from window-flow CSV")
    parser.add_argument("--csv", required=True, help="训练 CSV（必须包含 __y__）")
    parser.add_argument("--model", default="models/first_stage_lgbm.pkl", help="输出模型路径（joblib）")
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--n-estimators", type=int, default=500, help="LightGBM 树数量")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="学习率")
    parser.add_argument("--class-weight", type=str, default=None, help="例如 balanced（或留空）")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"找不到 CSV：{args.csv}")

    try:
        import numpy as np
        import pandas as pd
        import joblib
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        raise SystemExit(f"缺少依赖：{e}\n建议安装：pip install pandas numpy scikit-learn lightgbm joblib")

    df = pd.read_csv(args.csv, low_memory=False)
    df.columns = df.columns.str.strip()
    if "__y__" not in df.columns:
        raise SystemExit("训练 CSV 缺少 __y__ 列。请在导出/合并时设置 --label 0/1，或手工补 __y__。")

    # 数值化 + 缺失处理
    for k in FEATURE_KEYS:
        if k not in df.columns:
            df[k] = 0
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)

    y = pd.to_numeric(df["__y__"], errors="coerce").fillna(0).astype(int).values
    X = df[FEATURE_KEYS].astype(float).values

    if len(set(y.tolist())) < 2:
        raise SystemExit(f"__y__ 只有单一类别：{set(y.tolist())}，无法训练。")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=float(args.test_size), random_state=42, stratify=y
    )

    clf = LGBMClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        random_state=42,
        n_jobs=-1,
        class_weight=args.class_weight,
    )
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)

    _ensure_dir(args.model)
    joblib.dump(clf, args.model)

    # 同时打印概率分位数，方便你选 low/high
    qs = np.quantile(p_val, [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).tolist()
    dist = {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}

    print(f"已训练并保存一级模型：{args.model}")
    print(f"样本数={len(y)} 标签分布={dist}  val_AUC={auc:.4f}")
    print("val 概率分位数 q=", [float(x) for x in qs])
    print("提示：可把 low 设在 q0.05~q0.20，高阈值 high 设在 q0.80~q0.95 作为初始搜索范围。")


if __name__ == "__main__":
    main()


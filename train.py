"""训练入口。

用法::

    python train.py
"""
import pandas as pd

from app.pipeline import run_training


def main() -> None:
    print("=" * 72)
    print("电信用户流失预测 - 模型训练")
    print("=" * 72)

    best_name, metrics, _, _ = run_training()

    report = pd.DataFrame(metrics).T
    report = report[["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]]
    print("\n各模型在测试集上的表现（按 ROC-AUC 降序）：")
    print(report.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n最佳模型：{best_name}")
    print("产物已写入 output/ 目录：模型、metrics.json、对比图与高风险用户名单。")


if __name__ == "__main__":
    main()

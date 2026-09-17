"""端到端训练编排：加载 → 特征工程 → 分层划分 → 训练 → 评估 → 落盘。

产物（全部写入 ``output/``）：
    churn_model.joblib          最佳模型管线（含预处理器，可直接 predict）
    metrics.json                全部模型指标 + 最佳模型名 + 数据规模
    roc_curves.png / pr_curves.png   模型对比图
    confusion_matrix.png        最佳模型混淆矩阵
    feature_importance.png      最佳模型特征重要性
    high_risk_customers.csv     测试集上的高流失风险用户名单（业务产物）
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app import evaluate as ev
from app.config import (
    CM_FIG_PATH,
    FI_FIG_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    PR_FIG_PATH,
    RANDOM_STATE,
    RISK_LIST_PATH,
    RISK_THRESHOLD,
    ROC_FIG_PATH,
    TEST_SIZE,
    TOP_RISK_EXPORT,
)
from app.data_loader import load_data
from app.features import engineer_features, split_X_y
from app.model import train_all_models


def export_high_risk_customers(best_model, X_test_raw: pd.DataFrame,
                               save_path=RISK_LIST_PATH,
                               threshold: float = RISK_THRESHOLD,
                               top_n: int = TOP_RISK_EXPORT) -> pd.DataFrame:
    """对测试集打分，导出高流失风险用户名单（保留原始可读字段）。

    典型业务用法：营销团队按名单从高到低投放挽留优惠。
    """
    X_feat = engineer_features(X_test_raw)
    scored = X_test_raw.copy()
    scored["churn_prob"] = ev.predict_proba(best_model, X_feat)
    scored["high_risk"] = scored["churn_prob"] >= threshold
    high_risk = (
        scored[scored["high_risk"]]
        .sort_values("churn_prob", ascending=False)
        .head(top_n)
    )
    display_cols = [
        "customerID", "gender", "tenure", "Contract", "InternetService",
        "MonthlyCharges", "TotalCharges", "PaymentMethod", "churn_prob",
    ]
    high_risk[display_cols].to_csv(save_path, index=False, encoding="utf-8-sig")
    return high_risk


def run_training(threshold: float = RISK_THRESHOLD, save: bool = True):
    """执行完整训练流程，返回 (最佳模型名, 指标字典, 最佳模型, 测试集)。"""
    # 1. 加载 + 特征工程（衍生特征不使用标签，划分前构造无泄漏）
    df_raw = load_data()
    df = engineer_features(df_raw)
    X, y = split_X_y(df)

    # 2. 分层划分，保持训练/测试集中流失比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 3. 训练全部候选模型
    models = train_all_models(X_train, y_train, random_state=RANDOM_STATE)

    # 4. 评估并选优（按 ROC-AUC）
    metrics = ev.evaluate_all(models, X_test, y_test, threshold)
    best_name = next(iter(metrics))
    best_model = models[best_name]

    # 5. 可视化
    ev.plot_roc_curves(models, X_test, y_test, ROC_FIG_PATH)
    ev.plot_pr_curves(models, X_test, y_test, PR_FIG_PATH)
    ev.plot_confusion_matrix(best_model, X_test, y_test, CM_FIG_PATH, threshold)
    ev.plot_feature_importance(best_model, FI_FIG_PATH)

    # 6. 导出测试集高风险名单（保留原始字段，便于业务核对）
    test_raw = df_raw.loc[X_test.index]
    high_risk = export_high_risk_customers(
        best_model, test_raw, threshold=threshold
    )

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        payload = {
            "best_model": best_name,
            "decision_threshold": threshold,
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "churn_rate": round(float(y.mean()), 4),
            "high_risk_count_in_test": int(len(high_risk)),
            "models": {
                name: {k: round(v, 4) for k, v in row.items()}
                for name, row in metrics.items()
            },
        }
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return best_name, metrics, best_model, (X_test, y_test)

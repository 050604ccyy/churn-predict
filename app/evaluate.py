"""模型评估与可视化。

指标选择的业务依据：用户挽留动作有营销成本，漏掉一个真实流失用户（FN，
流失带走收入）的代价通常高于误挽留一个忠诚用户（FP，只损失一次优惠），
因此除通用的 ROC-AUC 外，重点关注**召回率**与 **PR-AUC**（在类别不平衡
场景下比准确率更能反映模型价值）。默认按 ROC-AUC 选择上线模型。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Windows 中文环境常用字体；缺失时 matplotlib 自动回退，不影响出图
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METRIC_COLUMNS = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]


def predict_proba(model, X) -> np.ndarray:
    """统一获取正类（流失）概率。"""
    return model.predict_proba(X)[:, 1]


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """计算单模型在给定阈值下的全套指标。"""
    proba = predict_proba(model, X)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def evaluate_all(models: dict, X, y, threshold: float = 0.5) -> dict:
    """评估所有模型并按 ROC-AUC 降序排列。"""
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X, y, threshold)
    return dict(sorted(
        results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True
    ))


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(models: dict, X, y, save_path) -> None:
    """多模型 ROC 曲线对比图。"""
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        proba = predict_proba(model, X)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        plt.plot(fpr, tpr, linewidth=1.6, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
    plt.xlabel("假正率 FPR")
    plt.ylabel("真正率 TPR（召回率）")
    plt.title("ROC 曲线对比")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(models: dict, X, y, save_path) -> None:
    """多模型 Precision-Recall 曲线对比图（不平衡场景更敏感）。"""
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        proba = predict_proba(model, X)
        precision, recall, _ = precision_recall_curve(y, proba)
        ap = average_precision_score(y, proba)
        plt.plot(recall, precision, linewidth=1.6, label=f"{name} (AP={ap:.3f})")
    baseline = float(np.mean(y))
    plt.axhline(baseline, color="k", linestyle="--", linewidth=1,
                label=f"正类占比={baseline:.2f}")
    plt.xlabel("召回率 Recall")
    plt.ylabel("精确率 Precision")
    plt.title("Precision-Recall 曲线对比")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(model, X, y, save_path, threshold: float = 0.5) -> None:
    """最佳模型混淆矩阵热力图。"""
    import seaborn as sns

    proba = predict_proba(model, X)
    cm = confusion_matrix(y, (proba >= threshold).astype(int))
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["预测不流失", "预测流失"],
        yticklabels=["实际不流失", "实际流失"],
    )
    plt.title(f"混淆矩阵（阈值={threshold:.2f}）")
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _transformed_feature_names(model) -> list[str]:
    """取预处理后的特征名（含 OneHot 展开后的列）。"""
    pre = model.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    return [str(n).replace("num__", "").replace("cat__", "") for n in names]


def plot_feature_importance(model, save_path, top_k: int = 20) -> None:
    """特征重要性图：树模型用 impurity 重要性，线性模型用系数绝对值。"""
    clf = model.named_steps["clf"]
    names = _transformed_feature_names(model)

    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
        title = f"Top{top_k} 特征重要性（树模型）"
    else:
        importance = np.abs(np.asarray(clf.coef_[0], dtype=float))
        title = f"Top{top_k} 特征重要性（逻辑回归 |系数|）"

    order = np.argsort(importance)[-top_k:]
    plt.figure(figsize=(8, max(4.0, 0.35 * len(order))))
    plt.barh([names[i] for i in order], importance[order], color="#4C72B0")
    plt.xlabel("重要性")
    plt.title(title)
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

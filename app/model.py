"""模型定义与训练。

四个参评模型，覆盖线性基线、集成树与不平衡处理手段：

==========================  ================================================
模型                         不平衡处理
==========================  ================================================
LogisticRegression          class_weight="balanced"（类别加权）
RandomForest                class_weight="balanced_subsample"（类别加权）
XGBoost                     scale_pos_weight（按正负样本比加权）
LogisticRegression+SMOTE    训练集内 SMOTE 过采样（重采样对照实验）
==========================  ================================================

所有模型都是 ``预处理 + 分类器`` 的 sklearn Pipeline，接口完全一致；
SMOTE 模型使用 imblearn Pipeline，保证重采样只发生在训练折内，
不会污染验证数据。
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.features import make_preprocessor

try:  # xgboost 为可选依赖，缺失时跳过对应模型而非整体报错
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

try:  # imbalanced-learn 为可选依赖
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    HAS_IMBLEARN = False


def build_models(scale_pos_weight: float = 1.0,
                 random_state: int = 42) -> dict[str, Pipeline]:
    """构造全部候选模型（未训练）。

    Parameters
    ----------
    scale_pos_weight:
        正类（流失）权重，取训练集负正样本比，用于 XGBoost。
    """
    models: dict[str, Pipeline] = {}

    models["LogisticRegression"] = Pipeline([
        ("preprocess", make_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state,
        )),
    ])

    models["RandomForest"] = Pipeline([
        ("preprocess", make_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=random_state,
        )),
    ])

    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("preprocess", make_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.05,
                min_child_weight=5, subsample=0.85, colsample_bytree=0.8,
                reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", n_jobs=-1, random_state=random_state,
            )),
        ])

    if HAS_IMBLEARN:
        models["LogisticRegression+SMOTE"] = ImbPipeline([
            ("preprocess", make_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])

    return models


def compute_scale_pos_weight(y) -> float:
    """负正样本数量比，作为 XGBoost 的 scale_pos_weight。"""
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """在训练集上拟合全部候选模型，返回 {模型名: 已拟合管线}。"""
    spw = compute_scale_pos_weight(y_train)
    fitted = {}
    for name, model in build_models(
        scale_pos_weight=spw, random_state=random_state
    ).items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted

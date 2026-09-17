"""特征工程层。

两阶段设计，边界清晰、杜绝数据泄漏：

1. :func:`engineer_features` —— 业务衍生特征。只使用单行自身的信息
   （不依赖标签、不依赖全局统计量），因此在 train/test 划分前后执行均可；
2. :func:`make_preprocessor` —— 由 ``StandardScaler + OneHotEncoder`` 组成的
   ``ColumnTransformer``，封装进模型 Pipeline 后只在训练集上 ``fit``，
   对验证/预测数据只做 ``transform``，编码规则不会泄漏测试集信息。
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import TARGET
from app.data_loader import ID_COLUMN, INTERNET_ADDONS

# Yes/No 二值字段（0/1 化后作为数值列）
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# 多类别字段（OneHot 编码）
CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "Contract",
    "PaymentMethod", "tenure_group",
] + INTERNET_ADDONS

# 数值字段（标准化）
NUMERICAL_COLS = [
    "SeniorCitizen", *BINARY_COLS,
    "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "avg_charge_per_month", "charge_per_service",
]

_YES_NO = {"Yes": 1, "No": 0}
_TENURE_BINS = [-0.1, 12.0, 24.0, 48.0, 60.0, 72.0]
_TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61-72"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造业务衍生特征并规范字段类型。

    新增特征：
        num_services           订购服务总数（电话 + 互联网 + 6 项增值服务）
        avg_charge_per_month   月均实际支出（TotalCharges / (tenure+1)）
        charge_per_service     单项服务均价（月度套餐性价比信号）
        tenure_group           在网时长分档（业务可解释的离散特征）
    """
    out = df.copy()
    out = out.drop(columns=[ID_COLUMN], errors="ignore")

    # 二值字段 0/1 化
    for col in BINARY_COLS:
        if col in out.columns:
            out[col] = out[col].map(_YES_NO).fillna(0).astype(int)

    # ---- 衍生特征 ----
    has_internet = (out["InternetService"] != "No").astype(int)
    addon_count = (out[INTERNET_ADDONS] == "Yes").sum(axis=1)
    out["num_services"] = out["PhoneService"] + has_internet + addon_count

    out["avg_charge_per_month"] = out["TotalCharges"] / (out["tenure"] + 1)
    out["charge_per_service"] = (
        out["MonthlyCharges"] / out["num_services"].clip(lower=1)
    )
    out["tenure_group"] = pd.cut(
        out["tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS
    ).astype(str)

    return out


def split_X_y(df: pd.DataFrame, target: str = TARGET):
    """拆分特征矩阵与 0/1 标签（Churn=Yes 为正类）。"""
    if target not in df.columns:
        raise ValueError(f"数据中缺少标签列 {target!r}")
    X = df.drop(columns=[target])
    y = (df[target] == "Yes").astype(int)
    return X, y


def make_preprocessor() -> ColumnTransformer:
    """数值标准化 + 类别独热的列变换器（随 Pipeline 一起 fit）。"""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )

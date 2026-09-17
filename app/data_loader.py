"""数据加载层。

数据获取策略（优先级从上到下）：
1. 若 ``data/telco_churn.csv`` 存在则直接读取——可放入 Kaggle 官方
   "Telco Customer Churn" 数据集（7043 行、21 列）无缝替换；
2. 否则生成一份**字段结构、取值域、边际分布均与真实数据一致**的模拟数据，
   且流失标签由服务组合特征经 logistic 函数生成（含真实信号而非随机标签），
   保证项目克隆后开箱即跑、模型能学到有效模式。
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATA_PATH, RANDOM_STATE, SYNTHETIC_N_ROWS

ID_COLUMN = "customerID"

# 与 Kaggle Telco 数据集完全一致的 21 个字段
COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# 六项互联网增值服务（无互联网用户取值 "No internet service"）
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# 各增值服务在有互联网用户中的开通率（贴近真实数据边际分布）
_ADDON_SUBSCRIBE_RATE = {
    "OnlineSecurity": 0.38,
    "OnlineBackup": 0.44,
    "DeviceProtection": 0.44,
    "TechSupport": 0.37,
    "StreamingTV": 0.47,
    "StreamingMovies": 0.48,
}


def generate_synthetic_data(n_rows: int = SYNTHETIC_N_ROWS,
                            seed: int = RANDOM_STATE) -> pd.DataFrame:
    """生成带真实流失信号的 Telco 同结构模拟数据。

    流失概率由一个 logistic 模型生成，强信号包括：月结合同、在网时长短、
    光纤用户、未开通安全/技术支持、电子支票支付、高月费等——与真实电信
    流失场景的经验规律一致。
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    # ---- 人口属性 ----
    gender = rng.choice(["Male", "Female"], size=n, p=[0.50, 0.50])
    senior = rng.binomial(1, 0.17, size=n)
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    # ---- 在网时长（月）：gamma 分布，新客占比偏高，截断到 0~72 ----
    tenure = np.clip(rng.gamma(shape=2.2, scale=14.0, size=n), 0, 72)
    tenure = tenure.round().astype(int)

    # ---- 电话服务 ----
    phone = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple = np.where(
        phone == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]),
    )

    # ---- 互联网服务与增值服务 ----
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]
    )
    addons: dict[str, np.ndarray] = {}
    for col in INTERNET_ADDONS:
        p = _ADDON_SUBSCRIBE_RATE[col]
        addons[col] = np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[p, 1.0 - p]),
        )

    # ---- 合同、账单与支付方式 ----
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.25, 0.20],
    )
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n, p=[0.34, 0.22, 0.22, 0.22],
    )

    # ---- 月费：由服务组合决定（电话 20 / DSL 30 / 光纤 55 / 每增值服务 6~10），
    #      校准后均值约 65 美元，与真实 Telco 数据一致 ----
    monthly = np.full(n, 0.0, dtype=float)
    monthly += (phone == "Yes") * 20.0
    monthly += (internet == "DSL") * 30.0
    monthly += (internet == "Fiber optic") * 55.0
    for col in INTERNET_ADDONS:
        monthly += (addons[col] == "Yes") * rng.uniform(6.0, 10.0, size=n)
    monthly += rng.normal(0.0, 3.0, size=n)
    monthly = np.clip(monthly, 18.25, 118.95).round(1)

    # ---- 总费用 ≈ 月费 × 在网月数（带轻微噪声）；
    #      tenure=0 的新客总费用为空字符串，与真实数据的脏数据形态一致 ----
    total = (monthly * tenure * rng.uniform(0.97, 1.03, size=n)).round(2)
    total = total.astype(object)
    total[tenure == 0] = " "

    # ---- 服务绑定数量（用于非线性信号）----
    num_services = (
        (phone == "Yes").astype(int)
        + (internet != "No").astype(int)
        + sum((addons[c] == "Yes").astype(int) for c in INTERNET_ADDONS)
    )

    # ---- 流失标签：logistic 生成，整体流失率约 26%（真实数据为 26.5%）----
    # 线性主效应
    logit = np.full(n, -3.5)
    logit += (contract == "Month-to-month") * 1.8
    logit += (contract == "One year") * 0.2
    logit -= 0.04 * tenure
    logit += (internet == "Fiber optic") * 0.8
    logit += (addons["OnlineSecurity"] == "No") * 0.6
    logit += (addons["TechSupport"] == "No") * 0.7
    logit += (payment == "Electronic check") * 0.7
    logit += 0.02 * (monthly - 65.0)
    logit += senior * 0.5
    logit += (partner == "Yes") * -0.3
    logit += (dependents == "Yes") * -0.2
    logit += (paperless == "Yes") * 0.3
    # 非线性交互效应（真实业务规律，树模型相对线性模型的优势所在）：
    # 月结合同 + 入网初期 = 流失最高危人群；
    # 光纤高月费 + 无技术支持 = 价格敏感型流失；
    # 月费超过 90 美元存在"账单冲击"阈值效应；
    # 服务绑定越少，越容易离开。
    logit += ((contract == "Month-to-month") & (tenure <= 6)) * 1.0
    logit += ((internet == "Fiber optic")
              & (addons["TechSupport"] == "No")) * 0.7
    logit += (tenure <= 3) * 0.7
    logit += (num_services <= 2) * 0.5
    # 月费呈 U 型风险：>90 有"账单冲击"，<30 多为低绑定的电话单服务用户
    logit += (monthly > 90.0) * 0.6
    logit += (monthly < 30.0) * 0.8
    logit += ((contract == "Month-to-month") & (tenure <= 3)
              & (internet == "Fiber optic")) * 0.8

    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    # ---- 客户 ID（形如 3668-QPYBK）----
    alphabet = np.array(list(string.ascii_uppercase + string.digits))
    ids = np.array([
        f"{i:04d}-" + "".join(rng.choice(alphabet, size=3)) for i in range(n)
    ])

    df = pd.DataFrame({
        "customerID": ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return df[COLUMNS]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """原始数据清洗：TotalCharges 脏值转数值、类型规范。"""
    out = df.copy()
    # 真实数据中 TotalCharges 含空格字符串，无法直接转 float
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # 少量新客 TotalCharges 缺失，用在网月数 × 月费回填
    fallback = out["tenure"] * out["MonthlyCharges"]
    out["TotalCharges"] = out["TotalCharges"].fillna(fallback)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out["tenure"] = out["tenure"].astype(int)
    return out


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """加载数据：优先读本地 CSV，否则生成模拟数据并缓存到 ``data/``。"""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = generate_synthetic_data()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    return clean_raw_data(df)

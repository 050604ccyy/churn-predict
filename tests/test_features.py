"""特征工程与预处理器测试。"""
import numpy as np
import pandas as pd

from app.data_loader import ID_COLUMN
from app.features import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    engineer_features,
    make_preprocessor,
    split_X_y,
)


def test_id_column_dropped(raw_df):
    df = engineer_features(raw_df)
    assert ID_COLUMN not in df.columns


def test_no_missing_values_after_engineering(raw_df):
    df = engineer_features(raw_df)
    assert df.isna().sum().sum() == 0


def test_derived_features(raw_df):
    df = engineer_features(raw_df)
    # 电话(0/1) + 互联网(0/1) + 6 项增值服务(0/1) => 0~8
    assert df["num_services"].between(0, 8).all()
    assert (df["avg_charge_per_month"] >= 0).all()
    assert (df["charge_per_service"] > 0).all()
    assert set(df["tenure_group"].unique()) <= {
        "0-12", "13-24", "25-48", "49-60", "61-72"
    }


def test_binary_columns_mapped(raw_df):
    df = engineer_features(raw_df)
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        assert set(df[col].unique()) <= {0, 1}


def test_split_xy(raw_df):
    df = engineer_features(raw_df)
    X, y = split_X_y(df)
    assert "Churn" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_preprocessor_output_shape_and_no_nan(split_data):
    X_train, X_test, _, _ = split_data
    pre = make_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    assert Xtr.ndim == 2
    assert Xtr.shape[0] == len(X_train)
    assert Xte.shape[1] == Xtr.shape[1]
    assert not np.isnan(Xtr).any()
    # OneHot + 数值列后的总维度不小于原始特征数
    assert Xtr.shape[1] >= X_train.shape[1]


def test_preprocessor_handles_unseen_category():
    """handle_unknown='ignore'：预测期出现训练时未见的类别不应报错。"""
    row = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": 0,
        "Dependents": 0, "tenure": 5, "PhoneService": 1,
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check", "MonthlyCharges": 55.0,
        "TotalCharges": 275.0, "num_services": 3,
        "avg_charge_per_month": 45.8, "charge_per_service": 18.3,
        "tenure_group": "0-12",
    }
    train_df = pd.DataFrame([row] * 4)
    pre = make_preprocessor()
    pre.fit(train_df)
    weird = train_df.iloc[[0]].copy()
    weird["Contract"] = "Unknown-contract-type"  # 模拟未见类别
    out = pre.transform(weird)
    assert out.shape[0] == 1

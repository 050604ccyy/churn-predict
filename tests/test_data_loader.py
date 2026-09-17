"""模拟数据生成与原始数据清洗测试。"""
import numpy as np
import pandas as pd

from app.data_loader import COLUMNS, INTERNET_ADDONS, clean_raw_data, generate_synthetic_data


def test_shape_and_columns():
    df = generate_synthetic_data(n_rows=500, seed=42)
    assert df.shape == (500, 21)
    assert list(df.columns) == COLUMNS


def test_churn_rate_close_to_real_data(raw_df):
    # 真实 Telco 数据流失率 26.5%，模拟数据应落在合理区间
    churn_rate = (raw_df["Churn"] == "Yes").mean()
    assert 0.15 < churn_rate < 0.40


def test_value_ranges(raw_df):
    assert raw_df["tenure"].between(0, 72).all()
    assert raw_df["MonthlyCharges"].between(18.0, 119.0).all()
    assert set(raw_df["Churn"].unique()) <= {"Yes", "No"}


def test_totalcharges_dirty_values_cleaned():
    df = generate_synthetic_data(n_rows=500, seed=42)
    # 原始数据中 tenure=0 的新客 TotalCharges 为空字符串（脏数据）
    cleaned = clean_raw_data(df)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert np.issubdtype(cleaned["TotalCharges"].dtype, np.floating)


def test_no_internet_implies_no_addon(raw_df):
    no_internet = raw_df[raw_df["InternetService"] == "No"]
    for col in INTERNET_ADDONS:
        assert (no_internet[col] == "No internet service").all()


def test_reproducible_with_seed():
    a = generate_synthetic_data(n_rows=300, seed=7)
    b = generate_synthetic_data(n_rows=300, seed=7)
    pd.testing.assert_frame_equal(a, b)

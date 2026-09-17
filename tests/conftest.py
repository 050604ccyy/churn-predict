"""pytest 共享夹具：小样本数据集（2,000 条）加速测试。"""
import pytest
from sklearn.model_selection import train_test_split

from app.config import RANDOM_STATE
from app.data_loader import clean_raw_data, generate_synthetic_data
from app.features import engineer_features, split_X_y


@pytest.fixture(scope="session")
def raw_df():
    """清洗后的原始结构小样本数据。"""
    return clean_raw_data(generate_synthetic_data(n_rows=2000, seed=RANDOM_STATE))


@pytest.fixture(scope="session")
def split_data(raw_df):
    """特征工程 + 分层划分后的训练/测试集。"""
    df = engineer_features(raw_df)
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

"""模型训练与评估测试。"""
import numpy as np
from sklearn.metrics import roc_auc_score

from app.evaluate import evaluate_model
from app.model import (
    HAS_IMBLEARN,
    HAS_XGBOOST,
    build_models,
    compute_scale_pos_weight,
    train_all_models,
)


def test_build_models_contains_baselines():
    models = build_models()
    assert "LogisticRegression" in models
    assert "RandomForest" in models
    if HAS_XGBOOST:
        assert "XGBoost" in models
    if HAS_IMBLEARN:
        assert "LogisticRegression+SMOTE" in models


def test_scale_pos_weight(split_data):
    _, _, y_train, _ = split_data
    spw = compute_scale_pos_weight(y_train)
    # 流失率约 1/4 => 负正比例约 3
    assert 1.5 < spw < 5.0


def test_trained_models_valid_probabilities(split_data):
    X_train, X_test, y_train, _ = split_data
    models = train_all_models(X_train, y_train)
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape == (len(X_test),)
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_models_have_predictive_power(split_data):
    """模拟数据含真实信号，各模型 AUC 应显著优于随机猜测（0.5）。"""
    X_train, X_test, y_train, y_test = split_data
    models = train_all_models(X_train, y_train)
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["roc_auc"] > 0.75, f"{name} AUC={metrics['roc_auc']:.3f}"


def test_smote_only_in_training_pipeline(split_data):
    """SMOTE 作为 Pipeline 步骤存在，重采样不会在 fit 前接触数据。"""
    if not HAS_IMBLEARN:
        return
    X_train, _, y_train, _ = split_data
    models = train_all_models(X_train, y_train)
    smote_model = models["LogisticRegression+SMOTE"]
    assert "smote" in smote_model.named_steps
    # SMOTE 后训练折内正负类样本数应相等
    X_res, y_res = smote_model.named_steps["smote"].fit_resample(
        smote_model.named_steps["preprocess"].transform(X_train), y_train
    )
    assert (y_res == 0).sum() == (y_res == 1).sum()


def test_xgboost_flags():
    # 环境检查：两个可选依赖在本项目标准环境中应当可用
    assert HAS_XGBOOST
    assert HAS_IMBLEARN

"""电信用户流失预测项目。

模块划分：
    data_loader  数据加载与模拟数据生成
    features     特征工程（衍生特征 + 预处理管线）
    model        模型定义与训练（LR / RF / XGBoost / SMOTE 对照）
    evaluate     指标计算与可视化
    pipeline     端到端训练编排
"""

__version__ = "1.0.0"

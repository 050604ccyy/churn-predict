"""全局配置：路径、随机种子与训练超参数，代码中不出现硬编码。"""
from pathlib import Path

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_PATH = DATA_DIR / "telco_churn.csv"
MODEL_PATH = OUTPUT_DIR / "churn_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
RISK_LIST_PATH = OUTPUT_DIR / "high_risk_customers.csv"

ROC_FIG_PATH = OUTPUT_DIR / "roc_curves.png"
PR_FIG_PATH = OUTPUT_DIR / "pr_curves.png"
CM_FIG_PATH = OUTPUT_DIR / "confusion_matrix.png"
FI_FIG_PATH = OUTPUT_DIR / "feature_importance.png"

# ---- 训练参数 ----
RANDOM_STATE = 42
TEST_SIZE = 0.2
SYNTHETIC_N_ROWS = 7043  # 与 Kaggle Telco 数据集行数一致

# ---- 业务参数 ----
TARGET = "Churn"
POS_LABEL = "Yes"
RISK_THRESHOLD = 0.5  # 概率达到该值判定为高流失风险
TOP_RISK_EXPORT = 200

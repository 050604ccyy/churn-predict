# ChurnPredict —— 电信用户流失预测系统

一个覆盖**机器学习完整生命周期**的二分类项目：数据清洗 → EDA 特征工程 → 不平衡处理 → 多模型对比选型 → 业务指标评估 → Streamlit 模型部署演示。基于 7,043 条电信用户订阅数据，预测下月可能流失的用户并输出高风险挽留名单。

## 一、技术栈

| 层次 | 技术 | 用途 |
| --- | --- | --- |
| 数据处理 | pandas / numpy | 数据清洗、类型规范、业务衍生特征 |
| 特征工程 | scikit-learn（ColumnTransformer / Pipeline） | 数值标准化 + 类别独热，随模型一起 fit 防泄漏 |
| 建模 | scikit-learn / XGBoost / imbalanced-learn | 逻辑回归、随机森林、XGBoost、SMOTE 对照 |
| 不平衡处理 | class_weight / scale_pos_weight / SMOTE | 加权与重采样两类方法对比 |
| 评估 | scikit-learn / matplotlib / seaborn | ROC-AUC、PR-AUC、召回率、混淆矩阵、特征重要性 |
| 模型部署 | Streamlit / joblib | 单用户风险预测、高风险名单导出、评估看板 |
| 测试 | pytest | 19 个用例覆盖数据、特征、模型三层 |

> 环境要求：Python 3.9+，本地运行，**不需要** GPU / Docker / 数据库。

## 二、项目结构与建模流程

```
                ┌─────────────────────────────────────────────┐
                │ data/telco_churn.csv（7,043 × 21）             │
                │ 缺失时自动生成同结构、同分布的模拟数据（开箱即跑） │
                └──────────────────────┬──────────────────────┘
                                       ▼
        data_loader.py   清洗：TotalCharges 脏值（空格字符串）转数值回填
                                       ▼
        features.py      特征工程：Yes/No 二值化、服务数/月均费用/单项均价
                          衍生特征、tenure 分档
                                       ▼
                分层 train_test_split（stratify=y，8:2）
                                       ▼
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
 LogisticRegression          RandomForest            XGBoost（scale_pos_weight）
 class_weight=balanced   class_weight=balanced     + LR+SMOTE 重采样对照
        └────────────────────────────┼────────────────────────────┘
                                     ▼
        evaluate.py   ROC/PR 曲线、混淆矩阵、召回率、F1、特征重要性
                                     ▼
                   按 ROC-AUC 选最佳模型 → joblib 落盘
                                     ▼
        demo_app.py   单用户预测 + 高风险名单 CSV + 评估看板
```

## 三、目录结构

```
churn-predict/
├── app/
│   ├── __init__.py
│   ├── config.py            # 路径、随机种子、阈值等全局配置
│   ├── data_loader.py       # 数据加载 + 模拟数据生成 + 脏数据清洗
│   ├── features.py          # 衍生特征 + ColumnTransformer 预处理器
│   ├── model.py             # 四个候选模型定义与训练
│   ├── evaluate.py          # 指标计算与四张可视化图
│   └── pipeline.py          # 端到端编排：训练→评估→落盘→导出名单
├── data/                    # 数据集（首次运行自动生成，可替换为官方 CSV）
├── output/                  # 模型、metrics.json、图表、高风险名单（运行生成）
├── tests/                   # pytest：6 数据 + 7 特征 + 6 模型，共 19 个用例
├── demo_app.py              # Streamlit 演示页面
├── train.py                 # 训练入口：python train.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## 四、快速开始

### 1. 安装依赖

```bash
cd churn-predict
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

首次运行会在 `data/` 自动生成模拟数据，随后完成四模型训练，控制台输出指标对比表，产物全部写入 `output/`。

### 3. 启动演示页面

```bash
streamlit run demo_app.py
```

浏览器自动打开后可使用三个功能：

- **单用户风险预测**：表单录入用户信息，实时输出流失概率、风险等级和针对性挽留建议；
- **高风险用户名单**：全量打分，阈值滑块筛选，一键导出 CSV（模拟真实营销工单）；
- **模型表现**：指标表、ROC/PR 曲线、混淆矩阵、Top20 特征重要性。

### 4. 运行测试

```bash
python -m pytest tests
```

### 5. 切换为 Kaggle 真实数据（可选）

下载 Kaggle [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 数据集，将 `WA_Fn-UseC_-Telco-Customer-Churn.csv` 重命名为 `telco_churn.csv` 放入 `data/` 目录，重新执行 `python train.py` 即可——字段结构与模拟数据完全一致，**代码无需任何改动**。

## 五、数据说明

21 个字段覆盖用户画像、账户、服务三类信息：

- **用户画像**：性别、是否老年、伴侣、家属；
- **服务信息**：在网时长 tenure、电话/多线/DSL/光纤、六项增值服务（网络安全、在线备份、设备保护、技术支持、流媒体电视/电影）；
- **账户信息**：合同类型（月付/一年/两年）、无纸化账单、支付方式、月费、总费用。

标签 `Churn` 正负类占比约 **26.6% : 73.4%**，属于典型的类别不平衡问题。

> 模拟数据并非随机标签：流失标签由包含主效应（月结合同 +1.8、tenure 负相关、光纤、电子支票等）与交互效应（月付×入网初期、光纤×无技术支持、月费 U 型风险）的 logistic 过程生成，流失率与各字段边际分布均对齐真实数据，模型可以学到有效且符合业务直觉的模式。

## 六、核心建模设计

### 1. 特征工程（防泄漏是重点）

- **业务衍生特征**：订购服务总数 `num_services`（衡量用户绑定深度）、月均实际支出、单项服务均价、tenure 五档分组；
- **两阶段防泄漏设计**：衍生特征只使用单行自身信息；标准化/独热封装在 `ColumnTransformer` 内并作为 Pipeline 第一步，**只在训练集 fit**，对测试/预测数据仅 transform；
- **脏数据处理**：真实数据 `TotalCharges` 含 11 条空格字符串，`pd.to_numeric(errors="coerce")` 后用 `tenure × MonthlyCharges` 回填；
- **鲁棒编码**：OneHotEncoder 设置 `handle_unknown="ignore"`，预测期遇到训练时未见的类别值不报错。

### 2. 类别不平衡的两种处理路线对比

| 方法 | 思想 | 落地方式 |
| --- | --- | --- |
| 类别加权 | 提高少数类错分代价 | LR/RF 的 `class_weight`、XGBoost 的 `scale_pos_weight`（训练集负正比例） |
| 过采样 | 合成少数类样本补齐分布 | SMOTE，且置于 imblearn Pipeline 中，**只对训练折重采样**，杜绝验证集信息泄漏 |

### 3. 模型对比结果（7,043 条，8:2 分层划分，随机种子 42）

| 模型 | ROC-AUC | PR-AUC | Accuracy | Precision | **Recall** | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **LogisticRegression（加权）** | **0.870** | **0.728** | 0.779 | 0.562 | **0.773** | 0.651 |
| LogisticRegression + SMOTE | 0.868 | 0.727 | 0.781 | 0.565 | 0.763 | 0.649 |
| XGBoost | 0.867 | 0.712 | 0.777 | 0.560 | 0.765 | 0.646 |
| RandomForest | 0.859 | 0.703 | 0.798 | 0.602 | 0.707 | 0.650 |

**选型结论**：四个模型 ROC-AUC 差距在 1 个百分点左右，XGBoost 并未显著优于逻辑回归——因为该数据集信号以近似线性的加性效应为主、样本量中等、独热后特征维度低。综合**可解释性**（系数可直接用于业务解读与监管解释）、**训练/部署成本**与效果，最终上线逻辑回归。这比"无脑上复杂模型"更接近真实工业界的决策方式。

### 4. 为什么不用准确率而用召回率/PR-AUC？

- 空模型把所有用户预测为"不流失"即可获得 73.4% 准确率，因此 Accuracy 在不平衡场景下具有误导性；
- 业务上漏掉一个流失用户（FN，损失月费与获客成本）远贵于误挽留一个忠诚用户（FP，只损失一张优惠券），因此**召回率优先**；
- PR 曲线比 ROC 曲线对类别不平衡更敏感，PR-AUC 作为补充指标；
- 判定阈值（默认 0.5）可在演示页面滑动调节：降阈值提高召回但增加营销成本，对应真实业务中的预算-覆盖权衡。


## 七、可扩展方向

- `GridSearchCV` 系统化调参并输出学习曲线；
- 用 SHAP 替代系数/重要性做单用户归因，让挽留建议从规则升级为模型解释；
- Streamlit 表单替换为 FastAPI + 前端，模型服务化；
- 漂移监控：对比每月打分分布与训练集 PSI，触发重训练。

"""Streamlit 流失预测演示页面。

启动方式（在 churn-predict 目录下）::

    streamlit run demo_app.py

三个页面：
    1. 单用户风险预测 —— 表单录入用户信息，实时输出流失概率与挽留建议；
    2. 高风险用户名单 —— 全量打分，按阈值筛选，支持导出 CSV；
    3. 模型表现 —— 指标表、ROC/PR 曲线、混淆矩阵、特征重要性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import joblib

from app.config import (
    CM_FIG_PATH,
    DATA_PATH,
    FI_FIG_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PR_FIG_PATH,
    ROC_FIG_PATH,
    RISK_THRESHOLD,
)
from app.data_loader import INTERNET_ADDONS, load_data
from app.features import engineer_features
from app.pipeline import run_training

st.set_page_config(page_title="电信用户流失预测系统", page_icon="📉", layout="wide")


# ---------------------------------------------------------------- 资源缓存
@st.cache_resource(show_spinner="正在加载模型…")
def load_trained_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner="正在加载数据…")
def load_full_dataset():
    return load_data(DATA_PATH)


def train_and_refresh():
    with st.spinner("正在执行完整训练流程（约 10~30 秒）…"):
        run_training()
    st.success("训练完成，模型与图表已更新。")
    load_trained_model.clear()
    load_full_dataset.clear()
    st.rerun()


# ---------------------------------------------------------------- 侧边栏
with st.sidebar:
    st.header("⚙️ 模型控制")
    model = load_trained_model()
    if model is None:
        st.warning("尚未训练模型，请先点击下方按钮。")
        st.button("🚀 一键训练模型", type="primary", on_click=train_and_refresh)
    else:
        st.success("模型已加载，可以开始预测。")
        if st.button("🔄 重新训练"):
            train_and_refresh()
    st.caption("数据为 7,043 条 Telco 用户记录（缺失时自动生成同结构模拟数据）。")

st.title("📉 电信用户流失预测系统")

tab_predict, tab_list, tab_report = st.tabs(
    ["🧑‍💼 单用户风险预测", "📋 高风险用户名单", "📊 模型表现"]
)

# ---------------------------------------------------------------- Tab 1
with tab_predict:
    st.subheader("录入用户信息，评估流失风险")

    if model is None:
        st.info("请先在左侧边栏训练模型。")
    else:
        with st.form("customer_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("性别", ["Male", "Female"])
                senior = st.selectbox("是否老年用户", ["否", "Yes"])
                partner = st.selectbox("是否有伴侣", ["Yes", "No"])
                dependents = st.selectbox("是否有家属", ["Yes", "No"])
            with col2:
                tenure = st.slider("在网时长（月）", 0, 72, 12)
                phone = st.selectbox("电话服务", ["Yes", "No"])
                multiple = st.selectbox(
                    "多条电话线", ["Yes", "No", "No phone service"]
                )
                internet = st.selectbox(
                    "互联网服务", ["Fiber optic", "DSL", "No"]
                )
            with col3:
                contract = st.selectbox(
                    "合同类型", ["Month-to-month", "One year", "Two year"]
                )
                paperless = st.selectbox("无纸化账单", ["Yes", "No"])
                payment = st.selectbox(
                    "支付方式",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly = st.slider("月消费（美元）", 18.25, 118.95, 65.0, 0.5)

            st.markdown("**互联网增值服务**")
            addon_cols = st.columns(3)
            addon_values = {}
            options = ["Yes", "No", "No internet service"]
            for i, addon in enumerate(INTERNET_ADDONS):
                with addon_cols[i % 3]:
                    default = 2 if internet == "No" else 1
                    addon_values[addon] = st.radio(
                        addon, options, index=default, horizontal=True
                    )

            submitted = st.form_submit_button("🔍 评估流失风险", type="primary")

        if submitted:
            row = {
                "customerID": "WEB-PREDICT",
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                **addon_values,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": float(monthly),
                "TotalCharges": float(tenure * monthly),
            }
            X_one = engineer_features(pd.DataFrame([row]))
            prob = float(model.predict_proba(X_one)[0, 1])

            st.markdown("---")
            m1, m2, m3 = st.columns([1, 1, 1.4])
            m1.metric("预测流失概率", f"{prob:.1%}")
            level = "🔴 高风险" if prob >= RISK_THRESHOLD else "🟢 低风险"
            m2.metric("风险等级", level)
            m3.metric("判定阈值", f"{RISK_THRESHOLD:.0%}")

            st.progress(min(prob, 1.0))

            suggestions = []
            if contract == "Month-to-month":
                suggestions.append("月结合同用户：推荐转签一年/两年合约（捆绑优惠）")
            if internet == "Fiber optic" and addon_values.get("TechSupport") != "Yes":
                suggestions.append("光纤高月费用户：附赠技术支持服务包可显著降低流失")
            if payment == "Electronic check":
                suggestions.append("电子支票支付：引导开通自动扣款（信用卡/银行转账）")
            if tenure <= 12:
                suggestions.append("在网不满一年的新客：首年续约优惠、定期满意度回访")
            if addon_values.get("OnlineSecurity") != "Yes" and internet != "No":
                suggestions.append("未开通网络安全服务：安全服务是流失的强保护因素，可试用赠送")

            if suggestions:
                st.markdown("**🎯 系统建议的挽留动作：**")
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.success("该用户画像较健康，维持常规关怀即可。")

# ---------------------------------------------------------------- Tab 2
with tab_list:
    st.subheader("全量用户打分与高风险名单")

    if model is None:
        st.info("请先在左侧边栏训练模型。")
    else:
        df_all = load_full_dataset()
        X_all = engineer_features(df_all)
        scored = df_all.copy()
        scored["churn_prob"] = model.predict_proba(X_all)[:, 1]

        c1, c2, c3 = st.columns(3)
        threshold = c1.slider("风险阈值", 0.1, 0.9, RISK_THRESHOLD, 0.05)
        high = scored[scored["churn_prob"] >= threshold].sort_values(
            "churn_prob", ascending=False
        )
        c2.metric("高风险用户数", f"{len(high):,}")
        c3.metric("占全体用户比例", f"{len(high) / len(scored):.1%}")

        st.bar_chart(
            (scored["churn_prob"] * 10).round() / 10,
            use_container_width=True,
        )
        st.caption("横轴：流失概率分档；纵轴：用户数量。")

        display_cols = [
            "customerID", "gender", "tenure", "Contract", "InternetService",
            "MonthlyCharges", "PaymentMethod", "churn_prob",
        ]
        st.dataframe(
            high[display_cols].style.format({"churn_prob": "{:.2%}"}),
            use_container_width=True,
            height=420,
        )

        csv = high.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 下载高风险名单 CSV",
            csv,
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- Tab 3
with tab_report:
    st.subheader("模型评估报告")

    if not METRICS_PATH.exists():
        st.info("暂无评估结果，请先在左侧边栏训练模型。")
    else:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最佳模型", report["best_model"])
        c2.metric("样本量", f"{report['n_samples']:,}")
        c3.metric("整体流失率", f"{report['churn_rate']:.1%}")
        c4.metric("判定阈值", f"{report['decision_threshold']:.0%}")

        st.markdown("#### 各模型指标对比")
        metrics_df = pd.DataFrame(report["models"]).T
        st.dataframe(
            metrics_df.style.format("{:.4f}").background_gradient(
                cmap="Blues", axis=None
            ),
            use_container_width=True,
        )

        st.markdown("#### ROC / PR 曲线")
        cc1, cc2 = st.columns(2)
        if ROC_FIG_PATH.exists():
            cc1.image(str(ROC_FIG_PATH), caption="ROC 曲线对比")
        if PR_FIG_PATH.exists():
            cc2.image(str(PR_FIG_PATH), caption="Precision-Recall 曲线对比")

        st.markdown("#### 混淆矩阵与特征重要性")
        cc3, cc4 = st.columns(2)
        if CM_FIG_PATH.exists():
            cc3.image(str(CM_FIG_PATH), caption="最佳模型混淆矩阵")
        if FI_FIG_PATH.exists():
            cc4.image(str(FI_FIG_PATH), caption="Top20 特征重要性")

        st.markdown(
            """
            > **指标解读**：业务上漏掉流失用户（FN）比误挽留忠诚用户（FP）
            > 代价更高，因此重点关注**召回率（Recall）**与 **PR-AUC**；
            > 阈值可在"高风险名单"页调节——降低阈值召回更高但营销成本上升，
            > 可结合挽留预算做决策。
            """
        )

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


ORANGE_XGBOOST_PARAMS = {
    # Orange3 Gradient Boosting(xgboost) screenshot values
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "reg_lambda": 0.8,
    "max_depth": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "colsample_bylevel": 1.00,
    "colsample_bynode": 1.00,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


st.set_page_config(page_title="Orange3 XGBoost Streamlit", layout="wide")
st.title("Orange3 XGBoost 분석 앱")
st.caption("Orange3 Gradient Boosting(xgboost) 파라미터 설정값을 동일하게 설정하여 Streamlit + XGBoost으로 재현하여 동일한 값이 나왔습니다.")


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".tab"):
        return pd.read_csv(uploaded_file, sep="\t")
    raise ValueError("CSV, Excel, TAB 파일만 지원합니다.")


def infer_task_type(y: pd.Series) -> str:
    """숫자형 연속값이면 회귀, 문자형/소수 클래스 숫자면 분류로 추정합니다."""
    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique(dropna=True)
        if unique_count <= 20 and unique_count <= max(2, len(y) * 0.05):
            return "classification"
        return "regression"
    return "classification"


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.copy().dropna(subset=[target_col])
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def make_model(task_type: str):
    params = ORANGE_XGBOOST_PARAMS.copy()
    if task_type == "classification":
        return XGBClassifier(**params)

    # 회귀에서는 logloss가 맞지 않으므로 제거합니다.
    params.pop("eval_metric", None)
    return XGBRegressor(**params)


def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def get_feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    """전처리 이후 XGBoost feature importance를 DataFrame으로 반환합니다."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False)


def draw_feature_importance(pipeline: Pipeline, top_n: int = 20):
    importance_df = get_feature_importance_df(pipeline).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
    ax.set_title(f"XGBoost Feature Importance Top {top_n}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    st.pyplot(fig)


def draw_classification_graphs(y_test, y_pred, pipeline: Pipeline):
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    with col2:
        st.write("실제값 / 예측값 분포")
        compare_df = pd.DataFrame(
            {
                "Actual": pd.Series(y_test).astype(str).values,
                "Predicted": pd.Series(y_pred).astype(str).values,
            }
        )
        melted = compare_df.melt(var_name="type", value_name="class")
        fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
        sns.countplot(data=melted, x="class", hue="type", ax=ax_dist)
        ax_dist.set_xlabel("Class")
        ax_dist.set_ylabel("Count")
        ax_dist.tick_params(axis="x", rotation=45)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

    st.write("변수 중요도")
    draw_feature_importance(pipeline)


def draw_regression_graphs(y_test, y_pred, pipeline: Pipeline):
    y_test_series = pd.Series(y_test).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    residuals = y_test_series - y_pred_series

    col1, col2 = st.columns(2)
    with col1:
        st.write("실제값 vs 예측값")
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
        sns.scatterplot(x=y_test_series, y=y_pred_series, ax=ax_scatter)
        min_val = min(y_test_series.min(), y_pred_series.min())
        max_val = max(y_test_series.max(), y_pred_series.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--")
        ax_scatter.set_xlabel("Actual")
        ax_scatter.set_ylabel("Predicted")
        fig_scatter.tight_layout()
        st.pyplot(fig_scatter)

    with col2:
        st.write("잔차 분포")
        fig_resid, ax_resid = plt.subplots(figsize=(6, 5))
        sns.histplot(residuals, kde=True, ax=ax_resid)
        ax_resid.axvline(0, color="red", linestyle="--")
        ax_resid.set_xlabel("Residual = Actual - Predicted")
        fig_resid.tight_layout()
        st.pyplot(fig_resid)

    st.write("변수 중요도")
    draw_feature_importance(pipeline)


with st.sidebar:
    st.header("Orange3 설정값")
    st.write("아래 기본값은 첨부하신 Orange3 화면을 반영했습니다.")

    n_estimators = st.number_input("Number of trees", min_value=1, value=2000, step=100)
    learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.03, step=0.01, format="%.4f")
    reg_lambda = st.number_input("Lambda regularization", min_value=0.0, value=0.8, step=0.1)
    max_depth = st.number_input("Max depth", min_value=1, value=10, step=1)
    subsample = st.number_input("Fraction of training instances", min_value=0.01, max_value=1.0, value=0.90, step=0.05)
    colsample_bytree = st.number_input("Fraction of features for each tree", min_value=0.01, max_value=1.0, value=0.90, step=0.05)
    colsample_bylevel = st.number_input("Fraction of features for each level", min_value=0.01, max_value=1.0, value=1.00, step=0.05)
    colsample_bynode = st.number_input("Fraction of features for each split/node", min_value=0.01, max_value=1.0, value=1.00, step=0.05)

    ORANGE_XGBOOST_PARAMS.update(
        {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "reg_lambda": float(reg_lambda),
            "max_depth": int(max_depth),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "colsample_bylevel": float(colsample_bylevel),
            "colsample_bynode": float(colsample_bynode),
        }
    )

uploaded_file = st.file_uploader("분석할 데이터 파일을 업로드하세요", type=["csv", "xlsx", "xls", "tab"])

if uploaded_file is None:
    st.info("Orange3에서 저장한 전체 테이블 파일(.csv, .xlsx, .tab)을 업로드하면 분석을 시작합니다.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as exc:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {exc}")
    st.stop()

st.subheader("데이터 미리보기")
st.write(f"행: {df.shape[0]:,} / 컬럼: {df.shape[1]:,}")
st.dataframe(df.head(50), use_container_width=True)

if df.empty or df.shape[1] < 2:
    st.error("최소 2개 이상의 컬럼이 있는 데이터가 필요합니다. 입력 변수와 타깃 컬럼이 모두 있어야 합니다.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    target_col = st.selectbox("예측할 타깃 컬럼 선택", options=df.columns)
with col2:
    task_default = infer_task_type(df[target_col])
    task_type = st.selectbox(
        "분석 유형",
        options=["classification", "regression"],
        index=0 if task_default == "classification" else 1,
        format_func=lambda x: "분류" if x == "classification" else "회귀",
    )
with col3:
    test_size = st.slider("테스트 데이터 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

if st.button("XGBoost 학습 실행", type="primary"):
    X, y = split_xy(df, target_col)

    if X.empty or y.empty:
        st.error("타깃 결측값 제거 후 학습 가능한 데이터가 없습니다.")
        st.stop()

    stratify = y if task_type == "classification" and y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

    preprocessor = build_preprocessor(X_train)
    model = make_model(task_type)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    with st.spinner("XGBoost 모델을 학습 중입니다..."):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    st.success("학습이 완료되었습니다.")

    result_df = X_test.copy()
    result_df[f"actual_{target_col}"] = y_test.values
    result_df[f"predicted_{target_col}"] = y_pred

    st.subheader("tabs() 로 탭 나누기")
    tab1, tab2 = st.tabs(["📈 그래프", "📋 데이터"])

    with tab1:
        st.subheader("XGBoost 분석 결과 그래프")
        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Weighted F1", f"{f1:.4f}")
            draw_classification_graphs(y_test, y_pred, pipeline)
        else:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric("R²", f"{r2:.4f}")
            draw_regression_graphs(y_test, y_pred, pipeline)
    with tab2:
        st.subheader("예측 결과 데이터")
        st.dataframe(result_df.head(100), use_container_width=True)
        st.download_button(
            "예측 결과 CSV 다운로드",
            data=make_download_csv(result_df),
            file_name="xgboost_predictions.csv",
            mime="text/csv",
        )

        st.subheader("상세 평가표")
        if task_type == "classification":
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            labels = sorted(pd.Series(y_test).dropna().unique().tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            st.write("Confusion matrix table")
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)

        importance_df = get_feature_importance_df(pipeline)
        st.write("Feature importance table")
        st.dataframe(importance_df, use_container_width=True)

        st.subheader("사용된 XGBoost 파라미터")
        shown_params = ORANGE_XGBOOST_PARAMS.copy()
        if task_type == "regression":
            shown_params.pop("eval_metric", None)
        st.json(shown_params)
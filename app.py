import io
import pickle
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bank Churn XGBoost", page_icon="🏦", layout="wide")

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
DROP_COLUMNS = ["id", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"
CATEGORICAL_COLUMNS = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]
MODEL_PARAMS = {
    "seed": 42,
    "objective": "binary:logistic",
    "gamma": 1.0,
    "learning_rate": 0.1,
    "max_depth": 5,
    "reg_lambda": 10.0,
    "scale_pos_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.5,
    "n_estimators": 400,
}


@dataclass
class TrainedBundle:
    model: xgb.XGBClassifier
    feature_names: list[str]
    validation_metrics: dict[str, float]


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner=False)
def train_from_dataframe(train_df: pd.DataFrame) -> TrainedBundle:
    data = train_df.copy()
    missing_target = TARGET_COLUMN not in data.columns
    if missing_target:
        raise ValueError(f"Training data must contain the target column '{TARGET_COLUMN}'.")

    for column in DROP_COLUMNS:
        if column in data.columns:
            data = data.drop(columns=column)

    y = data[TARGET_COLUMN].astype(int)
    X = data.drop(columns=[TARGET_COLUMN])
    X = pd.get_dummies(X, columns=[column for column in CATEGORICAL_COLUMNS if column in X.columns])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    fit_kwargs = dict(
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    try:
        model.fit(X_train, y_train, early_stopping_rounds=20, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train, **fit_kwargs)

    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_valid, valid_pred),
        "roc_auc": roc_auc_score(y_valid, valid_proba),
        "precision": precision_score(y_valid, valid_pred, zero_division=0),
        "recall": recall_score(y_valid, valid_pred, zero_division=0),
        "f1": f1_score(y_valid, valid_pred, zero_division=0),
    }

    return TrainedBundle(model=model, feature_names=list(X.columns), validation_metrics=metrics)


def preprocess_for_model(input_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    data = input_df.copy()
    for column in DROP_COLUMNS:
        if column in data.columns:
            data = data.drop(columns=column)

    data = pd.get_dummies(data, columns=[column for column in CATEGORICAL_COLUMNS if column in data.columns])
    data = data.reindex(columns=feature_names, fill_value=0)
    return data


def build_single_row_input() -> pd.DataFrame:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)

    credit_score = st.slider("CreditScore", 300, 850, 650)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    num_products = st.selectbox("NumOfProducts", [1, 2, 3, 4], index=0)
    has_cr_card = st.selectbox("HasCrCard", [0, 1], index=1)
    is_active = st.selectbox("IsActiveMember", [0, 1], index=1)
    estimated_salary = st.number_input("EstimatedSalary", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

    return pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )


def render_probability_card(probability: float) -> None:
    percentage = probability * 100
    if percentage >= 70:
        label = "High churn risk"
        color = "#D64545"
    elif percentage >= 40:
        label = "Medium churn risk"
        color = "#D48B1A"
    else:
        label = "Low churn risk"
        color = "#2D8C5B"

    st.markdown(
        f"""
        <div style="padding: 1.2rem; border-radius: 16px; border: 1px solid rgba(0,0,0,0.08); background: linear-gradient(135deg, rgba(18,34,65,0.96), rgba(9,15,31,0.96)); color: white;">
            <div style="font-size: 0.95rem; opacity: 0.8;">Model output</div>
            <div style="font-size: 2.3rem; font-weight: 700; line-height: 1.05; margin-top: 0.35rem; color: {color};">{percentage:.2f}%</div>
            <div style="font-size: 1rem; margin-top: 0.25rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #F8FAFC 0%, #F0F4F9 100%); }
        [data-testid="stSidebar"] * { color: #1F2937; }
        .hero {
            padding: 1.6rem 1.7rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #122241 0%, #0F172A 55%, #1F4E79 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2rem; }
        .hero p { margin-top: 0.5rem; opacity: 0.9; }
        .stat-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Bank Churn Predictor</h1>
        <p>Streamlit app built from the XGBoost churn model in your notebook. Train from CSV or load a saved model, then score one customer or a whole file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model setup")
    train_file = st.file_uploader("Upload training CSV with Exited", type=["csv"])
    model_file = st.file_uploader("Or upload a saved model (.pkl/.joblib)", type=["pkl", "joblib"])
    st.caption("Expected training columns: id, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited")

    if st.button("Clear cached model"):
        st.cache_resource.clear()
        st.session_state.pop("bundle", None)
        st.rerun()

bundle: Optional[TrainedBundle] = None
if "bundle" in st.session_state:
    bundle = st.session_state["bundle"]
elif model_file is not None:
    try:
        loaded_model = joblib.load(model_file)
    except Exception:
        loaded_model = pickle.load(model_file)
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    if feature_names is None:
        st.error("Model file loaded, but feature names were not found. Use a model saved after fitting on a DataFrame.")
    else:
        bundle = TrainedBundle(model=loaded_model, feature_names=list(feature_names), validation_metrics={})
        st.session_state["bundle"] = bundle
elif train_file is not None:
    try:
        train_df = load_csv(train_file)
        bundle = train_from_dataframe(train_df)
        st.session_state["bundle"] = bundle
    except Exception as exc:
        st.error(f"Training failed: {exc}")

col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.subheader("1. Predict one customer")
    st.write("Fill the form below. The app will automatically apply the same feature engineering as the notebook.")
    input_df = build_single_row_input()

    predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

    if predict_clicked:
        if bundle is None:
            st.warning("Upload a training CSV or a saved model first.")
        else:
            processed = preprocess_for_model(input_df, bundle.feature_names)
            probability = float(bundle.model.predict_proba(processed)[:, 1][0])
            prediction = int(probability >= 0.5)
            render_probability_card(probability)
            st.write(f"Predicted class: {'Exited' if prediction == 1 else 'Not Exited'}")
            st.progress(min(max(probability, 0.0), 1.0))

            result_df = input_df.copy()
            result_df["PredictedExitedProbability"] = probability
            result_df["PredictedClass"] = prediction
            st.dataframe(result_df, use_container_width=True)

with col_right:
    st.subheader("2. Model status")
    if bundle is None:
        st.info("No model is active yet. Load a saved model or upload training data to fit one inside the app.")
    else:
        st.success(f"Model ready with {len(bundle.feature_names)} input features after encoding.")
        metrics = bundle.validation_metrics
        metric_cols = st.columns(2)
        metric_cols[0].metric("ROC AUC", f"{metrics.get('roc_auc', 0.0):.4f}" if metrics else "N/A")
        metric_cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}" if metrics else "N/A")
        if metrics:
            summary_cols = st.columns(3)
            summary_cols[0].metric("Precision", f"{metrics['precision']:.4f}")
            summary_cols[1].metric("Recall", f"{metrics['recall']:.4f}")
            summary_cols[2].metric("F1", f"{metrics['f1']:.4f}")

        st.subheader("3. Batch scoring")
        batch_file = st.file_uploader("Upload a CSV to score", type=["csv"], key="batch_file")
        if batch_file is not None:
            batch_df = load_csv(batch_file)
            scored = preprocess_for_model(batch_df, bundle.feature_names)
            probs = bundle.model.predict_proba(scored)[:, 1]
            scored_output = batch_df.copy()
            scored_output["ExitedProbability"] = probs
            scored_output["PredictedClass"] = (probs >= 0.5).astype(int)
            st.dataframe(scored_output.head(20), use_container_width=True)
            csv_data = scored_output.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored CSV",
                data=csv_data,
                file_name="bank_churn_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()
st.subheader("How this app maps to the notebook")
notes = st.columns(3)
notes[0].markdown("- Drops id, CustomerId, Surname\n- Keeps the 10 business features\n- One-hot encodes the categorical columns")
notes[1].markdown("- Uses XGBoost with the tuned hyperparameters\n- Scores via `predict_proba`\n- Threshold is 0.5 by default")
notes[2].markdown("- Upload train.csv to fit inside app\n- Or load a pickled model\n- Batch scoring is supported")
